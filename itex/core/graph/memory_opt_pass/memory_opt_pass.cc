/* Copyright (c) 2022 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "itex/core/graph/memory_opt_pass/memory_opt_pass.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/layout_utils.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/attr_value_util.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace graph {

namespace {
// Auxiliary information for Inplace Inference
std::vector<SearchInfo> sinfo;
}  // namespace

// Forwarding from input:0 to output:0
const auto regular_inplace_rule = gtl::FlatSet<string>{
    "_ITEXSoftmax",          "_ITEXInstanceNorm",     "_ITEXFusedInstanceNorm",
    "_ITEXMklLayerNorm",     "_ITEXLayerNorm",        "_ITEXFusedBatchNorm",
    "_ITEXFusedBatchNormV2", "_ITEXFusedBatchNormV3", "_ITEXFusedBatchNormEx"};

const auto add_inplace_rule = gtl::FlatSet<string>{
    "_ITEXFusedConv2DWithSum", "_ITEXFusedAccMatMulWithSum",
    "_ITEXFusedMatMulWithSum", "_OneDnnFusedConv2D", "_OneDnnFusedMatMul"};

const auto onednngraph_inplace_rule =
    gtl::FlatSet<string>{"_OneDnnGraph", "OneDnnGraph"};

static constexpr int MAX_LLGA_SEARCH_NODES = 50;

std::vector<int> GetCandidateForwardPort(const MutableNodeView* node_view) {
  const auto* node_def = node_view->node();

  if (regular_inplace_rule.count(node_def->op())) return {0};

  if (add_inplace_rule.count(node_def->op())) {
    // TODO(yifeng): Remove this work-around after binary add is ready.

    if (node_def->op() == "_ITEXFusedAccMatMulWithSum") {
      if (GetDataTypeFromAttr(*node_def, "Tout") !=
          GetDataTypeFromAttr(*node_def, "Tpost")) {
        return {};
      }
    }

    int add_tensor_port = 2;
    for (const string& fused_op : node_def->attr().at("fused_ops").list().s()) {
      if (fused_op == "BiasAdd") ++add_tensor_port;
      if (fused_op == "Add") return {add_tensor_port};
    }
  }

  if (onednngraph_inplace_rule.count(node_def->op())) {
    // We will check all input tensors inplace status for OneDnnGraph op
    int input_size = node_view->GetRegularFanins().size();

    if (IsOneDnnLayoutDependentOp(node_def->op())) {
      // For block layout op, only data tensors can be forwarded
      input_size /= 2;
    }

    std::vector<int> index_vector(input_size);
    std::iota(index_vector.begin(), index_vector.end(), 0);
    return index_vector;
  }
  return {};
}

bool IsInPreserveSet(const MemoryOptContext* ctx, const NodeDef* node) {
  return ctx->nodes_to_preserve.count(node->name()) > 0;
}

bool IsOnSameDevice(const MutableNodeView* node_view_x,
                    const MutableNodeView* node_view_y) {
  return node_view_x->node()->device() == node_view_y->node()->device();
}

int GetOutPort(const MutableNodeView* cur_node_view,
               const MutableNodeView* tgt_node_view) {
  for (int id = 0; id < tgt_node_view->GetRegularFanouts().size(); ++id)
    for (auto node_out : tgt_node_view->GetRegularFanout(id))
      if (cur_node_view == node_out.node_view()) return id;

  return -1;
}

int GetStaticRefCount(const MutableNodeView* node_view,
                      const int forward_port) {
  const auto* tgt_node_view =
      node_view->GetRegularFanin(forward_port).node_view();

  int out_port = GetOutPort(node_view, tgt_node_view);

  ITEX_CHECK(out_port >= 0);

  return tgt_node_view->GetRegularFanout(out_port).size();
}

// Suppose that the current node must have Add Fusion.
bool IsSafeForwarding(const MutableNodeView* node_view,
                      const int forward_port) {
  if (forward_port < 0) return false;

  const auto* tgt_node_view =
      node_view->GetRegularFanin(forward_port).node_view();

  int out_port = GetOutPort(node_view, tgt_node_view);

  const auto& out_port_node_vec = tgt_node_view->GetRegularFanout(out_port);

  // Forwarding is safe due to AddInput with only one fanout.
  if (out_port_node_vec.size() == 1) return true;

  // AddInput with more than two fanouts is not supported in this work-around.
  if (out_port_node_vec.size() > 2) return false;

  // Get the other consumer of the buffer to be forwarded.
  const auto* ref_node_view = (out_port_node_vec[0].node_view() != node_view)
                                  ? out_port_node_vec[0].node_view()
                                  : out_port_node_vec[1].node_view();

  // Prevent forwarding when AddInput is the same as the input to the fused op.
  //      AddInput
  //        /  |
  //       /   |
  //    Conv   |
  //       \   |
  //        \  |
  //         Add
  if (ref_node_view == node_view) return false;

  // For safe forwarding, contraction should have a larger topological index
  // than precursors.
  if (node_view->node_index() < ref_node_view->node_index()) return false;

  // To simplify this work-around, BFS is regulated by the number of nodes
  // `remain` to be searched rather than depth. `remain = 10` is enough for
  // ResNet50.
  int remain = 10;
  std::queue<const utils::MutableNodeView*> queue;

  if (IsAnyOneDnnGraph(*node_view->node())) {
    // LLGA op forward tensor can from any input tensor
    remain = MAX_LLGA_SEARCH_NODES;
    for (int i = 0; i < node_view->NumRegularFanins(); ++i) {
      queue.push(node_view->GetRegularFanin(i).node_view());
    }
  } else {
    // Start BFS from input:0 due to the hypothesis of add fusion.
    queue.push(node_view->GetRegularFanin(0).node_view());
  }

  while (!queue.empty() && remain) {
    --remain;
    const auto* node = queue.front();
    queue.pop();

    if (node == ref_node_view) return true;

    for (const auto& in_node : node->GetRegularFanins())
      queue.push(in_node.node_view());
  }

  return false;
}

void CheckDependence(MemoryOptContext* ctx, const MutableNodeView* node_view,
                     const int forward_port) {
  static std::set<std::string> contraction_nodes = {
      "Conv2D",
      "Conv3D",
      "DepthwiseConv2dNative",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "Conv3DBackpropFilterV2",
      "Conv3DBackpropInputV2",
      "DepthwiseConv2dNativeBackpropFilter",
      "DepthwiseConv2dNativeBackpropInput",
      "MatMul",
      "BatchMatMulV2"};

  const int node_index = node_view->node_index();

  // TODO(yifeng): Remove this work-around after binary add is ready.
  if (add_inplace_rule.count(node_view->node()->op())) {
    if (IsSafeForwarding(node_view, forward_port)) {
      auto* new_attr = node_view->node()->mutable_attr();

      SetAttrValue(true, &(*new_attr)["inplace_sum"]);
      sinfo[node_index].is_inplace = true;
    }
    return;
  }

  if (onednngraph_inplace_rule.count(node_view->node()->op())) {
    if (IsSafeForwarding(node_view, forward_port)) {
      auto* new_attr = node_view->node()->mutable_attr();
      bool has_contraction_node = false;

      // TODO(itex): relax the restrction here to allow non-contraction
      // inplace
      for (auto op : node_view->node()->attr().at("framework_ops").list().s()) {
        if (contraction_nodes.find(op) != contraction_nodes.end()) {
          has_contraction_node = true;
          break;
        }
      }

      if (!has_contraction_node) return;

      ITEX_VLOG(2) << "Find LLGA inplace node: " << node_view->node()->name()
                   << " forward input port: " << forward_port;

      auto candidate_inplace_input_edge =
          node_view->node()
              ->attr()
              .at("candidate_inplace_input_edge")
              .list()
              .b();
      candidate_inplace_input_edge[forward_port] = true;

      SetAttrValue(candidate_inplace_input_edge,
                   &(*new_attr)["candidate_inplace_input_edge"]);
      sinfo[node_index].is_inplace = true;
    }
    return;
  }

  int ref_count = GetStaticRefCount(node_view, forward_port);

  // Just for exception
  if (ref_count < 1) return;

  // Safe forwarding
  if (ref_count == 1) {
    sinfo[node_index].is_inplace = true;
    auto* new_attr = node_view->node()->mutable_attr();
    SetAttrValue(true, &(*new_attr)["is_inplace"]);
    return;
  }

  // Unsafe forwarding
  // TODO(yifeng): Count controlled fanins
  // and adjust condition for block format.
  if (node_view->NumRegularFanins() <= 1) {
    sinfo[node_index].is_inplace = false;
    return;
  }

  // TODO(yifeng): Create queries for ref_count > 1 && in_degree > 1
}

void DetectUnvisitedNode(MemoryOptContext* ctx,
                         const MutableNodeView* node_view) {
  const int node_index = node_view->node_index();

  // The flag is_visited is always equal to false here due to the pruning,
  // but is_visited here may be true in the future.
  if (sinfo[node_index].is_visited) return;

  // Try to get the port of input tensor may be forwarded with explicit rules.
  // TODO(yifeng): Introduce shape inference in the future.
  std::vector<int> forward_ports = GetCandidateForwardPort(node_view);

  for (auto forward_port : forward_ports) {
    const auto* tgt_node_view =
        node_view->GetRegularFanin(forward_port).node_view();
    const auto* tgt_node_def = tgt_node_view->node();

    // Const and fetch nodes should not be forwarded.
    if (IsInPreserveSet(ctx, tgt_node_def) || IsAnyConst(*tgt_node_def))
      continue;

    // Current node and target node must be on the same device.
    if (!IsOnSameDevice(node_view, tgt_node_view)) continue;

    CheckDependence(ctx, node_view, forward_port);
  }
}

void InplaceInference(MemoryOptContext* ctx, const MutableNodeView* node_view) {
  const int node_index = node_view->node_index();

  // Execute pruning if visited
  if (sinfo[node_index].is_visited) {
    return;
  }

  // TODO(yifeng): Consider control edges in dependence check.
  // Ignore Control Fanins and skip nodes have Control Fanouts
  if (node_view->NumControlledFanouts() > 0) return;

  // TODO(yifeng): Enable ExecuteQueries if necessary in the future.
  // ExecuteQueries(node_view);

  DetectUnvisitedNode(ctx, node_view);

  // Set as visited and instack
  sinfo[node_index].is_visited = true;
  sinfo[node_index].is_instack = true;

  // Visit precursor nodes
  for (const auto& in_node : node_view->GetRegularFanins())
    InplaceInference(ctx, in_node.node_view());

  sinfo[node_index].is_instack = false;
}

void StaticInplaceOpt(MemoryOptContext* ctx, const char* device_name) {
  // Skip nodes that were invalidated
  int num_nodes = ctx->graph_view.graph()->node_size();

  sinfo.reserve(num_nodes);
  for (int node_index = 0; node_index < num_nodes; ++node_index) {
    sinfo[node_index].is_inplace = false;
    sinfo[node_index].is_visited = false;
    sinfo[node_index].is_instack = false;
  }

  ITEX_VLOG(1) << "MemoryOptPass: Start to rewrite nodes.";

  for (int node_index = num_nodes - 1; node_index >= 0; --node_index) {
    const auto* node_view = ctx->graph_view.GetNode(node_index);
    const auto* node_def = node_view->node();

    // Start new DFS form an unvisited node only
    if (sinfo[node_index].is_visited) continue;

    // Check if node can run on current optimizer device.
    if (!NodeIsOnDevice(device_name, node_def)) continue;

    InplaceInference(ctx, node_view);
  }
}

void WeightCacheOpt(MemoryOptContext* ctx) {
  int num_nodes = ctx->graph_view.graph()->node_size();

  for (int node_index = num_nodes - 1; node_index >= 0; --node_index) {
    const auto* node_view = ctx->graph_view.GetNode(node_index);

    CheckConstFilter(node_view, ctx->nodes_to_preserve);
  }
}

Status RunMemoryOptPass(OptimizerContext* opt_ctx, const GrapplerItem& item,
                        const GraphDef& graph_def, GraphDef* optimized_graph) {
  Status status;
  GraphDef mutable_graph_def = graph_def;
  MemoryOptContext ctx(item, &mutable_graph_def, &status);

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_ABORT_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  StaticInplaceOpt(&ctx, opt_ctx->device_name);

  WeightCacheOpt(&ctx);

  // Introduce more optimization if needed.

  *optimized_graph = std::move(mutable_graph_def);
  return Status::OK();
}

}  // namespace graph
}  // namespace itex
