/* Copyright (c) 2021-2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/graph/generic_layout_optimizer/generic_layout_optimizer.h"

#include <cstring>
#include <memory>
#include <utility>

#include "itex/core/graph/remapper/remapper.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/tensor_id.h"

namespace itex {
namespace graph {

inline bool GetValueAttrFromConstInputNode(
    const utils::MutableNodeView& node,
    const std::function<bool(const NodeDef&)>& predicate, int index,
    Tensor* tensor) {
  if (!predicate(*node.node())) {
    return false;
  }
  const auto& regular_fanin = node.GetRegularFanin(index);
  auto* regular_fanin_node = regular_fanin.node_view();
  if (!IsConstant(*regular_fanin_node->node())) {
    return false;
  }
  const auto* value_attr = regular_fanin_node->GetAttr("value");
  if (value_attr == nullptr || value_attr->tensor().dtype() != DT_INT32) {
    return false;
  }
  if (!tensor->FromProto(value_attr->tensor())) {
    return false;
  }

  return true;
}

inline bool IsCancellableConstPermTransposeNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
  Tensor fanout_tensor;
  if (!GetValueAttrFromConstInputNode(fanout_transpose, IsTranspose, 1,
                                      &fanout_tensor)) {
    return false;
  }
  Tensor fanin_tensor;
  if (!GetValueAttrFromConstInputNode(fanin_transpose, IsTranspose, 1,
                                      &fanin_tensor)) {
    return false;
  }
  if (fanout_tensor.NumElements() != fanin_tensor.NumElements()) {
    return false;
  }

  // Using dst->src to permute on src->dst will result in
  // seq(0, ..., num_elements - 1) if they are cancellable.
  const auto& fanout_tensor_data = fanout_tensor.unaligned_flat<int32>();
  const auto& fanin_tensor_data = fanin_tensor.unaligned_flat<int32>();
  const int num_elements = fanout_tensor.NumElements();
  for (int i = 0; i < num_elements; ++i) {
    if (fanout_tensor_data(fanin_tensor_data(i)) != i) {
      return false;
    }
  }
  return true;
}

// From: Transpose[NCHWD->NDHWC] -> X -> Transpose[NDHWC->NCHWD]
// To:   newX[NCHWD]
inline Status EraseCancellableNodesAroundContraction(
    TransposeContext* context) {
  ITEX_VLOG(3) << "Start to run EraseCancellableNodesAroundContraction pass";
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();

  absl::flat_hash_set<utils::MutableNodeView*> cancelled_transposes;

  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    // Transpose node after Contraction.
    auto* transpose_after = graph_view->GetNode(i);
    if (!IsTranspose(*transpose_after->node())) continue;

    NodeDef* transpose_after_def = transpose_after->node();
    if (!NodeIsOnGpu(transpose_after_def)) {
      continue;
    }

    if (context->nodes_to_preserve.count(transpose_after_def->name()) > 0)
      continue;

    // This transpose was already cancelled in previous loop iteration.
    if (cancelled_transposes.contains(transpose_after)) continue;

    const auto valid_transpose_perm =
        [&](const utils::MutableNodeView& transpose) -> bool {
      auto* const_nodeview = transpose.GetRegularFanin(1).node_view();
      NodeDef* const_nodedef = const_nodeview->node();
      Tensor shape_tensor;
      std::vector<int32_t> shape_value;
      if (!IsConstant(*const_nodedef)) {
        return false;
      }
      TensorProto tensor_proto = const_nodedef->attr().at("value").tensor();
      if (!shape_tensor.FromProto(tensor_proto)) {
        return false;
      }
      for (int i = 0; i < shape_tensor.NumElements(); ++i) {
        shape_value.push_back(shape_tensor.flat<int32_t>()(i));
      }
      int32_t perm_dim_1 = 0;
      perm_dim_1 = shape_value[1];
      return perm_dim_1 == 4 ? true : false;
    };
    if (!valid_transpose_perm(*transpose_after)) {
      continue;
    }

    // Contraction node.
    const auto& transpose_after_fanin = transpose_after->GetRegularFanin(0);
    auto* contraction_view = transpose_after_fanin.node_view();
    if (!IsConv3D(*contraction_view->node())) continue;
    if (!HaveSameDataType(transpose_after->node(), contraction_view->node()) ||
        !HasAtMostOneFanoutAtPort0(*contraction_view))
      return Status(TF_INVALID_ARGUMENT, "Invalid Value");

    // Transpose node before Contraction.
    const auto& contraction_fanin_0 = contraction_view->GetRegularFanin(0);
    auto* transpose_before = contraction_fanin_0.node_view();
    if (!IsTranspose(*transpose_before->node())) continue;
    // Transpose before output used once by the Pad node.
    if (transpose_before->NumRegularFanouts() != 1) continue;

    // Transposes are cancellable.
    if (!IsCancellableConstPermTransposeNodePair(*transpose_after,
                                                 *transpose_before))
      continue;

    // Pad output might be used multiple times by different Transpose nodes. If
    // they all have identical permutation, we can cancel all of them.
    std::vector<utils::MutableNodeView*> contraction_fanout_transposes;
    contraction_fanout_transposes.emplace_back(transpose_after);

    string old_data_format;
    std::vector<int64> dilations;
    std::vector<int64> strides;
    TF_ABORT_IF_ERROR(GetNodeAttr(*contraction_view->node(), "data_format",
                                  &old_data_format));
    TF_ABORT_IF_ERROR(
        GetNodeAttr(*contraction_view->node(), "dilations", &dilations));
    TF_ABORT_IF_ERROR(
        GetNodeAttr(*contraction_view->node(), "strides", &strides));

    string new_data_format;
    if ("NDHWC" == old_data_format) {
      new_data_format = "NCDHW";
    } else {
      return Status(TF_INVALID_ARGUMENT, "Unsupported data format");
    }
    std::swap(strides[1], strides[4]);
    std::swap(dilations[1], dilations[4]);

    ITEX_VLOG(3) << "Cancel Transpose nodes around Conv:"
                 << " transpose_before=" << transpose_before->node()->name()
                 << " Conv=" << contraction_view->node()->name()
                 << " transpose_after=" << transpose_after->node()->name();

    NodeDef* contraction_node = contraction_view->node();
    auto* contraction_attr = contraction_node->mutable_attr();
    SetAttrValue(new_data_format, &(*contraction_attr)["data_format"]);
    SetAttrValue(strides, &(*contraction_attr)["strides"]);
    SetAttrValue(dilations, &(*contraction_attr)["dilations"]);
    // Transform Transpose nodes into Identity nodes.
    const auto transpose_to_identity =
        [&cancelled_transposes,
         &mutation](utils::MutableNodeView* transpose) -> void {
      mutation->UpdateNodeOp(transpose, "Identity");
      mutation->RemoveNodeAttr(transpose, "Tperm");
      mutation->RemoveRegularFanin(transpose, 1);
      cancelled_transposes.insert(transpose);
    };

    transpose_to_identity(transpose_before);

    absl::c_for_each(contraction_fanout_transposes, transpose_to_identity);
  }
  return mutation->Apply();
}

inline Status EraseCancellableIdenityNodes(TransposeContext* context) {
  ITEX_VLOG(3) << "Start to run EraseCancellableIdenityNodes pass.";
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();

  absl::flat_hash_set<utils::MutableNodeView*> cancelled_idenity;

  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    auto* idenity_node = graph_view->GetNode(i);
    if (!IsIdentity(*idenity_node->node())) continue;

    NodeDef* identity_node_def = idenity_node->node();
    if (context->nodes_to_preserve.count(identity_node_def->name()) > 0)
      continue;
    // This node was already cancelled in previous loop iteration.
    if (cancelled_idenity.contains(idenity_node)) continue;

    if (idenity_node->NumRegularFanouts() != 1 &&
        idenity_node->NumRegularFanins() != 1)
      continue;

    const auto& fanin_to_forward = idenity_node->GetRegularFanin(0);
    TensorId fanin_id_to_forward(fanin_to_forward.node_view()->GetName(),
                                 fanin_to_forward.index());

    for (const auto& regular_fanout : idenity_node->GetRegularFanout(0)) {
      mutation->AddOrUpdateRegularFanin(regular_fanout.node_view(),
                                        regular_fanout.index(),
                                        fanin_id_to_forward);
    }
    mutation->RemoveNode(idenity_node);
  }
  return mutation->Apply();
}

Status TransposeContext::InitializeTransposeContext(bool assume_valid_feeds,
                                                    const GrapplerItem& item,
                                                    const GraphDef& graph_def,
                                                    TransposeContext* context) {
  // DCHECK(context != nullptr);
  context->graph = graph_def;
  context->graph_properties = std::make_unique<GraphProperties>(item);
  TF_RETURN_IF_ERROR(
      context->graph_properties->InferStatically(assume_valid_feeds));
  Status status;
  context->graph_view =
      std::make_unique<utils::MutableGraphView>(&context->graph, &status);
  TF_RETURN_IF_ERROR(status);
  context->num_nodes = context->graph.node_size();
  const auto& nodes_to_preserve = item.NodesToPreserve();
  context->nodes_to_preserve = absl::flat_hash_set<string>(
      nodes_to_preserve.begin(), nodes_to_preserve.end());
  ITEX_VLOG(2) << "TransposeContext is initialized.";
  return OkStatus();
}

Status GenericLayoutOptimizer::Optimize(const char* device_name,
                                        const GrapplerItem& item,
                                        const GraphDef& graph_def,
                                        GraphDef* optimized_graph) {
  TransposeContext trans_context;
  // needs to be checked
  const bool is_aggressive = false;
  TF_RETURN_IF_ERROR(TransposeContext::InitializeTransposeContext(
      /*assume_valid_feeds=*/is_aggressive, item, graph_def, &trans_context));

  TF_RETURN_IF_ERROR(EraseCancellableNodesAroundContraction(&trans_context));
  // TF_RETURN_IF_ERROR(EraseCancellableIdenityNodes(&trans_context));
  TF_RETURN_IF_ERROR(
      trans_context.graph_view->SortTopologically(/*ignore_cycles=*/false, {}));

  *optimized_graph = trans_context.graph;
  return OkStatus();
}

}  // namespace graph
}  // namespace itex
