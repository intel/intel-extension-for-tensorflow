/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/graph/auto_mixed_precision/auto_mixed_precision.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "itex/core/graph/auto_mixed_precision/auto_mixed_precision_lists.h"
#include "itex/core/graph/graph_view/mutable_graph_view.h"
#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/node_type_attr_map.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/symbolic_shapes.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/cpu_info.h"
#include "itex/core/utils/device_name_utils.h"
#include "itex/core/utils/env_time.h"
#include "itex/core/utils/function.h"
#include "itex/core/utils/op_def_util.h"
#include "itex/core/utils/path.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace graph {
namespace {

// class Cluster;
const char kSuffix[] = "AutoMixedPrecision";
const char kCastToFp16[] = "CastToFp16";
const char kCastToBf16[] = "CastToBf16";
const char kCastToFp32[] = "CastToFp32";

// Get AutoMixedPrecision Mode through device_name.
Status GetAutoMixedPrecisionMode(const char* device_name,
                                 AutoMixedPrecisionMode* model) {
  string mode_type;
  string mode_type_env = "ITEX_AUTO_MIXED_PRECISION_DATA_TYPE";

  auto cfg_ = itex::itex_get_config();
  // For CPU and GPU, the default data type is bfloat16.
  if (cfg_.graph_options().auto_mixed_precision_options().data_type()) {
    mode_type = "BFLOAT16";
    if (cfg_.graph_options().auto_mixed_precision_options().data_type() ==
        itex::BFLOAT16) {
      mode_type = "BFLOAT16";
    } else if (cfg_.graph_options()
                   .auto_mixed_precision_options()
                   .data_type() == itex::FLOAT16) {
      mode_type = "FLOAT16";
    } else {
      return errors::InvalidArgument(
          "`auto_mixed_precision_options.data_type` should be set BFLOAT16 or "
          "FLOAT16.");
    }
  } else {
    ITEX_CHECK_OK(ReadStringFromEnvVar(
        mode_type_env, (device_name == DEVICE_CPU) ? "BFLOAT16" : "FLOAT16",
        &mode_type));
    mode_type = absl::AsciiStrToUpper(mode_type);
  }

  // Set AutoMixedPrecision mode on CPU device. CPU support BF16 and FP16.
  if (device_name == DEVICE_CPU) {
    if (mode_type == "FLOAT16") {
      if (port::HasCpuFP16Support()) {
        *model = AutoMixedPrecisionMode::CPU_FLOAT16;
      } else {
        return errors::InvalidArgument(
            "Auto Mixed Precision data type should be set BFLOAT16, "
            "Because user's CPU only support bfloat16 data type.");
      }
    } else if (mode_type == "BFLOAT16") {
      *model = AutoMixedPrecisionMode::CPU_BFLOAT16;
    } else {
      return errors::InvalidArgument(
          "Auto Mixed Precision data type should be set BFLOAT16 or "
          "FLOAT16.");
    }
  }
  // Set AutoMixedPrecision mode on GPU device. GPUs support BF16 and FP16.
  if (device_name == DEVICE_XPU) {
    if (mode_type == "FLOAT16") {
      *model = AutoMixedPrecisionMode::GPU_FLOAT16;
    } else if (mode_type == "BFLOAT16") {
      mode_type = "BFLOAT16";
      *model = AutoMixedPrecisionMode::GPU_BFLOAT16;
    } else {
      return errors::InvalidArgument(
          "Auto Mixed Precision data type should be set BFLOAT16 or "
          "FLOAT16.");
    }
  }
  ITEX_LOG(INFO) << "Run advanced auto mixed precision datatype " << mode_type
                 << " on " << device_name;
  return Status::OK();
}

// Sets the data type of the given type attribute. Returns false if the type
// attribute is invalid, otherwise true.
bool SetDataType(NodeDef* node, const TypeAttrId& type_attr, DataType type) {
  if (type_attr.attr_name.empty() || !node->attr().count(type_attr.attr_name)) {
    return false;
  }
  AttrValue& attr_value = node->mutable_attr()->at(type_attr.attr_name);
  if (type_attr.type_index == TypeAttrId::kSingleType) {
    attr_value.set_type(type);
  } else {
    if (type_attr.type_index < 0 ||
        type_attr.type_index >= attr_value.list().type_size()) {
      return false;
    }
    attr_value.mutable_list()->set_type(type_attr.type_index, type);
  }
  return true;
}

struct NodeTypeId {
  NodeTypeId(const NodeDef* _node, const TypeAttrId& _type_attr)
      : node(_node), type_attr(_type_attr) {}

  const NodeDef* node;
  TypeAttrId type_attr;

  bool operator==(const NodeTypeId& other) const {
    return node == other.node && type_attr == other.type_attr;
  }

  template <typename H>
  friend H AbslHashValue(H h, const NodeTypeId& nt) {
    return H::combine(std::move(h), nt.node, nt.type_attr);
  }
};

struct NodeTypeIdEdge {
  NodeTypeIdEdge(const NodeTypeId& _src, const NodeTypeId& _dst)
      : src(_src), dst(_dst) {}
  NodeTypeId src;
  NodeTypeId dst;
};

// TODO(itex): Investigate whether the existing GraphTopologyView can be
// used instead of this modified version.
// This is just like GraphTopologyView but with (NodeDef, TypeAttrId) pairs as
// the vertices instead of just NodeDef.
// For example, if node A has output A:0 with TypeAttrId 'T', and node B has
// input B:0 with TypeAttrId 'U', and input B:0 connects to output A:0, there
// will be an edge from (A, T) to (B, U).
class GraphTypeTopologyView {
 public:
  GraphTypeTopologyView() = default;
  explicit GraphTypeTopologyView(bool skip_invalid_edges)
      : skip_invalid_edges_(skip_invalid_edges) {}

  // Initialize graph topology view from the graph. It's possible to pass
  // additional edges that do not exist in a graph, but must be respected when
  // computing graph topology. Example: Tensorflow runtime allows concurrent
  // execution of dequeue/enqueue ops from the same queue resource, but we might
  // want to enforce ordering between them for the purpose of graph analysis.
  Status InitializeFromGraph(const GraphDef& graph,
                             const NodeTypeAttrMap& node_type_map);

  Status AddEphemeralEdges(absl::Span<const NodeTypeIdEdge> ephemeral_edges);

  bool is_initialized() const { return graph_ != nullptr; }
  int num_nodes() const { return num_nodes_; }
  const GraphDef* graph() const { return graph_; }

  //  // Returns true iff the node exists in the underlying graph.
  //  bool HasNode(absl::string_view node_name, const TypeAttrId& type_attr)
  //  const;

  // Finds a node by name or returns `nullptr` if it's not in the graph.
  const NodeTypeId* GetNode(absl::string_view node_name,
                            const TypeAttrId& type_attr) const;
  // Returns a node corresponding to the given node index.
  const NodeTypeId* GetNode(int node_idx) const;

  // Returns a node index for the given node name, if the name exists in the
  // underlying graph. Otherwise returns empty optional.
  const absl::optional<int> GetNodeIndex(absl::string_view node_name,
                                         const TypeAttrId& type_attr) const;
  // Returns a node index for the given node, if the node belongs to the
  // underlying graph. Otherwise returns empty optional.
  const absl::optional<int> GetNodeIndex(const NodeTypeId& node) const;

  // Returns all the node indexes that are in the direct fanin of the given
  // node. If the `node_idx` is outside of [0, num_nodes_) returns empty vector.
  const absl::InlinedVector<int, 4>& GetFanin(int node_idx) const;
  // Returns all the node indexes that are in the direct fanout of the given
  // node. If the `node_idx` is outside of [0, num_nodes_) returns empty vector.
  const absl::InlinedVector<int, 2>& GetFanout(int node_idx) const;

 private:
  // The key type used to uniquely identify a type attribute on a node.
  struct NodeTypeKey : public std::pair<absl::string_view, TypeAttrId> {
    typedef std::pair<absl::string_view, TypeAttrId> Base;

    // Inherit the set of constructors.
    using Base::pair;

    template <typename H>
    friend H AbslHashValue(H h, const NodeTypeKey& nt) {
      return H::combine(std::move(h), nt.first, nt.second);
    }
  };

  // If true, all invalid edges and inputs (srd, dst or input node not found in
  // a graph) will be skipped, otherwise initialization will fail with error.
  bool skip_invalid_edges_ = false;

  // WARN: `graph_` must outlive this object and graph nodes must not be
  // destructed, because node names captured with absl::string_view.
  const GraphDef* graph_ = nullptr;  // do not own
  int num_nodes_ = 0;
  std::vector<NodeTypeId> node_type_attrs_;
  absl::flat_hash_map<absl::string_view, int> node_name_to_index_;
  absl::flat_hash_map<NodeTypeKey, int> node_type_name_to_index_;

  std::vector<absl::InlinedVector<int, 4>> fanins_;
  std::vector<absl::InlinedVector<int, 2>> fanouts_;

  // We need a valid reference to return from GetFanin/GetFanout if the
  // `node_idx` argument is outside of the [0, num_nodes_) range.
  absl::InlinedVector<int, 4> empty_fanin_;
  absl::InlinedVector<int, 2> empty_fanout_;
};

template <typename T>
inline void SortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

Status GraphTypeTopologyView::InitializeFromGraph(
    const GraphDef& graph, const NodeTypeAttrMap& node_type_map) {
  if (graph_ != nullptr) {
    return errors::InvalidArgument(
        "GraphTypeTopologyView is already initialized.");
  }

  graph_ = &graph;
  int num_nodedefs = graph.node_size();
  node_name_to_index_.rehash(num_nodedefs);

  // Build maps from name to index.
  node_type_attrs_.reserve(num_nodedefs);         // Only approximate.
  node_type_name_to_index_.rehash(num_nodedefs);  // Only approximate.
  for (int node_idx = 0; node_idx < num_nodedefs; ++node_idx) {
    const NodeDef& node = graph.node(node_idx);
    node_name_to_index_.emplace(node.name(), node_idx);

    for (const TypeAttrId& type_attr : node_type_map.GetTypeAttrs(node)) {
      int node_type_idx = node_type_attrs_.size();
      node_type_name_to_index_.emplace(NodeTypeKey(node.name(), type_attr),
                                       node_type_idx);
      node_type_attrs_.emplace_back(&node, type_attr);
    }
  }
  num_nodes_ = node_type_attrs_.size();
  fanins_.resize(num_nodes_);
  fanouts_.resize(num_nodes_);

  // Add graph edges to the adjacency lists.
  for (int node_type_idx = 0; node_type_idx < num_nodes_; ++node_type_idx) {
    const NodeTypeId& node_type = node_type_attrs_.at(node_type_idx);
    auto input_ports =
        node_type_map.GetInputPorts(*node_type.node, node_type.type_attr);
    fanins_[node_type_idx].reserve(input_ports.size());
    for (int port : input_ports) {
      const string& input = node_type.node->input(port);
      TensorId tensor = ParseTensorName(input);
      const auto it = node_name_to_index_.find(tensor.node());
      const bool valid_input = it != node_name_to_index_.end();

      if (!valid_input) {
        const string error_message = strings::StrCat(
            "Non-existent input ", input, " in node ", node_type.node->name());
        if (skip_invalid_edges_) {
          ITEX_VLOG(2) << "Skip error: " << error_message;
        } else {
          return errors::InvalidArgument(error_message);
        }
      }

      if (valid_input) {
        const int input_idx = it->second;
        const NodeDef& input_node = graph_->node(input_idx);
        TypeAttrId input_type_attr =
            node_type_map.GetOutputTypeAttr(input_node, tensor.index());
        const auto it2 = node_type_name_to_index_.find(
            NodeTypeKey(input_node.name(), input_type_attr));
        if (it2 == node_type_name_to_index_.end()) {
          if (!skip_invalid_edges_) {
            return errors::InvalidArgument("Did not find type attr ",
                                           input_type_attr.DebugString(),
                                           " in node ", input_node.name());
          }
          continue;
        }
        int input_node_type_idx = it2->second;
        fanins_[node_type_idx].push_back(input_node_type_idx);
        fanouts_[input_node_type_idx].push_back(node_type_idx);
      }
    }

    // Dedup the input list while it's still hot in cache.
    SortAndRemoveDuplicates(&fanins_[node_type_idx]);
  }

  // Dedup outputs for all the graph nodes.
  for (int node_type_idx = 0; node_type_idx < num_nodes_; ++node_type_idx) {
    SortAndRemoveDuplicates(&fanouts_[node_type_idx]);
  }

  return Status::OK();
}

Status GraphTypeTopologyView::AddEphemeralEdges(
    absl::Span<const NodeTypeIdEdge> ephemeral_edges) {
  // Add ephemeral edges to the adjacency lists.
  for (const NodeTypeIdEdge& edge : ephemeral_edges) {
    const auto src = node_name_to_index_.find(edge.src.node->name());
    const bool valid_src = src != node_name_to_index_.end();

    if (!valid_src) {
      const string error_message =
          strings::StrCat("Non-existent src node: ", edge.src.node->name());
      if (skip_invalid_edges_) {
        ITEX_VLOG(0) << "Skip error: " << error_message;
      } else {
        return errors::InvalidArgument(error_message);
      }
    }

    const auto dst = node_name_to_index_.find(edge.dst.node->name());
    const bool valid_dst = dst != node_name_to_index_.end();

    if (!valid_dst) {
      const string error_message =
          strings::StrCat("Non-existent dst node: ", edge.dst.node->name());
      if (skip_invalid_edges_) {
        ITEX_VLOG(0) << "Skip error: " << error_message;
      } else {
        return errors::InvalidArgument(error_message);
      }
    }

    if (valid_dst && valid_src) {
      int src_node_type_idx = node_type_name_to_index_.at(
          NodeTypeKey(edge.src.node->name(), edge.src.type_attr));
      int dst_node_type_idx = node_type_name_to_index_.at(
          NodeTypeKey(edge.dst.node->name(), edge.dst.type_attr));
      fanins_[dst_node_type_idx].push_back(src_node_type_idx);
      fanouts_[src_node_type_idx].push_back(dst_node_type_idx);
    }
  }

  // Dedup inputs and outputs for all the graph nodes.
  for (int node_type_idx = 0; node_type_idx < num_nodes_; ++node_type_idx) {
    SortAndRemoveDuplicates(&fanins_[node_type_idx]);
    SortAndRemoveDuplicates(&fanouts_[node_type_idx]);
  }

  return Status::OK();
}

const NodeTypeId* GraphTypeTopologyView::GetNode(
    absl::string_view node_name, const TypeAttrId& type_attr) const {
  ITEX_DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  NodeTypeKey key(node_name, type_attr);
  const auto it = node_type_name_to_index_.find(key);
  return it == node_type_name_to_index_.end()
             ? nullptr
             : &node_type_attrs_.at(it->second);
}

const NodeTypeId* GraphTypeTopologyView::GetNode(int node_idx) const {
  ITEX_DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  ITEX_DCHECK(node_idx >= 0 && node_idx < num_nodes_)
      << "node_idx is out of range";
  return &node_type_attrs_.at(node_idx);
}

const absl::optional<int> GraphTypeTopologyView::GetNodeIndex(
    absl::string_view node_name, const TypeAttrId& type_attr) const {
  ITEX_DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  NodeTypeKey key(node_name, type_attr);
  const auto it = node_type_name_to_index_.find(key);
  ITEX_DCHECK(it != node_type_name_to_index_.end())
      << "Node doesn't exist in a graph";
  return it == node_type_name_to_index_.end() ? absl::nullopt
                                              : absl::make_optional(it->second);
}

const absl::optional<int> GraphTypeTopologyView::GetNodeIndex(
    const NodeTypeId& node) const {
  return GetNodeIndex(node.node->name(), node.type_attr);
}

const absl::InlinedVector<int, 4>& GraphTypeTopologyView::GetFanin(
    int node_idx) const {
  ITEX_DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  const bool is_valid_node_idx = node_idx >= 0 && node_idx < num_nodes_;
  ITEX_DCHECK(is_valid_node_idx) << "node_idx is out of range";
  return is_valid_node_idx ? fanins_[node_idx] : empty_fanin_;
}

const absl::InlinedVector<int, 2>& GraphTypeTopologyView::GetFanout(
    int node_idx) const {
  ITEX_DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  const bool is_valid_node_idx = node_idx >= 0 && node_idx < num_nodes_;
  ITEX_DCHECK(is_valid_node_idx) << "node_idx is out of range";
  return is_valid_node_idx ? fanouts_[node_idx] : empty_fanout_;
}

enum class TypeTraversalDirection {
  kFollowInputs,
  kFollowOutputs,
  kFollowInputsAndOutputs,
};

// Encapsulate DFS callbacks that will be called during the graph traversal.
//
// If non-empty, the `pre_order` and `post_order` functors will be called on
// each reachable node (including the `from` nodes) in pre and post order. If
// loops are found, the `on_back_edge` functor will be called on the
// corresponding back edges. Moreover, the pre and post order will assume that
// these back edges will be cut.
struct DfsTypeCallbacks {
  DfsTypeCallbacks() = default;
  DfsTypeCallbacks(std::function<void(int)> pre, std::function<void(int)> post,
                   std::function<void(int, int)> back_edge)
      : pre_order(std::move(pre)),
        post_order(std::move(post)),
        on_back_edge(std::move(back_edge)) {}

  static DfsTypeCallbacks PreOrder(std::function<void(int)> pre) {
    return DfsTypeCallbacks(std::move(pre), nullptr, nullptr);
  }

  static DfsTypeCallbacks PostOrder(std::function<void(int)> post) {
    return DfsTypeCallbacks(nullptr, std::move(post), nullptr);
  }

  std::function<void(int)> pre_order;
  std::function<void(int)> post_order;
  std::function<void(int, int)> on_back_edge;
};

// Encapsulate DFS predicates for traversing the graph.
//
// The `enter` predicate decides if traversal should enter the node, and the
// `advance` predicate decides if the traversal should follow inputs/outputs
// from the node.
//
// If predicates are empty (default initialized), it's assumed that we can enter
// into any node and advance from any node respectively.
struct DfsTypePredicates {
  DfsTypePredicates() = default;
  DfsTypePredicates(std::function<bool(int)> enter,
                    std::function<bool(int)> advance)
      : enter(std::move(enter)), advance(std::move(advance)) {}

  static DfsTypePredicates Enter(std::function<bool(int)> enter) {
    return DfsTypePredicates(std::move(enter), nullptr);
  }

  static DfsTypePredicates Advance(std::function<bool(int)> advance) {
    return DfsTypePredicates(nullptr, std::move(advance));
  }

  std::function<bool(int)> enter;
  std::function<bool(int)> advance;
};

struct DfsStackElem {
  DfsStackElem(int node, bool children_visited, int src)
      : node(node), children_visited(children_visited), src(src) {}
  explicit DfsStackElem(int node) : DfsStackElem(node, false, -1) {}

  // Index of the node in the graph ∊ [0, num_nodes).
  int node;
  // `True` if visited all the input/output nodes (pushed all input/output nodes
  // to the stack).
  bool children_visited;
  // Index of the node in the graph, from which we entered the `node`.
  int src;
};

enum class NodeState { kNotVisited, kVisiting, kDone };

void DfsTypeTraversal(const GraphTypeTopologyView& graph_type_view,
                      const absl::Span<const NodeTypeId* const> from,
                      const TypeTraversalDirection direction,
                      const DfsTypePredicates& predicates,
                      const DfsTypeCallbacks& callbacks) {
  std::vector<DfsStackElem> stack;
  stack.reserve(from.size());

  for (const NodeTypeId* node : from) {
    const absl::optional<int> node_idx = graph_type_view.GetNodeIndex(*node);
    ITEX_DCHECK(node_idx.has_value())
        << "Illegal start node: " << node->node->name();
    if (node_idx.has_value()) {
      stack.emplace_back(node_idx.value());
    }
  }

  absl::flat_hash_map<int, NodeState> node_state;
  while (!stack.empty()) {
    DfsStackElem w = stack.back();
    stack.pop_back();

    NodeState& state = node_state[w.node];
    if (state == NodeState::kDone) continue;

    // Skip nodes that we should not enter.
    if (predicates.enter && !predicates.enter(w.node)) {
      state = NodeState::kDone;
      continue;
    }

    // We've processed all the children of this node.
    if (w.children_visited) {
      state = NodeState::kDone;
      if (callbacks.post_order) {
        callbacks.post_order(w.node);
      }
      continue;
    }

    // Loop detected.
    if (state == NodeState::kVisiting) {
      if (callbacks.on_back_edge) {
        callbacks.on_back_edge(w.src, w.node);
      }
      continue;
    }

    state = NodeState::kVisiting;
    if (callbacks.pre_order) {
      callbacks.pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.emplace_back(w.node, true, w.src);

    // Check if we can continue traversal from the current node.
    if (predicates.advance && !predicates.advance(w.node)) {
      continue;
    }

    // Now enqueue the fanin/fanout nodes.
    if (direction == TypeTraversalDirection::kFollowInputs ||
        direction == TypeTraversalDirection::kFollowInputsAndOutputs) {
      for (const int fanin : graph_type_view.GetFanin(w.node)) {
        stack.emplace_back(fanin, false, w.node);
      }
    }
    if (direction == TypeTraversalDirection::kFollowOutputs ||
        direction == TypeTraversalDirection::kFollowInputsAndOutputs) {
      for (const int fanout : graph_type_view.GetFanout(w.node)) {
        stack.emplace_back(fanout, false, w.node);
      }
    }
  }
}

DataTypeSet AllowedDataTypes(const OpDef::AttrDef& attr_def) {
  const auto& allowed_types = attr_def.allowed_values().list().type();
  if (allowed_types.empty()) {
    return AllTypes();
  }
  uint32 dtype_mask = 0;
  for (int dtype : allowed_types) {
    dtype_mask |= 1u << dtype;
  }
  return DataTypeSet(dtype_mask);
}

DataTypeSet AllowedDataTypes(const OpDef& op_def, const TypeAttrId& t_attr_id) {
  if (t_attr_id.attr_name.empty()) {
    return ToSet(t_attr_id.fixed_type);
  }
  const OpDef::AttrDef* attr_def = FindAttr(t_attr_id.attr_name, op_def);
  ITEX_CHECK(attr_def);  // Crash Ok
  return AllowedDataTypes(*attr_def);
}

Status ValidateLists(const gtl::FlatSet<string>& allow_list,
                     const gtl::FlatSet<string>& deny_list,
                     const gtl::FlatSet<string>& infer_list,
                     const gtl::FlatSet<string>& clear_list) {
  std::vector<gtl::FlatSet<string>> lists{allow_list, deny_list, infer_list,
                                          clear_list};
  std::multiset<string> counts;
  for (const auto& list : lists) {
    counts.insert(list.begin(), list.end());
  }
  bool duplicates = false;
  for (const auto& s : counts) {
    if (counts.count(s) > 1) {
      duplicates = true;
      ITEX_LOG(ERROR) << "Op present in multiple lists: " << s;
    }
  }
  if (duplicates) {
    return errors::InvalidArgument("Op lists have conflicting entries");
  } else {
    return Status::OK();
  }
}

// TODO(itex): after supporting virtual_placer_ and , please add them.
class AutoMixedPrecisionImpl {
 public:
  AutoMixedPrecisionImpl(const std::unordered_set<string>& nodes_to_preserve,
                         GraphDef* graph, AutoMixedPrecisionMode mode)
      : nodes_to_preserve_(nodes_to_preserve),
        graph_(graph),
        function_library_(*graph),
        graph_view_(graph),
        mode_(mode),
        target_dtype_((mode_ == AutoMixedPrecisionMode::GPU_FLOAT16 ||
                       mode_ == AutoMixedPrecisionMode::CPU_FLOAT16)
                          ? DT_HALF
                          : DT_BFLOAT16) {}

  Status Optimize();

 private:
  typedef absl::flat_hash_set<NodeTypeId> NodeTypeIdSet;
  std::unique_ptr<AutoMixedPrecisionLists> get_mixed_precision_lists() const {
    switch (mode_) {
      case AutoMixedPrecisionMode::GPU_FLOAT16:
        return absl::make_unique<AutoMixedPrecisionListsGPU>();
      case AutoMixedPrecisionMode::GPU_BFLOAT16:
        return absl::make_unique<AutoMixedPrecisionListsGPU>();
      case AutoMixedPrecisionMode::CPU_FLOAT16:
        return absl::make_unique<AutoMixedPrecisionListsCPU>();
      case AutoMixedPrecisionMode::CPU_BFLOAT16:
        return absl::make_unique<AutoMixedPrecisionListsCPU>();
      default:
        ITEX_CHECK(false) << "Unsupported ITEX AMP mode";
    }
  }
  Status PrintDebugLogs(bool preop, size_t timestamp);
  void LogSkippedNode(const NodeDef& node) const;
  bool MustPreserve(const NodeDef& node) const;
  bool IsOnDevice(const NodeDef& node, const string& device_type) const;
  bool ShouldProcess(const NodeDef& node) const;
  bool NodeHasF16KernelForTypeAttr(const NodeDef& node, TypeAttrId taid) const;
  bool NodeImplicitlyReadsNonResourceVariable(const NodeDef& node) const;
  void ConvertBatchNormOpsToV2();
  bool HasInputOrOutputRefs(const NodeDef& node) const;
  bool CanForceFP16(const NodeDef& node) const;
  bool SupportsF16(const NodeTypeId& node_type) const;
  bool IsQuantized(const NodeTypeId& node_type) const;
  const NodeTypeId* GetTensorListFloat32NodeTypeId(const NodeDef& node) const;
  bool IsSourceOrSinkOp(const string& op) const;
  void FindFloat32TensorListOpClustersAndDenylistUnsafe(
      std::vector<absl::flat_hash_set<const NodeDef*>>* clusters,
      absl::flat_hash_set<int>* deny_set) const;
  void FindTensorListImplicitFloat32Edges(
      const absl::flat_hash_set<const NodeDef*>& tensor_list_nodes,
      std::vector<NodeTypeIdEdge>* implicit_data_edges) const;
  void AddAllowlistOps(absl::flat_hash_set<int>* allow_set) const;
  void RemoveAllowsetWithFp32(absl::flat_hash_set<int>* allow_set) const;
  void PropagateDenyFwdThroughClearAndInfer(
      absl::flat_hash_set<int>* deny_set) const;
  void ForceColorMatchBetweenTensorListOps(
      const absl::flat_hash_set<const NodeDef*>& tensor_list_nodes,
      absl::flat_hash_set<int>* allow_set,
      absl::flat_hash_set<int>* deny_set) const;
  void AddClearAndInferToAllowIfBetweenAllow(
      const absl::flat_hash_set<int>& deny_set,
      absl::flat_hash_set<int>* allow_set) const;
  void AddInferToAllowIfFollowAllow(const absl::flat_hash_set<int>& deny_set,
                                    absl::flat_hash_set<int>* allow_set) const;
  void PropagateAllowThroughClear(const absl::flat_hash_set<int>& deny_set,
                                  absl::flat_hash_set<int>* allow_set) const;
  Status ForceColorMatchOnRecurrentEdges(
      absl::flat_hash_set<int>* allow_set) const;
  void MakeCastsAllowIfAllOutputsAllow(
      absl::flat_hash_set<int>* allow_set) const;
  NodeDef BuildCastNode(const MutableGraphView::OutputPort& src, bool to_f16,
                        const string& device) const;
  Status ChangeTypeAttrsAndAddCasts(const absl::flat_hash_set<int>& allow_set);

  std::unordered_map<string, DeviceProperties> devices_;
  std::unordered_set<string> nodes_to_preserve_;
  GraphDef* graph_;
  FunctionLibraryDefinition function_library_;
  MutableGraphView graph_view_;
  NodeTypeAttrMap node_type_map_;
  GraphTypeTopologyView graph_type_view_;
  bool force_all_f16_;
  AutoMixedPrecisionMode mode_;
  gtl::FlatSet<string> f16_allowlist_;
  gtl::FlatSet<string> f16_denylist_;
  gtl::FlatSet<string> f16_inferlist_;
  gtl::FlatSet<string> f16_clearlist_;
  absl::flat_hash_set<const NodeDef*> should_process_nodes_;
  DataType target_dtype_;  // Either DT_HALF or DT_BFLOAT16
};

NodeDef AutoMixedPrecisionImpl::BuildCastNode(
    const MutableGraphView::OutputPort& src, bool to_f16,
    const string& device) const {
  DataType src_type = to_f16 ? DT_FLOAT : target_dtype_;
  DataType dst_type = to_f16 ? target_dtype_ : DT_FLOAT;
  const char* cast_string = !to_f16                    ? kCastToFp32
                            : target_dtype_ == DT_HALF ? kCastToFp16
                                                       : kCastToBf16;
  string name = strings::StrCat(src.node->name(), "-", src.port_id, "-",
                                cast_string, "-", kSuffix);
  NodeDef node;
  node.set_name(name);
  node.set_op("Cast");
  node.set_device(device);
  if (src.port_id == 0) {
    node.add_input(src.node->name());
  } else {
    node.add_input(strings::StrCat(src.node->name(), ":", src.port_id));
  }
  (*node.mutable_attr())["SrcT"].set_type(src_type);
  (*node.mutable_attr())["DstT"].set_type(dst_type);
  (*node.mutable_attr())["Truncate"].set_b(false);
  return node;
}

bool AutoMixedPrecisionImpl::HasInputOrOutputRefs(const NodeDef& node) const {
  OpDef op_def;
  Status status = function_library_.LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return true;
  }
  for (const auto& input : op_def.input_arg()) {
    if (input.is_ref()) {
      return true;
    }
  }
  for (const auto& output : op_def.output_arg()) {
    if (output.is_ref()) {
      return true;
    }
  }
  return false;
}

bool AutoMixedPrecisionImpl::CanForceFP16(const NodeDef& node) const {
  return node.op() != "Const" && !IsStateful(node) &&
         !HasInputOrOutputRefs(node);
}

bool AutoMixedPrecisionImpl::NodeHasF16KernelForTypeAttr(
    const NodeDef& node, TypeAttrId taid) const {
  NodeDef node_copy(node);
  if (node.device().empty()) {
    // TODO(itex): After add virtual_placer, please add this code.
    // string device_name = virtual_placer_.get_canonical_device_name(node);
    // node_copy.set_device(device_name);
  }
  if (!SetDataType(&node_copy, taid, target_dtype_)) {
    return false;
  }
  return IsKernelRegisteredForNode(node_copy).ok();
}

Status AutoMixedPrecisionImpl::PrintDebugLogs(bool preop, size_t timestamp) {
  string prepend_path;
  auto cfg_ = itex::itex_get_config();
  prepend_path = cfg_.debug_options().auto_mixed_precision_log_path();
  if (prepend_path.empty()) {
    TF_RETURN_IF_ERROR(ReadStringFromEnvVar(
        "ITEX_AUTO_MIXED_PRECISION_LOG_PATH", "", &prepend_path));
  }

  if (prepend_path.empty()) return Status::OK();

  string suffix =
      strings::StrCat("_", preop ? "preop" : kSuffix, "_", timestamp);

  string fname = itex::io::JoinPath(prepend_path,
                                    strings::StrCat("graphdef", suffix, ".pb"));
  std::fstream f;
  f.open(fname.c_str(), std::fstream::out | std::fstream::binary);
  if (f.is_open()) {
    f << graph_->SerializeAsString();
    f.close();
    ITEX_LOG(INFO) << "Saved "
                   << (preop ? "pre-optimization" : "post-optimization")
                   << " graph as binary to " << fname;
  } else {
    f.close();
    ITEX_LOG(ERROR) << "failed to open " << fname;
  }

  fname = itex::io::JoinPath(prepend_path,
                             strings::StrCat("graphdef", suffix, ".pb.txt"));
  f.open(fname.c_str(), std::fstream::out);
  if (f.is_open()) {
    f << graph_->DebugString();
    f.close();
    ITEX_LOG(INFO) << "Saved "
                   << (preop ? "pre-optimization" : "post-optimization")
                   << " graph as text to " << fname;
  } else {
    f.close();
    ITEX_LOG(ERROR) << "failed to open " << fname;
  }

  if (!preop) {
    fname = itex::io::JoinPath(prepend_path,
                               strings::StrCat("paintbuckets", suffix, ".txt"));
    f.open(fname.c_str(), std::fstream::out);
    if (f.is_open()) {
      std::unique_ptr<AutoMixedPrecisionLists> mp_lists =
          get_mixed_precision_lists();
      f << "AllowList:\n";
      for (const auto& x : mp_lists->AllowList()) {
        f << x << "\n";
      }
      f << "\nDenyList:\n";
      for (const auto& x : mp_lists->DenyList()) {
        f << x << "\n";
      }
      f << "\nInferList:\n";
      for (const auto& x : mp_lists->InferList()) {
        f << x << "\n";
      }
      f << "\nClearList:\n";
      for (const auto& x : mp_lists->ClearList()) {
        f << x << "\n";
      }
      f.close();
      ITEX_LOG(INFO) << "Saved paint bucket info to " << fname;
    } else {
      f.close();
      ITEX_LOG(ERROR) << "failed to open " << fname;
    }
  }
  return Status::OK();
}

void AutoMixedPrecisionImpl::LogSkippedNode(const NodeDef& node) const {
  ITEX_VLOG(2) << "Skipping " << node.op() << " node " << node.name()
               << " because it "
               << (MustPreserve(node)
                       ? "must be preserved"
                       : "is not on the device, or the device is not suitable");
}

bool AutoMixedPrecisionImpl::MustPreserve(const NodeDef& node) const {
  return nodes_to_preserve_.count(node.name());
}

bool AutoMixedPrecisionImpl::IsOnDevice(const NodeDef& node,
                                        const string& device_type) const {
  string device_name;
  if (node.device().empty()) {
    // TODO(itex): After adding virtual_placer_, please uncomment.
    // device_name = virtual_placer_.get_canonical_device_name(node);
  } else {
    device_name = node.device();
  }
  string device;
  string not_used;
  if (DeviceNameUtils::SplitDeviceName(device_name, &not_used, &device) &&
      absl::StrContains(absl::AsciiStrToLower(device),
                        absl::AsciiStrToLower(device_type))) {
    return true;
  }
  return false;
}

bool AutoMixedPrecisionImpl::ShouldProcess(const NodeDef& node) const {
  return should_process_nodes_.count(&node);
}

bool IsFloat32(const NodeTypeId& node_type) {
  return GetDataType(*node_type.node, node_type.type_attr) ==
         DataType::DT_FLOAT;
}

bool IsTensorListOp(const string& op) {
  return op.find("TensorList") != string::npos;
}

bool IsTensorListReaderOp(const string& op) {
  static const gtl::FlatSet<string> tensor_list_reader_ops = {
      "TensorListConcat",  "TensorListConcatV2", "TensorListGather",
      "TensorListGetItem", "TensorListPopBack",  "TensorListStack"};
  return tensor_list_reader_ops.count(op);
}

bool IsTensorListWriterOp(const string& op) {
  static const gtl::FlatSet<string> tensor_list_writer_ops = {
      "TensorListFromTensor",    "TensorListPushBack",
      "TensorListPushBackBatch", "TensorListScatter",
      "TensorListScatterV2",     "TensorListScatterIntoExistingList",
      "TensorListSetItem",       "TensorListSplit"};
  return tensor_list_writer_ops.count(op);
}

bool AutoMixedPrecisionImpl::SupportsF16(const NodeTypeId& node_type) const {
  OpDef op_def;
  Status status = function_library_.LookUpOpDef(node_type.node->op(), &op_def);
  if (!status.ok()) return false;
  return AllowedDataTypes(op_def, node_type.type_attr)
             .Contains(target_dtype_) &&
         NodeHasF16KernelForTypeAttr(*node_type.node, node_type.type_attr);
}

bool AutoMixedPrecisionImpl::IsQuantized(const NodeTypeId& node_type) const {
  for (const TypeAttrId& type_attr :
       node_type_map_.GetTypeAttrs(*node_type.node)) {
    if (DataTypeIsQuantized(GetDataType(*node_type.node, type_attr))) {
      return true;
    }
  }
  return false;
}

// TODO(itex): Make this change the node's name (to aid debugging). Need to
// make sure that doing this won't break anything.
void AutoMixedPrecisionImpl::ConvertBatchNormOpsToV2() {
  for (int node_idx = 0; node_idx < graph_->node_size(); ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    if (!ShouldProcess(*node)) continue;
    bool changed = false;
    if (node->op() == "FusedBatchNorm") {
      ITEX_VLOG(2) << "Changing op of " << node->op() << " node "
                   << node->name() << " to FusedBatchNormV2";
      node->set_op("FusedBatchNormV2");
      changed = true;
    } else if (node->op() == "FusedBatchNormGrad") {
      ITEX_VLOG(2) << "Changing op of " << node->op() << " node "
                   << node->name() << " to FusedBatchNormGradV2";
      node->set_op("FusedBatchNormGradV2");
      changed = true;
    }
    if (changed) {
      (*node->mutable_attr())["U"].set_type(DT_FLOAT);
    }
  }
}

Status AutoMixedPrecisionImpl::Optimize() {
  auto cfg_ = itex::itex_get_config();
  if (cfg_.graph_options().auto_mixed_precision_options().unsafe_force_all()) {
    force_all_f16_ = true;
  } else {
    TF_RETURN_IF_ERROR(ReadBoolFromEnvVar(
        "ITEX_AUTO_MIXED_PRECISION_UNSAFE_FORCE_ALL", 0, &force_all_f16_));
  }

  if (force_all_f16_) {
    // Many ops do not support bfloat16 on CPU and GPU, and for GPU float16 only
    // support forward ops. so we disallowing forcing to bfloat16/float16.
    return errors::InvalidArgument(
        "Currently Auto Mixed Precision unsafe_force_all cannot be set to "
        "true.");
  }

  std::unique_ptr<AutoMixedPrecisionLists> mp_lists =
      get_mixed_precision_lists();
  f16_allowlist_ = mp_lists->AllowList();
  f16_denylist_ = mp_lists->DenyList();
  f16_inferlist_ = mp_lists->InferList();
  f16_clearlist_ = mp_lists->ClearList();
  TF_RETURN_IF_ERROR(ValidateLists(f16_allowlist_, f16_denylist_,
                                   f16_inferlist_, f16_clearlist_));

  size_t timestamp = EnvTime::NowMicros() / 1000;
  TF_RETURN_IF_ERROR(PrintDebugLogs(/* preop = */ true, timestamp));

  ITEX_VLOG(2) << "Identifying nodes that should be processed";
  for (const NodeDef& node : graph_->node()) {
    bool should_process = false;
    switch (mode_) {
      case AutoMixedPrecisionMode::GPU_FLOAT16:
        should_process = !MustPreserve(node) && IsOnDevice(node, DEVICE_XPU);
        break;
      case AutoMixedPrecisionMode::GPU_BFLOAT16:
        should_process = !MustPreserve(node) && IsOnDevice(node, DEVICE_XPU);
        break;
      case AutoMixedPrecisionMode::CPU_FLOAT16:
        should_process = !MustPreserve(node) && IsOnDevice(node, DEVICE_CPU);
        break;
      case AutoMixedPrecisionMode::CPU_BFLOAT16:
        should_process = !MustPreserve(node) && IsOnDevice(node, DEVICE_CPU);
        break;
    }
    if (should_process) {
      should_process_nodes_.insert(&node);
    } else {
      LogSkippedNode(node);
    }
  }

  ITEX_VLOG(2) << "Converting FusedBatchNorm* ops to V2";
  ConvertBatchNormOpsToV2();

  ITEX_VLOG(2) << "Building node type map for graph";
  TF_RETURN_IF_ERROR(node_type_map_.Init(*graph_));

  ITEX_VLOG(2) << "Constructing graph type attribute topology view";
  TF_RETURN_IF_ERROR(
      graph_type_view_.InitializeFromGraph(*graph_, node_type_map_));

  absl::flat_hash_set<int> deny_set;

  std::vector<absl::flat_hash_set<const NodeDef*>> tensor_list_clusters;
  FindFloat32TensorListOpClustersAndDenylistUnsafe(&tensor_list_clusters,
                                                   &deny_set);
  std::vector<NodeTypeIdEdge> ephemeral_edges;
  for (const auto& cluster : tensor_list_clusters) {
    ITEX_VLOG(2) << "Found safe Tensor List cluster of size " << cluster.size();
    for (const NodeDef* node : cluster) {
      ITEX_VLOG(2) << "  Cluster member: " << node->op() << " node "
                   << node->name();
    }
    FindTensorListImplicitFloat32Edges(cluster, &ephemeral_edges);
  }
  TF_RETURN_IF_ERROR(graph_type_view_.AddEphemeralEdges(ephemeral_edges));

  // The goal here is to change performance-critical ops to fp16 or bf16, and to
  // do so with the minimal number of casts, subject to the constraint that the
  // model's convergence is not affected. This is achieved by first identifying
  // which nodes should be changed to f16 and then inserting casts at the
  // boundaries between f16/non-f16 nodes.

  // The algorithm for deciding which nodes to change to f16 is as follows:
  // 1) Add all performance-critical ops (aka "allowlist" ops) to the allow_set.
  //    This is done under the assumption that allowlist ops are always
  //    numerically-safe in f16 and that they are the most important ops for
  //    improving performance.
  // 2) Add nodes to the deny_set iff they are numerically-dangerous (aka
  //    "denylist" ops) or they are on a forward path from a denylist node to
  //    a deny/infer node (including the node at the end of the path) through
  //    non-numerically-dangerous ops (aka "inferlist" and "clearlist" ops).
  //    This is done to prevent numerically-dangerous ops and their downstream
  //    effects from being changed to f16, which would risk breaking the
  //    numerical accuracy of the model.
  // 3) For all remaining nodes that are not considered dangerous (inferlist
  //    and clearlist ops), find those that are between (i.e., both upstream
  //    and downstream of) allow nodes, and add them to the allow_set.
  //    This is done to avoid unnecessary casts between allowlist ops.
  // 4) For the remaining inferlist nodes, add them to the allow_set if they
  //    are immediate downstream of allow_set node.
  // 5) For all remaining clearlist nodes, add them to the allow_set if they are
  //    connected to a node in the allow_set via other clearlist nodes.
  //    This is done to increase the number of ops in the allow_set without
  //    affecting numerical stability.

  absl::flat_hash_set<int> allow_set;
  ITEX_VLOG(2) << "Beginning pass 1 to add allowlist ops";
  AddAllowlistOps(&allow_set);
  ITEX_VLOG(2) << "Finished pass 1";

  if (allow_set.empty()) {
    ITEX_LOG(INFO) << "No allowlist ops found, nothing to do";
    return Status::OK();
  }

  ITEX_VLOG(2)
      << "Beginning pass 2 to propagate deny forwards from denylist ops "
         "through clear/inferlist ops";
  PropagateDenyFwdThroughClearAndInfer(&deny_set);
  ITEX_VLOG(2) << "Finished pass 2";

  ITEX_VLOG(2) << "Forcing color match between data structure ops";
  for (const auto& cluster : tensor_list_clusters) {
    ForceColorMatchBetweenTensorListOps(cluster, &allow_set, &deny_set);
  }

  ITEX_VLOG(2)
      << "Beginning pass 3 to set clear and infer nodes to allow if they "
         "are between allow ops";
  AddClearAndInferToAllowIfBetweenAllow(deny_set, &allow_set);
  ITEX_VLOG(2) << "Finished pass 3";

  ITEX_VLOG(2) << "Beginning pass 4 to add infer list ops to allow if they "
                  "directly follow allow nodes";
  AddInferToAllowIfFollowAllow(deny_set, &allow_set);
  ITEX_VLOG(2) << "Finished pass 4";

  ITEX_VLOG(2)
      << "Beginning pass 5 to propagate allow from allow nodes through "
         "clearlist ops";
  PropagateAllowThroughClear(deny_set, &allow_set);
  ITEX_VLOG(2) << "Finished pass 5";

  ITEX_VLOG(2)
      << "Beginning pass 6 to remove some nodes which could not be changed "
         "to F16"
         "from allow set";
  RemoveAllowsetWithFp32(&allow_set);
  ITEX_VLOG(2) << "Finished pass 6";

  ITEX_VLOG(2) << "Forcing color match between data structure ops";
  for (const auto& cluster : tensor_list_clusters) {
    ForceColorMatchBetweenTensorListOps(cluster, &allow_set, &deny_set);
  }

  ITEX_VLOG(2) << "Forcing color match on loop edges";
  TF_RETURN_IF_ERROR(ForceColorMatchOnRecurrentEdges(&allow_set));

  ITEX_VLOG(2) << "Finding existing casts that can be made allow";
  MakeCastsAllowIfAllOutputsAllow(&allow_set);

  ITEX_VLOG(2)
      << "Beginning final pass to change type attributes and insert Cast "
         "ops at paint boundaries";
  TF_RETURN_IF_ERROR(ChangeTypeAttrsAndAddCasts(allow_set));
  ITEX_VLOG(2) << "Finished final pass";

  TF_RETURN_IF_ERROR(PrintDebugLogs(/* preop = */ false, timestamp));

  return Status::OK();
}

// If node is a Tensor List op with a float32 data type attribute then this
// returns a pointer to the NodeTypeId representing that type attribute. In
// all other cases this returns nullptr.
const NodeTypeId* AutoMixedPrecisionImpl::GetTensorListFloat32NodeTypeId(
    const NodeDef& node) const {
  if (!IsTensorListOp(node.op())) return nullptr;
  for (const TypeAttrId& type_attr : node_type_map_.GetTypeAttrs(node)) {
    const NodeTypeId* node_type =
        graph_type_view_.GetNode(node.name(), type_attr);
    // This assumes that the float32 data type on a Tensor List op is always a
    // non-fixed type attribute containing a single type, and that this type
    // attribute represents the dtype of the values in the list.
    // TODO(itex): A new Tensor List op could theoretically break these
    // assumptions.
    if (node_type && node_type->type_attr.fixed_type == DT_INVALID &&
        node_type->type_attr.type_index == TypeAttrId::kSingleType &&
        IsFloat32(*node_type)) {
      return node_type;
    }
  }
  return nullptr;
}

bool AutoMixedPrecisionImpl::IsSourceOrSinkOp(const string& op) const {
  const gtl::FlatSet<string> source_and_sink_ops = {
      "_Arg",
      "_Retval",
      "OptionalFromValue",
      "OptionalGetValue",
      "PartitionedCall",
      "Placeholder",
      "StatefulPartitionedCall",
  };
  return source_and_sink_ops.count(op) || function_library_.Find(op);
}

// Finds all clusters of float32 Tensor List nodes that are connected via their
// handle edges. Unsafe clusters (those with unprocessable nodes, or with edges
// that cross untraversable boundaries via _Arg, _Ret, PartitionedCall etc.
// nodes) are added to deny_set. The caller should paint all nodes in a cluster
// the same color, as they may all refer to the same Tensor List.
void AutoMixedPrecisionImpl::FindFloat32TensorListOpClustersAndDenylistUnsafe(
    std::vector<absl::flat_hash_set<const NodeDef*>>* tensor_list_clusters,
    absl::flat_hash_set<int>* deny_set) const {
  absl::flat_hash_set<const NodeDef*> tensor_list_prop_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) ||
        root.type_attr.fixed_type != DataType::DT_VARIANT ||
        !GetTensorListFloat32NodeTypeId(*root.node) ||
        tensor_list_prop_set.count(root.node)) {
      continue;
    }
    const NodeTypeId* root_fp32 = GetTensorListFloat32NodeTypeId(*root.node);
    const absl::optional<int> maybe_root_fp32_idx =
        graph_type_view_.GetNodeIndex(*root_fp32);
    ITEX_DCHECK(maybe_root_fp32_idx.has_value())
        << "Type attribute " << root_fp32->type_attr.DebugString()
        << " of node " << root.node->name() << " not found in graph view";
    int root_fp32_idx = maybe_root_fp32_idx.value();
    // Traverse Tensor List handle edges (DT_VARIANT) to find cluster of all
    // connected Tensor List nodes.
    absl::flat_hash_set<const NodeDef*> cluster({root.node});
    DfsTypeTraversal(graph_type_view_, {&root},
                     TypeTraversalDirection::kFollowInputsAndOutputs,
                     DfsTypePredicates::Enter([&](int idx) -> bool {
                       const NodeTypeId& item = *graph_type_view_.GetNode(idx);
                       return !tensor_list_prop_set.count(item.node);
                     }),
                     DfsTypeCallbacks::PreOrder([&](int idx) {
                       const NodeTypeId& item = *graph_type_view_.GetNode(idx);
                       const NodeDef* node = item.node;
                       if (GetTensorListFloat32NodeTypeId(*node)) {
                         cluster.insert(node);
                         if (!ShouldProcess(*node)) {
                           // The cluster contains an un-processable node.
                           deny_set->insert(root_fp32_idx);
                         }
                         // TODO(itex): In a theoretical pathological
                         // case of a Tensor List of Tensor List handles, the
                         // Tensor List itself would need to be treated as a
                         // sink.
                       } else if (IsSourceOrSinkOp(node->op())) {
                         // The cluster crosses an untraversable boundary.
                         deny_set->insert(root_fp32_idx);
                       }
                     }));
    tensor_list_clusters->push_back(cluster);
  }
}

// Finds all writer -> reader pairs in the given set that are connected via
// their handles, and adds corresponding float32 edges to *implicit_fp32_edges.
void AutoMixedPrecisionImpl::FindTensorListImplicitFloat32Edges(
    const absl::flat_hash_set<const NodeDef*>& tensor_list_nodes,
    std::vector<NodeTypeIdEdge>* implicit_fp32_edges) const {
  for (const NodeDef* root_node : tensor_list_nodes) {
    if (!IsTensorListReaderOp(root_node->op())) continue;
    NodeTypeId root(root_node, TypeAttrId(DataType::DT_VARIANT));
    const NodeTypeId* root_fp32 = GetTensorListFloat32NodeTypeId(*root.node);
    ITEX_CHECK(root_fp32) << "No float32 type attribute found for "  // Crash OK
                          << root.node->op() << " node " << root.node->name();
    // Search backwards through handle edges (DT_VARIANT) for all writer ops,
    // adding direct implicit edges between them and the reader.
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowInputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return ShouldProcess(*item.node);
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          if (IsTensorListWriterOp(item.node->op())) {
            const NodeTypeId* item_fp32 =
                GetTensorListFloat32NodeTypeId(*item.node);
            ITEX_CHECK(item_fp32)  // Crash OK
                << "No float32 type attribute found for " << item.node->op()
                << " node " << item.node->name();
            ITEX_VLOG(2) << "Adding ephemeral float32 edge from "
                         << item_fp32->node->op() << " node "
                         << item_fp32->node->name() << " to "
                         << root_fp32->node->op() << " node "
                         << root_fp32->node->name();
            implicit_fp32_edges->emplace_back(*item_fp32, *root_fp32);
          }
        }));
  }
}

void AutoMixedPrecisionImpl::AddAllowlistOps(
    absl::flat_hash_set<int>* allow_set) const {
  // Add allowlisted ops to allow_set.
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node)) continue;
    bool force_allow = force_all_f16_ && CanForceFP16(*root.node);
    if (f16_allowlist_.count(root.node->op()) || force_allow) {
      bool inserted = allow_set->insert(root_idx).second;
      if (ITEX_VLOG_IS_ON(2) && inserted) {
        ITEX_VLOG(2) << "Painting type " << root.type_attr.DebugString()
                     << " of node " << root.node->name()
                     << " ALLOW because its op " << root.node->op()
                     << " is on the allowlist";
      }
    }
  }
}

// Adds nodes to deny_set iff they are on the denylist or they are on a
// forward path from a denylist node to a deny/infer node (including the node
// at the end of the path) through clear and infer nodes.
// E.g., deny -> infer -> clear -> infer -> clear -> allow -> infer
// becomes: deny -> deny -> deny -> deny -> clear -> allow -> infer.
void AutoMixedPrecisionImpl::PropagateDenyFwdThroughClearAndInfer(
    absl::flat_hash_set<int>* deny_set) const {
  if (force_all_f16_) return;

  // Find clear nodes that are upstream of deny or infer.
  absl::flat_hash_set<int> upstream_of_deny_or_infer_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!(f16_denylist_.count(root.node->op()) ||
          f16_inferlist_.count(root.node->op()))) {
      continue;
    }
    DfsTypeTraversal(graph_type_view_, {&root},
                     TypeTraversalDirection::kFollowInputs,
                     DfsTypePredicates::Enter([&](int idx) -> bool {
                       const NodeTypeId& item = *graph_type_view_.GetNode(idx);
                       return idx == root_idx ||
                              (!upstream_of_deny_or_infer_set.count(idx) &&
                               f16_clearlist_.count(item.node->op()));
                     }),
                     DfsTypeCallbacks::PreOrder([&](int idx) {
                       upstream_of_deny_or_infer_set.insert(idx);
                     }));
  }

  // Propagate deny forward through nodes in upstream_of_deny_or_infer_set.
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (deny_set->count(root_idx) || !f16_denylist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          return idx == root_idx || (!deny_set->count(idx) &&
                                     upstream_of_deny_or_infer_set.count(idx));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          bool inserted = deny_set->insert(idx).second;
          if (ITEX_VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            ITEX_VLOG(2) << "Painting type " << item.type_attr.DebugString()
                         << " of " << item.node->op() << " node "
                         << item.node->name() << " DENY";
          }
        }));
  }
}

void AutoMixedPrecisionImpl::AddClearAndInferToAllowIfBetweenAllow(
    const absl::flat_hash_set<int>& deny_set,
    absl::flat_hash_set<int>* allow_set) const {
  // Find clear/inferlist ops that are downstream of allow ops.
  absl::flat_hash_set<int> downstream_of_allow_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || !f16_allowlist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return idx == root_idx ||
                 (!downstream_of_allow_set.count(idx) &&
                  !f16_allowlist_.count(item.node->op()) &&
                  !deny_set.count(idx) && ShouldProcess(*item.node) &&
                  // TODO(itex): Consider allowing propagation through
                  // ops that are already float16 in order to reduce the number
                  // of casts.
                  IsFloat32(item) && SupportsF16(item) &&
                  (f16_clearlist_.count(item.node->op()) ||
                   f16_inferlist_.count(item.node->op())));
        }),
        DfsTypeCallbacks::PreOrder(
            [&](int idx) { downstream_of_allow_set.insert(idx); }));
  }

  // Set nodes that are both downstream and upstream of allow ops to allow.
  absl::flat_hash_set<int> upstream_of_allow_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || upstream_of_allow_set.count(root_idx) ||
        !f16_allowlist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowInputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          return idx == root_idx || (!upstream_of_allow_set.count(idx) &&
                                     downstream_of_allow_set.count(idx));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          upstream_of_allow_set.insert(idx);
          bool inserted = allow_set->insert(idx).second;
          if (ITEX_VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            ITEX_VLOG(2) << "Painting type " << item.type_attr.DebugString()
                         << " of " << item.node->op() << " node "
                         << item.node->name() << " ALLOW";
          }
        }));
  }
}

void AutoMixedPrecisionImpl::PropagateAllowThroughClear(
    const absl::flat_hash_set<int>& deny_set,
    absl::flat_hash_set<int>* allow_set) const {
  // Propagate allow from allow nodes through clearlist ops.
  absl::flat_hash_set<int> clear_prop_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || clear_prop_set.count(root_idx) ||
        !allow_set->count(root_idx)) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root},
        TypeTraversalDirection::kFollowInputsAndOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return idx == root_idx ||
                 (!allow_set->count(idx) && !deny_set.count(idx) &&
                  ShouldProcess(*item.node) && IsFloat32(item) &&
                  SupportsF16(item) &&
                  (f16_clearlist_.count(item.node->op())) &&
                  // We don't propagate (backwards) through nodes that read
                  // Variables because it can break the behavior of TensorBoard
                  // visualization and/or (in the case of Enter nodes) the model
                  // itself. This is only a problem for non-resource variables.
                  !NodeImplicitlyReadsNonResourceVariable(*item.node));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          clear_prop_set.insert(idx);
          bool inserted = allow_set->insert(idx).second;
          if (ITEX_VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            ITEX_VLOG(2) << "Painting type " << item.type_attr.DebugString()
                         << " of " << item.node->op() << " node "
                         << item.node->name() << " ALLOW";
          }
        }));
  }
}

// Set infer node to allow if its immediate upstream node is in allow set
void AutoMixedPrecisionImpl::AddInferToAllowIfFollowAllow(
    const absl::flat_hash_set<int>& deny_set,
    absl::flat_hash_set<int>* allow_set) const {
  for (int item_idx = 0; item_idx < graph_type_view_.num_nodes(); ++item_idx) {
    const NodeTypeId& item = *graph_type_view_.GetNode(item_idx);
    if (!ShouldProcess(*item.node) || deny_set.count(item_idx) ||
        allow_set->count(item_idx) || !f16_inferlist_.count(item.node->op()) ||
        !IsFloat32(item) || !SupportsF16(item)) {
      continue;
    }

    bool has_allow_fanin = false;
    for (const int fanin : graph_type_view_.GetFanin(item_idx)) {
      if (deny_set.count(fanin)) {
        has_allow_fanin = false;
        break;
      }
      if (allow_set->count(fanin)) {
        has_allow_fanin = true;
      }
    }
    if (has_allow_fanin) {
      bool inserted = allow_set->insert(item_idx).second;
      if (ITEX_VLOG_IS_ON(2) && inserted) {
        ITEX_VLOG(2) << "Painting type " << item.type_attr.DebugString()
                     << " of " << item.node->op() << " node "
                     << item.node->name() << " ALLOW";
      }
    }
  }
}

// If ops have one or more type_attr, But this type_attr could not be converted
// to F16. Such as FusedBatchNormV2/FusedBatchNormV3, its type_attr 'U' only
// support float. So we will remove this node from allow_set.
void AutoMixedPrecisionImpl::RemoveAllowsetWithFp32(
    absl::flat_hash_set<int>* allow_set) const {
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (f16_allowlist_.count(root.node->op()) && allow_set->count(root_idx) &&
        (!SupportsF16(root) || IsQuantized(root))) {
      auto erased = allow_set->erase(root_idx);
      if (ITEX_VLOG_IS_ON(2) && erased) {
        ITEX_VLOG(2) << "UnPainting type " << root.type_attr.DebugString()
                     << " of node " << root.node->name()
                     << " ALLOW because its op " << root.node->op()
                     << " is not support F16 DataType";
      }
    }
  }
}

// Forces NextIteration nodes and their output Merge node(s) to have the same
// color. Specifically, it removes them all from allow_set if any of the Merge
// nodes is not in allow_set, otherwise it adds the NextIteration node to
// allow_set.
Status AutoMixedPrecisionImpl::ForceColorMatchOnRecurrentEdges(
    absl::flat_hash_set<int>* allow_set) const {
  for (const NodeDef& node : graph_->node()) {
    if (node.op() == "NextIteration") {
      GraphView::OutputPort output_port(&node, 0);
      const auto& fanout = graph_view_.GetFanout(output_port);
      std::vector<int> merge_idxs;
      merge_idxs.reserve(fanout.size());
      bool any_merge_is_not_allow = false;
      for (const auto& output : fanout) {
        const NodeDef& merge_node = *output.node;
        if (merge_node.op() != "Merge") {
          return errors::FailedPrecondition(
              "Expected Merge node after NextIteration, got ", merge_node.op());
        }
        const absl::optional<int> maybe_merge_idx =
            graph_type_view_.GetNodeIndex(merge_node.name(), TypeAttrId("T"));
        if (!maybe_merge_idx.has_value()) {
          return errors::Internal("Type attribute T of Merge node ",
                                  merge_node.name(),
                                  " not found in graph view");
        }
        int merge_idx = maybe_merge_idx.value();
        merge_idxs.push_back(merge_idx);
        any_merge_is_not_allow =
            any_merge_is_not_allow || !allow_set->count(merge_idx);
      }
      const absl::optional<int> maybe_nextiter_idx =
          graph_type_view_.GetNodeIndex(node.name(), TypeAttrId("T"));
      if (!maybe_nextiter_idx.has_value()) {
        return errors::Internal("Type attribute T of NextIteration node ",
                                node.name(), " not found in graph view");
      }
      int nextiter_idx = maybe_nextiter_idx.value();
      if (any_merge_is_not_allow) {
        for (int merge_idx : merge_idxs) {
          if (allow_set->erase(merge_idx)) {
            ITEX_VLOG(2)
                << "Painting type T of Merge node "
                << graph_type_view_.GetNode(merge_idx)->node->name()
                << " DENY to match the color of its sibling Merge nodes "
                   "with common NextIteration node "
                << node.name();
          }
        }
        if (allow_set->erase(nextiter_idx)) {
          ITEX_VLOG(2)
              << "Painting type T of NextIteration node " << node.name()
              << " DENY to match the color of its output Merge node(s)";
        }
      } else {
        if (allow_set->insert(nextiter_idx).second) {
          ITEX_VLOG(2)
              << "Painting type T of NextIteration node " << node.name()
              << " ALLOW to match the color of its output Merge node(s)";
        }
      }
    }
  }
  return Status::OK();
}

// Forces all of the given Tensor List nodes into the same color set.
void AutoMixedPrecisionImpl::ForceColorMatchBetweenTensorListOps(
    const absl::flat_hash_set<const NodeDef*>& tensor_list_nodes,
    absl::flat_hash_set<int>* allow_set,
    absl::flat_hash_set<int>* deny_set) const {
  bool any_deny = false;
  bool any_allow = false;
  std::vector<int> node_type_idxs;
  node_type_idxs.reserve(tensor_list_nodes.size());
  for (const NodeDef* node : tensor_list_nodes) {
    const NodeTypeId& node_type = *GetTensorListFloat32NodeTypeId(*node);
    const absl::optional<int> maybe_node_type_idx =
        graph_type_view_.GetNodeIndex(node_type);
    ITEX_DCHECK(maybe_node_type_idx.has_value())
        << "Type attribute " << node_type.type_attr.DebugString() << " of node "
        << node->name() << " not found in graph view";
    node_type_idxs.push_back(maybe_node_type_idx.value());
  }
  for (int node_type_idx : node_type_idxs) {
    if (deny_set->count(node_type_idx)) {
      any_deny = true;
      break;
    } else if (allow_set->count(node_type_idx)) {
      any_allow = true;
    }
  }
  if (!any_deny && !any_allow) return;
  for (int node_type_idx : node_type_idxs) {
    const NodeTypeId& node_type = *graph_type_view_.GetNode(node_type_idx);
    ITEX_VLOG(2) << "Painting type " << node_type.type_attr.DebugString()
                 << " of " << node_type.node->op() << " node "
                 << node_type.node->name() << " "
                 << (any_deny ? "DENY" : "ALLOW")
                 << " because at least one of its siblings is "
                 << (any_deny ? "DENY" : "ALLOW");
    if (any_deny) {
      allow_set->erase(node_type_idx);
      deny_set->insert(node_type_idx);
    } else {
      allow_set->insert(node_type_idx);
    }
  }
}

bool AutoMixedPrecisionImpl::NodeImplicitlyReadsNonResourceVariable(
    const NodeDef& node) const {
  if (node.op() == "Identity" || node.op() == "Enter") {
    GraphView::InputPort node_input(&node, 0);
    MutableGraphView::OutputPort prev_output =
        graph_view_.GetRegularFanin(node_input);
    const NodeDef* input = prev_output.node;
    if (input && ((node.op() == "Identity" && (input->op() == "Variable" ||
                                               input->op() == "VariableV2")) ||
                  (node.op() == "Enter" &&
                   NodeImplicitlyReadsNonResourceVariable(*input)))) {
      return true;
    }
  }
  return false;
}

// This adds existing Cast nodes to allow_set if all of their outputs are allow,
// avoiding the need to add a new Cast node after an existing Cast.
void AutoMixedPrecisionImpl::MakeCastsAllowIfAllOutputsAllow(
    absl::flat_hash_set<int>* allow_set) const {
  int num_nodes_preop = graph_->node_size();
  for (int node_idx = 0; node_idx < num_nodes_preop; ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    NodeTypeId node_type(node, TypeAttrId("DstT"));
    if (node->op() != "Cast" || !IsFloat32(node_type)) {
      continue;
    }
    bool all_fanouts_allow = true;
    MutableGraphView::OutputPort src(node, 0);
    const auto& fanout = graph_view_.GetFanout(src);
    for (const MutableGraphView::InputPort& dst : fanout) {
      TypeAttrId dst_type_attr =
          node_type_map_.GetInputTypeAttr(*dst.node, dst.port_id);
      const absl::optional<int> maybe_dst_type_idx =
          graph_type_view_.GetNodeIndex(dst.node->name(), dst_type_attr);
      ITEX_DCHECK(maybe_dst_type_idx.has_value())
          << "Type attribute " << dst_type_attr.DebugString() << " of node "
          << dst.node->name() << " not found in graph view";
      int dst_type_idx = maybe_dst_type_idx.value();
      bool dst_is_allow = allow_set->count(dst_type_idx);
      if (!dst_is_allow) {
        all_fanouts_allow = false;
        break;
      }
    }
    if (!fanout.empty() && all_fanouts_allow) {
      const absl::optional<int> maybe_node_type_idx =
          graph_type_view_.GetNodeIndex(node_type);
      ITEX_DCHECK(maybe_node_type_idx.has_value())
          << "Type attribute " << node_type.type_attr.DebugString()
          << " of node " << node_type.node->name()
          << " not found in graph view";
      int node_type_idx = maybe_node_type_idx.value();
      allow_set->insert(node_type_idx);
    }
  }
}

// Changes all allow-painted type attributes to DT_HALF or DT_BFLOAT16, and
// inserts Cast nodes at node outputs for all edges that connect
// allow-painted <-> non-allow-painted type attributes.
Status AutoMixedPrecisionImpl::ChangeTypeAttrsAndAddCasts(
    const absl::flat_hash_set<int>& allow_set) {
  int num_nodes_changed = 0;
  int num_nonvar_casts_to_f16 = 0;
  int num_nodes_preop = graph_->node_size();
  for (int node_idx = 0; node_idx < num_nodes_preop; ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    for (const TypeAttrId& type_attr : node_type_map_.GetTypeAttrs(*node)) {
      const absl::optional<int> maybe_node_type_idx =
          graph_type_view_.GetNodeIndex(node->name(), type_attr);
      if (!maybe_node_type_idx.has_value()) {
        return errors::Internal("Type attribute ", type_attr.DebugString(),
                                " of ", node->op(), " node ", node->name(),
                                " not found in graph view");
      }
      int node_type_idx = maybe_node_type_idx.value();
      if (!IsFloat32(*graph_type_view_.GetNode(node_type_idx))) continue;
      bool src_is_allow = allow_set.count(node_type_idx);
      if (src_is_allow) {
        ITEX_VLOG(2) << "Changing type " << type_attr.DebugString() << " of "
                     << node->op() << " node " << node->name() << " to "
                     << DataTypeString(target_dtype_);
        if (!SetDataType(node, type_attr, target_dtype_)) {
          return errors::Internal("Failed to set type attribute");
        }
        ++num_nodes_changed;
      }
      for (int output_port : node_type_map_.GetOutputPorts(*node, type_attr)) {
        MutableGraphView::OutputPort src(node, output_port);
        NodeDef* added_cast_node = nullptr;
        // Note: This is copied so that edges can be modified inside the loop.
        auto fanout = graph_view_.GetFanout(src);
        for (const MutableGraphView::InputPort& dst : fanout) {
          TypeAttrId dst_type_attr =
              node_type_map_.GetInputTypeAttr(*dst.node, dst.port_id);
          const absl::optional<int> maybe_dst_type_idx =
              graph_type_view_.GetNodeIndex(dst.node->name(), dst_type_attr);
          if (!maybe_dst_type_idx.has_value()) {
            return errors::Internal("Type attribute ",
                                    dst_type_attr.DebugString(), " of ",
                                    dst.node->op(), " node ", dst.node->name(),
                                    " not found in graph view");
          }
          int dst_type_idx = maybe_dst_type_idx.value();
          bool dst_is_allow = allow_set.count(dst_type_idx);
          if (src_is_allow != dst_is_allow) {
            if (!added_cast_node) {
              bool to_f16 = dst_is_allow;
              ITEX_VLOG(2) << "Inserting cast to "
                           << (to_f16 ? DataTypeString(target_dtype_)
                                      : "DT_FLOAT")
                           << " at " << src.node->op() << " "
                           << src.node->name() << ":" << src.port_id;
              added_cast_node = graph_view_.AddNode(
                  BuildCastNode(src, to_f16, src.node->device()));
              if (to_f16 && !IsConstant(*node) && !IsVariable(*node) &&
                  !NodeImplicitlyReadsNonResourceVariable(*node)) {
                ++num_nonvar_casts_to_f16;
              }
            }
            TF_RETURN_IF_ERROR(graph_view_.UpdateRegularFaninByPort(
                dst.node->name(), dst.port_id, {added_cast_node->name(), 0}));
          }
        }
      }
    }
  }
  // Use Python type names (e.g. float16) instead of C++ type names (e.g. half)
  // since many Python users will see this message.
  const char* type_str = target_dtype_ == DT_HALF ? "float16" : "bfloat16";
  ITEX_LOG(INFO) << "Converted " << num_nodes_changed << "/" << num_nodes_preop
                 << " nodes to " << type_str << " precision using "
                 << num_nonvar_casts_to_f16 << " cast(s) to " << type_str
                 << " (excluding Const and Variable casts)";
  return Status::OK();
}

}  // end namespace
Status RunAutoMixedPrecision(OptimizerContext* opt_ctx,
                             const GrapplerItem& item,
                             const GraphDef& graph_def, GraphDef* output) {
  auto mode = AutoMixedPrecisionMode::GPU_FLOAT16;
  Status status = GetAutoMixedPrecisionMode(opt_ctx->device_name, &mode);
  // Start by copying input graph to output.
  *output = graph_def;

  TF_RETURN_IF_ERROR(status);

  // Optimize the output graph in-place.
  AutoMixedPrecisionImpl optimizer(item.NodesToPreserve(), output, mode);
  status = optimizer.Optimize();
  if (!status.ok()) {
    // Restore the original graph.
    *output = graph_def;
    ITEX_LOG(WARNING) << " graph optimizer FAILED: " << status.ToString();
  }
  return status;
}

}  // namespace graph
}  // namespace itex
