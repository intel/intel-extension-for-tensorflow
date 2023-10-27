/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef ITEX_CORE_GRAPH_ONEDNN_LAYOUT_ONEDNN_LAYOUT_H_
#define ITEX_CORE_GRAPH_ONEDNN_LAYOUT_ONEDNN_LAYOUT_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/grappler_item.h"
#include "itex/core/graph/utils/layout_utils.h"
#include "itex/core/graph/utils/node_type_attr_map.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/node_def_util.h"
#include "protos/graph.pb.h"

namespace itex {
namespace graph {

struct OneDnnLayoutContext {
  explicit OneDnnLayoutContext(const GrapplerItem& item, GraphDef* g_def,
                               Status* status)
      : graph_view(g_def, status), nodes_to_preserve(item.NodesToPreserve()) {
    TF_ABORT_IF_ERROR(node_type_map.Init(*g_def));
  }

  utils::MutableGraphView graph_view;
  std::unordered_set<string> nodes_to_preserve;
  NodeTypeAttrMap node_type_map;
};

/// Structure to specify the name of an original node, its new name after
/// rewrite, the number of inputs to the original node, the function to
/// be used to copy attributes for the op, and the rule (if any) which
/// must hold for rewriting the node
typedef struct {
  string name;      // Original name of op of the node in the graph
  string new_name;  // New name of the op of the node in the graph
  // A function handler to copy attributes from an old node to a new node.
  std::function<void(const utils::MutableNodeView*, NodeDef*)> copy_attrs;
  // A rule under which to rewrite this node
  std::function<bool(const utils::MutableNodeView&)> rewrite_rule;
} RewriteInfo;

// Is OpDef::ArgDef a list type? It could be N * T or list(type).
// Refer to opdef.proto for details of list type.
inline bool ArgIsList(const OpDef::ArgDef& arg) {
  return !arg.type_list_attr().empty() || !arg.number_attr().empty();
}

void GetDummyOneDnnTensorNode(const NodeDef& input, NodeDef* dummy);

const RewriteInfo* CheckForNodeRewrite(const utils::MutableNodeView& node_view);

Status RewriteNode(const char* device_name, OneDnnLayoutContext* ctx,
                   int node_index, const RewriteInfo* ri);

Status RunOneDnnLayout(OptimizerContext* opt_ctx, const GrapplerItem& item,
                       const GraphDef& graph_def, GraphDef* optimized_graph);

#ifdef INTEL_CPU_ONLY
static constexpr const char* onednngrap_op_name = "OneDnnGraphCPU";
static constexpr const char* _onednngrap_op_name = "_OneDnnGraphCPU";
#else
static constexpr const char* onednngrap_op_name = "OneDnnGraph";
static constexpr const char* _onednngrap_op_name = "_OneDnnGraph";
#endif

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_ONEDNN_LAYOUT_ONEDNN_LAYOUT_H_
