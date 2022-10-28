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

/// Structure to specify a forward op, a backward op, and the slot numbers
/// in the forward and backward ops where we will add a workspace edge.
typedef struct {
  string bwd_op;    // Name of a backward op in the graph
  int bwd_slot;     // Input slot in the backward op node where actual
                    // input tensor resides
  int ws_fwd_slot;  // Output slot in the forward op node where workspace
                    // edge is added
} WorkSpaceInfo;

// Is OpDef::ArgDef a list type? It could be N * T or list(type).
// Refer to opdef.proto for details of list type.
inline bool ArgIsList(const OpDef::ArgDef& arg) {
  return !arg.type_list_attr().empty() || !arg.number_attr().empty();
}

void GetDummyOneDnnTensorNode(const NodeDef& input, NodeDef* dummy);

const RewriteInfo* CheckForNodeRewrite(const utils::MutableNodeView& node_view);

string GetInputName(const NodeDef* input, int out_slot);

Status RewriteNode(const char* device_name, OneDnnLayoutContext* ctx,
                   int node_index, const RewriteInfo* ri);

Status FixOneDnnMetaDataEdges(OneDnnLayoutContext* ctx, int node_index);

Status RunOneDnnLayout(const char* device_name, const GrapplerItem& item,
                       const GraphDef& graph_def, GraphDef* optimized_graph);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_ONEDNN_LAYOUT_ONEDNN_LAYOUT_H_
