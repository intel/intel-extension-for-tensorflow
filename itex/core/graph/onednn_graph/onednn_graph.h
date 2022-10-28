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

#ifndef ITEX_CORE_GRAPH_ONEDNN_GRAPH_ONEDNN_GRAPH_H_
#define ITEX_CORE_GRAPH_ONEDNN_GRAPH_ONEDNN_GRAPH_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/grappler_item.h"
#include "itex/core/graph/utils/node_type_attr_map.h"
#include "itex/core/utils/node_def_util.h"

#include "protos/graph.pb.h"

namespace itex {
namespace graph {

struct OneDnnGraphContext {
  explicit OneDnnGraphContext(const GrapplerItem& item, GraphDef* g_def,
                              Status* status)
      : graph_view(g_def, status),
        fetch_tensors(item.fetch),
        nodes_to_preserve(item.NodesToPreserve()),
        graph_properties(item),
        inferred_graph_properties(false) {
    TF_ABORT_IF_ERROR(node_type_map.Init(*g_def));
  }
  utils::MutableGraphView graph_view;
  NodeTypeAttrMap node_type_map;
  std::vector<string> fetch_tensors;
  std::unordered_set<string> nodes_to_preserve;
  GraphProperties graph_properties;
  bool inferred_graph_properties;
};

Status RunOneDnnGraph(const GrapplerItem& item, const GraphDef& graph_def,
                      GraphDef* optimized_graph);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_ONEDNN_GRAPH_ONEDNN_GRAPH_H_
