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

#ifndef ITEX_CORE_GRAPH_GENERIC_LAYOUT_OPTIMIZER_GENERIC_LAYOUT_OPTIMIZER_H_
#define ITEX_CORE_GRAPH_GENERIC_LAYOUT_OPTIMIZER_GENERIC_LAYOUT_OPTIMIZER_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/attr_value_util.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace graph {

// TransposeContext owns all data members. Must initialize GraphProperties,
// FrameView, GraphDef and MutableGraphView with the same graph. NodeDef
// pointers in FrameView, GraphDef and MutableGraphView must point to nodes in
// the same GraphDef instance.
struct TransposeContext {
  // Initializes TransposeContext with given GrapplerItem. Because initializing
  // FrameMap and GraphProperties may return error, we initialize
  // TransposeContext outside constructor.
  static Status InitializeTransposeContext(bool assume_valid_feeds,
                                           const GrapplerItem& item,
                                           const GraphDef& graph_def,
                                           TransposeContext* context);

  static Status InitializeTransposeContext(const GrapplerItem& item,
                                           const GraphDef& graph_def,
                                           TransposeContext* context) {
    return InitializeTransposeContext(false, item, graph_def, context);
  }

  GraphDef graph;
  // Number of nodes in the original graph. As new nodes are appended to the end
  // of the graph, all new nodes should have a node index greater than or equal
  // to this.
  int num_nodes;
  absl::flat_hash_set<string> nodes_to_preserve;
  std::unique_ptr<GraphProperties> graph_properties;
  std::unique_ptr<utils::MutableGraphView> graph_view;

  string enforced_layout;
};

// Optimize the data layout for convolutional models.
class GenericLayoutOptimizer {
 public:
  GenericLayoutOptimizer() {}

  string name() const { return "layout"; }

  Status Optimize(const char* device_name, const GrapplerItem& item,
                  const GraphDef& graph_def, GraphDef* optimized_graph);
};

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_GENERIC_LAYOUT_OPTIMIZER_GENERIC_LAYOUT_OPTIMIZER_H_
