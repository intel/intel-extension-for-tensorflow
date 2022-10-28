/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_GRAPH_REMAPPER_REMAPPER_H_
#define ITEX_CORE_GRAPH_REMAPPER_REMAPPER_H_

#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/grappler_item.h"
#include "itex/core/utils/tf_buffer.h"
#include "protos/graph.pb.h"

namespace itex {
namespace graph {

struct RemapperContext {
  explicit RemapperContext(const GrapplerItem& item, GraphDef* g_def,
                           Status* status)
      : nodes_to_preserve(item.NodesToPreserve()),
        graph_view(g_def, status),
        graph_properties(item),
        inferred_graph_properties(false) {}

  std::unordered_set<string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
  GraphProperties graph_properties;
  bool inferred_graph_properties;

  GraphProperties& GetGraphProperties() {
    if (!inferred_graph_properties) {
      Status s = graph_properties.InferStatically(
          /*assume_valid_feeds=*/true,
          /*aggressive_shape_inference=*/false,
          /*include_input_tensor_values=*/true,
          /*include_output_tensor_values=*/true);

      // TODO(itex) Is there any case that InferStatically will return an
      // unsuccessful state?
      TF_ABORT_IF_ERROR(s);
      inferred_graph_properties = true;
    }

    return graph_properties;
  }
};

bool HasDataType(const NodeDef* node, const DataType& expected,
                 const string& type_attr = "T");

Status RunRemapper(const char* device_name, const GrapplerItem& item,
                   const GraphDef& graph_def, GraphDef* optimized_graph,
                   bool is_full = true);

void SetFusedOpAttributes(NodeDef* fused,
                          const absl::Span<const absl::string_view> fused_ops,
                          int num_args);
}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_REMAPPER_REMAPPER_H_
