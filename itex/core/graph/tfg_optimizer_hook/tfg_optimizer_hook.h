/* Copyright (c) 2023 Intel Corporation

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_GRAPH_TFG_OPTIMIZER_HOOK_TFG_OPTIMIZER_HOOK_H_
#define ITEX_CORE_GRAPH_TFG_OPTIMIZER_HOOK_TFG_OPTIMIZER_HOOK_H_

#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/grappler_item.h"
#include "itex/core/utils/status.h"
#include "protos/graph.pb.h"

namespace mlir {
namespace tfg {

itex::Status RunAutoShard(const itex::graph::GrapplerItem& item,
                          const itex::GraphDef& graph_def,
                          itex::GraphDef* optimized_graph,
                          bool have_matmul_or_conv);

}  // end namespace tfg
}  // end namespace mlir

#endif  // ITEX_CORE_GRAPH_TFG_OPTIMIZER_HOOK_TFG_OPTIMIZER_HOOK_H_
