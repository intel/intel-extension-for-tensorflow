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

#ifndef ITEX_CORE_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_H_
#define ITEX_CORE_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_H_

#include "itex/core/graph/utils/grappler_item.h"
#include "itex/core/utils/tf_buffer.h"
#include "protos/graph.pb.h"

namespace itex {
namespace graph {

enum class AutoMixedPrecisionMode { GPU_FLOAT16, GPU_BFLOAT16, CPU_BFLOAT16 };

Status RunAutoMixedPrecision(const char* device_name, const GrapplerItem& item,
                             const GraphDef& graph_def, GraphDef* output);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_H_
