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

#ifndef ITEX_CORE_UTILS_ONEDNN_ONEDNN_GRAPH_UTIL_H_
#define ITEX_CORE_UTILS_ONEDNN_ONEDNN_GRAPH_UTIL_H_

#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "protos/graph.pb.h"

#ifndef INTEL_CPU_ONLY
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif  // INTEL_CPU_ONLY

namespace itex {
namespace graph {

// Set partition id in grappler and get partition id when executing.
dnnl::graph::partition GetOneDnnGraphPartition(int pid);
void SetOneDnnGraphPartition(dnnl::graph::partition partition);

// Extract H/W (2D) or D/H/W (3D) based on format.
void ExtractSpatialDims(bool is_channel_last, const std::vector<int32_t>& src,
                        std::vector<int64_t>* dst);

// Extract Pad dims on H/W (2D) or D/H/W (3D) based on format.
void ExtractSpatialPadDims(bool is_channel_last,
                           const std::vector<int32_t>& src,
                           std::vector<int64_t>* pads_begin,
                           std::vector<int64_t>* pads_end);

// Utility function which maps TF Dtype to LLGA Dtype.
dnnl::graph::logical_tensor::data_type GetOneDnnGraphDataType(DataType dt);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_UTILS_ONEDNN_ONEDNN_GRAPH_UTIL_H_
