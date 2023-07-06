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

#include "itex/core/utils/onednn/onednn_graph_util.h"

#include <unordered_map>
#include <utility>

#include "itex/core/utils/mutex.h"

namespace itex {
namespace graph {

static mutex partition_map_mutex;

namespace {
std::unordered_map<int, dnnl::graph::partition>* GetPartitionMap() {
  static std::unordered_map<int, dnnl::graph::partition>
      partition_id_to_partition_ =
          std::unordered_map<int, dnnl::graph::partition>();
  return &partition_id_to_partition_;
}

}  // namespace

dnnl::graph::partition GetOneDnnGraphPartition(int pid) {
  // TODO(itex): check whether partition map remains unchanged during
  // execution. If true, we may remove this lock at all.
  tf_shared_lock mu(&partition_map_mutex);
  const auto it = GetPartitionMap()->find(pid);
  if (it != GetPartitionMap()->end()) return it->second;
  return dnnl::graph::partition();
}

void SetOneDnnGraphPartition(dnnl::graph::partition partition) {
  // TODO(itex): Check do we need to keep this mutex to handle multi graphs
  // are optimized parallel
  mutex_lock mu(&partition_map_mutex);
  GetPartitionMap()->insert({partition.get_id(), std::move(partition)});
}

void ExtractSpatialDims(bool is_channel_last, const std::vector<int32_t>& src,
                        std::vector<int64_t>* dst) {
  int spatial_dim_num = src.size() - 2;
  int spatial_dim_start_index = is_channel_last ? 1 : 2;

  for (int i = 0; i < spatial_dim_num; ++i) {
    dst->at(i) = src[i + spatial_dim_start_index];
  }
}

void ExtractSpatialPadDims(bool is_channel_last,
                           const std::vector<int32_t>& src,
                           std::vector<int64_t>* pads_begin,
                           std::vector<int64_t>* pads_end) {
  int spatial_dim_num = src.size() / 2 - 2;
  int spatial_dim_start_index = is_channel_last ? 1 : 2;

  for (int i = 0; i < spatial_dim_num; ++i) {
    pads_begin->at(i) = src[(i + spatial_dim_start_index) * 2];
    pads_end->at(i) = src[(i + spatial_dim_start_index) * 2 + 1];
  }
}

dnnl::graph::logical_tensor::data_type GetOneDnnGraphDataType(DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return dnnl::graph::logical_tensor::data_type::f32;
    case DT_HALF:
      return dnnl::graph::logical_tensor::data_type::f16;
    case DT_BFLOAT16:
      return dnnl::graph::logical_tensor::data_type::bf16;
    case DT_INT32:
    case DT_QINT32:
      return dnnl::graph::logical_tensor::data_type::s32;
    case DT_INT8:
    case DT_QINT8:
      return dnnl::graph::logical_tensor::data_type::s8;
    case DT_UINT8:
    case DT_QUINT8:
      return dnnl::graph::logical_tensor::data_type::u8;
      // TODO(itex): bring boolean back, once gc backend merge in master
#ifndef ITEX_ONEDNN_3_0
    case DT_BOOL:
      return dnnl::graph::logical_tensor::data_type::boolean;
#endif
    default:
      return dnnl::graph::logical_tensor::data_type::undef;
  }
}

}  // namespace graph
}  // namespace itex
