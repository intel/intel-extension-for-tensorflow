/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_GRAPH_MEMORY_OPT_PASS_WEIGHT_PREPACK_H_
#define ITEX_CORE_GRAPH_MEMORY_OPT_PASS_WEIGHT_PREPACK_H_

#include <vector>

#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/utils/onednn/onednn_util.h"

namespace itex {
namespace graph {

using utils::MutableNodeView;

// TODO(itex): support ops with multiple filters in future.
typedef struct {
  int filter_index;  // input index of filter node
                     // update to `std::vector<int> filter_list` if needed
  OneDnnTensorFormat onednn_format;  // for mapping to tf data format
} WeightPrePackInfo;

void WeightPrePack(const MutableNodeView* node_view);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_MEMORY_OPT_PASS_WEIGHT_PREPACK_H_
