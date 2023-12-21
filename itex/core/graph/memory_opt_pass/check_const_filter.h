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

#ifndef ITEX_CORE_GRAPH_MEMORY_OPT_PASS_CHECK_CONST_FILTER_H_
#define ITEX_CORE_GRAPH_MEMORY_OPT_PASS_CHECK_CONST_FILTER_H_

#include <string>
#include <unordered_set>

#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/utils/cpu_info.h"
#include "itex/core/utils/function.h"
#include "itex/core/utils/op_def_util.h"

namespace itex {
namespace graph {

// Check and set filter attribute
void CheckConstFilter(const utils::MutableNodeView* node_view,
                      const std::unordered_set<string>& nodes_to_preserve);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_MEMORY_OPT_PASS_CHECK_CONST_FILTER_H_
