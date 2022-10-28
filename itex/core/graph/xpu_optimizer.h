/* Copyright (c) 2021 Intel Corporation

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

#ifndef ITEX_CORE_GRAPH_XPU_OPTIMIZER_H_
#define ITEX_CORE_GRAPH_XPU_OPTIMIZER_H_

#include "tensorflow/c/experimental/grappler/grappler.h"

namespace itex {
namespace graph {

typedef struct Optimizer {
  // Point it to a global string, no need to delete it in destructor.
  const char* device_name = nullptr;
} Optimizer;

void* Optimizer_CPU_Create();

void* Optimizer_GPU_Create();

void* Optimizer_XPU_Create();

void Optimizer_Destroy(void* optimizer);

void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf,
                        const TF_GrapplerItem* tf_item,
                        TF_Buffer* optimized_graph_buf, TF_Status* tf_status);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_XPU_OPTIMIZER_H_
