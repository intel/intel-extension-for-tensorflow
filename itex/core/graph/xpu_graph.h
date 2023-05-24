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

#ifndef ITEX_CORE_GRAPH_XPU_GRAPH_H_
#define ITEX_CORE_GRAPH_XPU_GRAPH_H_

#ifdef __cplusplus
extern "C" {
#endif

// TF_InitGraph_Internal is used to do graph optimizer registration.
// Plugin should implement TF_InitGraph to register graph optimizers.
void TF_InitGraph_Internal(TP_OptimizerRegistrationParams* params,
                           TF_Status* status);

#ifdef __cplusplus
}
#endif

#endif  // ITEX_CORE_GRAPH_XPU_GRAPH_H_
