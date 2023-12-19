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

#ifndef ITEX_CORE_DEVICES_XPU_DEVICE_H_
#define ITEX_CORE_DEVICES_XPU_DEVICE_H_

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

void SE_InitPlugin_Internal(SE_PlatformRegistrationParams* const params,
                            TF_Status* const status);

#ifdef __cplusplus
}
#endif
#endif  // ITEX_CORE_DEVICES_XPU_DEVICE_H_
