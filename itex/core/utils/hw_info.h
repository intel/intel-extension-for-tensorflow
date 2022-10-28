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

#ifndef ITEX_CORE_UTILS_HW_INFO_H_
#define ITEX_CORE_UTILS_HW_INFO_H_
#include "itex/core/utils/macros.h"
#ifndef INTEL_CPU_ONLY
#include "third_party/build_option/dpcpp/runtime/dpcpp_runtime.h"
TF_EXPORT extern const char* const XeHPC_name;
TF_EXPORT extern const char* const XeHPC_name_448;

bool IsXeHPC(sycl::device* device_ptr = nullptr);

#endif

#endif  // ITEX_CORE_UTILS_HW_INFO_H_
