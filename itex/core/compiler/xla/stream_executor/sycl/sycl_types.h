/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// GPU (ROCm / CUDA) specific type handle resolution

#ifndef ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_TYPES_H_
#define ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_TYPES_H_

#include <level_zero/ze_api.h>

#include <CL/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

namespace stream_executor {
namespace gpu {

// using GpuContextHandle = hipCtx_t;
using GpuStreamHandle = ITEX_GPUStream*;
using GpuEventHandle = ITEX_GPUEvent*;
using GpuFunctionHandle = ::sycl::kernel*;
// using GpuFunctionAttribute = hipDeviceAttribute_t;  // not a typo!
using GpuDeviceHandle = int;  // DPCPPDevice*;
// using GpuDevicePtr = hipDeviceptr_t;
// using GpuDeviceAttribute = hipDeviceAttribute_t;
// using GpuDeviceProperty = hipDeviceProp_t;
using GpuModuleHandle = ze_module_handle_t;
// using GpuStatus = hipError_t;
// using GpuFuncCachePreference = hipFuncCache_t;
// using GpuSharedMemConfig = hipSharedMemConfig;
// using GpuComplexType = hipComplex;
// using GpuDoubleComplexType = hipDoubleComplex;
// using GpuRngHandle = hiprandGenerator_t;

}  // namespace gpu
}  // namespace stream_executor

#endif  // ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_TYPES_H_
