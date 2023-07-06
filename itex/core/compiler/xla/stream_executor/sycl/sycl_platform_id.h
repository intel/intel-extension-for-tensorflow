/* Copyright (c) 2023 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_ID_H_
#define ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_ID_H_

#include "itex/core/compiler/xla/stream_executor/platform.h"

namespace stream_executor {
namespace gpu {

// Opaque and unique identifier for the cuda platform.
// This is needed so that plugins can refer to/identify this platform without
// instantiating a CudaPlatform object.
// This is broken out here to avoid a circular dependency between CudaPlatform
// and CudaExecutor.
extern const Platform::Id kSyclPlatformId;

}  // namespace gpu
}  // namespace stream_executor

#endif  // ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_ID_H_
