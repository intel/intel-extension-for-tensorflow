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

#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"

#include "itex/core/compiler/xla/stream_executor/lib/status.h"
#include "itex/core/compiler/xla/stream_executor/stream.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_executor.h"

namespace stream_executor {
namespace gpu {

bool GpuStream::Init() {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, parent_->gpu_device());
  ITEX_GPUCreateStream(device_handle, &gpu_stream_);
  return true;
}

void GpuStream::Destroy() {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, parent_->gpu_device());
  ITEX_GPUDestroyStream(device_handle, gpu_stream_);
}

bool GpuStream::IsIdle() const { return true; }

GpuStream* AsGpuStream(Stream* stream) {
  ITEX_DCHECK(stream != nullptr);
  return static_cast<GpuStream*>(stream->implementation());
}

GpuStreamHandle AsGpuStreamValue(Stream* stream) {
  ITEX_DCHECK(stream != nullptr);
  return AsGpuStream(stream)->gpu_stream();
}

}  // namespace gpu
}  // namespace stream_executor
