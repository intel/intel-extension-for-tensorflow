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

#include "itex/core/compiler/xla/stream_executor/sycl/sycl_event.h"

#include "itex/core/compiler/xla/stream_executor/lib/statusor.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_executor.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"

namespace stream_executor {
namespace gpu {

GpuEvent::GpuEvent(GpuExecutor* parent)
    : parent_(parent), gpu_event_(nullptr) {}

GpuEvent::~GpuEvent() {}

port::Status GpuEvent::Init() {
  // ITEX_GPUDevice* device_handle;
  // ITEX_GPUGetDevice(&device_handle, parent_->gpu_device());
  gpu_event_ = new ITEX_GPUEvent;
  return port::Status::OK();
}

port::Status GpuEvent::Destroy() {
  // ITEX_GPUDevice* device_handle;
  // ITEX_GPUGetDevice(&device_handle, parent_->gpu_device());
  delete gpu_event_;
  return port::Status::OK();
}

port::Status GpuEvent::Record(GpuStream* stream) {
  *gpu_event_ = stream->gpu_stream()->ext_oneapi_submit_barrier();
  return port::Status::OK();
}

GpuEventHandle GpuEvent::gpu_event() { return gpu_event_; }

Event::Status GpuEvent::PollForStatus() {
  auto event_status =
      gpu_event_->get_info<cl::sycl::info::event::command_execution_status>();

  switch (event_status) {
    case cl::sycl::info::event_command_status::submitted:
    case cl::sycl::info::event_command_status::running:
      return Event::Status::kPending;
    case cl::sycl::info::event_command_status::complete:
      return Event::Status::kComplete;
    default:
      return Event::Status::kUnknown;
  }
}

}  // namespace gpu
}  // namespace stream_executor
