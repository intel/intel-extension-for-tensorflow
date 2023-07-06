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

#include "itex/core/compiler/xla/stream_executor/scratch_allocator.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "itex/core/compiler/xla/stream_executor/lib/status_macros.h"

namespace stream_executor {

ScratchAllocator::ScratchAllocator(int device_ordinal,
                                   DeviceMemoryAllocator* memory_allocator)
    : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

int64_t ScratchAllocator::GetMemoryLimitInBytes() {
  constexpr int64_t kScratchSize = 1LL << 32;  // 4GB by default.
  return kScratchSize;
}

itex::StatusOr<DeviceMemory<uint8_t>> ScratchAllocator::AllocateBytes(
    int64_t byte_size) {
  ITEX_CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes()) {
    return itex::Status(
        TF_RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes()));
  }

  TF_ASSIGN_OR_RETURN(OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  DeviceMemoryBase buffer_addr = *allocated_buffer;
  allocated_buffers_.push_back(std::move(allocated_buffer));
  return DeviceMemory<uint8_t>(buffer_addr);
}

ScratchAllocator::~ScratchAllocator() {}

itex::Status AllocateWorkspace(void** workspace,
                               ScratchAllocator* scratch_allocator,
                               size_t num_bytes) {
  SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> workspace_bytes,
                      scratch_allocator->AllocateBytes(num_bytes));
  *workspace = static_cast<void*>(workspace_bytes.opaque());
  return itex::Status::OK();
}

/*
OneTimeScratchAllocator::OneTimeScratchAllocator(Stream* stream)
    : stream_(stream) {}
OneTimeScratchAllocator::~OneTimeScratchAllocator() {}

int64_t OneTimeScratchAllocator::GetMemoryLimitInBytes() { return -1; }

port::StatusOr<DeviceMemory<uint8>> OneTimeScratchAllocator::AllocateBytes(
    int64_t byte_size) {
  CHECK(temporary_ == nullptr);
  SE_ASSIGN_OR_RETURN(temporary_,
                      stream_->AllocateTemporaryArray<uint8>(byte_size));
  return temporary_->device_memory();
}
*/
}  // namespace stream_executor
