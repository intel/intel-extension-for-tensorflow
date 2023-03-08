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

#ifndef ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SCRATCH_ALLOCATOR_H_
#define ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SCRATCH_ALLOCATOR_H_

#include <memory>
#include <vector>

#include "itex/core/compiler/xla/stream_executor/device_memory_allocator.h"
#include "itex/core/compiler/xla/stream_executor/lib/statusor.h"
#include "itex/core/compiler/xla/stream_executor/platform/port.h"

namespace stream_executor {

// class Stream;

// Interface for "scratch" allocator for device memory, which deallocates all
// buffers it has allocated at destruction. Returned memory pointers are not
// owning.
//
// Used by stream operations (e.g. Stream::ThenConvolveWithScratch) to
// optionally request scratch space to speed up the operation.
class ScratchAllocator {
 public:
  ScratchAllocator(int device_ordinal, DeviceMemoryAllocator* memory_allocator);

  // Returns a limit of memory this scratch allocator wants to produce, in
  // bytes. This information may be used to help select an algorithm.
  //
  // Returns values < 0 to indicate that there is no recommended limit.
  virtual int64_t GetMemoryLimitInBytes();

  int64_t TotalAllocatedBytes() { return total_allocated_bytes_; }

  // Returns an allocation on byte_size bytes for use in an operation on stream.
  //
  // This is a temporary allocation, and the caller is responsible for
  // deallocating at some known-safe point. See the class comment above.
  virtual itex::StatusOr<DeviceMemory<uint8_t>> AllocateBytes(
      int64_t byte_size);

  virtual ~ScratchAllocator();

 private:
  const int device_ordinal_;
  DeviceMemoryAllocator* memory_allocator_;
  std::vector<OwningDeviceMemory> allocated_buffers_;
  int64_t total_allocated_bytes_ = 0;
};

itex::Status AllocateWorkspace(void** workspace,
                               ScratchAllocator* scratch_allocator,
                               size_t num_bytes);
}  // namespace stream_executor

#endif  // ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SCRATCH_ALLOCATOR_H_
