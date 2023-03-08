/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/service/gpu/infeed_manager.h"

#include <limits>
#include <memory>
#include <utility>

#include "itex/core/compiler/xla/service/gpu/xla_executor_state.h"
#include "itex/core/compiler/xla/shape_util.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_executor.h"

namespace itex_xla {
namespace gpu {

constexpr int kMaxInfeedsInFlight = 8;

InfeedManager::InfeedManager(se::StreamExecutor* executor)
    : BlockingXfeedQueue(/*max_pending_xfeeds=*/kMaxInfeedsInFlight),
      stream_(std::make_unique<se::Stream>(executor)) {
  stream_->Init();
}

// InfeedManager::~InfeedManager() {
//   executor_->DeallocateStream(stream_);
// }

static StatusOr<se::ScopedDeviceMemory<uint8_t>> CopyBufferToDevice(
    se::Stream* stream, int64_t size, const void* source) {
  if (size > std::numeric_limits<int32_t>::max()) {
    return InvalidArgument("GPU infeed of %d bytes exceeds maximum of %d bytes",
                           size, std::numeric_limits<int32_t>::max());
  }

  if (size == 0) {
    return InvalidArgument("Infeed shape needs 0 bytes");
  }

  se::StreamExecutor* executor = stream->parent();
  se::ScopedDeviceMemory<uint8_t> buffer(
      executor, executor->AllocateArray<uint8_t>(size));
  stream->ThenMemcpy(buffer.ptr(), source, size);

  return std::move(buffer);
}

Status InfeedManager::TransferLiteralToInfeed(se::StreamExecutor* executor,
                                              const LiteralSlice& literal) {
  const Shape& literal_shape = literal.shape();
  ITEX_VLOG(2) << "Transferring literal to infeed with shape: "
               << ShapeUtil::HumanString(literal_shape);

  BlockUntilEnqueueSlotAvailable();

  // For a tuple, we transfer each of its elements to the device and enqueue the
  // resulting destination device addresses with the infeed manager.
  ShapeTree<se::ScopedDeviceMemory<uint8_t>> buffer_tree(literal_shape);
  for (auto& leaf : buffer_tree.leaves()) {
    const Shape& sub_shape = ShapeUtil::GetSubshape(literal_shape, leaf.first);
    ITEX_CHECK(sub_shape.IsArray())
        << ShapeUtil::HumanStringWithLayout(sub_shape);
    TF_ASSIGN_OR_RETURN(
        leaf.second,
        CopyBufferToDevice(stream(), ShapeUtil::ByteSizeOf(sub_shape),
                           literal.untyped_data(leaf.first)));
  }

  // TODO(b/30467474): Since this stream is shared across different infeed
  // requests, blocking on the stream might be heavy-handed. Figure out if
  // finer-grained acknowledgement is possible.
  Status block_status = stream()->BlockHostUntilDone();
  if (!block_status.ok()) {
    return InternalError("Failed to complete data transfer on stream %p: %s",
                         stream(), block_status.error_message());
  }

  EnqueueDestination(std::move(buffer_tree));
  return Status::OK();
}

InfeedManager* GetOrCreateInfeedManager(se::StreamExecutor* executor) {
  stream_executor::gpu::GpuExecutor* gpu_executor =
      stream_executor::gpu::ExtractGpuExecutor(executor);
  auto* xla_state =
      gpu_executor->getOrCreateXLAState<GpuExecutorXLAState>(executor);
  return xla_state->getOrCreateInfeedManager(executor);
}

}  // namespace gpu
}  // namespace itex_xla
