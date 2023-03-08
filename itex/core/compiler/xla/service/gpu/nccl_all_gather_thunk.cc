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

#include "itex/core/compiler/xla/service/gpu/nccl_all_gather_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "itex/core/compiler/xla/layout_util.h"
#include "itex/core/compiler/xla/service/gpu/ir_emission_utils.h"
#include "itex/core/compiler/xla/service/gpu/nccl_ops.h"
#include "itex/core/compiler/xla/service/hlo_casting_utils.h"
#include "itex/core/compiler/xla/service/hlo_instructions.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"
#include "itex/core/compiler/xla/util.h"

namespace itex_xla {
namespace gpu {

/*static*/ NcclAllGatherConfig NcclAllGatherThunk::GetNcclAllGatherConfig(
    mlir::lmhlo::AllGatherOp op) {
  NcclAllGatherConfig config;
  config.config =
      GetNcclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds());
  return config;
}

/*static*/ bool NcclAllGatherThunk::CanImplement(mlir::lmhlo::AllGatherOp op) {
  return absl::c_all_of(op.getInputs(), [&](mlir::Value operand) {
    Shape shape = GetShape(operand);
    return LayoutUtil::IsDenseArray(shape) &&
           IsTypeSupportedByNccl(shape.element_type()) &&
           LayoutUtil::MinorToMajor(shape).back() == op.getAllGatherDimension();
  });
}

NcclAllGatherThunk::NcclAllGatherThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllGatherOp op,
    std::vector<NcclAllGatherThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllGather, thunk_info),
      config_(GetNcclAllGatherConfig(op)),
      buffers_(std::move(buffers)) {
  ITEX_CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllGatherThunk::RunNcclCollective(const ExecuteParams& params,
                                             ncclComm_t comm) {
  int device_ordinal = params.stream->parent()->device_ordinal();
  ITEX_VLOG(1) << "Performing AllGather from device ordinal: "
               << device_ordinal;
#if ITEX_USE_CCL
  auto gpu_stream = params.stream->stream_handle;
  auto ccl_stream = ccl::create_stream(*gpu_stream);

  for (size_t i = 0; i < buffers_.size(); ++i) {
    const Buffer& buffer = buffers_[i];
    const void* send_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
            .opaque();
    void* recv_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
            .opaque();

    PrimitiveType element_type = config_.config.operand_element_type[i];
    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(element_type));
    auto dtype = dtype_and_multiplier.first;
    int element_count = buffer.element_count * dtype_and_multiplier.second;

    ITEX_VLOG(3) << absl::StreamFormat(
        "Calling ccl::allgather(send_buffer=%p, recv_buffer=%p, sendcount=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, element_count, static_cast<const void*>(comm),
        gpu_stream);

    std::vector<size_t> recv_counts(buffers_.size(), element_count);
    ccl::allgatherv(send_buffer, element_count, recv_buffer, recv_counts, dtype,
                    *comm, ccl_stream);
  }
#else   // ITEX_USE_CCL
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(params.stream);
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const Buffer& buffer = buffers_[i];
    const void* send_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
            .opaque();
    void* recv_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
            .opaque();

    PrimitiveType element_type = config_.config.operand_element_type[i];
    int element_count = buffer.element_count *
                        (primitive_util::IsComplexType(element_type) ? 2 : 1);

    ITEX_VLOG(3) << absl::StreamFormat(
        "Calling ccl::allgather(send_buffer=%p, recv_buffer=%p, sendcount=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, element_count, static_cast<const void*>(comm),
        gpu_stream);

    itex_allgather(send_buffer, recv_buffer, element_count, element_type,
                   gpu_stream, comm);
  }
#endif  // ITEX_USE_CCL
  ITEX_VLOG(1) << "Done performing AllGather for ordinal: " << device_ordinal;
  return Status::OK();
}

}  // namespace gpu
}  // namespace itex_xla
