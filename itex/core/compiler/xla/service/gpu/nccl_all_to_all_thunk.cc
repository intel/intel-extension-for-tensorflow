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

#include "itex/core/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "itex/core/compiler/xla/layout_util.h"
#include "itex/core/compiler/xla/service/gpu/ir_emission_utils.h"
#include "itex/core/compiler/xla/service/gpu/nccl_ops.h"
#include "itex/core/compiler/xla/service/hlo_casting_utils.h"
#include "itex/core/compiler/xla/service/hlo_instructions.h"
#include "itex/core/compiler/xla/shape_util.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"
#include "itex/core/compiler/xla/util.h"

namespace itex_xla {
namespace gpu {

/*static*/ NcclAllToAllConfig NcclAllToAllThunk::GetNcclAllToAllConfig(
    mlir::lmhlo::AllToAllOp op) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfigForMlir(op, absl::nullopt);
  config.has_split_dimension = op.getSplitDimension().has_value();
  return config;
}

/*static*/ bool NcclAllToAllThunk::CanImplement(mlir::lmhlo::AllToAllOp op) {
  return absl::c_all_of(op.getInputs(), [&op](mlir::Value operand) {
    Shape shape = GetShape(operand);
    return LayoutUtil::IsDenseArray(shape) &&
           IsTypeSupportedByNccl(shape.element_type()) &&
           (!op.getSplitDimension() ||
            LayoutUtil::MinorToMajor(shape).back() == *op.getSplitDimension());
  });
}

NcclAllToAllThunk::NcclAllToAllThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllToAllOp op,
    std::vector<NcclAllToAllThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllToAll, thunk_info),
      config_(GetNcclAllToAllConfig(op)),
      buffers_(std::move(buffers)) {
  ITEX_CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllToAllThunk::RunNcclCollective(const ExecuteParams& params,
                                            ncclComm_t comm) {
  int device_ordinal = params.stream->parent()->device_ordinal();
  ITEX_VLOG(1) << "Performing AllToAll from device ordinal: " << device_ordinal;
#if ITEX_USE_CCL
  auto gpu_stream = params.stream->stream_handle;
  auto ccl_stream = ccl::create_stream(*gpu_stream);

  int num_participants = comm->nranks;
  // XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_participants));

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (config_.has_split_dimension) {
    for (size_t i = 0; i < buffers_.size(); ++i) {
      const Buffer& buffer = buffers_[i];
      const uint8_t* send_buffer = static_cast<uint8_t*>(
          params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
              .opaque());
      uint8_t* recv_buffer = static_cast<uint8_t*>(
          params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
              .opaque());

      PrimitiveType element_type = config_.config.operand_element_type[i];
      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(element_type));
      auto dtype = dtype_and_multiplier.first;
      int element_count = buffer.element_count * dtype_and_multiplier.second;

      TF_RET_CHECK(element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_elements = element_count / num_participants;
      size_t chunk_bytes =
          chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);

      for (int rank = 0; rank < num_participants; ++rank) {
        ccl::alltoall(
            static_cast<const void*>(send_buffer + rank * chunk_bytes),
            static_cast<void*>(recv_buffer + rank * chunk_bytes),
            chunk_elements, dtype, *comm, ccl_stream);
      }
    }
  } else {
    TF_RET_CHECK(buffers_.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

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

      ccl::alltoall(send_buffer, recv_buffer, element_count, dtype, *comm,
                    ccl_stream);
    }
  }
#else   // ITEX_USE_CCL
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(params.stream);
  int num_participants = comm->nranks;

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (config_.has_split_dimension) {
    for (size_t i = 0; i < buffers_.size(); ++i) {
      const Buffer& buffer = buffers_[i];
      const uint8_t* send_buffer = static_cast<uint8_t*>(
          params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
              .opaque());
      uint8_t* recv_buffer = static_cast<uint8_t*>(
          params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
              .opaque());

      PrimitiveType element_type = config_.config.operand_element_type[i];
      int element_count = buffer.element_count *
                          (primitive_util::IsComplexType(element_type) ? 2 : 1);

      TF_RET_CHECK(element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_elements = element_count / num_participants;
      size_t chunk_bytes =
          chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);

      return Unimplemented("AllToAll has_split_dimension is not supported.");
      // for (int rank = 0; rank < num_participants; ++rank) {
      //   ccl::alltoall(
      //       static_cast<const void*>(send_buffer + rank * chunk_bytes),
      //       static_cast<void*>(recv_buffer + rank * chunk_bytes),
      //       chunk_elements, dtype, *comm, ccl_stream);
      // }
    }
  } else {
    TF_RET_CHECK(buffers_.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    std::vector<const void*> send_buffers;
    std::vector<void*> recv_buffers;
    for (size_t i = 0; i < buffers_.size(); ++i) {
      const Buffer& buffer = buffers_[i];
      const void* send_buffer =
          params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
              .opaque();
      void* recv_buffer =
          params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
              .opaque();
      send_buffers.push_back(send_buffer);
      recv_buffers.push_back(recv_buffer);
    }

    PrimitiveType element_type = config_.config.operand_element_type[0];
    int element_count = buffers_[0].element_count *
                        (primitive_util::IsComplexType(element_type) ? 2 : 1);

    itex_alltoall(send_buffers, recv_buffers, element_count, element_type,
                  gpu_stream, comm);
  }
#endif  // ITEX_USE_CCL
  ITEX_VLOG(1) << "Done performing AllToAll for ordinal: " << device_ordinal;
  return Status::OK();
}

}  // namespace gpu
}  // namespace itex_xla
