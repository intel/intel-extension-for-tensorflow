/* Copyright (c) 2023 Intel Corporation

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
#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_NCCL_OPS_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_NCCL_OPS_H_
#include <vector>

#include "itex/core/compiler/xla/service/collective_ops_utils.h"
#include "itex/core/compiler/xla/service/gpu/nccl_collective_thunk.h"

#if !ITEX_USE_CCL

namespace itex_xla {
namespace gpu {

void itex_allreduce(const void* send_buffer, void* recv_buffer,
                    int element_count, PrimitiveType dtype,
                    ReductionKind reduction_kind, ITEX_GPUStream* gpu_stream,
                    ncclComm_t comm);

void itex_allgather(const void* send_buffer, void* recv_buffer,
                    int element_count, PrimitiveType dtype,
                    ITEX_GPUStream* gpu_stream, ncclComm_t comm);

void itex_alltoall(std::vector<const void*> send_buffer,
                   std::vector<void*> recv_buffer, int element_count,
                   PrimitiveType dtype, ITEX_GPUStream* gpu_stream,
                   ncclComm_t comm);

void itex_reduce_scatter(const void* send_buffer, void* recv_buffer,
                         int element_count, PrimitiveType dtype,
                         ReductionKind reduction_kind,
                         ITEX_GPUStream* gpu_stream, ncclComm_t comm);

void itex_collective_permute(const void* send_buffer, void* recv_buffer,
                             int element_count, PrimitiveType dtype,
                             const absl::optional<int64_t>& source_id,
                             const absl::optional<int64_t>& target_id,
                             ITEX_GPUStream* gpu_stream, ncclComm_t comm);
}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_USE_CCL
#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_NCCL_OPS_H_
