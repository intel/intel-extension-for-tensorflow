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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_CONVOLUTION_THUNK_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_CONVOLUTION_THUNK_H_

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "itex/core/compiler/xla/service/buffer_assignment.h"
#include "itex/core/compiler/xla/service/gpu/buffer_allocations.h"
#include "itex/core/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "itex/core/compiler/xla/service/gpu/gpu_executable.h"
#include "itex/core/compiler/xla/service/gpu/thunk.h"
#include "itex/core/compiler/xla/service/hlo_instruction.h"
#include "itex/core/compiler/xla/service/hlo_instructions.h"
#include "itex/core/compiler/xla/types.h"
#include "itex/core/utils/status.h"
#include "protos/xla_data.pb.h"

namespace itex_xla {
namespace gpu {

class ConvolutionThunk : public Thunk {
 public:
  // Construct a thunk for launching a DNN convolution.
  //
  // This is thread-compatible.
  ConvolutionThunk(ThunkInfo thunk_info, GpuConvDescriptor descriptor,
                   std::vector<BufferAllocation::Slice> operand_slices,
                   BufferAllocation::Slice result_slice,
                   BufferAllocation::Slice scratch_slice);

  ConvolutionThunk(const ConvolutionThunk&) = delete;
  ConvolutionThunk& operator=(const ConvolutionThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::vector<BufferAllocation::Slice> operand_buffers_;
  BufferAllocation::Slice result_buffer_;
  BufferAllocation::Slice scratch_buffer_;

  const GpuConvDescriptor descriptor_;
  absl::Mutex mu_;
  absl::flat_hash_map<const se::Stream*, std::unique_ptr<OneDnnConvPrimitive>>
      onednn_primitives_;
  OneDnnConvPrimitive& GetOrCreateOneDnnConvPrimitive(
      se::Stream*, const std::vector<se::DeviceMemoryBase>& operand_se_buffers,
      const se::DeviceMemoryBase& result_buffer, const ExecuteParams& params);
};

}  // namespace gpu

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_CONVOLUTION_THUNK_H_
