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

#include "itex/core/compiler/xla/service/gpu/custom_call_thunk.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "itex/core/compiler/xla/service/buffer_assignment.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/errors.h"
namespace itex_xla {
namespace gpu {

CustomCallThunk::CustomCallThunk(ThunkInfo thunk_info,
                                 CustomCallTarget call_target,
                                 std::vector<OptionalSlice> operands,
                                 std::vector<OptionalSlice> results,
                                 const std::string& opaque)
    : Thunk(Thunk::kCustomCall, thunk_info),
      call_target_(std::move(call_target)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      opaque_(opaque) {}

Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());
  for (const std::vector<OptionalSlice>& slices : {operands_, results_}) {
    for (const OptionalSlice& slice : slices) {
      if (slice) {
        if (!slice->allocation())
          return InternalError("custom call input missing buffer allocation");
        buffers.push_back(
            params.buffer_allocations->GetDeviceAddress(*slice).opaque());
      } else {
        buffers.push_back(nullptr);
      }
    }
  }

  auto gpu_stream = se::gpu::AsGpuStreamValue(params.stream);
  ItexXlaCustomCallStatus custom_call_status;
  call_target_(gpu_stream, buffers.data(), opaque_.data(), opaque_.size(),
               &custom_call_status);
  auto message = CustomCallStatusGetMessage(&custom_call_status);
  if (message) {
    return InternalError("CustomCall failed: %s", *message);
  } else {
    return Status::OK();
  }
}

}  // namespace gpu
}  // namespace itex_xla
