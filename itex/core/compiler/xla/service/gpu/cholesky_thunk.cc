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

#if ITEX_USE_MKL
#include "itex/core/compiler/xla/service/gpu/cholesky_thunk.h"

#include <complex>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "itex/core/compiler/xla/service/gpu/mkl.h"
#include "itex/core/compiler/xla/stream_executor/device_memory.h"
#include "itex/core/compiler/xla/stream_executor/scratch_allocator.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"
#include "itex/core/compiler/xla/types.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/logging.h"
#include "protos/xla_data.pb.h"

namespace itex_xla {
namespace gpu {

CholeskyThunk::CholeskyThunk(ThunkInfo thunk_info,
                             const CholeskyOptions& options,
                             // const se::GpuAsmOpts asm_opts,
                             BufferAllocation::Slice a_buffer,
                             BufferAllocation::Slice workspace_buffer,
                             BufferAllocation::Slice info_buffer,
                             PrimitiveType type, int64_t batch_size, int64_t n)
    : Thunk(Kind::kCholesky, thunk_info),
      uplo_(options.lower() ? oneapi::mkl::uplo::L : oneapi::mkl::uplo::U),
      a_buffer_(a_buffer),
      workspace_buffer_(workspace_buffer),
      type_(type),
      batch_size_(batch_size),
      n_(n) {}

Status CholeskyThunk::ExecuteOnStream(const ExecuteParams& params) {
  ITEX_VLOG(3) << "type=" << PrimitiveType_Name(type_)
               << " uplo=" << UpperLowerString(uplo_)
               << " batch_size=" << batch_size_ << " n=" << n_
               << " a=" << a_buffer_.ToString()
               << " workspace=" << workspace_buffer_.ToString();
  switch (type_) {
    case F32:
      return DoPotrfBatched<float>(params);
    case F64:
      return DoPotrfBatched<double>(params);
    case C64:
      return DoPotrfBatched<std::complex<float>>(params);
    case C128:
      return DoPotrfBatched<std::complex<double>>(params);
    default:
      return InvalidArgument("Invalid type for cholesky %s",
                             PrimitiveType_Name(type_));
  }
  return Status::OK();
}

template <typename T>
Status CholeskyThunk::DoPotrfBatched(const ExecuteParams& params) {
  // auto stream = params.stream->stream_handle;
  auto stream = stream_executor::gpu::AsGpuStreamValue(params.stream);
  auto& buffer_allocations = *params.buffer_allocations;
  T* a_base =
      static_cast<T*>(buffer_allocations.GetDeviceAddress(a_buffer_).opaque());
  T* scratch_data = static_cast<T*>(
      buffer_allocations.GetDeviceAddress(workspace_buffer_).opaque());
  int64_t scratchpad_size = workspace_buffer_.size() / sizeof(T);

  const int64_t stride_a = n_ * n_;
  auto lda = n_;
  try {
    oneapi::mkl::lapack::potrf_batch(*stream, uplo_, n_, a_base, lda, stride_a,
                                     batch_size_, scratch_data,
                                     scratchpad_size);
  } catch (oneapi::mkl::lapack::batch_error const& be) {
    int i = 0;
    auto& ids = be.ids();
    for (auto const& e : be.exceptions()) {
      try {
        std::rethrow_exception(e);
      } catch (oneapi::mkl::lapack::exception& e) {
        ITEX_LOG(ERROR) << "Exception " << ids[i++]
                        << " in a batch says: " << e.what()
                        << " (info code: " << e.info() << ")";
      }
    }
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_USE_MKL
