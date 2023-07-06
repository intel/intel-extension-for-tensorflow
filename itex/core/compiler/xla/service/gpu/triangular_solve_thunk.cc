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
#include "itex/core/compiler/xla/service/gpu/triangular_solve_thunk.h"

#include <string>

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

namespace {
// populate an array of pointers:
//   [base + stride * i for i in range(n)].
//
void MakeBatchPointers_singlethread(char* base, int stride, int n,
                                    void** ptrs_out) {
  for (int idx = 0; idx < n; idx++) {
    ptrs_out[idx] = base + idx * stride;
  }
}

Status MakeBatchPointers(se::DeviceMemoryBase base_ptr, int stride_bytes, int n,
                         void** ptrs_out) {
  MakeBatchPointers_singlethread(reinterpret_cast<char*>(base_ptr.opaque()),
                                 stride_bytes, n, ptrs_out);
  return Status::OK();
}
}  // namespace

TriangularSolveThunk::TriangularSolveThunk(
    ThunkInfo thunk_info, const TriangularSolveOptions& options,
    // se::GpuAsmOpts asm_opts,  //
    const BufferAllocation::Slice& a_buffer,
    const BufferAllocation::Slice& b_buffer,
    const BufferAllocation::Slice& temp_buffer,  //
    PrimitiveType type, int64_t batch_size, int64_t m, int64_t n,
    int64_t a_batch_stride, int64_t b_batch_stride)
    : Thunk(Kind::kTriangularSolve, thunk_info),
      // asm_opts_(asm_opts),
      uplo_(options.lower() ? oneapi::mkl::uplo::L : oneapi::mkl::uplo::U),
      side_(options.left_side() ? oneapi::mkl::side::L : oneapi::mkl::side::R),
      unit_diagonal_(options.unit_diagonal() ? oneapi::mkl::diag::U
                                             : oneapi::mkl::diag::N),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      temp_buffer_(temp_buffer),
      type_(type),
      batch_size_(batch_size),
      m_(m),
      n_(n),
      a_batch_stride_(a_batch_stride),
      b_batch_stride_(b_batch_stride) {
  transpose_a_ = [&] {
    switch (options.transpose_a()) {
      case TriangularSolveOptions::NO_TRANSPOSE:
        return oneapi::mkl::transpose::N;
      case TriangularSolveOptions::TRANSPOSE:
        return oneapi::mkl::transpose::T;
      case TriangularSolveOptions::ADJOINT:
        return oneapi::mkl::transpose::C;
      default:
        ITEX_LOG(ERROR) << "Invalid triangular solve transpose value "
                        << options.transpose_a();
        return oneapi::mkl::transpose::N;
    }
  }();
}

Status TriangularSolveThunk::ExecuteOnStream(const ExecuteParams& params) {
  // auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;
  se::ScratchAllocator scratch_allocator(buffer_allocations.device_ordinal(),
                                         buffer_allocations.memory_allocator());

  ITEX_VLOG(3) << "uplo=" << UpperLowerString(uplo_)
               << " side=" << SideString(side_)
               << " diagonal=" << DiagonalString(unit_diagonal_)
               << " batch_size=" << batch_size_ << " m=" << m_ << " n=" << n_
               << " a_batch_stride=" << a_batch_stride_
               << " b_batch_stride=" << b_batch_stride_;

  const int lda = side_ == oneapi::mkl::side::L ? m_ : n_;
  const int ldb = m_;
  const int stride_a = lda * lda;
  const int stride_b = m_ * n_;

  se::DeviceMemoryBase a_data = buffer_allocations.GetDeviceAddress(a_buffer_);
  se::DeviceMemoryBase b_data = buffer_allocations.GetDeviceAddress(b_buffer_);
  void* workspace = nullptr;

  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(params.stream);
  // bool launch_ok;
  if (batch_size_ == 1) {
    switch (type_) {
      case F32: {
        oneapi::mkl::blas::trsm(*gpu_stream, side_, uplo_, transpose_a_,
                                unit_diagonal_, m_, n_, 1.0f,
                                static_cast<float*>(a_data.opaque()), lda,
                                static_cast<float*>(b_data.opaque()), ldb);
        break;
      }
      case F64: {
        oneapi::mkl::blas::trsm(*gpu_stream, side_, uplo_, transpose_a_,
                                unit_diagonal_, m_, n_, 1.0f,
                                static_cast<double*>(a_data.opaque()), lda,
                                static_cast<double*>(b_data.opaque()), ldb);
        break;
      }
      case C64: {
        oneapi::mkl::blas::trsm(
            *gpu_stream, side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
            1.0f, static_cast<std::complex<float>*>(a_data.opaque()), lda,
            static_cast<std::complex<float>*>(b_data.opaque()), ldb);
        break;
      }
      case C128: {
        oneapi::mkl::blas::trsm(
            *gpu_stream, side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
            1.0f, static_cast<std::complex<double>*>(a_data.opaque()), lda,
            static_cast<std::complex<double>*>(b_data.opaque()), ldb);
        break;
      }
      default:
        return InvalidArgument("Invalid type for triangular solve %d", type_);
    }
  } else {
    switch (type_) {
      case F32: {
        oneapi::mkl::blas::trsm_batch(
            *gpu_stream, side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
            1.0f, static_cast<float*>(a_data.opaque()), lda, stride_a,
            static_cast<float*>(b_data.opaque()), ldb, stride_b, batch_size_);
        break;
      }
      case F64: {
        oneapi::mkl::blas::trsm_batch(
            *gpu_stream, side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
            1.0f, static_cast<double*>(a_data.opaque()), lda, stride_a,
            static_cast<double*>(b_data.opaque()), ldb, stride_b, batch_size_);
        break;
      }
      case C64: {
        oneapi::mkl::blas::trsm_batch(
            *gpu_stream, side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
            1.0f, static_cast<std::complex<float>*>(a_data.opaque()), lda,
            stride_a, static_cast<std::complex<float>*>(b_data.opaque()), ldb,
            stride_b, batch_size_);
        break;
      }
      case C128: {
        oneapi::mkl::blas::trsm_batch(
            *gpu_stream, side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
            1.0f, static_cast<std::complex<double>*>(a_data.opaque()), lda,
            stride_a, static_cast<std::complex<double>*>(b_data.opaque()), ldb,
            stride_b, batch_size_);
        break;
      }
      default:
        return InvalidArgument("Invalid type for triangular solve %d", type_);
    }
  }

  return Status::OK();
}

}  // namespace gpu
}  // namespace itex_xla
#endif  // ITEX_USE_MKL
