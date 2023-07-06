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

#if ITEX_USE_MKL
#include "itex/core/compiler/xla/service/gpu/fft_thunk.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"
#include "itex/core/compiler/xla/types.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/logging.h"

namespace itex_xla {
namespace gpu {

namespace {

se::fft::Type FftTypeToSeType(FftType type, bool double_precision) {
  switch (type) {
    case FftType::FFT:
      return double_precision ? se::fft::Type::kZ2ZForward
                              : se::fft::Type::kC2CForward;
    case FftType::IFFT:
      return double_precision ? se::fft::Type::kZ2ZInverse
                              : se::fft::Type::kC2CInverse;
    case FftType::IRFFT:
      return double_precision ? se::fft::Type::kZ2D : se::fft::Type::kC2R;
    case FftType::RFFT:
      return double_precision ? se::fft::Type::kD2Z : se::fft::Type::kR2C;
    default:
      ITEX_LOG(FATAL) << "unsupported fft type";
  }
}

std::string FftTypeToString(se::fft::Type type) {
  switch (type) {
    case se::fft::Type::kC2CForward:
    case se::fft::Type::kZ2ZForward:
      return "FFT";
    case se::fft::Type::kC2CInverse:
    case se::fft::Type::kZ2ZInverse:
      return "IFFT";
    case se::fft::Type::kC2R:
    case se::fft::Type::kZ2D:
      return "IRFFT";
    case se::fft::Type::kR2C:
    case se::fft::Type::kD2Z:
      return "RFFT";
    default:
      ITEX_LOG(FATAL) << "unknown fft type";
  }
}

template <typename T>
class XLAFillCongugateSymmetry;

template <typename T>
void LaunchXLAFillConjugateSymmetry(se::Stream* stream, T* tensor_ptr,
                                    T* tmp_tensor_ptr, int64_t stride,
                                    int64_t tmp_stride, int64_t tensor_size) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  const auto num_elements = tensor_size / stride;
  const int work_group_size =
      (*gpu_stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const auto num_work_groups =
      (num_elements + work_group_size - 1) / work_group_size;

  gpu_stream->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<XLAFillCongugateSymmetry<T>>(
        sycl::nd_range<1>(sycl::range<1>(num_work_groups * work_group_size),
                          sycl::range<1>(work_group_size)),
        [=](sycl::nd_item<1> item) {
          auto idx = item.get_global_linear_id();
          if (idx >= num_elements) {
            return;
          }

          T* tensor_begin = tensor_ptr + idx * stride;
          T* tmp_tensor_begin = tmp_tensor_ptr + idx * tmp_stride;
          for (int i = 0; i < stride; ++i) {
            tensor_begin[i] = tmp_tensor_begin[i];
          }
        });
  });
}

}  // namespace

FftThunk::FftThunk(ThunkInfo thunk_info, FftType fft_type,
                   absl::Span<const int64_t> fft_length,
                   const BufferAllocation::Slice& input_buffer,
                   const BufferAllocation::Slice& output_buffer,
                   const Shape& input_shape, const Shape& output_shape)
    : Thunk(Kind::kFft, thunk_info),
      fft_type_(
          FftTypeToSeType(fft_type, input_shape.element_type() == F64 ||
                                        input_shape.element_type() == C128)),
      fft_length_(fft_length.begin(), fft_length.end()),
      scale_factor_(1.0),
      input_buffer_(input_buffer),
      output_buffer_(output_buffer),
      input_shape_(input_shape),
      output_shape_(output_shape) {}

Status FftThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  ITEX_VLOG(3) << "FFT type: " << FftTypeToString(fft_type_);
  ITEX_VLOG(3) << "Input shape: "
               << ShapeUtil::HumanStringWithLayout(input_shape_);
  ITEX_VLOG(3) << "Output shape: "
               << ShapeUtil::HumanStringWithLayout(output_shape_);

  se::ScratchAllocator scratch_allocator(buffer_allocations.device_ordinal(),
                                         buffer_allocations.memory_allocator());

  switch (fft_type_) {
    case se::fft::Type::kC2CForward:
    case se::fft::Type::kC2CInverse: {
      se::DeviceMemory<complex64> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex64> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      float* in = reinterpret_cast<float*>(
          stream_executor::gpu::GpuMemoryMutable(&input_data));
      float* out = reinterpret_cast<float*>(
          stream_executor::gpu::GpuMemoryMutable(&output_data));
      DoFFTInternal<oneapi::mkl::dft::precision::SINGLE,
                    oneapi::mkl::dft::domain::COMPLEX, float>(
          stream, in, out, scratch_allocator);
      break;
    }
    case se::fft::Type::kZ2ZForward:
    case se::fft::Type::kZ2ZInverse: {
      se::DeviceMemory<complex128> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex128> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      double* in = reinterpret_cast<double*>(
          stream_executor::gpu::GpuMemoryMutable(&input_data));
      double* out = reinterpret_cast<double*>(
          stream_executor::gpu::GpuMemoryMutable(&output_data));
      DoFFTInternal<oneapi::mkl::dft::precision::DOUBLE,
                    oneapi::mkl::dft::domain::COMPLEX, double>(
          stream, in, out, scratch_allocator);
      break;
    }
    case se::fft::Type::kR2C: {
      se::DeviceMemory<float> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex64> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      float* in = reinterpret_cast<float*>(
          stream_executor::gpu::GpuMemoryMutable(&input_data));
      float* out = reinterpret_cast<float*>(
          stream_executor::gpu::GpuMemoryMutable(&output_data));
      DoFFTInternal<oneapi::mkl::dft::precision::SINGLE,
                    oneapi::mkl::dft::domain::REAL, float>(stream, in, out,
                                                           scratch_allocator);
      break;
    }
    case se::fft::Type::kC2R: {
      se::DeviceMemory<complex64> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<float> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      float* in = reinterpret_cast<float*>(
          stream_executor::gpu::GpuMemoryMutable(&input_data));
      float* out = reinterpret_cast<float*>(
          stream_executor::gpu::GpuMemoryMutable(&output_data));
      DoFFTInternal<oneapi::mkl::dft::precision::SINGLE,
                    oneapi::mkl::dft::domain::REAL, float>(stream, in, out,
                                                           scratch_allocator);
      break;
    }
    case se::fft::Type::kD2Z: {
      se::DeviceMemory<double> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex128> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      double* in = reinterpret_cast<double*>(
          stream_executor::gpu::GpuMemoryMutable(&input_data));
      double* out = reinterpret_cast<double*>(
          stream_executor::gpu::GpuMemoryMutable(&output_data));
      DoFFTInternal<oneapi::mkl::dft::precision::DOUBLE,
                    oneapi::mkl::dft::domain::REAL, double>(stream, in, out,
                                                            scratch_allocator);
      break;
    }
    case se::fft::Type::kZ2D: {
      se::DeviceMemory<complex128> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<double> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      double* in = reinterpret_cast<double*>(
          stream_executor::gpu::GpuMemoryMutable(&input_data));
      double* out = reinterpret_cast<double*>(
          stream_executor::gpu::GpuMemoryMutable(&output_data));
      DoFFTInternal<oneapi::mkl::dft::precision::DOUBLE,
                    oneapi::mkl::dft::domain::REAL, double>(stream, in, out,
                                                            scratch_allocator);
      break;
    }
    default:
      ITEX_LOG(FATAL) << "unsupported fft type";
  }
}

template <oneapi::mkl::dft::precision P, oneapi::mkl::dft::domain D, typename T>
Status FftThunk::DoFFTInternal(se::Stream* stream, T* in_buffer, T* out_buffer,
                               se::ScratchAllocator& allocator) {
  const int64_t fft_rank = fft_length_.size();
  ITEX_CHECK_LE(fft_rank, 3);

  int64_t fft_length[3];
  int64_t input_embed[3];
  const uint64_t input_stride = 1;
  uint64_t input_distance = 1;
  int64_t output_embed[3];
  const uint64_t output_stride = 1;
  uint64_t output_distance = 1;

  for (int i = 0; i < fft_rank; ++i) {
    auto dim_offset = input_shape_.dimensions_size() - fft_rank + i;
    fft_length[i] = fft_length_[i];
    input_embed[i] = input_shape_.dimensions(dim_offset);
    input_distance *= input_shape_.dimensions(dim_offset);
    output_embed[i] = output_shape_.dimensions(dim_offset);
    output_distance *= output_shape_.dimensions(dim_offset);
  }

  int batch_size = 1;
  for (int i = 0; i < input_shape_.dimensions_size() - fft_rank; ++i) {
    batch_size *= input_shape_.dimensions(i);
  }
  bool is_input_complex;
  bool is_output_complex;
  bool is_forward;
  bool is_real;

  switch (fft_type_) {
    case se::fft::Type::kC2CForward:
    case se::fft::Type::kZ2ZForward: {
      is_forward = true;
      is_real = false;
      break;
    }
    case se::fft::Type::kR2C:
    case se::fft::Type::kD2Z: {
      is_forward = true;
      is_real = true;
      break;
    }
    case se::fft::Type::kC2CInverse:
    case se::fft::Type::kZ2ZInverse: {
      is_forward = false;
      is_real = false;
      break;
    }
    case se::fft::Type::kC2R:
    case se::fft::Type::kZ2D: {
      is_forward = false;
      is_real = true;
      break;
    }
    default:
      ITEX_LOG(FATAL) << "unsupported fft type";
  }

  std::vector<std::int64_t> dims_vec(fft_rank, 0);
  switch (fft_type_) {
    case se::fft::Type::kC2CForward:
    case se::fft::Type::kZ2ZForward:
    case se::fft::Type::kR2C:
    case se::fft::Type::kD2Z: {
      scale_factor_ = 1.0 / input_distance;
      for (int i = 0; i < fft_rank; ++i) {
        dims_vec[i] = input_embed[i];
      }
      break;
    }
    case se::fft::Type::kC2CInverse:
    case se::fft::Type::kZ2ZInverse:
    case se::fft::Type::kC2R:
    case se::fft::Type::kZ2D: {
      scale_factor_ = 1.0 / output_distance;
      for (int i = 0; i < fft_rank; ++i) {
        dims_vec[i] = output_embed[i];
      }
      break;
    }
    default:
      ITEX_LOG(FATAL) << "unsupported fft type";
  }

  oneapi::mkl::dft::descriptor<P, D> desc(dims_vec);
  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 batch_size);

  if (is_real) {
    std::vector<int64_t> mkl_istrides(1 + fft_rank, 0);
    std::vector<int64_t> mkl_ostrides(1 + fft_rank, 0);
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);

    int64_t tmp_istride = 1, tmp_ostride = 1;
    for (int64_t i = fft_rank; i > 0; --i) {
      if (is_input_complex && !is_output_complex) {
        if (i == (fft_rank - 1)) {
          tmp_istride = input_embed[i];
          tmp_ostride = (output_embed[i] / 2 + 1) * 2;
        }
      }
      mkl_istrides[i] = tmp_istride;
      mkl_ostrides[i] = tmp_ostride;
      tmp_istride *= input_embed[i - 1];
      tmp_ostride *= output_embed[i - 1];
    }
    desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                   mkl_istrides.data());
    desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   mkl_ostrides.data());
    if (is_forward) {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, tmp_istride);
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, tmp_ostride);
    } else {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, tmp_ostride);
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, tmp_istride);
      desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                     static_cast<T>(scale_factor_));
    }
  } else {
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   input_distance);
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   output_distance);
    desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                   static_cast<T>(scale_factor_));
  }

  desc.commit(*stream_executor::gpu::AsGpuStreamValue(stream));

  sycl::event fft_event;
  if (is_forward) {
    fft_event = oneapi::mkl::dft::compute_forward(desc, in_buffer, out_buffer);
  } else {
    if (is_input_complex && !is_output_complex && fft_rank > 1) {
      int workspace_size = 1;
      for (int i = 0; i < output_shape_.dimensions_size() - 1; ++i) {
        workspace_size *= output_shape_.dimensions(i);
      }
      int out_size = workspace_size * output_shape_.dimensions(
                                          output_shape_.dimensions_size() - 1);
      workspace_size =
          workspace_size * (output_embed[fft_rank - 1] / 2 + 1) * 2;

      void* workspace;
      TF_RETURN_IF_ERROR(se::AllocateWorkspace(&workspace, &allocator,
                                               workspace_size * sizeof(T)));
      T* tmp_out_buffer = reinterpret_cast<T*>(workspace);

      fft_event =
          oneapi::mkl::dft::compute_backward(desc, in_buffer, tmp_out_buffer);
      LaunchXLAFillConjugateSymmetry<T>(
          stream, out_buffer, tmp_out_buffer,
          output_shape_.dimensions(output_shape_.dimensions_size() - 1),
          (output_embed[fft_rank - 1] / 2 + 1) * 2, out_size);
    } else {
      fft_event =
          oneapi::mkl::dft::compute_backward(desc, in_buffer, out_buffer);
    }
  }
  fft_event.wait();
  return Status::OK();
}

}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_USE_MKL
