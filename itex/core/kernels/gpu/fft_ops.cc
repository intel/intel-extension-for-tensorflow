/* Copyright (c) 2021-2022 Intel Corporation

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

#if ITEX_USE_MKL
#include "itex/core/kernels/gpu/fft_ops.h"

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "mkl.h"  // NOLINT(build/include_subdir)
#include "oneapi/mkl/dfti.hpp"
#include "oneapi/mkl/exceptions.hpp"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct FillCongugateSymmetry {
  FillCongugateSymmetry(size_t num_elements, int64_t stride, int64_t tmp_stride,
                        T* tensor_ptr, T* tmp_tensor_ptr)
      : num_elements(num_elements),
        stride(stride),
        tmp_stride(tmp_stride),
        tensor_ptr(tensor_ptr),
        tmp_tensor_ptr(tmp_tensor_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto idx = item.get_global_linear_id();
    if (idx >= num_elements) {
      return;
    }

    T* tensor_begin = tensor_ptr + idx * stride;
    T* tmp_tensor_begin = tmp_tensor_ptr + idx * tmp_stride;
    for (int i = 0; i < stride; ++i) {
      tensor_begin[i] = tmp_tensor_begin[i];
    }
  }

 private:
  size_t num_elements;
  int64_t stride;
  int64_t tmp_stride;
  T* tensor_ptr;
  T* tmp_tensor_ptr;
};

template <typename T>
void LaunchFillConjugateSymmetry(OpKernelContext* ctx, T* tensor_ptr,
                                 T* tmp_tensor_ptr, int64_t stride,
                                 int64_t tmp_stride, int64_t tensor_size) {
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  const auto num_elements = tensor_size / stride;
  const int work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const auto num_work_groups =
      (num_elements + work_group_size - 1) / work_group_size;

  stream->submit([&](sycl::handler& cgh) {
    FillCongugateSymmetry<T> task(num_elements, stride, tmp_stride, tensor_ptr,
                                  tmp_tensor_ptr);
    cgh.parallel_for<FillCongugateSymmetry<T>>(
        sycl::nd_range<1>(sycl::range<1>(num_work_groups * work_group_size),
                          sycl::range<1>(work_group_size)),
        task);
  });
}

class FFTGPUBase : public FFTBase {
 public:
  using FFTBase::FFTBase;

 protected:
  void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
             Tensor* out) override {
    const TensorShape& input_shape = in.shape();
    const TensorShape& output_shape = out->shape();
    const int fft_rank = Rank();
    bool is_input_complex =
        (in.dtype() == DT_COMPLEX64 || in.dtype() == DT_COMPLEX128);
    bool is_output_complex =
        (out->dtype() == DT_COMPLEX64 || out->dtype() == DT_COMPLEX128);
    bool is_complex_domain = IsForward() ? is_input_complex : is_output_complex;
    int batch_size = 1;
    for (int i = 0; i < input_shape.dims() - fft_rank; ++i) {
      batch_size *= input_shape.dim_size(i);
    }
    std::vector<std::int64_t> input_embed(fft_rank, 0);
    std::vector<std::int64_t> output_embed(fft_rank, 0);
    uint64 input_distance = 1, output_distance = 1;
    for (int i = 0; i < fft_rank; ++i) {
      auto dim_offset = input_shape.dims() - fft_rank + i;
      input_embed[i] = input_shape.dim_size(dim_offset);
      input_distance *= input_shape.dim_size(dim_offset);
      output_embed[i] = output_shape.dim_size(dim_offset);
      output_distance *= output_shape.dim_size(dim_offset);
    }

    if (in.dtype() == DT_FLOAT || in.dtype() == DT_COMPLEX64) {
      if (is_complex_domain) {
        DoFFTInternal<oneapi::mkl::dft::precision::SINGLE,
                      oneapi::mkl::dft::domain::COMPLEX, float>(
            ctx, in, out, fft_rank, batch_size, is_input_complex,
            is_output_complex, input_embed, output_embed, input_distance,
            output_distance);
      } else {
        DoFFTInternal<oneapi::mkl::dft::precision::SINGLE,
                      oneapi::mkl::dft::domain::REAL, float>(
            ctx, in, out, fft_rank, batch_size, is_input_complex,
            is_output_complex, input_embed, output_embed, input_distance,
            output_distance);
      }
    } else if (in.dtype() == DT_DOUBLE || in.dtype() == DT_COMPLEX128) {
      if (is_complex_domain) {
        DoFFTInternal<oneapi::mkl::dft::precision::DOUBLE,
                      oneapi::mkl::dft::domain::COMPLEX, double>(
            ctx, in, out, fft_rank, batch_size, is_input_complex,
            is_output_complex, input_embed, output_embed, input_distance,
            output_distance);
      } else {
        DoFFTInternal<oneapi::mkl::dft::precision::DOUBLE,
                      oneapi::mkl::dft::domain::REAL, double>(
            ctx, in, out, fft_rank, batch_size, is_input_complex,
            is_output_complex, input_embed, output_embed, input_distance,
            output_distance);
      }
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument(
                      "Unsupported data type, in=", DataTypeString(in.dtype()),
                      ", out=", DataTypeString(out->dtype())));
    }
  }

 private:
  template <oneapi::mkl::dft::precision P, oneapi::mkl::dft::domain D,
            typename T>
  void DoFFTInternal(OpKernelContext* ctx, const Tensor& in, Tensor* out,
                     int fft_rank, int batch_size, bool is_input_complex,
                     bool is_output_complex,
                     const std::vector<std::int64_t>& input_embed,
                     const std::vector<std::int64_t>& output_embed,
                     uint64 input_distance, uint64 output_distance) {
    auto* stream = ctx->GetDeviceStream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
    std::int64_t signal_numel = 1;
    T *in_buffer, *out_buffer;
    if (is_input_complex) {
      in_buffer = const_cast<T*>(
          reinterpret_cast<const T*>(in.flat<std::complex<T>>().data()));
    } else {
      in_buffer =
          const_cast<T*>(reinterpret_cast<const T*>(in.flat<T>().data()));
    }
    if (is_output_complex) {
      out_buffer = reinterpret_cast<T*>(out->flat<std::complex<T>>().data());
    } else {
      out_buffer = reinterpret_cast<T*>(out->flat<T>().data());
    }
    std::vector<std::int64_t> dims_vec(fft_rank, 0);
    for (int i = 0; i < fft_rank; ++i) {
      if (IsForward()) {
        dims_vec[i] = input_embed[i];
        signal_numel *= input_embed[i];
      } else {
        dims_vec[i] = output_embed[i];
        signal_numel *= output_embed[i];
      }
    }
    double double_scale = 1.0 / static_cast<double>(signal_numel);
    try {
      oneapi::mkl::dft::descriptor<P, D> desc(dims_vec);

      desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                     DFTI_NOT_INPLACE);
      desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                     batch_size);

      if (IsReal()) {
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
        if (IsForward()) {
          desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                         tmp_istride);
          desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                         tmp_ostride);
        } else {
          desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                         tmp_ostride);
          desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                         tmp_istride);
          desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                         static_cast<T>(double_scale));
        }
      } else {
        desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                       input_distance);
        desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                       output_distance);
        desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                       static_cast<T>(double_scale));
      }

      desc.commit(*stream);

      sycl::event fft_event;
      if (IsForward()) {
        fft_event =
            oneapi::mkl::dft::compute_forward(desc, in_buffer, out_buffer);
      } else {
        if (is_input_complex && !is_output_complex && fft_rank > 1) {
          Tensor tmp_out;
          TensorShape tmp_out_shape;
          for (int i = 0; i < out->dims() - 1; ++i) {
            tmp_out_shape.AddDim(out->dim_size(i));
          }
          tmp_out_shape.AddDim((output_embed[fft_rank - 1] / 2 + 1) * 2);

          OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                                 tmp_out_shape, &tmp_out));
          T* tmp_out_buffer = reinterpret_cast<T*>(tmp_out.flat<T>().data());
          fft_event = oneapi::mkl::dft::compute_backward(desc, in_buffer,
                                                         tmp_out_buffer);
          LaunchFillConjugateSymmetry<T>(
              ctx, out_buffer, tmp_out_buffer, out->dim_size(out->dims() - 1),
              tmp_out.dim_size(tmp_out.dims() - 1), out->NumElements());
        } else {
          fft_event =
              oneapi::mkl::dft::compute_backward(desc, in_buffer, out_buffer);
        }
      }
      fft_event.wait();
    } catch (const oneapi::mkl::exception& e) {
      ITEX_LOG(ERROR)
          << "Unexpected exception caught during call to dft API, error "
             "messages: "
          << e.what();
      OP_REQUIRES(ctx, false,
                  errors::Aborted("Aborted during call to dft API."));
    }
  }
};

template <bool Forward, bool _Real, int FFTRank>
class FFTGPU : public FFTGPUBase {
 public:
  static_assert(FFTRank >= 1 && FFTRank <= 3,
                "Only 1D, 2D and 3D FFTs supported.");
  explicit FFTGPU(OpKernelConstruction* ctx) : FFTGPUBase(ctx) {}

 protected:
  int Rank() const override { return FFTRank; }
  bool IsForward() const override { return Forward; }
  bool IsReal() const override { return _Real; }
};

REGISTER_KERNEL_BUILDER(Name("FFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("IFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("FFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("IFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 3>);

REGISTER_KERNEL_BUILDER(
    Name("RFFT").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 1>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 1>);
REGISTER_KERNEL_BUILDER(
    Name("RFFT2D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 2>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT2D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 2>);
REGISTER_KERNEL_BUILDER(
    Name("RFFT3D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 3>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT3D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 3>);

// Deprecated kernels.
REGISTER_KERNEL_BUILDER(Name("BatchFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 3>);

}  // namespace itex
#endif  // ITEX_USE_MKL
