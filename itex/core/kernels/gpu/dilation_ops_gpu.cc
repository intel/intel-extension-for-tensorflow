/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <cfloat>
#include <vector>

#include "itex/core/kernels/gpu/dilation_ops.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
struct DilationKernel {
  DilationKernel(const T* input_ptr, const T* filter_ptr, int total_count,
                 int batch, int input_rows, int input_cols, int depth,
                 int filter_rows, int filter_cols, int output_rows,
                 int output_cols, int stride_rows, int stride_cols,
                 int rate_rows, int rate_cols, int pad_top, int pad_left,
                 T* output_ptr)
      : input_ptr_(input_ptr),
        filter_ptr_(filter_ptr),
        output_ptr_(output_ptr),
        total_count_(total_count),
        batch_(batch),
        input_rows_(input_rows),
        input_cols_(input_cols),
        depth_(depth),
        filter_rows_(filter_rows),
        filter_cols_(filter_cols),
        output_rows_(output_rows),
        output_cols_(output_cols),
        stride_rows_(stride_rows),
        stride_cols_(stride_cols),
        rate_rows_(rate_rows),
        rate_cols_(rate_cols),
        pad_top_(pad_top),
        pad_left_(pad_left) {}

  void operator()(sycl::nd_item<1> item) const {
    auto out_idx = item.get_global_linear_id();

    if (out_idx >= total_count_) return;

    // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
    const int d = out_idx % depth_;
    const int out_idx2 = out_idx / depth_;
    const int w_out = out_idx2 % output_cols_;
    const int out_idx3 = out_idx2 / output_cols_;
    const int h_out = out_idx3 % output_rows_;
    const int b = out_idx3 / output_rows_;
    int h_beg = h_out * stride_rows_ - pad_top_;
    int w_beg = w_out * stride_cols_ - pad_left_;
    T cur_val = Eigen::NumTraits<T>::lowest();
    for (int h = 0; h < filter_rows_; ++h) {
      const int h_in = h_beg + h * rate_rows_;
      if (h_in >= 0 && h_in < input_rows_) {
        for (int w = 0; w < filter_cols_; ++w) {
          const int w_in = w_beg + w * rate_cols_;
          if (w_in >= 0 && w_in < input_cols_) {
            const T val =
                input_ptr_[d + depth_ * (w_in + input_cols_ *
                                                    (h_in + input_rows_ * b))] +
                filter_ptr_[d + depth_ * (w + filter_cols_ * h)];
            if (val > cur_val) {
              cur_val = val;
            }
          }
        }
      }
    }
    output_ptr_[out_idx] = cur_val;
  }

 private:
  const T* input_ptr_;
  const T* filter_ptr_;
  T* output_ptr_;
  int total_count_, batch_, input_rows_, input_cols_, depth_;
  int filter_rows_, filter_cols_, output_rows_, output_cols_;
  int stride_rows_, stride_cols_, rate_rows_, rate_cols_;
  int pad_top_, pad_left_;
};

template <typename T, typename OutT = float>
struct DilationBackpropInputKernel {
  DilationBackpropInputKernel(const T* input_ptr, const T* filter_ptr,
                              const T* out_backprop_ptr, int total_count,
                              int batch, int input_rows, int input_cols,
                              int depth, int filter_rows, int filter_cols,
                              int output_rows, int output_cols, int stride_rows,
                              int stride_cols, int rate_rows, int rate_cols,
                              int pad_top, int pad_left, OutT* in_backprop_ptr)
      : input_ptr_(input_ptr),
        filter_ptr_(filter_ptr),
        out_backprop_ptr_(out_backprop_ptr),
        in_backprop_ptr_(in_backprop_ptr),
        total_count_(total_count),
        batch_(batch),
        input_rows_(input_rows),
        input_cols_(input_cols),
        depth_(depth),
        filter_rows_(filter_rows),
        filter_cols_(filter_cols),
        output_rows_(output_rows),
        output_cols_(output_cols),
        stride_rows_(stride_rows),
        stride_cols_(stride_cols),
        rate_rows_(rate_rows),
        rate_cols_(rate_cols),
        pad_top_(pad_top),
        pad_left_(pad_left) {}

  void operator()(sycl::nd_item<1> item) const {
    auto out_idx = item.get_global_linear_id();

    if (out_idx >= total_count_) return;
    // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
    const int d = out_idx % depth_;
    const int out_idx2 = out_idx / depth_;
    const int w_out = out_idx2 % output_cols_;
    const int out_idx3 = out_idx2 / output_cols_;
    const int h_out = out_idx3 % output_rows_;
    const int b = out_idx3 / output_rows_;
    int h_beg = h_out * stride_rows_ - pad_top_;
    int w_beg = w_out * stride_cols_ - pad_left_;
    T cur_val = Eigen::NumTraits<T>::lowest();
    int h_in_max = (h_beg < 0) ? 0 : h_beg;
    int w_in_max = (w_beg < 0) ? 0 : w_beg;
    // In the case of multiple argmax branches, we only back-propagate along the
    // last branch, i.e., the one with largest value of `h * filter_cols + w`,
    // similarly to the max-pooling backward routines.
    for (int h = 0; h < filter_rows_; ++h) {
      const int h_in = h_beg + h * rate_rows_;
      if (h_in >= 0 && h_in < input_rows_) {
        for (int w = 0; w < filter_cols_; ++w) {
          const int w_in = w_beg + w * rate_cols_;
          if (w_in >= 0 && w_in < input_cols_) {
            const T val =
                input_ptr_[d + depth_ * (w_in + input_cols_ *
                                                    (h_in + input_rows_ * b))] +
                filter_ptr_[d + depth_ * (w + filter_cols_ * h)];
            if (val > cur_val) {
              cur_val = val;
              h_in_max = h_in;
              w_in_max = w_in;
            }
          }
        }
      }
    }
    ItexAtomicAdd(
        in_backprop_ptr_ + d +
            depth_ * (w_in_max + input_cols_ * (h_in_max + input_rows_ * b)),
        static_cast<OutT>(out_backprop_ptr_[out_idx]));
  }

 private:
  const T* input_ptr_;
  const T* filter_ptr_;
  const T* out_backprop_ptr_;
  OutT* in_backprop_ptr_;
  int total_count_, batch_, input_rows_, input_cols_, depth_;
  int filter_rows_, filter_cols_, output_rows_, output_cols_;
  int stride_rows_, stride_cols_, rate_rows_, rate_cols_;
  int pad_top_, pad_left_;
};

template <typename T, bool enable_slm, typename OutT = float>
struct DilationBackpropFilterKernel {
  using LocalMem = sycl::local_accessor<OutT, 1>;
  DilationBackpropFilterKernel(const T* input_ptr, const T* filter_ptr,
                               const T* out_backprop_ptr, int total_count,
                               int batch, int input_rows, int input_cols,
                               int depth, int filter_rows, int filter_cols,
                               int output_rows, int output_cols,
                               int stride_rows, int stride_cols, int rate_rows,
                               int rate_cols, int pad_top, int pad_left,
                               OutT* filter_backprop_ptr, LocalMem scratch,
                               int mem_size)
      : input_ptr_(input_ptr),
        filter_ptr_(filter_ptr),
        out_backprop_ptr_(out_backprop_ptr),
        filter_backprop_ptr_(filter_backprop_ptr),
        total_count_(total_count),
        batch_(batch),
        input_rows_(input_rows),
        input_cols_(input_cols),
        depth_(depth),
        filter_rows_(filter_rows),
        filter_cols_(filter_cols),
        output_rows_(output_rows),
        output_cols_(output_cols),
        stride_rows_(stride_rows),
        stride_cols_(stride_cols),
        rate_rows_(rate_rows),
        rate_cols_(rate_cols),
        pad_top_(pad_top),
        pad_left_(pad_left),
        scratch_(scratch),
        mem_size_(mem_size) {}

  void operator()(sycl::nd_item<1> item) const {
    auto out_idx = item.get_global_linear_id();

    int offset = 0;
    if (out_idx < total_count_) {
      // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
      const int d = out_idx % depth_;
      const int out_idx2 = out_idx / depth_;
      const int w_out = out_idx2 % output_cols_;
      const int out_idx3 = out_idx2 / output_cols_;
      const int h_out = out_idx3 % output_rows_;
      const int b = out_idx3 / output_rows_;
      int h_beg = h_out * stride_rows_ - pad_top_;
      int w_beg = w_out * stride_cols_ - pad_left_;
      T cur_val = Eigen::NumTraits<T>::lowest();
      int h_max = 0;
      int w_max = 0;
      // In the case of multiple argmax branches, we only back-propagate along
      // the last branch, i.e., the one with largest value of `h * filter_cols +
      // w`, similarly to the max-pooling backward routines.
      for (int h = 0; h < filter_rows_; ++h) {
        const int h_in = h_beg + h * rate_rows_;
        if (h_in >= 0 && h_in < input_rows_) {
          for (int w = 0; w < filter_cols_; ++w) {
            const int w_in = w_beg + w * rate_cols_;
            if (w_in >= 0 && w_in < input_cols_) {
              const T val =
                  input_ptr_[d +
                             depth_ * (w_in + input_cols_ *
                                                  (h_in + input_rows_ * b))] +
                  filter_ptr_[d + depth_ * (w + filter_cols_ * h)];
              if (val > cur_val) {
                cur_val = val;
                h_max = h;
                w_max = w;
              }
            }
          }
        }
      }
      offset = d + depth_ * (w_max + filter_cols_ * h_max);
    }

    if (enable_slm) {
      auto local_id = item.get_local_linear_id();
      for (auto i = local_id; i < mem_size_; i += item.get_local_range(0)) {
        scratch_[i] = 0;
      }
      item.barrier(sycl::access::fence_space::local_space);
      if (out_idx < total_count_) {
        sycl::atomic_ref<OutT, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            atm_dest_local(scratch_[offset]);
        atm_dest_local += static_cast<OutT>(out_backprop_ptr_[out_idx]);
      }
      item.barrier(sycl::access::fence_space::local_space);
      for (int i = local_id; i < mem_size_; i += item.get_local_range(0)) {
        ItexAtomicAdd<OutT>(filter_backprop_ptr_ + i, scratch_[i]);
      }
    } else {
      if (out_idx < total_count_) {
        ItexAtomicAdd<OutT>(filter_backprop_ptr_ + offset,
                            static_cast<OutT>(out_backprop_ptr_[out_idx]));
      }
    }
  }

 private:
  const T* input_ptr_;
  const T* filter_ptr_;
  const T* out_backprop_ptr_;
  OutT* filter_backprop_ptr_;
  int total_count_, batch_, input_rows_, input_cols_, depth_;
  int filter_rows_, filter_cols_, output_rows_, output_cols_;
  int stride_rows_, stride_cols_, rate_rows_, rate_cols_;
  int pad_top_, pad_left_;
  LocalMem scratch_;
  int mem_size_;
};

}  // namespace

namespace functor {

template <typename T>
struct Dilation<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter, int stride_rows,
                  int stride_cols, int rate_rows, int rate_cols, int pad_top,
                  int pad_left, typename TTypes<T, 4>::Tensor output) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = output.dimension(1);
    const int output_cols = output.dimension(2);

    const int total_count = batch * output_rows * output_cols * depth;
    if (total_count == 0) return;

    auto stream = d.stream();

    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (total_count + group_size - 1) / group_size;

    const T* input_ptr = input.data();
    const T* filter_ptr = filter.data();
    T* output_ptr = output.data();

    auto event = stream->submit([&](sycl::handler& cgh) {
      DilationKernel<T> task(input_ptr, filter_ptr, total_count, batch,
                             input_rows, input_cols, depth, filter_rows,
                             filter_cols, output_rows, output_cols, stride_rows,
                             stride_cols, rate_rows, rate_cols, pad_top,
                             pad_left, output_ptr);

      cgh.parallel_for<DilationKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          task);
    });
  }
};

template <typename T, typename OutT>
struct DilationBackpropInput<GPUDevice, T, OutT> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<OutT, 4>::Tensor in_backprop) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = out_backprop.dimension(1);
    const int output_cols = out_backprop.dimension(2);

    // Initialize in_backprop with all zeros.
    in_backprop.device(d) = in_backprop.constant(static_cast<OutT>(0));

    // Accumulate.
    const int total_count = batch * output_rows * output_cols * depth;
    auto stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (total_count + group_size - 1) / group_size;

    const T* input_ptr = input.data();
    const T* filter_ptr = filter.data();
    const T* out_backprop_ptr = out_backprop.data();
    OutT* in_backprop_ptr = in_backprop.data();

    auto event = stream->submit([&](sycl::handler& cgh) {
      DilationBackpropInputKernel<T, OutT> task(
          input_ptr, filter_ptr, out_backprop_ptr, total_count, batch,
          input_rows, input_cols, depth, filter_rows, filter_cols, output_rows,
          output_cols, stride_rows, stride_cols, rate_rows, rate_cols, pad_top,
          pad_left, in_backprop_ptr);

      cgh.parallel_for<DilationBackpropInputKernel<T, OutT>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          task);
    });
  }
};

template <typename T, typename OutT>
struct DilationBackpropFilter<GPUDevice, T, OutT> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<OutT, 3>::Tensor filter_backprop) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = out_backprop.dimension(1);
    const int output_cols = out_backprop.dimension(2);

    // Initialize filter_backprop with all zeros.
    filter_backprop.device(d) = filter_backprop.constant(static_cast<OutT>(0));

    // Accumulate.
    const int total_count = batch * output_rows * output_cols * depth;
    auto stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (total_count + group_size - 1) / group_size;

    const T* input_ptr = input.data();
    const T* filter_ptr = filter.data();
    const T* out_backprop_ptr = out_backprop.data();
    OutT* filter_backprop_ptr = filter_backprop.data();

    auto slm_size =
        stream->get_device()
            .template get_info<sycl::info::device::local_mem_size>();
    auto slm_used = filter_backprop.size() * sizeof(OutT);

#define SubmitDilationBackPropFilterKernel(mem_size, use_slm)                 \
  stream->submit([&](sycl::handler& cgh) {                                    \
    sycl::local_accessor<OutT, 1> scratch(mem_size, cgh);                     \
    DilationBackpropFilterKernel<T, use_slm, OutT> task(                      \
        input_ptr, filter_ptr, out_backprop_ptr, total_count, batch,          \
        input_rows, input_cols, depth, filter_rows, filter_cols, output_rows, \
        output_cols, stride_rows, stride_cols, rate_rows, rate_cols, pad_top, \
        pad_left, filter_backprop_ptr, scratch, mem_size);                    \
    cgh.parallel_for<DilationBackpropFilterKernel<T, use_slm, OutT>>(         \
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),         \
                          sycl::range<1>(group_size)),                        \
        task);                                                                \
  });
    if (slm_used <= slm_size) {
      SubmitDilationBackPropFilterKernel(filter_backprop.size(), true);
    } else {
      SubmitDilationBackPropFilterKernel(0, false);
    }
  }
};
}  // namespace functor

#define DEFINE_GPU_SPECS(T) template struct functor::Dilation<GPUDevice, T>;

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_GPU_SPECS);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_float(DEFINE_GPU_SPECS);
TF_CALL_bfloat16(DEFINE_GPU_SPECS);
TF_CALL_half(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

#define DEFINE_GPU_GRAD_SPECS(T)                                       \
  template struct functor::DilationBackpropInput<GPUDevice, T, float>; \
  template struct functor::DilationBackpropFilter<GPUDevice, T, float>;

TF_CALL_float(DEFINE_GPU_GRAD_SPECS);
TF_CALL_bfloat16(DEFINE_GPU_GRAD_SPECS);
TF_CALL_half(DEFINE_GPU_GRAD_SPECS);
#ifdef ITEX_ENABLE_DOUBLE
template struct functor::DilationBackpropInput<GPUDevice, double, double>;
template struct functor::DilationBackpropFilter<GPUDevice, double, double>;
#endif  // ITEX_ENABLE_DOUBLE
#undef DEFINE_GPU_GRAD_SPECS
}  // namespace itex
