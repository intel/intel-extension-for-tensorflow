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

#include "itex/core/kernels/gpu/split_lib.h"

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, int vec_size>
struct SplitGpuKernel {
  using Tvec = AlignedVector<T, vec_size>;
  using FastDivisor = Eigen::internal::TensorIntDivisor<int>;
  SplitGpuKernel(const T* input, int input_slice_size, int split_slice_size,
                 int per_split_elements, int work_item_count,
                 int per_vectorize_num, int vectorize_num, int per_tail_num,
                 int offset, FastDivisor split_slice_size_fast_divisor,
                 FastDivisor per_vectorize_num_fast_divisor,
                 FastDivisor per_tail_num_fast_divisor,
                 GpuDeviceArrayStruct<T*> output_ptr_data)
      : input_(input),
        input_slice_size_(input_slice_size),
        split_slice_size_(split_slice_size),
        per_split_elements_(per_split_elements),
        work_item_count_(work_item_count),
        per_vectorize_num_(per_vectorize_num),
        vectorize_num_(vectorize_num),
        per_tail_num_(per_tail_num),
        offset_(offset),
        split_slice_size_fast_divisor_(split_slice_size_fast_divisor),
        per_vectorize_num_fast_divisor_(per_vectorize_num_fast_divisor),
        per_tail_num_fast_divisor_(per_tail_num_fast_divisor),
        output_ptr_data_(output_ptr_data) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= work_item_count_) return;

    if (id < vectorize_num_) {
      int out_id = static_cast<int>(id) / per_vectorize_num_fast_divisor_;
      int output_offset =
          (static_cast<int>(id) - out_id * per_vectorize_num_) * vec_size;
      T* output_ptr = output_ptr_data_.inline_values[out_id];
      int row_id = output_offset / split_slice_size_fast_divisor_;
      int col_id = output_offset - row_id * split_slice_size_;
      int input_offset = row_id * input_slice_size_ +
                         (offset_ + out_id) * split_slice_size_ + col_id;
      Tvec in;
      if (input_offset % vec_size == 0 &&
          col_id + vec_size <= split_slice_size_) {
        in = *(reinterpret_cast<const Tvec*>(input_ + input_offset));
      } else {
        for (int i = 0; i < vec_size; ++i) {
          in[i] = input_[input_offset];
          if (col_id + 1 >= split_slice_size_) {
            input_offset = input_offset - col_id + input_slice_size_;
            col_id = 0;
          } else {
            ++col_id;
            ++input_offset;
          }
        }
      }
      *(reinterpret_cast<Tvec*>(output_ptr + output_offset)) = in;
    } else {
      int out_id =
          (static_cast<int>(id) - vectorize_num_) / per_tail_num_fast_divisor_;
      int output_offset =
          per_split_elements_ - per_tail_num_ +
          (static_cast<int>(id) - vectorize_num_ - out_id * per_tail_num_);
      T* output_ptr = output_ptr_data_.inline_values[out_id];
      int row_id = output_offset / split_slice_size_fast_divisor_;
      int col_id = output_offset - row_id * split_slice_size_;
      int input_offset = row_id * input_slice_size_ +
                         (offset_ + out_id) * split_slice_size_ + col_id;
      output_ptr[output_offset] = input_[input_offset];
    }
  }

 private:
  const T* input_;
  int input_slice_size_, split_slice_size_, per_split_elements_;
  int work_item_count_;
  int per_vectorize_num_, vectorize_num_;
  int per_tail_num_;
  int offset_;
  FastDivisor split_slice_size_fast_divisor_;
  FastDivisor per_vectorize_num_fast_divisor_;
  FastDivisor per_tail_num_fast_divisor_;
  GpuDeviceArrayStruct<T*> output_ptr_data_;
};

namespace detail {
struct LaunchSplitGpuKernelVectorized {
  template <int vec_size>
  struct Impl {
    template <typename T>
    void operator()(const GPUDevice& d, const T* input, int prefix_dim_size,
                    int split_dim_size, int suffix_dim_size,
                    int split_dim_output_size, int offset,
                    const GpuDeviceArrayStruct<T*>& output_ptr_data) {
      int split_slice_size = split_dim_output_size * suffix_dim_size;
      int input_slice_size = split_dim_size * suffix_dim_size;
      int per_split_elements = prefix_dim_size * split_slice_size;
      int per_vectorize_num = per_split_elements / vec_size;
      int per_tail_num = per_split_elements - per_vectorize_num * vec_size;
      int vectorize_num = per_vectorize_num * output_ptr_data.size;
      int tail_num = per_tail_num * output_ptr_data.size;
      int work_item_count = vectorize_num + tail_num;

      Eigen::internal::TensorIntDivisor<int> split_slice_size_fast_divisor(
          split_slice_size);

#define EigenFastDivisor(divisor, num)                     \
  Eigen::internal::TensorIntDivisor<int> divisor;          \
  if (num != 0) {                                          \
    divisor = Eigen::internal::TensorIntDivisor<int>(num); \
  }
      EigenFastDivisor(per_vectorize_num_fast_divisor, per_vectorize_num);
      EigenFastDivisor(per_tail_num_fast_divisor, per_tail_num);

#undef EigenFastDivisor

      auto& stream = d.stream();
      auto workgroup_size =
          (*stream)
              .get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      auto num_workgroups =
          (work_item_count + workgroup_size - 1) / workgroup_size;

      stream->submit([&](sycl::handler& cgh) {
        SplitGpuKernel<T, vec_size> task(
            input, input_slice_size, split_slice_size, per_split_elements,
            work_item_count, per_vectorize_num, vectorize_num, per_tail_num,
            offset, split_slice_size_fast_divisor,
            per_vectorize_num_fast_divisor, per_tail_num_fast_divisor,
            output_ptr_data);
        cgh.parallel_for<SplitGpuKernel<T, vec_size>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    }
  };
};

};  // namespace detail

namespace functor {

template <typename T, int NDims>
void Split<T, NDims>::operator()(
    const Eigen::GpuDevice& d, typename TTypes<T, NDims>::Tensor output,
    typename TTypes<T, NDims>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_sizes) {
  output.device(d) = input.slice(slice_indices, slice_sizes);
}

template <typename T>
void SplitGpuFunctor<T>::operator()(
    const Eigen::GpuDevice& d, const T* input, int prefix_dim_size,
    int split_dim_size, int suffix_dim_size, int split_dim_output_size,
    int offset, const GpuDeviceArrayStruct<T*>& output_ptr_data) {
  DispatchToVectorized<T,
                       detail::LaunchSplitGpuKernelVectorized::template Impl>(
      alignment_of(input), d, input, prefix_dim_size, split_dim_size,
      suffix_dim_size, split_dim_output_size, offset, output_ptr_data);
}

#define DEFINE_DPCPP_KERNELS(T) \
  template struct Split<T, 2>;  \
  template struct Split<T, 3>;  \
  template struct SplitGpuFunctor<T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_DPCPP_KERNELS);
TF_CALL_int64(DEFINE_DPCPP_KERNELS);
TF_CALL_int32(DEFINE_DPCPP_KERNELS);
TF_CALL_complex64(DEFINE_DPCPP_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_DPCPP_KERNELS);
TF_CALL_complex128(DEFINE_DPCPP_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE

#undef DEFINE_DPCPP_KERNELS
}  // namespace functor
}  // namespace itex
