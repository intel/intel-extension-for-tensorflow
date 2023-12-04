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

#ifndef ITEX_CORE_KERNELS_GPU_ONE_HOT_OP_H_
#define ITEX_CORE_KERNELS_GPU_ONE_HOT_OP_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace itex {
typedef Eigen::GpuDevice GPUDevice;

namespace impl {

template <typename T, typename TI, bool readInCol>
struct OneHotEncodingKernel {
  using FastDivisor = Eigen::internal::TensorIntDivisor<int>;
  OneHotEncodingKernel(int32 elem_cnts, int32 outElems_in_single_batch,
                       int32 final_depth, int32 indices_col,
                       const TI* indices_ptr, const T* on_value_ptr,
                       const T* off_value_ptr,
                       FastDivisor outElems_in_single_batch_fast_divisor,
                       FastDivisor final_depth_fast_divisor, T* output_ptr)
      : elem_cnts(elem_cnts),
        outElems_in_single_batch(outElems_in_single_batch),
        final_depth(final_depth),
        indices_col(indices_col),
        indices_ptr(indices_ptr),
        on_value_ptr(on_value_ptr),
        off_value_ptr(off_value_ptr),
        outElems_in_single_batch_fast_divisor_(
            outElems_in_single_batch_fast_divisor),
        final_depth_fast_divisor_(final_depth_fast_divisor),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto global_id = item.get_global_id(0);
    auto global_range = item.get_global_range(0);
    for (int32 i = global_id, step = global_range; i < elem_cnts; i += step) {
      const int32 batch_index =
          i / outElems_in_single_batch_fast_divisor_;  // batch id
      const int32 flatten_index = i - outElems_in_single_batch * batch_index;

      const int32 row = flatten_index / final_depth_fast_divisor_;
      const int32 col = flatten_index - final_depth * row;

      if (readInCol) {
        const int32 col_index =
            batch_index * indices_col + col;  // col index in indices
        const int32 idx = indices_ptr[col_index];

        assert(idx >= 0 && idx < final_depth);
        output_ptr[i] = (idx == row) ? *on_value_ptr : *off_value_ptr;
      } else {
        const int32 row_index =
            batch_index * indices_col + row;  // row index in indices
        const int32 idx = indices_ptr[row_index];

        assert(idx >= 0 && idx < final_depth);
        output_ptr[i] = (idx == col) ? *on_value_ptr : *off_value_ptr;
      }
    }
  }

 private:
  int32 elem_cnts;
  int32 outElems_in_single_batch;
  int32 final_depth;
  int32 indices_col;
  const TI* indices_ptr;
  const T* on_value_ptr;
  const T* off_value_ptr;
  FastDivisor outElems_in_single_batch_fast_divisor_;
  FastDivisor final_depth_fast_divisor_;
  T* output_ptr;
};
/*
@param: indices
            batch>1:
                [batch, features]
            batch=1:
                [features]
@ret:   output
            batch>1:
                [batch, features, depth] if axis == -1
                [batch, depth, features] if axis == 1
            batch=1:
                [features, depth] if axis == -1
                [depth, features] if axis == 0
*/
template <typename T, typename TI>
struct OneHotDefaultKernel {
  Status operator()(const GPUDevice& device, const Tensor& indices,
                    const typename TTypes<T>::ConstScalar& on_value,
                    const typename TTypes<T>::ConstScalar& off_value,
                    const int32& elem_cnts, const int32& depth, const int& axis,
                    Tensor* output) {
    const int32 indices_col =
        indices.dim_size(indices.shape().dims() - 1);  // last dimention
    const int32 outElems_in_single_batch = indices_col * depth;

    auto stream = device.stream();
    auto max_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (elem_cnts + max_group_size - 1) / max_group_size;

    const TI* indices_ptr = indices.flat<TI>().data();
    const T* on_value_ptr = on_value.data();
    const T* off_value_ptr = off_value.data();
    T* output_ptr = output->flat<T>().data();

    int32 final_depth =
        1;  // change depth value and output format according to axis
    if (axis != output->shape().dims() - 1) {
      final_depth = indices_col;
#define EigenFastDivisor(divisor, num)                     \
  Eigen::internal::TensorIntDivisor<int> divisor;          \
  if (num != 0) {                                          \
    divisor = Eigen::internal::TensorIntDivisor<int>(num); \
  }
      EigenFastDivisor(outElems_in_single_batch_fast_divisor,
                       outElems_in_single_batch);
      EigenFastDivisor(final_depth_fast_divisor, final_depth);

#undef EigenFastDivisor

      stream->submit([&](sycl::handler& cgh) {
        OneHotEncodingKernel<T, TI, true> task(
            elem_cnts, outElems_in_single_batch, final_depth, indices_col,
            indices_ptr, on_value_ptr, off_value_ptr,
            outElems_in_single_batch_fast_divisor, final_depth_fast_divisor,
            output_ptr);
        cgh.parallel_for<OneHotEncodingKernel<T, TI, true>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * max_group_size),
                              sycl::range<1>(max_group_size)),
            task);
      });
    } else {
      final_depth = depth;

#define EigenFastDivisor(divisor, num)                     \
  Eigen::internal::TensorIntDivisor<int> divisor;          \
  if (num != 0) {                                          \
    divisor = Eigen::internal::TensorIntDivisor<int>(num); \
  }
      EigenFastDivisor(outElems_in_single_batch_fast_divisor,
                       outElems_in_single_batch);
      EigenFastDivisor(final_depth_fast_divisor, final_depth);

#undef EigenFastDivisor

      stream->submit([&](sycl::handler& cgh) {
        OneHotEncodingKernel<T, TI, false> task(
            elem_cnts, outElems_in_single_batch, final_depth, indices_col,
            indices_ptr, on_value_ptr, off_value_ptr,
            outElems_in_single_batch_fast_divisor, final_depth_fast_divisor,
            output_ptr);
        cgh.parallel_for<OneHotEncodingKernel<T, TI, false>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * max_group_size),
                              sycl::range<1>(max_group_size)),
            task);
      });
    }

    return Status::OK();
  }
};

}  // namespace impl

namespace functor {
template <typename T, typename TI>
struct OneHot {
  void Compute(OpKernelContext* context, const Tensor& indices,
               const typename TTypes<T>::ConstScalar& on_value,
               const typename TTypes<T>::ConstScalar& off_value,
               const int& axis, const int32& depth, Tensor* output) {
    const int32 out_elem_cnts = output->shape().num_elements();

    auto status = impl::OneHotDefaultKernel<T, TI>()(
        context->eigen_gpu_device(), indices, on_value, off_value,
        out_elem_cnts, depth, axis, output);
  }
};
}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_ONE_HOT_OP_H_
