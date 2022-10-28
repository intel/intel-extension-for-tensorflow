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

#ifndef ITEX_CORE_KERNELS_GPU_STRIDED_SLICE_OP_IMPL_H_
#define ITEX_CORE_KERNELS_GPU_STRIDED_SLICE_OP_IMPL_H_

#include "itex/core/kernels/gpu/slice_op.h"
#include "itex/core/kernels/gpu/strided_slice_op.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types_traits.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, int NDIM>
void HandleStridedSliceCase(OpKernelContext* context,
                            const gtl::ArraySlice<int64>& begin,
                            const gtl::ArraySlice<int64>& end,
                            const gtl::ArraySlice<int64>& strides,
                            const TensorShape& processing_shape,
                            bool is_simple_slice, Tensor* result);

template <typename T, int NDIM>
void HandleStridedSliceGradCase(OpKernelContext* context,
                                const gtl::ArraySlice<int64>& begin,
                                const gtl::ArraySlice<int64>& end,
                                const gtl::ArraySlice<int64>& strides,
                                const TensorShape& processing_shape,
                                bool is_simple_slice, Tensor* result);

template <typename Device, typename T, int NDIM>
class HandleStridedSliceAssignCase {
 public:
  void operator()(OpKernelContext* context, const gtl::ArraySlice<int64>& begin,
                  const gtl::ArraySlice<int64>& end,
                  const gtl::ArraySlice<int64>& strides,
                  const TensorShape& processing_shape, bool is_simple_slice,
                  Tensor* result);
};
}  // namespace itex

// The actual implementation. This is designed so multiple
// translation units can include this file in the form
namespace itex {
template <typename Device, typename T, int NDIM>
void HandleStridedSliceCase(OpKernelContext* context,
                            const gtl::ArraySlice<int64>& begin,
                            const gtl::ArraySlice<int64>& end,
                            const gtl::ArraySlice<int64>& strides,
                            const TensorShape& processing_shape,
                            bool is_simple_slice, Tensor* result) {
  typedef typename proxy_type<Device, T>::type Proxy;

  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();
  if (is_simple_slice) {
    Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes_di;
    for (int i = 0; i < NDIM; ++i) {
      begin_di[i] = begin[i];
      sizes_di[i] = end[i] - begin[i];
    }
    functor::Slice<Device, Proxy, NDIM> slice_functor;
    slice_functor(context->eigen_device<Device>(),
                  result->bit_casted_shaped<Proxy, NDIM>(processing_dims),
                  context->input(0).bit_casted_tensor<Proxy, NDIM>(), begin_di,
                  sizes_di);
  } else {
    Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
    for (int i = 0; i < NDIM; ++i) {
      begin_di[i] = begin[i];
      end_di[i] = end[i];
      strides_di[i] = strides[i];
    }
    functor::StridedSlice<Device, Proxy, NDIM> strided_slice_functor;
    strided_slice_functor(
        context->eigen_device<Device>(),
        result->bit_casted_shaped<Proxy, NDIM>(processing_dims),
        context->input(0).bit_casted_tensor<Proxy, NDIM>(), begin_di, end_di,
        strides_di);
  }
}

template <typename Device, typename T, int NDIM>
void HandleStridedSliceGradCase(OpKernelContext* context,
                                const gtl::ArraySlice<int64>& begin,
                                const gtl::ArraySlice<int64>& end,
                                const gtl::ArraySlice<int64>& strides,
                                const TensorShape& processing_shape,
                                bool is_simple_slice, Tensor* result) {
  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();

  Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
  for (int i = 0; i < NDIM; ++i) {
    begin_di[i] = begin[i];
    end_di[i] = end[i];
    strides_di[i] = strides[i];
  }

  typedef typename proxy_type<Device, T>::type Proxy;
  functor::StridedSliceGrad<Device, Proxy, NDIM> strided_slice_grad_functor;
  strided_slice_grad_functor(
      context->eigen_device<Device>(), result->bit_casted_tensor<Proxy, NDIM>(),
      context->input(4).bit_casted_shaped<Proxy, NDIM>(processing_dims),
      begin_di, end_di, strides_di);
}

template <typename Device, typename T, int NDIM>
void HandleStridedSliceAssignCase<Device, T, NDIM>::operator()(
    OpKernelContext* context, const gtl::ArraySlice<int64>& begin,
    const gtl::ArraySlice<int64>& end, const gtl::ArraySlice<int64>& strides,
    const TensorShape& processing_shape, bool is_simple_slice, Tensor* result) {
  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();
  typedef typename proxy_type<Device, T>::type Proxy;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
  for (int i = 0; i < NDIM; ++i) {
    begin_di[i] = begin[i];
    end_di[i] = end[i];
    strides_di[i] = strides[i];
  }
  functor::StridedSliceAssign<Device, Proxy, NDIM> strided_slice_assign_functor;
  strided_slice_assign_functor(
      context->eigen_device<Device>(), result->bit_casted_tensor<Proxy, NDIM>(),
      context->input(4).bit_casted_shaped<Proxy, NDIM>(processing_dims),
      begin_di, end_di, strides_di);
}

template <typename Device, typename T>
class HandleStridedSliceAssignCase<Device, T, 0> {
 public:
  enum { NDIM_PROXY = 1 };
  void operator()(OpKernelContext* context, const gtl::ArraySlice<int64>& begin,
                  const gtl::ArraySlice<int64>& end,
                  const gtl::ArraySlice<int64>& strides,
                  const TensorShape& processing_shape, bool is_simple_slice,
                  Tensor* result) {
    gtl::InlinedVector<int64, 1> processing_dims(1);
    processing_dims[0] = 1;

    typedef typename proxy_type<Device, T>::type Proxy;
    functor::StridedSliceAssignScalar<Device, Proxy> strided_slice_functor;
    strided_slice_functor(
        context->eigen_device<Device>(),
        result->bit_casted_shaped<Proxy, 1>(processing_dims),
        context->input(4).bit_casted_shaped<Proxy, 1>(processing_dims));
  }
};

#define INSTANTIATE_DIM0_AND_UP_HANDLERS(DEVICE, T, DIM) \
  template class HandleStridedSliceAssignCase<DEVICE, T, DIM>;

#define DECLARE_FOR_N_GPU(T)                        \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 0) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 1) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 2) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 3) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 4) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 5) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 6) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 7) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(GPUDevice, T, 8)

TF_CALL_int32(DECLARE_FOR_N_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_int64(DECLARE_FOR_N_GPU);
#endif
TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_N_GPU);

#define DECLARE_FOR_N_CPU(T)                        \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 0) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 1) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 2) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 3) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 4) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 5) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 6) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 7) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(CPUDevice, T, 8)

TF_CALL_int32(DECLARE_FOR_N_CPU);
#if defined(INTEL_CPU_ONLY) || defined(ITEX_ENABLE_DOUBLE)
TF_CALL_int64(DECLARE_FOR_N_CPU);
#endif
TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_N_CPU);
#undef DECLARE_FOR_N_GPU
#undef DECLARE_FOR_N_CPU
#undef INSTANTIATE_DIM0_AND_UP_HANDLERS
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_STRIDED_SLICE_OP_IMPL_H_
