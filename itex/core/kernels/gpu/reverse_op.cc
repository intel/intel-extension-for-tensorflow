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

#include "itex/core/kernels/gpu/reverse_op.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, int NDIMS>
void HandleReverseCase(OpKernelContext* context,
                       typename TTypes<bool, 1>::ConstTensor dims,
                       Tensor* result) {
  const Tensor& input = context->input(0);
  const bool can_use_32bit = std::is_same<Eigen::GpuDevice, Device>::value &&
                             input.NumElements() < kint32max;

  typename Eigen::array<bool, NDIMS> axes_di;
  for (int i = 0; i < NDIMS; i++) {
    axes_di[i] = dims(i);
  }
  functor::Reverse<Device, T, NDIMS>()(
      context->eigen_device<Device>(), input.tensor<T, NDIMS>(), axes_di,
      result->tensor<T, NDIMS>(), can_use_32bit);
}

template <typename Device, typename T, int NDIMS>
void HandleReverseV2Case(OpKernelContext* context,
                         const gtl::ArraySlice<bool>& axes, Tensor* result) {
  const Tensor& input = context->input(0);
  const bool can_use_32bit = std::is_same<Eigen::GpuDevice, Device>::value &&
                             input.NumElements() < kint32max;

  typename Eigen::array<bool, NDIMS> axes_di;
  for (int i = 0; i < NDIMS; i++) {
    axes_di[i] = axes[i];
  }
  functor::Reverse<Device, T, NDIMS>()(
      context->eigen_device<Device>(), input.tensor<T, NDIMS>(), axes_di,
      result->tensor<T, NDIMS>(), can_use_32bit);
}

template <typename Device, typename T>
class ReverseOp : public OpKernel {
 public:
  explicit ReverseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dims = context->input(1);

    if (TensorShapeUtils::IsScalar(input.shape())) {
      context->set_output(0, input);
    } else {
      const int input_dims = input.dims();
      OP_REQUIRES(context, TensorShapeUtils::IsVector(dims.shape()),
                  errors::InvalidArgument("'dims' must be 1-dimension, not ",
                                          dims.dims()));

      OP_REQUIRES(
          context, input_dims == dims.dim_size(0),
          errors::InvalidArgument(
              "'dims' must have the same number of values as 'input' has "
              "dimensions. 'input' has ",
              input_dims, "'dims' has ", dims.dim_size(0), " values"));
      OP_REQUIRES(context, input_dims <= 8,
                  errors::Unimplemented(
                      "reverse is not implemented for tensors of rank > 8."));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, input.shape(), &output));

#define HANDLE_REVERSE(NDIMS)                                               \
  case NDIMS:                                                               \
    HandleReverseCase<Device, T, NDIMS>(context, dims.vec<bool>(), output); \
    return;

      switch (input_dims) {
        HANDLE_REVERSE(0);
        HANDLE_REVERSE(1);
        HANDLE_REVERSE(2);
        HANDLE_REVERSE(3);
        HANDLE_REVERSE(4);
        HANDLE_REVERSE(5);
        HANDLE_REVERSE(6);
        HANDLE_REVERSE(7);
        HANDLE_REVERSE(8);
      }
#undef HANDLE_REVERSE
    }
  }
};

template <typename Device, typename T, typename Tidx>
class ReverseV2Op : public OpKernel {
 public:
  explicit ReverseV2Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& sparse_dims = context->input(1);

    if (TensorShapeUtils::IsScalar(input.shape()) || input.NumElements() == 0) {
      context->set_output(0, input);
    } else {
      const int input_dims = input.dims();
      const TensorShape& sparse_dims_shape = sparse_dims.shape();
      const auto& axes_sparse_flat = sparse_dims.flat<Tidx>();

      OP_REQUIRES(context, TensorShapeUtils::IsVector(sparse_dims_shape),
                  errors::InvalidArgument("'dims' must be 1-dimension, not ",
                                          sparse_dims.dims()));
      gtl::InlinedVector<bool, 8> axes_dense(input_dims, false);
      for (int dummy = 0; dummy < axes_sparse_flat.size(); dummy++) {
        Tidx axis = internal::SubtleMustCopy<Tidx>(axes_sparse_flat(dummy));
        Tidx canonical_axis = axis < 0 ? input_dims + axis : axis;
        OP_REQUIRES(context, canonical_axis >= 0 && canonical_axis < input_dims,
                    errors::InvalidArgument("'axis'[", dummy, "] = ", axis,
                                            " is out of valid range [", 0, ", ",
                                            input_dims - 1));
        OP_REQUIRES(context, !axes_dense[canonical_axis],
                    errors::InvalidArgument("axis ", canonical_axis,
                                            " specified more than once."));
        axes_dense[canonical_axis] = true;
      }

      OP_REQUIRES(context, input_dims <= 8,
                  errors::Unimplemented(
                      "reverse is not implemented for tensors of rank > 8."));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, input.shape(), &output));

      // TODO(itex): we can do dimension folding to reduce, e.g., a reverse
      // of a single dimension to the dims=3 or dims=2 case, regardless of the
      // number of dimensions in the tensor. This would let some ops use faster
      // lower-dimension code (and use optimized versions).

#define HANDLE_REVERSE(NDIMS)                                           \
  case NDIMS:                                                           \
    HandleReverseV2Case<Device, T, NDIMS>(context, axes_dense, output); \
    return;

      switch (input_dims) {
        HANDLE_REVERSE(0);
        HANDLE_REVERSE(1);
        HANDLE_REVERSE(2);
        HANDLE_REVERSE(3);
        HANDLE_REVERSE(4);
        HANDLE_REVERSE(5);
        HANDLE_REVERSE(6);
        HANDLE_REVERSE(7);
        HANDLE_REVERSE(8);
      }
#undef HANDLE_REVERSE
    }
  }
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(T)                              \
  REGISTER_KERNEL_BUILDER(Name("Reverse")                    \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<T>("T")        \
                              .HostMemory("dims"),           \
                          ReverseOp<GPUDevice, T>)           \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                  \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<int32>("Tidx") \
                              .HostMemory("axis"),           \
                          ReverseV2Op<GPUDevice, T, int32>)  \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                  \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<int64>("Tidx") \
                              .HostMemory("axis"),           \
                          ReverseV2Op<GPUDevice, T, int64>)
TF_CALL_bool(REGISTER_GPU_KERNELS);
TF_CALL_int32(REGISTER_GPU_KERNELS);
TF_CALL_uint8(REGISTER_GPU_KERNELS);
TF_CALL_int8(REGISTER_GPU_KERNELS);
TF_CALL_complex64(REGISTER_GPU_KERNELS);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
TF_CALL_complex128(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNEL

}  // namespace itex
