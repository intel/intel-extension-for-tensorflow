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

#include "itex/core/kernels/gpu/argmin_op.h"

#include <memory>

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tout, typename ArgFunctor>
class ArgOp : public OpKernel {
 public:
  explicit ArgOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dimension = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dimension.shape()),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension.shape().DebugString()));

    const int32 dim = internal::SubtleMustCopy(dimension.scalar<int32>()());
    const int input_dims = input.dims();

    int axis = dim < 0 ? dim + input_dims : dim;

    OP_REQUIRES(context, FastBoundsCheck(axis, input_dims),
                errors::InvalidArgument("Expected dimension in the range [",
                                        -input_dims, ", ", input_dims,
                                        "), but got ", dim));
    OP_REQUIRES(
        context, input.dim_size(axis) > 0,
        errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                                input.shape().DebugString()));

    TensorShape output_shape;
    const TensorShape& input_shape = input.shape();
    for (int d = 0; d < input_dims - 1; ++d) {
      output_shape.AddDim(input_shape.dim_size((d < axis) ? d : d + 1));
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() == 0) {
      return;
    }

#define HANDLE_DIM(NDIM)                                        \
  case NDIM:                                                    \
    ArgFunctor::Reduce##NDIM(context->eigen_device<Device>(),   \
                             input.tensor<T, NDIM>(), axis,     \
                             output->tensor<Tout, NDIM - 1>()); \
    break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Argmin only support up "
                                            "to 7 input dimensions, but got ",
                                            input_dims, ". Inputs shape: ",
                                            input.shape().DebugString()));
    }
  }
#undef HANDLE_DIM

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ArgOp);
};

template <typename Device, typename T, typename Tout>
class ArgMinOp
    : public ArgOp<Device, T, Tout, functor::ArgMin<Device, T, Tout> > {
 public:
  explicit ArgMinOp(OpKernelConstruction* context)
      : ArgOp<Device, T, Tout, functor::ArgMin<Device, T, Tout> >(context) {}
};

// Registration of the GPU implementations.
#define REGISTER_ARGMIN_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                            \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64>("output_type") \
                              .TypeConstraint<int32>("Tidx")        \
                              .HostMemory("dimension"),             \
                          ArgMinOp<GPUDevice, type, int64>);        \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                            \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("output_type") \
                              .TypeConstraint<int32>("Tidx")        \
                              .HostMemory("dimension"),             \
                          ArgMinOp<GPUDevice, type, int32>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ARGMIN_GPU);
TF_CALL_bool(REGISTER_ARGMIN_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ARGMIN_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_ARGMIN_GPU

}  // namespace itex
