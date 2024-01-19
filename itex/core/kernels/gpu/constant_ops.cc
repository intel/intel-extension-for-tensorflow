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

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class FillOp : public OpKernel {
 public:
  explicit FillOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& dims = context->input(0);
    OP_REQUIRES(context, dims.shape().dims() == 1 || dims.shape().dims() == 0,
                errors::InvalidArgument("dims must be a vector, got shape ",
                                        dims.shape().DebugString()));
    const Tensor& value = context->input(1);
    OP_REQUIRES(context,
                value.shape().dims() == 0 ||
                    (TensorShapeUtils::IsVector(value.shape()) &&
                     value.shape().dim_size(0) == 1),
                errors::InvalidArgument("value must be a scalar, got shape ",
                                        value.shape().DebugString()));

    TensorShape shape;
    auto flat_dims = dims.flat<Index>();
    OP_REQUIRES_OK(context,
                   TensorShapeUtils::MakeShape(
                       reinterpret_cast<const Index*>(flat_dims.data()),
                       flat_dims.size(), &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));

    if (value.NumElements() > 0 && shape.num_elements() > 0) {
      functor::FillFunctor<Device, T> fill;
      fill(context->eigen_device<Device>(), output->flat<T>(),
           value.scalar<T>());
    } else {
      ITEX_VLOG(1) << "Warning FillOP receive empty input.";
    }
  }
};

#define REGISTER_KERNEL(D, TYPE)                                     \
  REGISTER_KERNEL_BUILDER(Name("Fill")                               \
                              .Device(DEVICE_##D)                    \
                              .TypeConstraint<TYPE>("T")             \
                              .TypeConstraint<int32>("index_type")   \
                              .HostMemory("dims"),                   \
                          FillOp<D##Device, TYPE, int32>);           \
  REGISTER_KERNEL_BUILDER(Name("Fill")                               \
                              .Device(DEVICE_##D)                    \
                              .TypeConstraint<TYPE>("T")             \
                              .TypeConstraint<int64_t>("index_type") \
                              .HostMemory("dims"),                   \
                          FillOp<D##Device, TYPE, int64>);

REGISTER_KERNEL(GPU, Eigen::half);
REGISTER_KERNEL(GPU, Eigen::bfloat16);
REGISTER_KERNEL(GPU, float);
REGISTER_KERNEL(GPU, uint8);
REGISTER_KERNEL(GPU, int8);
REGISTER_KERNEL(GPU, uint16);
REGISTER_KERNEL(GPU, int16);
REGISTER_KERNEL(GPU, int32);
REGISTER_KERNEL(GPU, int64_t);
REGISTER_KERNEL(GPU, complex64);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL(GPU, double);
REGISTER_KERNEL(GPU, complex128);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_KERNEL

template <typename Device, typename T>
class ZerosLikeOp : public OpKernel {
 public:
  explicit ZerosLikeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Device& d = ctx->eigen_gpu_device();

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {0}, 0, input.shape(), &out));
    functor::SetZeroFunctor<Device, T> f;
    // This to handle the input is empty case. Otherwise, would got
    // null vptr.
    if (input.NumElements() > 0) {
      f(d, out->flat<T>());
    }
  }
};

template <typename Device>
class ZerosLikeOp<Device, Variant> : public OpKernel {
 public:
  explicit ZerosLikeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    OP_REQUIRES(ctx, input.dims() == 0,
                errors::InvalidArgument("ZerosLike non-scalar Tensor with "
                                        "dtype=DT_VARIANT is not supported."));
    auto zeros_like_func = [](TF_OpKernelContext* tf_ctx, TF_Tensor* tf_input,
                              TF_Tensor* tf_out) {
      OpKernelContext ctx(tf_ctx);
      Tensor out(tf_out);

#ifdef USING_NEXTPLUGGABLE_DEVICE
      create_pjrt_buffer_to_tensor(tf_ctx, tf_out, out.shape(), out.dtype());
#endif
      switch (out.dtype()) {
#define DTYPE_CASE(dtype)                                           \
  case DataTypeToEnum<dtype>::value:                                \
    out.template flat<dtype>().device(ctx.eigen_device<Device>()) = \
        out.template flat<dtype>().constant(dtype(0));              \
    break;
        TF_CALL_POD_TYPES(DTYPE_CASE)
        default:
          break;
#undef DTYPE_CASE
      }
    };
    TF_OpKernelContext* tf_ctx = ctx->Get();
    TF_Status* tf_status = TF_NewStatus();
    TF_ZerosLikeVariant(tf_ctx, zeros_like_func, tf_status);
    Status status = StatusFromTF_Status(tf_status);
    ITEX_CHECK_OK(status);
    TF_DeleteStatus(tf_status);
  }
};

#define REGISTER_KERNEL(type, dev)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ZerosLike").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      ZerosLikeOp<dev##Device, type>)

REGISTER_KERNEL(bool, GPU);
REGISTER_KERNEL(Eigen::half, GPU);
REGISTER_KERNEL(Eigen::bfloat16, GPU);
REGISTER_KERNEL(float, GPU);
REGISTER_KERNEL(int64, GPU);
REGISTER_KERNEL(Variant, GPU);
REGISTER_KERNEL(int32, GPU);
REGISTER_KERNEL(complex64, GPU);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL(double, GPU);
REGISTER_KERNEL(complex128, GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_KERNEL

// Tensorflow has no python api for this kernel, so we do not have ut for this
// kernel. As for array_ops.ones_like, tensorflow already implemented this op in
// python level. For more details, see tensorflow/python/ops/array_ops.py in
// tensorflow repo.
template <typename Device, typename T>
class OnesLikeOp : public OpKernel {
 public:
  explicit OnesLikeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {0}, 0, input.shape(), &out));
    functor::SetOneFunctor<Device, T> f;
    f(ctx->eigen_gpu_device(), out->flat<T>());
  }
};

#define REGISTER_KERNEL(type, dev)                                     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("OnesLike").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      OnesLikeOp<dev##Device, type>)
REGISTER_KERNEL(bool, GPU);
REGISTER_KERNEL(Eigen::half, GPU);
REGISTER_KERNEL(Eigen::bfloat16, GPU);
REGISTER_KERNEL(float, GPU);
REGISTER_KERNEL(int64, GPU);
REGISTER_KERNEL(int32, GPU);
REGISTER_KERNEL(complex64, GPU);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL(double, GPU);
REGISTER_KERNEL(complex128, GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNEL
};  // namespace itex
