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
#include "itex/core/kernels/gpu/inplace_ops_functor.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {
template <typename Device, typename T>
class ParallelConcatUpdate : public OpKernel {
 public:
  explicit ParallelConcatUpdate(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("loc", &loc_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto value = ctx->input(0);
    auto update = ctx->input(1);

    OP_REQUIRES(
        ctx, value.dims() == update.dims(),
        errors::InvalidArgument("value and update shape doesn't match: ",
                                value.shape().DebugString(), " vs. ",
                                update.shape().DebugString()));
    for (int i = 1; i < value.dims(); ++i) {
      OP_REQUIRES(
          ctx, value.dim_size(i) == update.dim_size(i),
          errors::InvalidArgument("value and update shape doesn't match ",
                                  value.shape().DebugString(), " vs. ",
                                  update.shape().DebugString()));
    }
    OP_REQUIRES(ctx, 1 == update.dim_size(0),
                errors::InvalidArgument("update shape doesn't match: ",
                                        update.shape().DebugString()));

    Tensor output = value;  // This creates an alias intentionally.
    const auto& d = ctx->eigen_device<Device>();
    ::itex::functor::DoParallelConcat<T>(d, update, loc_, &output);
    ctx->set_output(0, output);
  }

 private:
  int32 loc_;
};

template <typename Device, typename T>
class ParallelConcatStart : public OpKernel {
 public:
  explicit ParallelConcatStart(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &out));
  }

 private:
  TensorShape shape_;
};

class FailureKernel : public OpKernel {
 public:
  explicit FailureKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   errors::Internal("Found instance of parallel_stack which "
                                    "could not be properly replaced."));
  }

  void Compute(OpKernelContext*) override {}
};

class InplaceOpBase : public OpKernel {
 public:
  explicit InplaceOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto x = ctx->input(0);
    auto i = ctx->input(1);
    auto v = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(i.shape()),
                errors::InvalidArgument("i must be a vector. ",
                                        i.shape().DebugString()));
    OP_REQUIRES(ctx, x.dims() == v.dims(),
                errors::InvalidArgument(
                    "x and v shape doesn't match (ranks differ): ",
                    x.shape().DebugString(), " vs. ", v.shape().DebugString()));
    for (int i = 1; i < x.dims(); ++i) {
      OP_REQUIRES(
          ctx, x.dim_size(i) == v.dim_size(i),
          errors::InvalidArgument("x and v shape doesn't match at index ", i,
                                  " : ", x.shape().DebugString(), " vs. ",
                                  v.shape().DebugString()));
    }
    OP_REQUIRES(ctx, i.dim_size(0) == v.dim_size(0),
                errors::InvalidArgument(
                    "i and x shape doesn't match at index 0: ",
                    i.shape().DebugString(), " vs. ", v.shape().DebugString()));

    Tensor y = x;  // This creates an alias intentionally.
    // Skip processing if tensors are empty.
    if (x.NumElements() > 0 || v.NumElements() > 0) {
      OP_REQUIRES_OK(ctx, DoCompute(ctx, i, v, &y));
    }
    ctx->set_output(0, y);
  }

 protected:
  virtual Status DoCompute(OpKernelContext* ctx, const Tensor& i,
                           const Tensor& v, Tensor* y) = 0;
};

template <typename Device, functor::InplaceOpType op>
class InplaceOp : public InplaceOpBase {
 public:
  explicit InplaceOp(OpKernelConstruction* ctx) : InplaceOpBase(ctx) {}

 protected:
  Status DoCompute(OpKernelContext* ctx, const Tensor& i, const Tensor& v,
                   Tensor* y) override {
    const auto& d = ctx->eigen_gpu_device();
    return ::itex::functor::DoInplace(d, op, i, v, y);
  }
};

template <typename Device, typename T>
class EmptyOp : public OpKernel {
 public:
  explicit EmptyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("init", &init_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape.shape()),
        errors::InvalidArgument("shape must be a vector of int32, got shape ",
                                shape.shape().DebugString()));
    auto dims = shape.flat<int32>();
    TensorShape out_shape;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                            reinterpret_cast<const int32*>(dims.data()),
                            dims.size(), &out_shape));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (init_) {
      functor::SetZeroFunctor<Device, T>()(ctx->eigen_device<Device>(),
                                           out->flat<T>());
    }
  }

 private:
  bool init_;
};

class CopyOpBase : public OpKernel {
 public:
  explicit CopyOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto x = ctx->input(0);
    Tensor* y;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, DoCompute(ctx, x, y));
  }

 protected:
  virtual Status DoCompute(OpKernelContext* ctx, const Tensor& x,
                           Tensor* y) = 0;
};

template <typename Device>
class CopyOp : public CopyOpBase {
 public:
  explicit CopyOp(OpKernelConstruction* ctx) : CopyOpBase(ctx) {}

 protected:
  Status DoCompute(OpKernelContext* ctx, const Tensor& x, Tensor* y) override {
    const auto& d = ctx->eigen_device<Device>();
    return ::itex::functor::DoCopy(d, x, y);
  }
};

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                                 \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("InplaceAdd").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      InplaceOp<GPUDevice, functor::I_ADD>);                           \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("InplaceSub").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      InplaceOp<GPUDevice, functor::I_SUB>);                           \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("DeepCopy").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),   \
      CopyOp<GPUDevice>);                                              \
  REGISTER_KERNEL_BUILDER(Name("Empty")                                \
                              .Device(DEVICE_GPU)                      \
                              .HostMemory("shape")                     \
                              .TypeConstraint<TYPE>("dtype"),          \
                          EmptyOp<GPUDevice, TYPE>)

REGISTER(float);
REGISTER(Eigen::bfloat16);
REGISTER(Eigen::half);
REGISTER(itex::int64);
#undef REGISTER

#define REGISTER_KERNEL_FOR_GPU(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")                     \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<TYPE>("T"),                  \
                          ParallelConcatStart<GPUDevice, TYPE>);           \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")                    \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<TYPE>("T"),                  \
                          ParallelConcatUpdate<GPUDevice, TYPE>);          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ParallelConcat").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      FailureKernel);                                                      \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("InplaceUpdate").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),  \
      InplaceOp<GPUDevice, functor::I_UPDATE>);

TF_CALL_float(REGISTER_KERNEL_FOR_GPU);
TF_CALL_bfloat16(REGISTER_KERNEL_FOR_GPU);
TF_CALL_half(REGISTER_KERNEL_FOR_GPU);
TF_CALL_int32(REGISTER_KERNEL_FOR_GPU);
TF_CALL_int64(REGISTER_KERNEL_FOR_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL_FOR_GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNEL_FOR_GPU

#define REGISTER_INPLACEADD_SUB_DOUBLE(TYPE)                           \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("InplaceAdd").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      InplaceOp<GPUDevice, functor::I_ADD>);                           \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("InplaceSub").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      InplaceOp<GPUDevice, functor::I_SUB>)

#ifdef ITEX_ENABLE_DOUBLE
REGISTER_INPLACEADD_SUB_DOUBLE(double);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_INPLACEADD_SUB_DOUBLE

REGISTER_KERNEL_BUILDER(Name("Empty")
                            .Device(DEVICE_GPU)
                            .HostMemory("shape")
                            .TypeConstraint<itex::int32>("dtype"),
                        EmptyOp<GPUDevice, itex::int32>)
REGISTER_KERNEL_BUILDER(
    Name("InplaceUpdate").Device(DEVICE_GPU).TypeConstraint<bool>("T"),
    InplaceOp<GPUDevice, functor::I_UPDATE>);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("DeepCopy").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    CopyOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("Empty")
                            .Device(DEVICE_GPU)
                            .HostMemory("shape")
                            .TypeConstraint<double>("dtype"),
                        EmptyOp<GPUDevice, double>)
#endif  // ITEX_ENABLE_DOUBLE
}  // namespace itex
