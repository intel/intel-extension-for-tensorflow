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

#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/kernels/gpu/training_ops.h"

namespace itex {

namespace functor {
template <typename T>
struct ApplyAdadelta<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat accum_update,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    accum.device(d) = accum * rho.reshape(single).broadcast(bcast) +
                      grad.square() * (grad.constant(T(1)) -
                                       rho.reshape(single).broadcast(bcast));
    const auto update =
        (accum_update + epsilon.reshape(single).broadcast(bcast)).sqrt() *
        (accum + epsilon.reshape(single).broadcast(bcast)).rsqrt() * grad;
    var.device(d) -= update * lr.reshape(single).broadcast(bcast);
    accum_update.device(d) =
        accum_update * rho.reshape(single).broadcast(bcast) +
        update.square() *
            (grad.constant(T(1)) - rho.reshape(single).broadcast(bcast));
  }
};
}  // namespace functor

template <typename Device, typename T>
class ApplyAdadeltaOp : public OpKernel {
 public:
  explicit ApplyAdadeltaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    DoValidate(ctx);
    if (!ctx->status().ok()) return;
    DoCompute(ctx);
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var;
    const bool sparse = false;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor accum_update;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_,
                                                   sparse, &accum_update));

    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, accum_update.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& epsilon = ctx->input(5);
    const Tensor& grad = ctx->input(6);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var;
    const bool sparse = false;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor accum_update;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_,
                                                   sparse, &accum_update));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& epsilon = ctx->input(5);
    const Tensor& grad = ctx->input(6);

    functor::ApplyAdadelta<Device, T>()(
        device, var.flat<T>(), accum.flat<T>(), accum_update.flat<T>(),
        lr.scalar<T>(), rho.scalar<T>(), epsilon.scalar<T>(), grad.flat<T>());
  }
};

#define REGISTER_KERNELS(D, T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyAdadelta").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdadeltaOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdadelta")                \
                              .Device(DEVICE_##D)                      \
                              .HostMemory("var")                       \
                              .HostMemory("accum")                     \
                              .HostMemory("accum_update")              \
                              .TypeConstraint<T>("T"),                 \
                          ApplyAdadeltaOp<D##Device, T>);

#define REGISTER_ITEX_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T)
TF_CALL_half(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_float(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_bfloat16(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_complex64(REGISTER_ITEX_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_complex128(REGISTER_ITEX_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_ITEX_GPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace itex
