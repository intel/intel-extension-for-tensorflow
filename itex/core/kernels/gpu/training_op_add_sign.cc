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
struct ApplyAddSign<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar alpha,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    // The following is the GPU equivalent of the CPU version:
    // m.device(d) = m * beta() + grad * (static_cast<T>(1) - beta());
    const auto one = static_cast<T>(1.0);
    auto beta_bcast = beta.reshape(single).broadcast(bcast);
    auto one_minus_beta =
        (beta.constant(one) - beta).reshape(single).broadcast(bcast);
    m.device(d) = m * beta_bcast + grad * one_minus_beta;

    // The following is the GPU equivalent of the CPU version:
    // var.device(d) -= lr() * (alpha() + sign_decay() * sign_gm) * grad;
    auto sign_gm = grad.sign() * m.sign();
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto alpha_bcast = alpha.reshape(single).broadcast(bcast);
    auto sign_decay_bcast = sign_decay.reshape(single).broadcast(bcast);
    var.device(d) -=
        lr_bcast * (alpha_bcast + sign_decay_bcast * sign_gm) * grad;
  }
};
}  // namespace functor

template <typename Device, typename T>
class ApplyAddSignOp : public OpKernel {
 public:
  explicit ApplyAddSignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, m.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& alpha = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha.shape()),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha.shape().DebugString()));
    const Tensor& sign_decay = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha.shape()),
                errors::InvalidArgument("sign_decay is not a scalar: ",
                                        sign_decay.shape().DebugString()));
    const Tensor& beta = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta.shape()),
                errors::InvalidArgument("beta is not a scalar: ",
                                        beta.shape().DebugString()));
    const Tensor& grad = ctx->input(6);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAddSign<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), lr.scalar<T>(), alpha.scalar<T>(),
        sign_decay.scalar<T>(), beta.scalar<T>(), grad.flat<T>());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ApplyAddSign").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAddSignOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAddSign")                \
                              .Device(DEVICE_##D)                     \
                              .HostMemory("var")                      \
                              .HostMemory("m")                        \
                              .TypeConstraint<T>("T"),                \
                          ApplyAddSignOp<D##Device, T>);

#define REGISTER_ITEX_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T)
TF_CALL_half(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_float(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_bfloat16(REGISTER_ITEX_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ITEX_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_ITEX_GPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace itex
