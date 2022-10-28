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
struct ApplyGradientDescent<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    var.device(d) -= lr.reshape(single).broadcast(bcast) * grad;
  }
};
}  // namespace functor

template <typename Device, typename T>
class ApplyGradientDescentOp : public OpKernel {
 public:
  explicit ApplyGradientDescentOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    const Tensor& alpha = ctx->input(1);
    OP_REQUIRES(ctx, IsLegacyScalar(alpha.shape()),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha.shape().DebugString()));
    const Tensor& delta = ctx->input(2);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(delta.shape()),
        errors::InvalidArgument("var and delta do not have the same shape",
                                var.shape().DebugString(), " ",
                                delta.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyGradientDescent<Device, T>()(
        device, var.flat<T>(), alpha.scalar<T>(), delta.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyGradientDescent").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyGradientDescentOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyGradientDescent")                \
                              .Device(DEVICE_##D)                             \
                              .HostMemory("var")                              \
                              .TypeConstraint<T>("T"),                        \
                          ApplyGradientDescentOp<D##Device, T>);

#define REGISTER_DPCPP_KERNELS(T) REGISTER_KERNELS(GPU, T)
TF_CALL_half(REGISTER_DPCPP_KERNELS);
TF_CALL_float(REGISTER_DPCPP_KERNELS);
TF_CALL_bfloat16(REGISTER_DPCPP_KERNELS);
TF_CALL_complex64(REGISTER_DPCPP_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_complex128(REGISTER_DPCPP_KERNELS);
TF_CALL_double(REGISTER_DPCPP_KERNELS);
#endif
#undef REGISTER_DPCPP_KERNELS
#undef REGISTER_KERNELS

}  // namespace itex
