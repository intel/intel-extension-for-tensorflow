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
struct ApplyRMSProp<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    ms.device(d) =
        ms + (rho.constant(one) - rho).reshape(single).broadcast(bcast) *
                 (grad.square() - ms);
    mom.device(d) =
        mom * momentum.reshape(single).broadcast(bcast) +
        lr.reshape(single).broadcast(bcast) * grad /
            ((epsilon.reshape(single).broadcast(bcast) + ms).sqrt());
    var.device(d) -= mom;
  }
};

template <typename T>
struct ApplyRMSPropComputeRMS<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstFlat ms,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::Flat output) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    output.device(d) =
        ms + (rho.constant(one) - rho).reshape(single).broadcast(bcast) *
                 (grad.square() - ms);
  }
};

template <typename T>
struct ApplyRMSPropVarUpdate<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstFlat var,
                  typename TTypes<T>::ConstFlat ms,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::Flat output) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    output.device(d) =
        var - lr.reshape(single).broadcast(bcast) * grad /
                  (epsilon.reshape(single).broadcast(bcast) + ms.sqrt());
  }
};

template <typename T>
struct ApplyCenteredRMSProp<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat mg, typename TTypes<T>::Flat ms,
                  typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    const auto one_minus_rho =
        (rho.constant(one) - rho).reshape(single).broadcast(bcast);
    ms.device(d) = ms + one_minus_rho * (grad.square() - ms);
    mg.device(d) = mg + one_minus_rho * (grad - mg);
    auto denom = (ms - mg.square()) + epsilon.reshape(single).broadcast(bcast);
    mom.device(d) = mom * momentum.reshape(single).broadcast(bcast) +
                    lr.reshape(single).broadcast(bcast) * grad / denom.sqrt();
    var.device(d) -= mom;
  }
};
}  // namespace functor

template <typename Device, typename T>
class ApplyRMSPropOp : public OpKernel {
 public:
  explicit ApplyRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor ms;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &ms));
    Tensor mom;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &mom));

    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, ms.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, mom.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& momentum = ctx->input(5);
    const Tensor& epsilon = ctx->input(6);
    const Tensor& grad = ctx->input(7);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyRMSProp<Device, T>()(device, var.flat<T>(), ms.flat<T>(),
                                       mom.flat<T>(), lr.scalar<T>(),
                                       rho.scalar<T>(), momentum.scalar<T>(),
                                       epsilon.scalar<T>(), grad.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename Device, typename T>
class ApplyRMSPropComputeRMSOp : public OpKernel {
 public:
  explicit ApplyRMSPropComputeRMSOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ms = ctx->input(0);
    const Tensor& rho = ctx->input(1);
    const Tensor& grad = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {0}, 0, ms.shape(), &output));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyRMSPropComputeRMS<Device, T>()(
        device, ms.flat<T>(), rho.scalar<T>(), grad.flat<T>(),
        output->flat<T>());
  }
};

template <typename Device, typename T>
class ApplyRMSPropVarUpdateOp : public OpKernel {
 public:
  explicit ApplyRMSPropVarUpdateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& var = ctx->input(0);
    const Tensor& ms = ctx->input(1);
    const Tensor& lr = ctx->input(2);
    const Tensor& epsilon = ctx->input(3);
    const Tensor& grad = ctx->input(4);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument(
                    "var and ms do not have the same shape ",
                    var.shape().DebugString(), " ", ms.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {0}, 0, var.shape(), &output));

    const Device& device = ctx->template eigen_device<Device>();

    functor::ApplyRMSPropVarUpdate<Device, T>()(
        device, var.flat<T>(), ms.flat<T>(), lr.scalar<T>(),
        epsilon.scalar<T>(), grad.flat<T>(), output->flat<T>());
  }
};

template <typename Device, typename T>
class ApplyCenteredRMSPropOp : public OpKernel {
 public:
  explicit ApplyCenteredRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2, 3});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor mg;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &mg));
    Tensor ms;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &ms));
    Tensor mom;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, sparse, &mom));

    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, mg.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, ms.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, mom.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));

    const Tensor& lr = ctx->input(4);
    const Tensor& rho = ctx->input(5);
    const Tensor& momentum = ctx->input(6);
    const Tensor& epsilon = ctx->input(7);
    const Tensor& grad = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mg.shape()),
                errors::InvalidArgument("var and mg do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyCenteredRMSProp<Device, T>()(
        device, var.flat<T>(), mg.flat<T>(), ms.flat<T>(), mom.flat<T>(),
        lr.scalar<T>(), rho.scalar<T>(), momentum.scalar<T>(),
        epsilon.scalar<T>(), grad.flat<T>());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyRMSProp").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      ApplyRMSPropOp<D##Device, T>);                                          \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyCenteredRMSProp").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyCenteredRMSPropOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyRMSProp")                        \
                              .Device(DEVICE_##D)                             \
                              .HostMemory("var")                              \
                              .HostMemory("ms")                               \
                              .HostMemory("mom")                              \
                              .TypeConstraint<T>("T"),                        \
                          ApplyRMSPropOp<D##Device, T>);                      \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyCenteredRMSProp")                \
                              .Device(DEVICE_##D)                             \
                              .HostMemory("var")                              \
                              .HostMemory("mg")                               \
                              .HostMemory("ms")                               \
                              .HostMemory("mom")                              \
                              .TypeConstraint<T>("T"),                        \
                          ApplyCenteredRMSPropOp<D##Device, T>);

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

#define REGISTER_KERNELS(D, T)                                                 \
  REGISTER_KERNEL_BUILDER(Name("ApplyRMSPropComputeRMS")                       \
                              .Device(DEVICE_##D)                              \
                              .TypeConstraint<T>("T"),                         \
                          ApplyRMSPropComputeRMSOp<D##Device, T>);             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ApplyRMSPropVarUpdate").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyRMSPropVarUpdateOp<D##Device, T>);

#define REGISTER_ITEX_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T)
TF_CALL_half(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_float(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_bfloat16(REGISTER_ITEX_GPU_KERNELS);
#undef REGISTER_ITEX_GPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace itex
