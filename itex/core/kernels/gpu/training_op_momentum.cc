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
using GPUDevice = Eigen::GpuDevice;
using Index = Eigen::Index;

namespace functor {
template <typename T>
struct ApplyMomentumUseNesterovKernel {
  ApplyMomentumUseNesterovKernel(int work_items, T* accum_ptr,
                                 const T* grad_ptr, T* var_ptr, const T* lr_ptr,
                                 const T* momentum_ptr)
      : work_items(work_items),
        accum_ptr(accum_ptr),
        grad_ptr(grad_ptr),
        var_ptr(var_ptr),
        lr_ptr(lr_ptr),
        momentum_ptr(momentum_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_id();
    if (index >= work_items) return;
    // load data
    auto accum_var = accum_ptr[index];
    auto grad_var = grad_ptr[index];
    auto var_ = var_ptr[index];
    auto lr_var = *lr_ptr;
    auto momentum_var = *momentum_ptr;

    // compute
    accum_var = accum_var * momentum_var + grad_var;
    var_ -= grad_var * lr_var + accum_var * momentum_var * lr_var;

    // write back
    accum_ptr[index] = accum_var;
    var_ptr[index] = var_;
  }

 private:
  int work_items;
  T* accum_ptr;
  const T* grad_ptr;
  T* var_ptr;
  const T* lr_ptr;
  const T* momentum_ptr;
};

template <typename T>
struct ApplyMomentumNoNesterovKernel {
  ApplyMomentumNoNesterovKernel(int work_items, T* accum_ptr, const T* grad_ptr,
                                T* var_ptr, const T* lr_ptr,
                                const T* momentum_ptr)
      : work_items(work_items),
        accum_ptr(accum_ptr),
        grad_ptr(grad_ptr),
        var_ptr(var_ptr),
        lr_ptr(lr_ptr),
        momentum_ptr(momentum_ptr) {}

  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_id();
    if (index >= work_items) return;
    // load data
    auto accum_var = accum_ptr[index];
    auto grad_var = grad_ptr[index];
    auto var_ = var_ptr[index];
    auto lr_var = *lr_ptr;
    auto momentum_var = *momentum_ptr;

    // compute
    accum_var = accum_var * momentum_var + grad_var;
    var_ -= accum_var * lr_var;

    // write back
    accum_ptr[index] = accum_var;
    var_ptr[index] = var_;
  }

 private:
  int work_items;
  T* accum_ptr;
  const T* grad_ptr;
  T* var_ptr;
  const T* lr_ptr;
  const T* momentum_ptr;
};

template <typename T>
struct ApplyMomentumITEX_GPU {
  void operator()(const GPUDevice& device, T* var_ptr, T* accum_ptr,
                  const T* lr_ptr, const T* grad_ptr, const T* momentum_ptr,
                  bool use_nesterov, const int work_items) {
    auto stream = device.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (work_items + group_size - 1) / group_size;
    if (use_nesterov) {
      stream->submit([&](sycl::handler& cgh) {
        ApplyMomentumUseNesterovKernel<T> task(work_items, accum_ptr, grad_ptr,
                                               var_ptr, lr_ptr, momentum_ptr);
        cgh.parallel_for<ApplyMomentumUseNesterovKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroup * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
    } else {
      stream->submit([&](sycl::handler& cgh) {
        ApplyMomentumNoNesterovKernel<T> task(work_items, accum_ptr, grad_ptr,
                                              var_ptr, lr_ptr, momentum_ptr);
        cgh.parallel_for<ApplyMomentumNoNesterovKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroup * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
    }
  }
};

template <typename T>
struct FusedApplyMomentumUseNesterovKernel {
  FusedApplyMomentumUseNesterovKernel(int work_items, T* accum_ptr,
                                      const T* mul_left, const T* mul_right,
                                      const T* addN_input, T* var_ptr,
                                      const T* lr_ptr, const T* momentum_ptr)
      : work_items(work_items),
        accum_ptr(accum_ptr),
        mul_left(mul_left),
        mul_right(mul_right),
        addN_input(addN_input),
        var_ptr(var_ptr),
        lr_ptr(lr_ptr),
        momentum_ptr(momentum_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_id();
    if (index >= work_items) return;
    // load data
    auto accum_var = accum_ptr[index];
    // auto grad_var = grad_ptr[index];
    auto grad_var = mul_left[index] * (*mul_right) + addN_input[index];
    auto var_ = var_ptr[index];
    auto lr_var = *lr_ptr;
    auto momentum_var = *momentum_ptr;

    // compute
    accum_var = accum_var * momentum_var + grad_var;
    var_ -= grad_var * lr_var + accum_var * momentum_var * lr_var;

    // write back
    accum_ptr[index] = accum_var;
    var_ptr[index] = var_;
  }

 private:
  int work_items;
  T* accum_ptr;
  const T* mul_left;
  const T* mul_right;
  const T* addN_input;
  T* var_ptr;
  const T* lr_ptr;
  const T* momentum_ptr;
};

template <typename T>
struct FusedApplyMomentumNoNesterovKernel {
  FusedApplyMomentumNoNesterovKernel(int work_items, T* accum_ptr,
                                     const T* mul_left, const T* mul_right,
                                     const T* addN_input, T* var_ptr,
                                     const T* lr_ptr, const T* momentum_ptr)
      : work_items(work_items),
        accum_ptr(accum_ptr),
        mul_left(mul_left),
        mul_right(mul_right),
        addN_input(addN_input),
        var_ptr(var_ptr),
        lr_ptr(lr_ptr),
        momentum_ptr(momentum_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_id();
    if (index >= work_items) return;
    // load data
    auto accum_var = accum_ptr[index];
    auto grad_var = mul_left[index] * (*mul_right) + addN_input[index];
    auto var_ = var_ptr[index];
    auto lr_var = *lr_ptr;
    auto momentum_var = *momentum_ptr;

    // compute
    accum_var = accum_var * momentum_var + grad_var;
    var_ -= accum_var * lr_var;

    // write back
    accum_ptr[index] = accum_var;
    var_ptr[index] = var_;
  }

 private:
  int work_items;
  T* accum_ptr;
  const T* mul_left;
  const T* mul_right;
  const T* addN_input;
  T* var_ptr;
  const T* lr_ptr;
  const T* momentum_ptr;
};

template <typename T>
struct FusedApplyMomentumITEX_GPU {
  void operator()(const GPUDevice& device, T* var_ptr, T* accum_ptr,
                  const T* lr_ptr, const T* mul_left, const T* mul_right,
                  const T* addN_input, const T* momentum_ptr, bool use_nesterov,
                  const int work_items) {
    auto stream = device.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (work_items + group_size - 1) / group_size;
    if (use_nesterov) {
      stream->submit([&](sycl::handler& cgh) {
        FusedApplyMomentumUseNesterovKernel<T> task(
            work_items, accum_ptr, mul_left, mul_right, addN_input, var_ptr,
            lr_ptr, momentum_ptr);
        cgh.parallel_for<FusedApplyMomentumUseNesterovKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroup * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
    } else {
      stream->submit([&](sycl::handler& cgh) {
        FusedApplyMomentumNoNesterovKernel<T> task(
            work_items, accum_ptr, mul_left, mul_right, addN_input, var_ptr,
            lr_ptr, momentum_ptr);
        cgh.parallel_for<FusedApplyMomentumNoNesterovKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroup * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
    }
  }
};

template <typename T>
struct ApplyKerasMomentum<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    accum.device(d) = (accum * momentum.reshape(single).broadcast(bcast) -
                       grad * lr.reshape(single).broadcast(bcast));
    if (use_nesterov) {
      var.device(d) += (accum * momentum.reshape(single).broadcast(bcast) -
                        grad * lr.reshape(single).broadcast(bcast));
    } else {
      var.device(d) += accum;
    }
  }
};

}  // namespace functor

template <typename Device, typename T>
class ApplyMomentumOp;

template <typename T>
class ApplyMomentumOp<GPUDevice, T> : public OpKernel {
 public:
  explicit ApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<GPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
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

    const Tensor& momentum = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    auto& device = ctx->eigen_gpu_device();
    functor::ApplyMomentumITEX_GPU<T>()(
        device, var.flat<T>().data(), accum.flat<T>().data(),
        lr.scalar<T>().data(), grad.flat<T>().data(),
        momentum.scalar<T>().data(), use_nesterov_, var.flat<T>().size());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

template <typename Device, typename T>
class FusedApplyMomentumOp;

template <typename T>
class FusedApplyMomentumOp<GPUDevice, T> : public OpKernel {
 public:
  explicit FusedApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES(ctx, fused_ops_[0] == "Mul" && fused_ops_[1] == "AddN",
                errors::Unimplemented(
                    "Only Mul + AddN + ApplyMomentumOp is implemented"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_addn_inputs", &num_addn_inputs_));
    OP_REQUIRES(
        ctx, num_addn_inputs_ == 1,
        errors::Unimplemented(
            "Only num_addn_inputs = 1 is supported by _FusedApplyMomentumOp"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_mul_inputs", &num_mul_inputs_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<GPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& momentum = ctx->input(3);

    Tensor mul_left, mul_right, addN_input;
    if (num_mul_inputs_ == 1) {
      mul_left = var;
      mul_right = ctx->input(4);
      addN_input = ctx->input(5);
    } else {
      // always treat the left input as a tenor, the right input as a scalar
      int left_index = 4;
      int right_index = 5;
      const TensorShape& left_shape = ctx->input(left_index).shape();
      const TensorShape& right_shape = ctx->input(right_index).shape();

      bool left_is_scalar = TensorShapeUtils::IsScalar(left_shape);
      bool right_is_scalar = TensorShapeUtils::IsScalar(right_shape);

      OP_REQUIRES(
          ctx, left_is_scalar || right_is_scalar,
          errors::InvalidArgument(
              "neither of mul's inputs is a scalar: ", left_shape.DebugString(),
              " ", right_shape.DebugString()));
      if (left_is_scalar) {
        left_index = 5;
        right_index = 4;
      }

      mul_left = ctx->input(left_index);
      mul_right = ctx->input(right_index);
      addN_input = ctx->input(6);
    }

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, mul_left.shape().IsSameSize(addN_input.shape()),
                errors::InvalidArgument(
                    "mul_left and addN_input do not have the same shape",
                    mul_left.shape().DebugString(), " ",
                    addN_input.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(addN_input.shape()),
        errors::InvalidArgument("var and addN_input do not have the same shape",
                                var.shape().DebugString(), " ",
                                addN_input.shape().DebugString()));

    auto& device = ctx->eigen_gpu_device();
    functor::FusedApplyMomentumITEX_GPU<T>()(
        device, var.flat<T>().data(), accum.flat<T>().data(),
        lr.scalar<T>().data(), mul_left.flat<T>().data(),
        mul_right.flat<T>().data(), addN_input.flat<T>().data(),
        momentum.scalar<T>().data(), use_nesterov_, var.flat<T>().size());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
  std::vector<std::string> fused_ops_;
  int num_addn_inputs_;
  int num_mul_inputs_;
};

#define REGISTER_KERNELS(D, T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyMomentum").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyMomentumOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyMomentum")                \
                              .Device(DEVICE_##D)                      \
                              .HostMemory("var")                       \
                              .HostMemory("accum")                     \
                              .TypeConstraint<T>("T"),                 \
                          ApplyMomentumOp<D##Device, T>);

#define REGISTER_ITEX_GPU_KERNELS(T)                                   \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyMomentum").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ApplyMomentumOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyMomentum")                \
                              .Device(DEVICE_GPU)                      \
                              .HostMemory("var")                       \
                              .HostMemory("accum")                     \
                              .TypeConstraint<T>("T"),                 \
                          ApplyMomentumOp<GPUDevice, T>);
#define REGISTER_FUSED_ITEX_GPU_KERNELS(T)                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_FusedApplyMomentum").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FusedApplyMomentumOp<GPUDevice, T>);                                   \
  REGISTER_KERNEL_BUILDER(Name("_FusedResourceApplyMomentum")                \
                              .Device(DEVICE_GPU)                            \
                              .HostMemory("var")                             \
                              .HostMemory("accum")                           \
                              .TypeConstraint<T>("T"),                       \
                          FusedApplyMomentumOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_complex64(REGISTER_ITEX_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_complex128(REGISTER_ITEX_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_float(REGISTER_FUSED_ITEX_GPU_KERNELS);
#undef REGISTER_ITEX_GPU_KERNELS
#undef REGISTER_FUSED_ITEX_GPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyKerasMomentumOp : public OpKernel {
 public:
  explicit ApplyKerasMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
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

    const Tensor& momentum = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyKerasMomentum<Device, T>()(
        device, var.flat<T>(), accum.flat<T>(), lr.scalar<T>(), grad.flat<T>(),
        momentum.scalar<T>(), use_nesterov_);
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(D, T)                               \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyKerasMomentum") \
                              .Device(DEVICE_##D)            \
                              .HostMemory("var")             \
                              .HostMemory("accum")           \
                              .TypeConstraint<T>("T"),       \
                          ApplyKerasMomentumOp<D##Device, T>);

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, Eigen::bfloat16);
REGISTER_KERNELS(GPU, complex64);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNELS(GPU, double);
REGISTER_KERNELS(GPU, complex128);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_KERNELS

}  // namespace itex
