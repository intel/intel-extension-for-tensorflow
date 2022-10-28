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

template <typename T>
struct ComputeAdaMaxKernel {
  ComputeAdaMaxKernel(const T* grad_ptr, T* var_ptr, T* m_ptr, T* v_ptr,
                      const T* beta1_power, const T* lr, const T* beta1,
                      const T* beta2, const T* epsilon, int total_size)
      : grad_ptr_(grad_ptr),
        var_ptr_(var_ptr),
        m_ptr_(m_ptr),
        v_ptr_(v_ptr),
        beta1_power_(beta1_power),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        total_size_(total_size) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    auto grad_ele = grad_ptr_[id];
    auto grad_abs = Eigen::numext::abs(grad_ele);
    auto m_ele = m_ptr_[id];
    auto v_ele = v_ptr_[id];
    auto m = m_ele + (T(1) - beta1_[0]) * (grad_ele - m_ele);
    auto tmp = beta2_[0] * v_ele;
    if (tmp < grad_abs) {
      v_ptr_[id] = grad_abs;
    } else {
      v_ptr_[id] = tmp;
    }
    m_ptr_[id] = m;
    v_ele = v_ptr_[id];

    auto var_ele = var_ptr_[id] - ((lr_[0] / (T(1) - beta1_power_[0])) *
                                   (m / (v_ele + epsilon_[0])));
    var_ptr_[id] = var_ele;
  }

 private:
  const T* grad_ptr_;
  T* var_ptr_;
  T* m_ptr_;
  T* v_ptr_;
  const T* beta1_power_;
  const T* lr_;
  const T* beta1_;
  const T* beta2_;
  const T* epsilon_;
  int total_size_;
};

template <typename T>
class AdaMaxKernel;

template <typename T>
struct ApplyAdaMax {
  void operator()(const GPUDevice& d, T* var, T* m, T* v, const T* beta1_power,
                  const T* lr, const T* beta1, const T* beta2, const T* epsilon,
                  const T* grad, OpKernelContext* ctx, int elements) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeAdaMaxKernel<T> task(grad, var, m, v, beta1_power, lr, beta1,
                                  beta2, epsilon, elements);

      cgh.parallel_for<AdaMaxKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

template <typename Device, typename T>
class ApplyAdaMaxOp : public OpKernel {
 public:
  explicit ApplyAdaMaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(ctx, m.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(ctx, v.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: "));

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1 = ctx->input(5);
    const Tensor& beta2 = ctx->input(6);
    const Tensor& epsilon = ctx->input(7);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(8);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    ApplyAdaMax<T>()(device, var.flat<T>().data(), m.flat<T>().data(),
                     v.flat<T>().data(), beta1_power.scalar<T>().data(),
                     lr.scalar<T>().data(), beta1.scalar<T>().data(),
                     beta2.scalar<T>().data(), epsilon.scalar<T>().data(),
                     grad.flat<T>().data(), ctx, grad.NumElements());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ApplyAdaMax").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdaMaxOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdaMax")                \
                              .HostMemory("var")                     \
                              .HostMemory("m")                       \
                              .HostMemory("v")                       \
                              .Device(DEVICE_##D)                    \
                              .TypeConstraint<T>("T"),               \
                          ApplyAdaMaxOp<D##Device, T>);

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, Eigen::bfloat16);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNELS(GPU, double);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNELS
}  // namespace itex
