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
struct ComputeAdagradV2Kernel {
  ComputeAdagradV2Kernel(const T* grad_ptr, T* var_ptr, T* accum_ptr,
                         const T* lr, const T* epsilon, int total_size,
                         bool update_slots)
      : grad_ptr_(grad_ptr),
        var_ptr_(var_ptr),
        accum_ptr_(accum_ptr),
        lr_(lr),
        epsilon_(epsilon),
        total_size_(total_size),
        update_slots_(update_slots) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    auto accum = update_slots_ ? accum_ptr_[id] + grad_ptr_[id] * grad_ptr_[id]
                               : accum_ptr_[id];
    accum_ptr_[id] = accum;
    auto var_ele =
        var_ptr_[id] -
        lr_[0] * grad_ptr_[id] / (Eigen::numext::sqrt(accum) + epsilon_[0]);
    var_ptr_[id] = var_ele;
  }

 private:
  const T* grad_ptr_;
  T* var_ptr_;
  T* accum_ptr_;
  const T* lr_;
  const T* epsilon_;
  int total_size_;
  bool update_slots_;
};

template <typename T>
class AdagradV2Kernel;

template <typename T>
struct ApplyAdagradV2DPCPP {
  void operator()(const GPUDevice& d, T* var, T* accum, const T* lr,
                  const T* epsilon, const T* grad, OpKernelContext* ctx,
                  int elements, bool update_slots) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeAdagradV2Kernel<T> task(grad, var, accum, lr, epsilon, elements,
                                     update_slots);

      cgh.parallel_for<AdagradV2Kernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

}  // namespace functor

template <typename T>
class ApplyAdagradV2Op : public OpKernel {
 public:
  explicit ApplyAdagradV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
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
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& epsilon = ctx->input(3);
    OP_REQUIRES(ctx, IsLegacyScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(4);
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

    auto device = ctx->eigen_gpu_device();
    functor::ApplyAdagradV2DPCPP<T>()(
        device, var.flat<T>().data(), accum.flat<T>().data(),
        lr.flat<T>().data(), epsilon.flat<T>().data(), grad.flat<T>().data(),
        ctx, grad.NumElements(), update_slots_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_DPCPP_KERNELS(T)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ApplyAdagradV2").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ApplyAdagradV2Op<T>);                                             \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdagradV2")                \
                              .HostMemory("var")                        \
                              .HostMemory("accum")                      \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<T>("T"),                  \
                          ApplyAdagradV2Op<T>);

TF_CALL_half(REGISTER_DPCPP_KERNELS);
TF_CALL_float(REGISTER_DPCPP_KERNELS);
TF_CALL_bfloat16(REGISTER_DPCPP_KERNELS);
TF_CALL_complex64(REGISTER_DPCPP_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_complex128(REGISTER_DPCPP_KERNELS);
TF_CALL_double(REGISTER_DPCPP_KERNELS);
#endif
#undef REGISTER_DPCPP_KERNELS

}  // namespace itex
