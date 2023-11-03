/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/kernels/gpu/full_reduction_kernels.h"
#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/kernels/gpu/training_ops.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

namespace functor {
template <typename T>
struct ComputeLAMBStage1Kernel {
  ComputeLAMBStage1Kernel(const T* grad_ptr, T* m_ptr, T* v_ptr, T* var_ptr,
                          T* update_ptr, T* vhat_ptr, const T* beta1_power,
                          const T* beta2_power, const T* beta1, const T* beta2,
                          const T* epsilon, const T* wd, bool use_amsgrad,
                          int total_size)
      : grad_ptr_(grad_ptr),
        m_ptr_(m_ptr),
        v_ptr_(v_ptr),
        var_ptr_(var_ptr),
        update_ptr_(update_ptr),
        vhat_ptr_(vhat_ptr),
        beta1_power_(beta1_power),
        beta2_power_(beta2_power),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        wd_(wd),
        use_amsgrad_(use_amsgrad),
        total_size_(total_size) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    auto beta1_sub = T(1) - beta1_[0];
    auto beta2_sub = T(1) - beta2_[0];
    auto grad_ele = grad_ptr_[id];
    auto m_ele = m_ptr_[id];
    auto v_ele = v_ptr_[id];
    auto var_ele = var_ptr_[id];
    auto m = m_ele + (grad_ele - m_ele) * beta1_sub;
    auto v = v_ele + (grad_ele * grad_ele - v_ele) * beta2_sub;
    v_ptr_[id] = v;
    m_ptr_[id] = m;
    m = m / (T(1) - beta1_power_[0]);
    v = v / (T(1) - beta2_power_[0]);
    if (use_amsgrad_) {
      auto vhat = std::max(vhat_ptr_[id], v);
      v = vhat;
      vhat_ptr_[id] = vhat;
    }
    auto update = m / (Eigen::numext::sqrt(v) + epsilon_[0]) + wd_[0] * var_ele;
    update_ptr_[id] = update;
  }

 private:
  const T* grad_ptr_;
  T* m_ptr_;
  T* v_ptr_;
  T* var_ptr_;
  T* update_ptr_;
  T* vhat_ptr_;
  const T* beta1_power_;
  const T* beta2_power_;
  const T* beta1_;
  const T* beta2_;
  const T* epsilon_;
  const T* wd_;
  bool use_amsgrad_;
  int total_size_;
};

template <typename T>
struct ComputeLAMBStage2Kernel {
  ComputeLAMBStage2Kernel(T* var_ptr, const T* update_ptr, const T* lr,
                          int total_size, const T* w_norm, const T* update_norm,
                          bool use_lamb)
      : var_ptr_(var_ptr),
        update_ptr_(update_ptr),
        lr_(lr),
        w_norm_(w_norm),
        update_norm_(update_norm),
        use_lamb_(use_lamb),
        total_size_(total_size) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    auto update_ele = update_ptr_[id];
    auto ratio = T(1);
    if (use_lamb_) {
      auto w_norm_ele = w_norm_[0];
      auto update_norm_ele = update_norm_[0];
      if (w_norm_ele != T(0) && update_norm_ele != T(0))
        ratio = Eigen::numext::sqrt(w_norm_ele) /
                Eigen::numext::sqrt(update_norm_ele);
    }
    auto var_ele = var_ptr_[id] - ratio * lr_[0] * update_ele;
    var_ptr_[id] = var_ele;
  }

 private:
  T* var_ptr_;
  const T* update_ptr_;
  const T* lr_;
  const T* w_norm_;
  const T* update_norm_;
  bool use_lamb_;
  int total_size_;
};

template <typename T>
struct square {
  inline T operator()(T x) const { return x * x; }
};

template <typename T, bool fused>
class LAMBStage1Kernel;

template <typename T>
struct ApplyLAMBStage1ITEX_GPU {
  void operator()(const GPUDevice& d, T* var, T* m, T* v, T* vhat,
                  const T* beta1_power, const T* beta2_power, const T* beta1,
                  const T* beta2, const T* epsilon, const T* grad, T* update,
                  OpKernelContext* ctx, const T* wd, bool use_amsgrad,
                  int elements) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeLAMBStage1Kernel<T> task(grad, m, v, var, update, vhat,
                                      beta1_power, beta2_power, beta1, beta2,
                                      epsilon, wd, use_amsgrad, elements);

      cgh.parallel_for<LAMBStage1Kernel<T, false>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

template <typename T, bool fused>
class LAMBStage2Kernel;

template <typename T>
struct ApplyLAMBStage2ITEX_GPU {
  void operator()(const GPUDevice& d, T* var, const T* update, const T* lr,
                  OpKernelContext* ctx, int elements, const T* w_norm,
                  const T* update_norm, bool use_lamb) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeLAMBStage2Kernel<T> task(var, update, lr, elements, w_norm,
                                      update_norm, use_lamb);

      cgh.parallel_for<LAMBStage2Kernel<T, false>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

}  // namespace functor

template <typename Device, typename T>
class ApplyLAMBOp : public OpKernel {
 public:
  explicit ApplyLAMBOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_amsgrad", &use_amsgrad_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_lamb", &use_lamb_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<GPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, m.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, v.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));

    const Tensor& beta1_power_dev = ctx->input(3);
    const Tensor& beta2_power_dev = ctx->input(4);
    const Tensor& lr_dev = ctx->input(5);
    const Tensor& beta1_dev = ctx->input(6);
    const Tensor& beta2_dev = ctx->input(7);
    const Tensor& epsilon_dev = ctx->input(8);
    const Tensor& wd_dev = ctx->input(9);

    Tensor vhat;
    T* vhat_ptr = nullptr;
    if (use_amsgrad_) {
      OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                              ctx, 10, use_exclusive_lock_, sparse, &vhat));
      vhat_ptr = vhat.flat<T>().data();
    }

    auto device = ctx->eigen_gpu_device();
    const Tensor& grad = ctx->input(11);

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

    Tensor update;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           var.shape(), &update));
    functor::ApplyLAMBStage1ITEX_GPU<T>()(
        device, var.flat<T>().data(), m.flat<T>().data(), v.flat<T>().data(),
        vhat_ptr, beta1_power_dev.flat<T>().data(),
        beta2_power_dev.flat<T>().data(), beta1_dev.flat<T>().data(),
        beta2_dev.flat<T>().data(), epsilon_dev.flat<T>().data(),
        grad.flat<T>().data(), update.flat<T>().data(), ctx,
        wd_dev.flat<T>().data(), use_amsgrad_, grad.NumElements());

    Tensor w_norm;
    T* w_norm_ptr = nullptr;
    Tensor update_norm;
    T* update_norm_ptr = nullptr;
    if (use_lamb_) {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape({}), &w_norm));
      w_norm_ptr = w_norm.flat<T>().data();
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape({}), &update_norm));
      update_norm_ptr = update_norm.flat<T>().data();

      LaunchFullReduction<const T, T, T, sycl::plus<T>, functor::square<T>>(
          ctx, var.flat<T>().data(), w_norm_ptr, T(0), var.flat<T>().size(),
          sycl::plus<T>(), functor::square<T>());
      LaunchFullReduction<const T, T, T, sycl::plus<T>, functor::square<T>>(
          ctx, update.flat<T>().data(), update_norm_ptr, T(0),
          update.flat<T>().size(), sycl::plus<T>(), functor::square<T>());
    }

    functor::ApplyLAMBStage2ITEX_GPU<T>()(
        device, var.flat<T>().data(), update.flat<T>().data(),
        lr_dev.flat<T>().data(), ctx, var.NumElements(), w_norm_ptr,
        update_norm_ptr, use_lamb_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_amsgrad_;
  bool use_lamb_;
};

#define REGISTER_ITEX_GPU_KERNELS(T)                                   \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ITEXApplyLAMB").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ApplyLAMBOp<GPUDevice, T>);                                      \
  REGISTER_KERNEL_BUILDER(Name("ITEXResourceApplyLAMB")                \
                              .HostMemory("var")                       \
                              .HostMemory("m")                         \
                              .HostMemory("v")                         \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<T>("T"),                 \
                          ApplyLAMBOp<GPUDevice, T>);

TF_CALL_float(REGISTER_ITEX_GPU_KERNELS);
#undef REGISTER_ITEX_GPU_KERNELS

}  // namespace itex
