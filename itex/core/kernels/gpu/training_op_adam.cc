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
struct ComputeAdamKernel {
  ComputeAdamKernel(const T* grad_ptr, T* m_ptr, T* v_ptr, T* var_ptr,
                    const T* beta1_power, const T* beta2_power, const T* lr,
                    const T* beta1, const T* beta2, const T* epsilon,
                    int total_size)
      : grad_ptr_(grad_ptr),
        m_ptr_(m_ptr),
        v_ptr_(v_ptr),
        var_ptr_(var_ptr),
        beta1_power_(beta1_power),
        beta2_power_(beta2_power),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        total_size_(total_size) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    const T alpha = lr_[0] * Eigen::numext::sqrt(T(1) - beta2_power_[0]) /
                    (T(1) - beta1_power_[0]);
    auto beta1_sub = T(1) - beta1_[0];
    auto beta2_sub = T(1) - beta2_[0];

    auto grad_ele = grad_ptr_[id];
    auto m_ele = m_ptr_[id];
    auto v_ele = v_ptr_[id];
    auto m = m_ele + (grad_ele - m_ele) * beta1_sub;
    auto v = v_ele + (grad_ele * grad_ele - v_ele) * beta2_sub;
    v_ptr_[id] = v;
    m_ptr_[id] = m;
    auto var_ele =
        var_ptr_[id] - (m * alpha) / (Eigen::numext::sqrt(v) + epsilon_[0]);
    var_ptr_[id] = var_ele;
  }

 private:
  const T* grad_ptr_;
  T* m_ptr_;
  T* v_ptr_;
  T* var_ptr_;
  const T* beta1_power_;
  const T* beta2_power_;
  const T* lr_;
  const T* beta1_;
  const T* beta2_;
  const T* epsilon_;
  int total_size_;
};

template <typename T>
struct ComputeFusedAdamKernel {
  ComputeFusedAdamKernel(const T* mul_left_ptr, const T* mul_right_ptr,
                         T* m_ptr, T* v_ptr, T* var_ptr, const T* beta1_power,
                         const T* beta2_power, const T* lr, const T* beta1,
                         const T* beta2, const T* epsilon, int total_size)
      : mul_left_ptr_(mul_left_ptr),
        mul_right_ptr_(mul_right_ptr),
        m_ptr_(m_ptr),
        v_ptr_(v_ptr),
        var_ptr_(var_ptr),
        beta1_power_(beta1_power),
        beta2_power_(beta2_power),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        total_size_(total_size) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    const T alpha = lr_[0] * Eigen::numext::sqrt(T(1) - beta2_power_[0]) /
                    (T(1) - beta1_power_[0]);
    auto beta1_sub = T(1) - beta1_[0];
    auto beta2_sub = T(1) - beta2_[0];

    auto grad_ele = mul_left_ptr_[id] * (*mul_right_ptr_);
    auto m_ele = m_ptr_[id];
    auto v_ele = v_ptr_[id];
    auto m = m_ele + (grad_ele - m_ele) * beta1_sub;
    auto v = v_ele + (grad_ele * grad_ele - v_ele) * beta2_sub;
    v_ptr_[id] = v;
    m_ptr_[id] = m;
    auto var_ele =
        var_ptr_[id] - (m * alpha) / (Eigen::numext::sqrt(v) + epsilon_[0]);
    var_ptr_[id] = var_ele;
  }

 private:
  const T* mul_left_ptr_;
  const T* mul_right_ptr_;
  T* m_ptr_;
  T* v_ptr_;
  T* var_ptr_;
  const T* beta1_power_;
  const T* beta2_power_;
  const T* lr_;
  const T* beta1_;
  const T* beta2_;
  const T* epsilon_;
  int total_size_;
};

template <typename T, bool fused>
class AdamKernel;

template <typename T>
struct ApplyAdamITEX_GPU {
  void operator()(const GPUDevice& d, T* var, T* m, T* v, const T* beta1_power,
                  const T* beta2_power, const T* lr, const T* beta1,
                  const T* beta2, const T* epsilon, const T* grad,
                  OpKernelContext* ctx, int elements) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeAdamKernel<T> task(grad, m, v, var, beta1_power, beta2_power, lr,
                                beta1, beta2, epsilon, elements);

      cgh.parallel_for<AdamKernel<T, false>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

template <typename T>
struct FusedApplyAdamITEX_GPU {
  void operator()(const GPUDevice& d, T* var, T* m, T* v, const T* beta1_power,
                  const T* beta2_power, const T* lr, const T* beta1,
                  const T* beta2, const T* epsilon, const T* mul_left,
                  const T* mul_right, OpKernelContext* ctx, int elements) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeFusedAdamKernel<T> task(mul_left, mul_right, m, v, var,
                                     beta1_power, beta2_power, lr, beta1, beta2,
                                     epsilon, elements);

      cgh.parallel_for<AdamKernel<T, true>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

template <typename T>
struct ComputeAdamWeightDecayKernel {
  ComputeAdamWeightDecayKernel(const T* grad_ptr, T* m_ptr, T* v_ptr,
                               T* vhat_ptr, T* var_ptr, const T* beta1_power,
                               const T* beta2_power, const T* lr,
                               const T* beta1, const T* beta2, const T* epsilon,
                               int total_size, const T* wd, bool use_amsgrad)
      : grad_ptr_(grad_ptr),
        m_ptr_(m_ptr),
        v_ptr_(v_ptr),
        vhat_ptr_(vhat_ptr),
        var_ptr_(var_ptr),
        beta1_power_(beta1_power),
        beta2_power_(beta2_power),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        wd_(wd),
        use_amsgrad_(use_amsgrad),
        total_size_(total_size) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    const T alpha = lr_[0] * Eigen::numext::sqrt(T(1) - beta2_power_[0]) /
                    (T(1) - beta1_power_[0]);
    auto beta1_sub = T(1) - beta1_[0];
    auto beta2_sub = T(1) - beta2_[0];
    auto wd_sub = T(1) - wd_[0] * lr_[0];

    auto grad_ele = grad_ptr_[id];
    auto m_ele = m_ptr_[id];
    auto v_ele = v_ptr_[id];
    auto m = m_ele + (grad_ele - m_ele) * beta1_sub;
    auto v = v_ele + (grad_ele * grad_ele - v_ele) * beta2_sub;
    v_ptr_[id] = v;
    m_ptr_[id] = m;
    if (use_amsgrad_) {
      auto vhat = std::max(vhat_ptr_[id], v);
      v = vhat;
      vhat_ptr_[id] = vhat;
    }
    auto var_ele = wd_sub * var_ptr_[id] -
                   (m * alpha) / (Eigen::numext::sqrt(v) + epsilon_[0]);
    var_ptr_[id] = var_ele;
  }

 private:
  const T* grad_ptr_;
  T* m_ptr_;
  T* v_ptr_;
  T* vhat_ptr_;
  T* var_ptr_;
  const T* beta1_power_;
  const T* beta2_power_;
  const T* lr_;
  const T* beta1_;
  const T* beta2_;
  const T* epsilon_;
  const T* wd_;
  bool use_amsgrad_;
  int total_size_;
};

template <typename T>
struct ComputeFusedAdamWeightDecayKernel {
  ComputeFusedAdamWeightDecayKernel(const T* mul_left_ptr,
                                    const T* mul_right_ptr, T* m_ptr, T* v_ptr,
                                    T* vhat_ptr, T* var_ptr,
                                    const T* beta1_power, const T* beta2_power,
                                    const T* lr, const T* beta1, const T* beta2,
                                    const T* epsilon, int total_size,
                                    const T* wd, bool use_amsgrad)
      : mul_left_ptr_(mul_left_ptr),
        mul_right_ptr_(mul_right_ptr),
        m_ptr_(m_ptr),
        v_ptr_(v_ptr),
        vhat_ptr_(vhat_ptr),
        var_ptr_(var_ptr),
        beta1_power_(beta1_power),
        beta2_power_(beta2_power),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        use_amsgrad_(use_amsgrad),
        total_size_(total_size),
        wd_(wd) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    const T alpha = lr_[0] * Eigen::numext::sqrt(T(1) - beta2_power_[0]) /
                    (T(1) - beta1_power_[0]);
    auto beta1_sub = T(1) - beta1_[0];
    auto beta2_sub = T(1) - beta2_[0];
    auto wd_sub = T(1) - wd_[0] * lr_[0];

    auto grad_ele = mul_left_ptr_[id] * (*mul_right_ptr_);
    auto m_ele = m_ptr_[id];
    auto v_ele = v_ptr_[id];
    auto m = m_ele + (grad_ele - m_ele) * beta1_sub;
    auto v = v_ele + (grad_ele * grad_ele - v_ele) * beta2_sub;
    v_ptr_[id] = v;
    m_ptr_[id] = m;
    if (use_amsgrad_) {
      auto vhat = std::max(vhat_ptr_[id], v);
      v = vhat;
      vhat_ptr_[id] = vhat;
    }
    auto var_ele = wd_sub * var_ptr_[id] -
                   (m * alpha) / (Eigen::numext::sqrt(v) + epsilon_[0]);
    var_ptr_[id] = var_ele;
  }

 private:
  const T* mul_left_ptr_;
  const T* mul_right_ptr_;
  T* m_ptr_;
  T* v_ptr_;
  T* vhat_ptr_;
  T* var_ptr_;
  const T* beta1_power_;
  const T* beta2_power_;
  const T* lr_;
  const T* beta1_;
  const T* beta2_;
  const T* epsilon_;
  bool use_amsgrad_;
  int total_size_;
  const T* wd_;
};

template <typename T, bool fused>
class AdamWeightDecayKernel;

template <typename T>
struct ApplyAdamWeightDecayITEX_GPU {
  void operator()(const GPUDevice& d, T* var, T* m, T* v, T* vhat,
                  const T* beta1_power, const T* beta2_power, const T* lr,
                  const T* beta1, const T* beta2, const T* epsilon,
                  const T* grad, OpKernelContext* ctx, int elements,
                  const T* wd, bool use_amsgrad) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeAdamWeightDecayKernel<T> task(grad, m, v, vhat, var, beta1_power,
                                           beta2_power, lr, beta1, beta2,
                                           epsilon, elements, wd, use_amsgrad);

      cgh.parallel_for<AdamWeightDecayKernel<T, false>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

template <typename T>
struct FusedApplyAdamWeightDecayITEX_GPU {
  void operator()(const GPUDevice& d, T* var, T* m, T* v, T* vhat,
                  const T* beta1_power, const T* beta2_power, const T* lr,
                  const T* beta1, const T* beta2, const T* epsilon,
                  const T* mul_left, const T* mul_right, OpKernelContext* ctx,
                  int elements, const T* wd, bool use_amsgrad) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeFusedAdamWeightDecayKernel<T> task(
          mul_left, mul_right, m, v, vhat, var, beta1_power, beta2_power, lr,
          beta1, beta2, epsilon, elements, wd, use_amsgrad);

      cgh.parallel_for<AdamWeightDecayKernel<T, true>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

template <typename T>
struct ApplyAdamWithAmsgrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat vhat,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    m.device(d) =
        m + (beta1.constant(one) - beta1).reshape(single).broadcast(bcast) *
                (grad - m);
    v.device(d) =
        v + (beta2.constant(one) - beta2).reshape(single).broadcast(bcast) *
                (grad.square() - v);
    vhat.device(d) = vhat.cwiseMax(v);

    var.device(d) -= (lr * (beta2_power.constant(one) - beta2_power).sqrt() /
                      (beta1_power.constant(one) - beta1_power))
                         .reshape(single)
                         .broadcast(bcast) *
                     m /
                     (epsilon.reshape(single).broadcast(bcast) + vhat.sqrt());
  }
};

}  // namespace functor

template <typename Device, typename T>
class ApplyAdamOp;

template <typename T>
class ApplyAdamOp<GPUDevice, T> : public OpKernel {
 public:
  explicit ApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
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

    auto device = ctx->eigen_gpu_device();
    const Tensor& grad = ctx->input(9);

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
    functor::ApplyAdamITEX_GPU<T>()(
        device, var.flat<T>().data(), m.flat<T>().data(), v.flat<T>().data(),
        beta1_power_dev.flat<T>().data(), beta2_power_dev.flat<T>().data(),
        lr_dev.flat<T>().data(), beta1_dev.flat<T>().data(),
        beta2_dev.flat<T>().data(), epsilon_dev.flat<T>().data(),
        grad.flat<T>().data(), ctx, grad.NumElements());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename Device, typename T>
class FusedApplyAdamOp;

template <typename T>
class FusedApplyAdamOp<GPUDevice, T> : public OpKernel {
 public:
  explicit FusedApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES(ctx, fused_ops_[0] == "Mul" && fused_ops_.size() == 1,
                errors::Unimplemented("Only Mul + ApplyAdam is implemented"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_addn_inputs", &num_addn_inputs_));
    OP_REQUIRES(ctx, num_addn_inputs_ == 0,
                errors::Unimplemented("Only Mul + ApplyAdam is implemented"));
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

    // always treat the left input as a tenor, the right input as a scalar
    int left_index = 9;
    int right_index = 10;
    const TensorShape& left_shape = ctx->input(left_index).shape();
    const TensorShape& right_shape = ctx->input(right_index).shape();

    bool left_is_scalar = TensorShapeUtils::IsScalar(left_shape);
    bool right_is_scalar = TensorShapeUtils::IsScalar(right_shape);

    OP_REQUIRES(ctx, left_is_scalar || right_is_scalar,
                errors::InvalidArgument("neither of mul's inputs is a scalar: ",
                                        left_shape.DebugString(), " ",
                                        right_shape.DebugString()));
    if (left_is_scalar) {
      left_index = 10;
      right_index = 9;
    }

    const Tensor& mul_left = ctx->input(left_index);
    const Tensor& mul_right = ctx->input(right_index);

    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(mul_left.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                mul_left.shape().DebugString()));

    auto device = ctx->eigen_gpu_device();
    functor::FusedApplyAdamITEX_GPU<T>()(
        device, var.flat<T>().data(), m.flat<T>().data(), v.flat<T>().data(),
        beta1_power_dev.flat<T>().data(), beta2_power_dev.flat<T>().data(),
        lr_dev.flat<T>().data(), beta1_dev.flat<T>().data(),
        beta2_dev.flat<T>().data(), epsilon_dev.flat<T>().data(),
        mul_left.flat<T>().data(), mul_right.flat<T>().data(), ctx,
        mul_left.NumElements());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  std::vector<std::string> fused_ops_;
  int num_addn_inputs_;
};

template <typename Device, typename T>
class ApplyAdamWithWeightDecayOp : public OpKernel {
 public:
  explicit ApplyAdamWithWeightDecayOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_amsgrad", &use_amsgrad_));
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
    functor::ApplyAdamWeightDecayITEX_GPU<T>()(
        device, var.flat<T>().data(), m.flat<T>().data(), v.flat<T>().data(),
        vhat_ptr, beta1_power_dev.flat<T>().data(),
        beta2_power_dev.flat<T>().data(), lr_dev.flat<T>().data(),
        beta1_dev.flat<T>().data(), beta2_dev.flat<T>().data(),
        epsilon_dev.flat<T>().data(), grad.flat<T>().data(), ctx,
        grad.NumElements(), wd_dev.flat<T>().data(), use_amsgrad_);
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_amsgrad_;
};

template <typename Device, typename T>
class FusedApplyAdamWithWeightDecayOp : public OpKernel {
 public:
  explicit FusedApplyAdamWithWeightDecayOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_amsgrad", &use_amsgrad_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES(ctx, fused_ops_[0] == "Mul" && fused_ops_.size() == 1,
                errors::Unimplemented(
                    "Only Mul + ApplyAdamWithWeightDecay is implemented"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_addn_inputs", &num_addn_inputs_));
    OP_REQUIRES(ctx, num_addn_inputs_ == 0,
                errors::Unimplemented(
                    "Only Mul + ApplyAdamWithWeightDecay is implemented"));
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

    // always treat the left input as a tenor, the right input as a scalar
    int left_index = 11;
    int right_index = 12;
    const TensorShape& left_shape = ctx->input(left_index).shape();
    const TensorShape& right_shape = ctx->input(right_index).shape();

    bool left_is_scalar = TensorShapeUtils::IsScalar(left_shape);
    bool right_is_scalar = TensorShapeUtils::IsScalar(right_shape);

    OP_REQUIRES(ctx, left_is_scalar || right_is_scalar,
                errors::InvalidArgument("neither of mul's inputs is a scalar: ",
                                        left_shape.DebugString(), " ",
                                        right_shape.DebugString()));
    if (left_is_scalar) {
      left_index = 12;
      right_index = 11;
    }

    const Tensor& mul_left = ctx->input(left_index);
    const Tensor& mul_right = ctx->input(right_index);

    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(mul_left.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                mul_left.shape().DebugString()));
    auto device = ctx->eigen_gpu_device();
    functor::FusedApplyAdamWeightDecayITEX_GPU<T>()(
        device, var.flat<T>().data(), m.flat<T>().data(), v.flat<T>().data(),
        vhat_ptr, beta1_power_dev.flat<T>().data(),
        beta2_power_dev.flat<T>().data(), lr_dev.flat<T>().data(),
        beta1_dev.flat<T>().data(), beta2_dev.flat<T>().data(),
        epsilon_dev.flat<T>().data(), mul_left.flat<T>().data(),
        mul_right.flat<T>().data(), ctx, mul_left.NumElements(),
        wd_dev.flat<T>().data(), use_amsgrad_);
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_amsgrad_;
  std::vector<std::string> fused_ops_;
  int num_addn_inputs_;
};

#define REGISTER_KERNELS(T)                                        \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyAdam").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ApplyAdamOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdam")                \
                              .HostMemory("var")                   \
                              .HostMemory("m")                     \
                              .HostMemory("v")                     \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          ApplyAdamOp<GPUDevice, T>);

TF_CALL_complex64(REGISTER_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNELS);
TF_CALL_complex128(REGISTER_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNELS

#define REGISTER_ITEX_GPU_KERNELS(T)                                         \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ApplyAdam").Device(DEVICE_GPU).TypeConstraint<T>("T"),           \
      ApplyAdamOp<GPUDevice, T>);                                            \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdam")                          \
                              .HostMemory("var")                             \
                              .HostMemory("m")                               \
                              .HostMemory("v")                               \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T"),                       \
                          ApplyAdamOp<GPUDevice, T>);                        \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXFusedApplyAdam").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FusedApplyAdamOp<GPUDevice, T>);                                       \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedResourceApplyAdam")                \
                              .HostMemory("var")                             \
                              .HostMemory("m")                               \
                              .HostMemory("v")                               \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T"),                       \
                          FusedApplyAdamOp<GPUDevice, T>);                   \
  REGISTER_KERNEL_BUILDER(Name("ITEXApplyAdamWithWeightDecay")               \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T"),                       \
                          ApplyAdamWithWeightDecayOp<GPUDevice, T>);         \
  REGISTER_KERNEL_BUILDER(Name("ITEXResourceApplyAdamWithWeightDecay")       \
                              .HostMemory("var")                             \
                              .HostMemory("m")                               \
                              .HostMemory("v")                               \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T"),                       \
                          ApplyAdamWithWeightDecayOp<GPUDevice, T>);         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedApplyAdamWithWeightDecay")         \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T"),                       \
                          FusedApplyAdamWithWeightDecayOp<GPUDevice, T>);    \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedResourceApplyAdamWithWeightDecay") \
                              .HostMemory("var")                             \
                              .HostMemory("m")                               \
                              .HostMemory("v")                               \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T"),                       \
                          FusedApplyAdamWithWeightDecayOp<GPUDevice, T>);

TF_CALL_half(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_float(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_bfloat16(REGISTER_ITEX_GPU_KERNELS);
#undef REGISTER_ITEX_GPU_KERNELS

template <typename Device, typename T>
class ApplyAdamWithAmsgradOp : public OpKernel {
 public:
  explicit ApplyAdamWithAmsgradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
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
    Tensor vhat;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, sparse, &vhat));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, m.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, v.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, vhat.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));

    const Tensor& beta1_power = ctx->input(4);
    const Tensor& beta2_power = ctx->input(5);
    const Tensor& lr = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
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

    const Tensor& grad = ctx->input(10);
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
    functor::ApplyAdamWithAmsgrad<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(), vhat.flat<T>(),
        beta1_power.scalar<T>(), beta2_power.scalar<T>(), lr.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>(),
        grad.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                 \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdamWithAmsgrad") \
                              .HostMemory("var")               \
                              .HostMemory("m")                 \
                              .HostMemory("v")                 \
                              .HostMemory("vhat")              \
                              .Device(DEVICE_##D)              \
                              .TypeConstraint<T>("T"),         \
                          ApplyAdamWithAmsgradOp<D##Device, T>);

#define REGISTER_ITEX_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ITEX_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ITEX_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_ITEX_GPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace itex
