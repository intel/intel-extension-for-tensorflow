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

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

namespace itex {

namespace {
template <class T>
inline T sgn(const T x) {
  T zero(0);
  T one(1);
  return (x == zero ? zero : (x < zero ? -one : one));
}
}  // namespace

namespace functor {
template <typename T>
struct ComputeFtrlKernel {
  ComputeFtrlKernel(T* var_ptr, T* accum_ptr, T* linear, const T* grad_ptr,
                    const T* lr, const T* l1, const T* l2, const T* lr_power,
                    int total_size)
      : var_ptr_(var_ptr),
        accum_ptr_(accum_ptr),
        linear_(linear),
        grad_ptr_(grad_ptr),
        lr_(lr),
        l1_(l1),
        l2_(l2),
        lr_power_(lr_power),
        total_size_(total_size) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    auto accum = accum_ptr_[id] + grad_ptr_[id] * grad_ptr_[id];
    if (lr_power_[0] == static_cast<T>(-0.5)) {
      linear_[id] =
          linear_[id] + grad_ptr_[id] -
          (Eigen::numext::sqrt(accum) - Eigen::numext::sqrt(accum_ptr_[id])) /
              lr_[0] * var_ptr_[id];
    } else {
      linear_[id] = linear_[id] + grad_ptr_[id] -
                    (Eigen::numext::pow(accum, -lr_power_[0]) -
                     Eigen::numext::pow(accum_ptr_[id], -lr_power_[0])) /
                        lr_[0] * var_ptr_[id];
    }
    auto x = l1_[0] * sgn<T>(linear_[id]) - linear_[id];
    if (lr_power_[0] == static_cast<T>(-0.5)) {
      auto y = Eigen::numext::sqrt(accum) / lr_[0] + static_cast<T>(2) * l2_[0];
      auto pre_shrink = x / y;
      var_ptr_[id] = Eigen::numext::abs(linear_[id]) > l1_[0]
                         ? pre_shrink
                         : static_cast<T>(0);
    } else {
      auto y = Eigen::numext::pow(accum, -lr_power_[0]) / lr_[0] +
               static_cast<T>(2) * l2_[0];
      auto pre_shrink = x / y;
      var_ptr_[id] = Eigen::numext::abs(linear_[id]) > l1_[0]
                         ? pre_shrink
                         : static_cast<T>(0);
    }
    accum_ptr_[id] += grad_ptr_[id] * grad_ptr_[id];
  }

 private:
  T* var_ptr_;
  T* accum_ptr_;
  T* linear_;
  const T* grad_ptr_;
  const T* lr_;
  const T* l1_;
  const T* l2_;
  const T* lr_power_;
  int total_size_;
};

template <typename T>
struct ComputeFtrlV2Kernel {
  ComputeFtrlV2Kernel(T* var_ptr, T* accum_ptr, T* linear, const T* grad_ptr,
                      const T* lr, const T* l1, const T* l2,
                      const T* l2_shrinkage, const T* lr_power, int total_size)
      : var_ptr_(var_ptr),
        accum_ptr_(accum_ptr),
        linear_(linear),
        grad_ptr_(grad_ptr),
        lr_(lr),
        l1_(l1),
        l2_(l2),
        l2_shrinkage_(l2_shrinkage),
        lr_power_(lr_power),
        total_size_(total_size) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    auto grad_with_shrinkage =
        grad_ptr_[id] + static_cast<T>(2) * l2_shrinkage_[0] * var_ptr_[id];
    auto accum = accum_ptr_[id] + grad_ptr_[id] * grad_ptr_[id];
    if (lr_power_[0] == static_cast<T>(-0.5)) {
      linear_[id] =
          linear_[id] + grad_with_shrinkage -
          (Eigen::numext::sqrt(accum) - Eigen::numext::sqrt(accum_ptr_[id])) /
              lr_[0] * var_ptr_[id];
    } else {
      linear_[id] = linear_[id] + grad_with_shrinkage -
                    (Eigen::numext::pow(accum, -lr_power_[0]) -
                     Eigen::numext::pow(accum_ptr_[id], -lr_power_[0])) /
                        lr_[0] * var_ptr_[id];
    }
    auto x = l1_[0] * sgn<T>(linear_[id]) - linear_[id];
    if (lr_power_[0] == static_cast<T>(-0.5)) {
      auto y = Eigen::numext::sqrt(accum) / lr_[0] + static_cast<T>(2) * l2_[0];
      auto pre_shrink = x / y;
      var_ptr_[id] = Eigen::numext::abs(linear_[id]) > l1_[0]
                         ? pre_shrink
                         : static_cast<T>(0);
    } else {
      auto y = Eigen::numext::pow(accum, -lr_power_[0]) / lr_[0] +
               static_cast<T>(2) * l2_[0];
      auto pre_shrink = x / y;
      var_ptr_[id] = Eigen::numext::abs(linear_[id]) > l1_[0]
                         ? pre_shrink
                         : static_cast<T>(0);
    }
    accum_ptr_[id] += grad_ptr_[id] * grad_ptr_[id];
  }

 private:
  T* var_ptr_;
  T* accum_ptr_;
  T* linear_;
  const T* grad_ptr_;
  const T* lr_;
  const T* l1_;
  const T* l2_;
  const T* l2_shrinkage_;
  const T* lr_power_;
  int total_size_;
};

template <typename T>
class FtrlKernel;

template <typename T>
class FtrlV2Kernel;

template <typename T>
struct ApplyFtrlDPCPP {
  void operator()(const GPUDevice& d, T* var, T* accum, T* linear,
                  const T* grad, const T* lr, const T* l1, const T* l2,
                  const T* lr_power, OpKernelContext* ctx, int elements) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeFtrlKernel<T> task(var, accum, linear, grad, lr, l1, l2, lr_power,
                                elements);

      cgh.parallel_for<FtrlKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

template <typename T>
struct ApplyFtrlV2DPCPP {
  void operator()(const GPUDevice& d, T* var, T* accum, T* linear,
                  const T* grad, const T* lr, const T* l1, const T* l2,
                  const T* l2_shrinkage, const T* lr_power,
                  OpKernelContext* ctx, int elements) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeFtrlV2Kernel<T> task(var, accum, linear, grad, lr, l1, l2,
                                  l2_shrinkage, lr_power, elements);

      cgh.parallel_for<FtrlV2Kernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

template <typename T, typename Tindex, bool has_l2_shrinkage>
struct SparseApplyFtrlKernel {
  SparseApplyFtrlKernel(T* var, T* accum, T* linear, const T* lr, const T* l1,
                        const T* l2, const T* l2_shrinkage, const T* lr_power,
                        const T* grad, const Tindex* indices, Tindex param_rows,
                        Tindex updates_size, Tindex indices_size,
                        bool multiply_linear_by_lr)
      : var_(var),
        accum_(accum),
        linear_(linear),
        lr_(lr),
        l1_(l1),
        l2_(l2),
        l2_shrinkage_(l2_shrinkage),
        lr_power_(lr_power),
        grad_(grad),
        indices_(indices),
        param_rows_(param_rows),
        updates_size_(updates_size),
        indices_size_(indices_size),
        multiply_linear_by_lr_(multiply_linear_by_lr) {}
  void operator()(sycl::item<1> item) const {
    int64_t grad_index = item.get_linear_id();
    const Tindex col_size = updates_size_ / indices_size_;
    const Tindex indices_row = grad_index / col_size;
    const Tindex param_row = indices_[indices_row];
    if (param_row < 0 || param_row >= param_rows_) {
      // Ignore indices that are out of range.
      return;
    }

    // Compute the index of var and accum.
    const Tindex param_index = param_row * col_size + (grad_index % col_size);

    // Read variables.
    T var_i = var_[param_index];
    T accum_i = accum_[param_index];
    T linear_i = linear_[param_index];
    const T grad_i = grad_[grad_index];
    const T lr_t = *lr_;
    const T l1_t = *l1_;
    const T l2_t = *l2_;
    const T lr_power_t = *lr_power_;

    const T grad_shr_i =
        has_l2_shrinkage ? grad_i + static_cast<T>(2) * (*l2_shrinkage_) * var_i
                         : grad_i;
    const T new_accum_i = accum_i + grad_i * grad_i;
    const bool lr_power_is_neg_half = lr_power_t == static_cast<T>(-0.5);
    const T pow_new_accum = lr_power_is_neg_half
                                ? Eigen::numext::sqrt(new_accum_i)
                                : Eigen::numext::pow(new_accum_i, -lr_power_t);
    const T pow_accum = lr_power_is_neg_half
                            ? Eigen::numext::sqrt(accum_i)
                            : Eigen::numext::pow(accum_i, -lr_power_t);
    T linear_change = grad_shr_i * lr_t - (pow_new_accum - pow_accum) * var_i;
    if (!multiply_linear_by_lr_) {
      linear_change /= lr_t;
    }
    linear_i += linear_change;

    T l1_mult = l1_t;
    if (multiply_linear_by_lr_) {
      l1_mult *= lr_t;
    }
    const T l1_reg_adjust =
        Eigen::numext::maxi(Eigen::numext::mini(linear_i, l1_mult), -l1_mult);
    const T x = l1_reg_adjust - linear_i;
    T y = pow_new_accum + static_cast<T>(2) * l2_t * lr_t;
    if (!multiply_linear_by_lr_) {
      y /= lr_t;
    }
    var_i = x / y;
    accum_i = new_accum_i;

    // Write update back to variables.
    var_[param_index] = var_i;
    accum_[param_index] = accum_i;
    linear_[param_index] = linear_i;
  }

 private:
  T* var_;
  T* accum_;
  T* linear_;
  const T* lr_;
  const T* l1_;
  const T* l2_;
  const T* l2_shrinkage_;
  const T* lr_power_;
  const T* grad_;
  const Tindex* indices_;
  Tindex param_rows_;
  Tindex updates_size_;
  Tindex indices_size_;
  bool multiply_linear_by_lr_;
};

template <typename T, typename Tindex, bool has_l2_shrinkage>
struct SparseApplyFtrl<GPUDevice, T, Tindex, has_l2_shrinkage> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                  typename TTypes<T>::Matrix accum,
                  typename TTypes<T>::Matrix linear,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar l2_shrinkage,
                  typename TTypes<T>::ConstScalar lr_power,
                  typename TTypes<T>::ConstMatrix grad,
                  typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,
                  bool multiply_linear_by_lr) {
    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();

    auto* dpcpp_stream = d.stream();
    dpcpp_stream->submit([&](sycl::handler& cgh) {
      SparseApplyFtrlKernel<T, Tindex, has_l2_shrinkage> task(
          var.data(), accum.data(), linear.data(), lr.data(), l1.data(),
          l2.data(), l2_shrinkage.data(), lr_power.data(), grad.data(),
          indices.data(), first_dim_size, grad_size, indices_size,
          multiply_linear_by_lr);
      cgh.parallel_for<SparseApplyFtrlKernel<T, Tindex, has_l2_shrinkage>>(
          sycl::range<1>(grad_size), task);
    });
  }
};

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T)                      \
  template struct SparseApplyFtrl<GPUDevice, T, int32,         \
                                  /*has_l2_shrinkage=*/false>; \
  template struct SparseApplyFtrl<GPUDevice, T, int64,         \
                                  /*has_l2_shrinkage=*/false>; \
  template struct SparseApplyFtrl<GPUDevice, T, int32,         \
                                  /*has_l2_shrinkage=*/true>;  \
  template struct SparseApplyFtrl<GPUDevice, T, int64,         \
                                  /*has_l2_shrinkage=*/true>

EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);

#undef EXPLICITLY_INSTANTIATE_FUNCTOR

}  // namespace functor

template <typename T, bool has_l2_shrinkage>
class ApplyFtrlOp : public OpKernel {
 public:
  explicit ApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<GPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor linear;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<GPUDevice, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &linear));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, linear.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));

    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(linear.shape()),
        errors::InvalidArgument("var and linear do not have the same shape",
                                var.shape().DebugString(), " ",
                                linear.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l1.shape()),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l2.shape()),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 8 : 7;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_power.shape()),
                errors::InvalidArgument("lr_power is not a"
                                        " non-positive scalar: ",
                                        lr_power.shape().DebugString()));

    auto device = ctx->eigen_gpu_device();
    if (has_l2_shrinkage) {
      const Tensor& l2_shrinkage = ctx->input(7);
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(l2_shrinkage.shape()),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage.shape().DebugString()));
      functor::ApplyFtrlV2DPCPP<T>()(
          device, var.flat<T>().data(), accum.flat<T>().data(),
          linear.flat<T>().data(), grad.flat<T>().data(), lr.scalar<T>().data(),
          l1.scalar<T>().data(), l2.scalar<T>().data(),
          l2_shrinkage.scalar<T>().data(), lr_power.scalar<T>().data(), ctx,
          grad.NumElements());
    } else {
      functor::ApplyFtrlDPCPP<T>()(
          device, var.flat<T>().data(), accum.flat<T>().data(),
          linear.flat<T>().data(), grad.flat<T>().data(), lr.scalar<T>().data(),
          l1.scalar<T>().data(), l2.scalar<T>().data(),
          lr_power.scalar<T>().data(), ctx, grad.NumElements());
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename Device, typename T, typename Tindex, bool has_l2_shrinkage>
class SparseApplyFtrlOp : public OpKernel {
 public:
  explicit SparseApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("multiply_linear_by_lr", &multiply_linear_by_lr_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor linear;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &linear));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(ctx, linear.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(linear.shape()),
        errors::InvalidArgument("var and linear do not have the same shape",
                                var.shape().DebugString(), " ",
                                linear.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    // Note: The range checks on lr, l1, l2, and lr_power below are disabled
    // for non-CPU devices because their values cannot be accessed directly from
    // the host. The GPU kernel will not crash if these conditions are not met,
    // it will simply produce a bogus answer (possibly inf/nan).
    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsScalar(lr.shape()) &&
            (!std::is_same<Device, CPUDevice>::value ||
             lr.scalar<T>()() > static_cast<T>(0) ||
             (multiply_linear_by_lr_ && lr.scalar<T>()() >= static_cast<T>(0))),
        errors::InvalidArgument("lr is not a positive scalar (or zero if "
                                "multiply_linear_by_lr is set): ",
                                lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l1.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l2.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 9 : 8;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     lr_power.scalar<T>()() <= static_cast<T>(0)),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));
    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tensor* l2_shrinkage;
    if (has_l2_shrinkage) {
      l2_shrinkage = &ctx->input(8);
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(l2_shrinkage->shape()) &&
              (!std::is_same<Device, CPUDevice>::value ||
               l2_shrinkage->scalar<T>()() >= static_cast<T>(0)),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage->shape().DebugString()));
    }

    const Device& device = ctx->template eigen_device<Device>();
    auto indices_vec = indices.vec<Tindex>();
    functor::SparseApplyFtrl<Device, T, Tindex, has_l2_shrinkage>()(
        device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
        linear.flat_outer_dims<T>(), lr.scalar<T>(), l1.scalar<T>(),
        l2.scalar<T>(),
        // Note: Passing l2 as a placeholder when not has_l2_shrinkage
        // (it will not be used).
        has_l2_shrinkage ? l2_shrinkage->scalar<T>() : l2.scalar<T>(),
        lr_power.scalar<T>(), grad.flat_outer_dims<T>(), indices_vec, inner_dim,
        multiply_linear_by_lr_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool multiply_linear_by_lr_;
};

#define REGISTER_DPCPP_KERNELS(T)                                      \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyFtrl").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      ApplyFtrlOp<T, /*has_l2_shrinkage=*/false>);                     \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyFtrl")                    \
                              .HostMemory("var")                       \
                              .HostMemory("accum")                     \
                              .HostMemory("linear")                    \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<T>("T"),                 \
                          ApplyFtrlOp<T, /*has_l2_shrinkage=*/false>); \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyFtrlV2").Device(DEVICE_GPU).TypeConstraint<T>("T"),   \
      ApplyFtrlOp<T, /*has_l2_shrinkage=*/true>);                      \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyFtrlV2")                  \
                              .HostMemory("var")                       \
                              .HostMemory("accum")                     \
                              .HostMemory("linear")                    \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<T>("T"),                 \
                          ApplyFtrlOp<T, /*has_l2_shrinkage=*/true>);

TF_CALL_half(REGISTER_DPCPP_KERNELS);
TF_CALL_float(REGISTER_DPCPP_KERNELS);
TF_CALL_bfloat16(REGISTER_DPCPP_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_DPCPP_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_DPCPP_KERNELS

#define REGISTER_DPCPP_KERNELS(D, T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseApplyFtrl")                                                 \
          .Device(DEVICE_##D)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      SparseApplyFtrlOp<D##Device, T, Tindices, /*has_l2_shrinkage=*/false>); \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ResourceSparseApplyFtrl")                                         \
          .Device(DEVICE_##D)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      SparseApplyFtrlOp<D##Device, T, Tindices, /*has_l2_shrinkage=*/false>); \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseApplyFtrlV2")                                               \
          .Device(DEVICE_##D)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      SparseApplyFtrlOp<D##Device, T, Tindices, /*has_l2_shrinkage=*/true>);  \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ResourceSparseApplyFtrlV2")                                       \
          .Device(DEVICE_##D)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      SparseApplyFtrlOp<D##Device, T, Tindices, /*has_l2_shrinkage=*/true>);

REGISTER_DPCPP_KERNELS(GPU, Eigen::half, int32);
REGISTER_DPCPP_KERNELS(GPU, Eigen::half, int64);
REGISTER_DPCPP_KERNELS(GPU, Eigen::bfloat16, int32);
REGISTER_DPCPP_KERNELS(GPU, Eigen::bfloat16, int64);
REGISTER_DPCPP_KERNELS(GPU, float, int32);
REGISTER_DPCPP_KERNELS(GPU, float, int64);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_DPCPP_KERNELS(GPU, double, int32);
REGISTER_DPCPP_KERNELS(GPU, double, int64);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_DPCPP_KERNELS

}  // namespace itex
