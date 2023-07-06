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

namespace {
template <class T>
inline T sgn(const T x) {
  T zero(0);
  T one(1);
  return (x == zero ? zero : (x < zero ? -one : one));
}
template <class T>
inline T max(const T x, const T y) {
  return (x > y ? x : y);
}
}  // namespace

namespace functor {
template <typename T>
struct ComputeProximalAdagradKernel {
  ComputeProximalAdagradKernel(const T* grad_ptr, T* var_ptr, T* accum_ptr,
                               const T* lr, const T* l1, const T* l2,
                               int total_size)
      : grad_ptr_(grad_ptr),
        var_ptr_(var_ptr),
        accum_ptr_(accum_ptr),
        lr_(lr),
        l1_(l1),
        l2_(l2),
        total_size_(total_size) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;

    accum_ptr_[id] += grad_ptr_[id] * grad_ptr_[id];
    auto learning_rate = lr_[0] / Eigen::numext::sqrt(accum_ptr_[id]);
    auto prox_var = var_ptr_[id];
    prox_var -= grad_ptr_[id] * learning_rate;
    if (l1_[0] > static_cast<T>(0)) {
      var_ptr_[id] =
          sgn<T>(prox_var) *
          max<T>(Eigen::numext::abs(prox_var) - learning_rate * l1_[0],
                 static_cast<T>(0)) /
          (static_cast<T>(1) + l2_[0] * learning_rate);
    } else {
      var_ptr_[id] = prox_var / (static_cast<T>(1) + l2_[0] * learning_rate);
    }
  }

 private:
  const T* grad_ptr_;
  T* var_ptr_;
  T* accum_ptr_;
  const T* lr_;
  const T* l1_;
  const T* l2_;
  int total_size_;
};

template <typename T>
class ProximalAdagradKernel;

template <typename T>
struct ApplyProximalAdagradITEX_GPU {
  void operator()(const GPUDevice& d, T* var, T* accum, const T* lr,
                  const T* l1, const T* l2, const T* grad, OpKernelContext* ctx,
                  int elements) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;

    stream->submit([&](sycl::handler& cgh) {
      ComputeProximalAdagradKernel<T> task(grad, var, accum, lr, l1, l2,
                                           elements);

      cgh.parallel_for<ProximalAdagradKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

}  // namespace functor

template <typename T>
class ApplyProximalAdagradOp : public OpKernel {
 public:
  explicit ApplyProximalAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
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
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l1.shape()),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l2.shape()),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const Tensor& grad = ctx->input(5);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    auto device = ctx->eigen_gpu_device();
    functor::ApplyProximalAdagradITEX_GPU<T>()(
        device, var.flat<T>().data(), accum.flat<T>().data(),
        lr.scalar<T>().data(), l1.scalar<T>().data(), l2.scalar<T>().data(),
        grad.flat<T>().data(), ctx, grad.NumElements());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_ITEX_GPU_KERNELS(T)                                          \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyProximalAdagrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ApplyProximalAdagradOp<T>);                                             \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyProximalAdagrad")                \
                              .HostMemory("var")                              \
                              .HostMemory("accum")                            \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T"),                        \
                          ApplyProximalAdagradOp<T>);

TF_CALL_half(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_float(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_bfloat16(REGISTER_ITEX_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ITEX_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_ITEX_GPU_KERNELS

namespace functor {

template <typename T, typename Tindex>
struct SparseApplyProximalAdagradKernel {
  SparseApplyProximalAdagradKernel(T* var, T* accum, const T* lr, const T* l1,
                                   const T* l2, const T* grad,
                                   const Tindex* indices, Tindex param_rows,
                                   Tindex updates_size, Tindex indices_size)

      : var_(var),
        accum_(accum),
        lr_(lr),
        l1_(l1),
        l2_(l2),
        grad_(grad),
        indices_(indices),
        param_rows_(param_rows),
        updates_size_(updates_size),
        indices_size_(indices_size) {}

  void operator()(sycl::item<1> item) const {
    int32_t grad_index = item.get_linear_id();
    Tindex col_size = updates_size_ / indices_size_;
    Tindex indices_row = grad_index / col_size;
    Tindex param_row = indices_[indices_row];
    if (param_row < 0 || param_row >= param_rows_) {
      // Ignore indices that are out of range.
      return;
    }

    // Compute the index of var and accum.
    Tindex param_index = param_row * col_size + (grad_index % col_size);

    // Read variables.
    T var_i = var_[param_index];
    T accum_i = accum_[param_index];
    T grad_i = grad_[grad_index];
    const T lr_t = *lr_;
    const T l1_t = *l1_;
    const T l2_t = *l2_;

    accum_i += grad_i * grad_i;
    T learning_rate = lr_t * rsqrt(accum_i);
    // compute v = w - lr * grad.
    T prox_var_i = var_i - grad_i * learning_rate;
    // compute sign(v) * max(|v| - lr * max(l1, 0), 0)
    var_i = (prox_var_i >= T(0) ? T(1.) : T(-1.)) *
            max(abs(prox_var_i) - learning_rate * max(l1_t, T(0)), T(0)) /
            (T(1.) + l2_t * learning_rate);

    // Write update back to variables.
    var_[param_index] = var_i;
    accum_[param_index] = accum_i;
  }

 private:
  T* var_;
  T* accum_;
  const T* lr_;
  const T* l1_;
  const T* l2_;
  const T* grad_;
  const Tindex* indices_;
  Tindex param_rows_;
  Tindex updates_size_;
  Tindex indices_size_;
};

template <typename T, typename Tindex>
struct SparseApplyProximalAdagrad<GPUDevice, T, Tindex> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                  typename TTypes<T>::Matrix accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstMatrix grad,
                  typename TTypes<Tindex>::ConstVec indices, int64 inner_dim) {
    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();

    auto* ITEX_GPU_stream = d.stream();
    ITEX_GPU_stream->submit([&](sycl::handler& cgh) {
      SparseApplyProximalAdagradKernel<T, Tindex> task(
          var.data(), accum.data(), lr.data(), l1.data(), l2.data(),
          grad.data(), indices.data(), first_dim_size, grad_size, indices_size);
      cgh.parallel_for<SparseApplyProximalAdagradKernel<T, Tindex>>(
          sycl::range<1>(grad_size), task);
    });
  }
};
}  // namespace functor

template <typename Device, typename T, typename Tindex>
class SparseApplyProximalAdagradOp : public OpKernel {
 public:
  explicit SparseApplyProximalAdagradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
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
                    "Attempting to use uninitialized first variable."));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized second variable."));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     lr.scalar<T>()() > static_cast<T>(0)),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(3);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l1.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(4);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l2.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));

    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64_t inner_dim = 1;
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

    const Device& device = ctx->template eigen_device<Device>();
    functor::SparseApplyProximalAdagrad<Device, T, Tindex>()(
        device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
        lr.scalar<T>(), l1.scalar<T>(), l2.scalar<T>(),
        grad.flat_outer_dims<T>(), indices.vec<Tindex>(), inner_dim);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T, Tindices)                     \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("SparseApplyProximalAdagrad")                     \
          .Device(DEVICE_##D)                                \
          .TypeConstraint<T>("T")                            \
          .TypeConstraint<Tindices>("Tindices"),             \
      SparseApplyProximalAdagradOp<D##Device, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("ResourceSparseApplyProximalAdagrad")             \
          .Device(DEVICE_##D)                                \
          .TypeConstraint<T>("T")                            \
          .TypeConstraint<Tindices>("Tindices"),             \
      SparseApplyProximalAdagradOp<D##Device, T, Tindices>);

namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex) \
  template struct SparseApplyProximalAdagrad<GPUDevice, T, Tindex>;
DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(float, int32)
DECLARE_GPU_SPEC(Eigen::bfloat16, int32)
DECLARE_GPU_SPEC(int64, int32)
#ifdef ITEX_ENABLE_DOUBLE
DECLARE_GPU_SPEC(double, int32)
#endif  // ITEX_ENABLE_DOUBLE
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half, int32);
REGISTER_KERNELS(GPU, float, int32);
REGISTER_KERNELS(GPU, Eigen::bfloat16, int32);
REGISTER_KERNELS(GPU, int64, int32);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNELS(GPU, double, int32);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNELS

}  // namespace itex
