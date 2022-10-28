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
struct ApplyAdagrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    var.device(d) -= lr.reshape(single).broadcast(bcast) * grad * accum.rsqrt();
  }
};
}  // namespace functor

template <typename Device, typename T>
class ApplyAdagradOp : public OpKernel {
 public:
  explicit ApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
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
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
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

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdagrad<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                       lr.scalar<T>(), grad.flat<T>(),
                                       update_slots_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_KERNELS(D, T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ApplyAdagrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdagradOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdagrad")                \
                              .HostMemory("var")                      \
                              .HostMemory("accum")                    \
                              .Device(DEVICE_##D)                     \
                              .TypeConstraint<T>("T"),                \
                          ApplyAdagradOp<D##Device, T>);

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, Eigen::bfloat16);
REGISTER_KERNELS(GPU, complex64);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNELS(GPU, double);
REGISTER_KERNELS(GPU, complex128);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_KERNELS

namespace functor {
template <typename T, typename Tindex, bool has_epsilon>
struct SparseApplyAdagradKernel {
  SparseApplyAdagradKernel(T* var, T* accum, const T* lr, const T* epsilon,
                           const T* grad, const Tindex* indices,
                           Tindex param_rows, Tindex updates_size,
                           Tindex indices_size, bool update_slots)
      : var_(var),
        accum_(accum),
        lr_(lr),
        epsilon_(epsilon),
        grad_(grad),
        indices_(indices),
        param_rows_(param_rows),
        updates_size_(updates_size),
        indices_size_(indices_size),
        update_slots_(update_slots) {}

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
    const T epsilon_t = *epsilon_;

    if (update_slots_) {
      accum_i += grad_i * grad_i;
    }
    if (has_epsilon) {
      var_i -= lr_t * grad_i / (Eigen::numext::sqrt(accum_i) + epsilon_t);
    } else {
      var_i -= lr_t * grad_i * rsqrt(accum_i);
    }

    // Write update back to variables.
    var_[param_index] = var_i;
    accum_[param_index] = accum_i;
  }

 private:
  T* var_;
  T* accum_;
  const T* lr_;
  const T* epsilon_;
  const T* grad_;
  const Tindex* indices_;
  Tindex param_rows_;
  Tindex updates_size_;
  Tindex indices_size_;
  bool update_slots_;
};

template <typename T, typename Tindex, bool has_epsilon>
struct SparseApplyAdagrad<GPUDevice, T, Tindex, has_epsilon> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                  typename TTypes<T>::Matrix accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstMatrix grad,
                  typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,
                  bool update_slots) {
    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    if (grad_size == 0) return;

    auto* dpcpp_stream = d.stream();
    dpcpp_stream->submit([&](sycl::handler& cgh) {
      SparseApplyAdagradKernel<T, Tindex, has_epsilon> task(
          var.data(), accum.data(), lr.data(), epsilon.data(), grad.data(),
          indices.data(), first_dim_size, grad_size, indices_size,
          update_slots);

      cgh.parallel_for<SparseApplyAdagradKernel<T, Tindex, has_epsilon>>(
          sycl::range<1>(grad_size), task);
    });
  }
};
}  // namespace functor

template <typename Device, typename T, typename Tindex>
class SparseApplyAdagradOp : public OpKernel {
 public:
  explicit SparseApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
  }

  void Compute(OpKernelContext* ctx) override {
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
                    "Attempting to use uninitialized first variable"));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized second variable"));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
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
    functor::SparseApplyAdagrad<Device, T, Tindex,
                                /*has_epsilon = */ false>()(
        device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
        // Note: Passing lr as a placeholder for unused epsilon.
        lr.scalar<T>(), lr.scalar<T>(), grad.flat_outer_dims<T>(),
        indices.vec<Tindex>(), inner_dim, update_slots_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_KERNELS(D, T, Tindices)                                 \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagrad")                     \
                              .Device(DEVICE_##D)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<Tindices>("Tindices"),     \
                          SparseApplyAdagradOp<D##Device, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdagrad")             \
                              .Device(DEVICE_##D)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<Tindices>("Tindices"),     \
                          SparseApplyAdagradOp<D##Device, T, Tindices>);

#define REGISTER_KERNELS_ALL(D, T) \
  REGISTER_KERNELS(D, T, int32);   \
  REGISTER_KERNELS(D, T, int64);

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                        \
  template struct SparseApplyAdagrad<GPUDevice, T, Tindex, \
                                     /*has_epsilon=*/false>;
#define DECLARE_GPU_SPEC_ALL(T) \
  DECLARE_GPU_SPEC(T, int32);   \
  DECLARE_GPU_SPEC(T, int64);

DECLARE_GPU_SPEC_ALL(Eigen::half);
DECLARE_GPU_SPEC_ALL(Eigen::bfloat16);
DECLARE_GPU_SPEC_ALL(float);
#ifdef ITEX_ENABLE_DOUBLE
DECLARE_GPU_SPEC_ALL(double);
#endif  // ITEX_ENABLE_DOUBLE
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_ALL
}  // namespace functor

REGISTER_KERNELS_ALL(GPU, Eigen::half);
REGISTER_KERNELS_ALL(GPU, Eigen::bfloat16);
REGISTER_KERNELS_ALL(GPU, float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNELS_ALL(GPU, double);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNELS
#undef REGISTER_KERNELS_ALL

template <typename Device, typename T, typename Tindex>
class SparseApplyAdagradV2Op : public OpKernel {
 public:
  explicit SparseApplyAdagradV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
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
                    "Attempting to use uninitialized first variables"));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized second variables"));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& epsilon = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);
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
    functor::SparseApplyAdagrad<Device, T, Tindex,
                                /*has_epsilon = */ true>()(
        device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
        lr.scalar<T>(), epsilon.scalar<T>(), grad.flat_outer_dims<T>(),
        indices.vec<Tindex>(), inner_dim, update_slots_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_KERNELS(D, T, Tindices)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagradV2")                     \
                              .Device(DEVICE_##D)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices"),       \
                          SparseApplyAdagradV2Op<D##Device, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdagradV2")             \
                              .Device(DEVICE_##D)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices"),       \
                          SparseApplyAdagradV2Op<D##Device, T, Tindices>);

#define REGISTER_KERNELS_ALL(D, T) \
  REGISTER_KERNELS(D, T, int32);   \
  REGISTER_KERNELS(D, T, int64);

namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                        \
  template struct SparseApplyAdagrad<GPUDevice, T, Tindex, \
                                     /*has_epsilon=*/true>;
#define DECLARE_GPU_SPEC_ALL(T) \
  DECLARE_GPU_SPEC(T, int32);   \
  DECLARE_GPU_SPEC(T, int64);

DECLARE_GPU_SPEC_ALL(Eigen::half);
DECLARE_GPU_SPEC_ALL(Eigen::bfloat16);
DECLARE_GPU_SPEC_ALL(float);
#ifdef ITEX_ENABLE_DOUBLE
DECLARE_GPU_SPEC_ALL(double);
#endif  // ITEX_ENABLE_DOUBLE
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_ALL
}  // namespace functor

REGISTER_KERNELS_ALL(GPU, Eigen::half);
REGISTER_KERNELS_ALL(GPU, Eigen::bfloat16);
REGISTER_KERNELS_ALL(GPU, float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNELS_ALL(GPU, double);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNELS
#undef REGISTER_KERNELS_ALL

}  // namespace itex
