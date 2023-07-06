/* Copyright (c) 2023 Intel Corporation
Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#include <sycl/sycl.hpp>

#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/kernels/gpu/training_ops.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"

namespace itex {
namespace functor {
template <typename T, typename Tindex>
struct SparseApplyAdadeltaKernel {
  SparseApplyAdadeltaKernel(T* var, T* accum, T* accum_update, const T* lr,
                            const T* rho, const T* epsilon, const T* grad,
                            const Tindex* indices, Tindex param_rows,
                            Tindex updates_size, Tindex indices_size)
      : var_(var),
        accum_(accum),
        accum_update_(accum_update),
        lr_(lr),
        rho_(rho),
        epsilon_(epsilon),
        grad_(grad),
        indices_(indices),
        param_rows_(param_rows),
        updates_size_(updates_size),
        indices_size_(indices_size) {}

  void operator()(sycl::item<1> item) const {
    const T lr_t = *lr_;
    const T rho_t = *rho_;
    const T epsilon_t = *epsilon_;

    int32_t grad_index = item.get_linear_id();
    Tindex col_size = updates_size_ / indices_size_;
    Tindex indices_row = grad_index / col_size;
    Tindex param_row = indices_[indices_row];

    if (param_row < 0 || param_row >= param_rows_ ||
        grad_index >= param_rows_) {
      // Ignore indices that are out of range.
      return;
    }
    // Compute the index of var and accum.
    Tindex param_index = param_row * col_size + (grad_index % col_size);

    // Read variables.
    T var_i = var_[param_index];
    T accum_i = accum_[param_index];
    T accum_update_i = accum_update_[param_index];
    T grad_i = grad_[grad_index];

    // Variable update computation.
    accum_i = accum_i * rho_t + grad_i * grad_i * (T(1.0) - rho_t);
    T update = Eigen::numext::sqrt(accum_update_i + epsilon_t) * grad_i /
               Eigen::numext::sqrt(accum_i + epsilon_t);
    var_i = var_i - update * lr_t;
    accum_update_i =
        accum_update_i * rho_t + update * update * (T(1.0) - rho_t);

    // Write update back to variables.
    var_[param_index] = var_i;
    accum_[param_index] = accum_i;
    accum_update_[param_index] = accum_update_i;
  }

 private:
  T* var_;
  T* accum_;
  T* accum_update_;
  const T* lr_;
  const T* rho_;
  const T* epsilon_;
  const T* grad_;
  const Tindex* indices_;
  Tindex param_rows_;
  Tindex updates_size_;
  Tindex indices_size_;
};

template <typename T, typename Tindex>
struct SparseApplyAdadelta<GPUDevice, T, Tindex> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                  typename TTypes<T>::Matrix accum,
                  typename TTypes<T>::Matrix accum_update,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstMatrix grad,
                  typename TTypes<Tindex>::ConstFlat indices) {
    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    auto* stream = d.stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (grad_size + total_threads - 1) / total_threads;

    auto e = stream->submit([&](sycl::handler& cgh) {
      SparseApplyAdadeltaKernel<T, Tindex> task(
          var.data(), accum.data(), accum_update.data(), lr.data(), rho.data(),
          epsilon.data(), grad.data(), indices.data(), first_dim_size,
          grad_size, indices_size);

      cgh.parallel_for<SparseApplyAdadeltaKernel<T, Tindex>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
    e.wait();
  }
};
}  // namespace functor

template <typename T, typename Device, typename Tindex>
class SparseApplyAdadeltaOp : public OpKernel {
 public:
  explicit SparseApplyAdadeltaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    DoCompute(ctx);
  }

  void DoCompute(OpKernelContext* ctx) {
    Tensor var;
    const bool sparse = true;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum_grad;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum_grad));
    Tensor accum_update;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_,
                                                   sparse, &accum_update));
    OP_REQUIRES(ctx, var.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, accum_grad.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(ctx, accum_update.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables"));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum_grad.shape()),
        errors::InvalidArgument("var and accum_grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum_grad.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(accum_update.shape()),
                errors::InvalidArgument(
                    "var and accum_update do not have the same shape",
                    var.shape().DebugString(), " ",
                    accum_update.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& rho = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    const Tensor& epsilon = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(6);
    const Tensor& indices = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      // Validate all the indices are in range
      auto indices_vec = indices.vec<Tindex>();
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        OP_REQUIRES(ctx,
                    (!std::is_same<Device, CPUDevice>::value ||
                     (index >= 0 && index < first_dim_size)),
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }

      const Device& device = ctx->template eigen_device<Device>();
      functor::SparseApplyAdadelta<Device, T, Tindex>()(
          device, var.flat_outer_dims<T>(), accum_grad.flat_outer_dims<T>(),
          accum_update.flat_outer_dims<T>(), lr.scalar<T>(), rho.scalar<T>(),
          epsilon.scalar<T>(), grad.flat_outer_dims<T>(), indices_vec);
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, D, Tindices)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdadelta")                     \
                              .Device(DEVICE_##D)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<Tindices>("Tindices"),      \
                          SparseApplyAdadeltaOp<T, D##Device, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdadelta")             \
                              .Device(DEVICE_##D)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<Tindices>("Tindices"),      \
                          SparseApplyAdadeltaOp<T, D##Device, Tindices>);
#define REGISTER_GPU_KERNELS(T)    \
  REGISTER_KERNELS(T, GPU, int32); \
  REGISTER_KERNELS(T, GPU, int64_t);

REGISTER_GPU_KERNELS(Eigen::half);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);
REGISTER_GPU_KERNELS(Eigen::bfloat16);
// REGISTER_GPU_KERNELS(complex64);
// REGISTER_GPU_KERNELS(complex128);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace itex
