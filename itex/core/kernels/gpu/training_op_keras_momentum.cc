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
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace functor {
template <typename T, typename Tindex>
struct SparseApplyKerasMomentumKernel {
  SparseApplyKerasMomentumKernel(T* var, T* accum, const T* lr, const T* grad,
                                 const Tindex* indices, const T* momentum,
                                 Tindex param_rows, Tindex updates_size,
                                 Tindex indices_size, bool use_nesterov)
      : var_(var),
        accum_(accum),
        lr_(lr),
        grad_(grad),
        indices_(indices),
        momentum_(momentum),
        param_rows_(param_rows),
        updates_size_(updates_size),
        indices_size_(indices_size),
        use_nesterov_(use_nesterov) {}
  void operator()(sycl::item<1> item) const {
    int64_t grad_index = item.get_linear_id();
    const Tindex col_size = updates_size_ / indices_size_;
    const Tindex indices_row = grad_index / col_size;
    const Tindex param_row = indices_[indices_row];
    if (param_row < 0 || param_row >= param_rows_) {
      // Ignore indices that are out of range.
      return;
    }

    const Tindex param_index = param_row * col_size + (grad_index % col_size);

    T var_i = var_[param_index];
    T accum_i = accum_[param_index];
    const T grad_i = grad_[grad_index];
    const T momentum_t = *momentum_;
    const T lr_t = *lr_;

    accum_i = momentum_t * accum_i - lr_t * grad_i;
    if (use_nesterov_) {
      var_i = var_i + (momentum_t * accum_i - lr_t * grad_i);
    } else {
      var_i = var_i + accum_i;
    }

    var_[param_index] = var_i;
    accum_[param_index] = accum_i;
  }

 private:
  T* var_;
  T* accum_;
  const T* lr_;
  const T* grad_;
  const Tindex* indices_;
  const T* momentum_;
  Tindex param_rows_;
  Tindex updates_size_;
  Tindex indices_size_;
  bool use_nesterov_;
};

template <typename T, typename Tindex>
struct SparseApplyKerasMomentumITEX_GPU {
  void operator()(const GPUDevice& d, T* var, T* accum, const T* lr,
                  const T* grad, const Tindex* indices, const T* momentum,
                  OpKernelContext* ctx, bool use_nesterov,
                  const Tindex first_dim_size, const Tindex grad_size,
                  const Tindex indices_size) {
    auto* stream = ctx->eigen_gpu_device().stream();

    stream->submit([&](sycl::handler& cgh) {
      SparseApplyKerasMomentumKernel<T, Tindex> task(
          var, accum, lr, grad, indices, momentum, first_dim_size, grad_size,
          indices_size, use_nesterov);

      cgh.parallel_for<SparseApplyKerasMomentumKernel<T, Tindex>>(
          sycl::range<1>(grad_size), task);
    });
  }
};
}  // namespace functor

template <typename T, typename Device, typename Tindex>
class SparseApplyKerasMomentumOp : public OpKernel {
 public:
  explicit SparseApplyKerasMomentumOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
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
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(ctx, accum.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: "));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
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

    const Tensor& momentum = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    auto indices_flat = indices.flat<Tindex>();

    const Tindex first_dim_size = var.flat_outer_dims<T>().dimension(0);
    const Tindex grad_size = grad.flat_outer_dims<T>().size();
    const Tindex indices_size = indices_flat.size();

    functor::SparseApplyKerasMomentumITEX_GPU<T, Tindex>()(
        device, var.flat_outer_dims<T>().data(),
        accum.flat_outer_dims<T>().data(), lr.scalar<T>().data(),
        grad.flat_outer_dims<T>().data(), indices_flat.data(),
        momentum.scalar<T>().data(), ctx, use_nesterov_, first_dim_size,
        grad_size, indices_size);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(T, D, Tindices)                             \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyKerasMomentum")   \
                              .Device(DEVICE_##D)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyKerasMomentumOp<T, D##Device, Tindices>);

#define REGISTER_ITEX_GPU_KERNELS(T) \
  REGISTER_KERNELS(T, GPU, int32);   \
  REGISTER_KERNELS(T, GPU, int64);

TF_CALL_complex64(REGISTER_ITEX_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_FLOAT_TYPES(REGISTER_ITEX_GPU_KERNELS);
TF_CALL_complex128(REGISTER_ITEX_GPU_KERNELS);
#else
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ITEX_GPU_KERNELS);
#endif

#undef REGISTER_ITEX_GPU_KERNELS
#undef REGISTER_KERNELS
}  // namespace itex
