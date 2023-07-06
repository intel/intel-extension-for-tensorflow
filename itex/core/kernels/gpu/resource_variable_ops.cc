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

#include "itex/core/kernels/gpu/dense_update_functor.h"
#include "itex/core/kernels/gpu/gather_functor.h"
#include "itex/core/kernels/gpu/gather_nd_op.h"
#include "itex/core/kernels/gpu/scatter_functor.h"
#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "itex/core/utils/util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AssignVariableOp : public OpKernel {
 public:
  explicit AssignVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    if (!c->GetAttr("_grappler_relax_allocator_constraints",
                    &relax_constraints_)
             .ok()) {
      relax_constraints_ = false;
    }
    if (c->HasAttr("validate_shape")) {
      OP_REQUIRES_OK(c, c->GetAttr("validate_shape", &validate_shape_));
    }
  }

  void Compute(OpKernelContext* context) override {
    const int input_index = 0;
    const int value_index = 1;
    OP_REQUIRES_OK(
        context, AssignVariableHelper<Device, T>(context, input_index,
                                                 value_index, validate_shape_));
  }

 private:
  DataType dtype_;
  bool relax_constraints_;
  bool validate_shape_ = false;
};

#define REGISTER_GPU_KERNELS(type)                           \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableOp")           \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("resource"),       \
                          AssignVariableOp<GPUDevice, type>);

TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_int32(REGISTER_GPU_KERNELS);
TF_CALL_half(REGISTER_GPU_KERNELS);
TF_CALL_bfloat16(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_bool(REGISTER_GPU_KERNELS);
TF_CALL_complex64(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
TF_CALL_complex128(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

template <typename Device, typename T, DenseUpdateType Op>
class AssignUpdateVariableOp : public OpKernel {
 public:
  explicit AssignUpdateVariableOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    const int input_index = 0;
    const int value_index = 1;

    // TODO(itex): why TF_AssignUpdateVariable will cause accuracy issue
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        context, /* do_lock */ true, /* sparse */ false, {input_index});

    Tensor params;
    OP_REQUIRES_OK(context, GetInputTensorFromVariable<Device, T>(
                                context, 0, /* lock_held */ true,
                                /* sparse */ false, &params));
    const Tensor& updates = context->input(value_index);

    OP_REQUIRES(context, params.shape().IsSameSize(updates.shape()),
                errors::InvalidArgument(
                    "Cannot update variable with shape ",
                    params.shape().DebugString(), " using a Tensor with shape ",
                    updates.shape().DebugString(), ", shapes must be equal."));

    functor::DenseUpdate<Device, T, Op> update_functor;
    update_functor(context->eigen_gpu_device(), params.flat<T>(),
                   updates.flat<T>());
  }
};

#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("AssignAddVariableOp")                    \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("resource")                    \
                              .TypeConstraint<type>("dtype"),            \
                          AssignUpdateVariableOp<GPUDevice, type, ADD>); \
  REGISTER_KERNEL_BUILDER(Name("AssignSubVariableOp")                    \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("resource")                    \
                              .TypeConstraint<type>("dtype"),            \
                          AssignUpdateVariableOp<GPUDevice, type, SUB>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int32(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

template <typename Device, typename T, typename Index>
class ResourceGatherOp : public OpKernel {
 private:
  int32 batch_dims_ = 0;

  // Add the batch offset derrived from params to each batch of indices.
  // Example: batch_dims = 1, indices = [[0, 1, 2], [0, 1, 2]]
  // If indexing into a params dimension of size 4, then the indices will become
  // [0, 1, 2, 4, 5, 6]
  void AddBatchOffsets(Tensor* indices, const Tensor& params) {
    int64 batch_size = 1;  // The size of all batch dimensions.
    for (int idx = 0; idx < batch_dims_; ++idx) {
      batch_size *= params.dim_size(idx);
    }

    auto indices_flat = indices->flat<Index>();
    int64 const index_inner_size = indices->NumElements() / batch_size;
    int64 const batch_offset = params.dim_size(batch_dims_);
    for (int64 batch_idx = 0, dest_idx = 0; batch_idx < batch_size;
         ++batch_idx) {
      for (int64 idx = 0; idx < index_inner_size; ++idx) {
        indices_flat(dest_idx++) += batch_offset * batch_idx;
      }
    }
  }

 public:
  explicit ResourceGatherOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
  }

  void Compute(OpKernelContext* c) override {
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        c, /* do_lock */ true, /* sparse */ true, {0});

    Tensor params;
    OP_REQUIRES_OK(
        c, GetInputTensorFromVariable<Device, T>(
               c, 0, /* lock_held unused */ true, /* sparse */ true, &params));

    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // Check that we have enough index space
    const int64 N = indices.NumElements();
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));
    // Check that we have enough index space
    int64 gather_dim_size = 1;
    for (int idx = 0; idx <= batch_dims_; ++idx) {
      gather_dim_size *= params.dim_size(idx);
    }

    // The result shape is params.shape[:axis] + indices.shape[batch_dims:] +
    // params.shape[axis + 1:].
    TensorShape result_shape;
    int64 outer_size = 1;
    int64 inner_size = 1;
    for (int i = 0; i < batch_dims_; i++) {
      result_shape.AddDim(params.dim_size(i));
      outer_size *= params.dim_size(i);
    }
    for (int i = batch_dims_; i < indices.dims(); ++i) {
      result_shape.AddDim(indices.dim_size(i));
    }
    for (int i = batch_dims_ + 1; i < params.dims(); i++) {
      result_shape.AddDim(params.dim_size(i));
      inner_size *= params.dim_size(i);
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N == 0) return;

    if (batch_dims_ > 0) {
      // TODO(itex): Switch to transpose / gather with axis=0 / transpose
      // on GPU, to avoid launching a lot of small kernels.

      // To avoid copying params (by transposing), run gather for each batch.
      int64 batch_size = 1;
      for (int i = 0; i < batch_dims_; ++i) {
        batch_size *= params.dim_size(i);
      }
      outer_size /= batch_size;
      auto batched_params =
          params.shaped<T, 2>({batch_size, params.NumElements() / batch_size});
      auto batched_indices =
          indices.shaped<Index, 2>({batch_size, N / batch_size});
      auto batched_out =
          out->shaped<T, 2>({batch_size, out->NumElements() / batch_size});

      // TODO(itex): Investigate the best performance, when the number of
      // batches is large, between parallel vs sequential runs.
      for (int64 batch = 0; batch < batch_size; ++batch) {
        auto params_flat = typename TTypes<T, 3>::ConstTensor(
            &batched_params(batch, 0), static_cast<Index>(outer_size),
            static_cast<Index>(gather_dim_size),
            static_cast<Index>(inner_size));
        auto indices_flat = typename TTypes<Index>::ConstFlat(
            &batched_indices(batch, 0), batched_indices.dimension(1));
        auto out_flat = typename TTypes<T, 3>::Tensor(
            &batched_out(batch, 0), static_cast<Index>(outer_size),
            static_cast<Index>(N), static_cast<Index>(inner_size));

        functor::GatherFunctor<Device, T, Index> functor;
        const int64 bad_i = functor(c, params_flat, indices_flat, out_flat);

        OP_REQUIRES(
            c, bad_i < 0,
            errors::InvalidArgument(
                "indices", SliceDebugString(indices.shape(), bad_i), " = ",
                indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
      }
    } else {
      auto params_flat = const_cast<const Tensor&>(params).shaped<T, 3>(
          {outer_size, gather_dim_size, inner_size});
      auto indices_flat = indices.flat<Index>();
      auto out_flat = out->shaped<T, 3>({outer_size, N, inner_size});

      functor::GatherFunctor<Device, T, Index> functor;
      const int64 bad_i = functor(c, params_flat, indices_flat, out_flat);

      OP_REQUIRES(
          c, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
    }
  }
};

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                       \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("resource")                  \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          ResourceGatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64)

#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_int32(REGISTER_GATHER_GPU);
TF_CALL_int64(REGISTER_GATHER_GPU);
TF_CALL_float(REGISTER_GATHER_GPU);
TF_CALL_half(REGISTER_GATHER_GPU);
TF_CALL_bfloat16(REGISTER_GATHER_GPU);
TF_CALL_bool(REGISTER_GATHER_GPU);
TF_CALL_complex64(REGISTER_GATHER_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GATHER_GPU);
TF_CALL_complex128(REGISTER_GATHER_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_GATHER_GPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

template <typename Device, typename T, typename Index>
class ResourceGatherNdOp : public OpKernel {
 public:
  explicit ResourceGatherNdOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        c, /* do_lock */ true, /* sparse */ true, {0});
    Tensor params;
    OP_REQUIRES_OK(
        c, GetInputTensorFromVariable<Device, T>(
               c, 0, /* lock_held unused */ true, /* sparse */ true, &params));

    const Tensor& indices = c->input(1);

    Tensor out;
    OP_REQUIRES_OK(
        c, functor::DoGatherNd<Device, T, Index>(c, params, indices, &out));
    c->set_output(0, out);
  }
};

#define REGISTER_GATHER_ND_FULL(dev, type, index_type)                 \
  REGISTER_KERNEL_BUILDER(Name("ResourceGatherNd")                     \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("resource")                  \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          ResourceGatherNdOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ND_ALL_INDICES(dev, type) \
  REGISTER_GATHER_ND_FULL(dev, type, int32);      \
  REGISTER_GATHER_ND_FULL(dev, type, int64)

// Registers GPU kernels.
#define REGISTER_GATHER_ND_GPU(type) REGISTER_GATHER_ND_ALL_INDICES(GPU, type)

TF_CALL_int32(REGISTER_GATHER_ND_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_ND_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GATHER_ND_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_GATHER_ND_GPU
#undef REGISTER_GATHER_ND_ALL_INDICES
#undef REGISTER_GATHER_ND_FULL

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
class ResourceScatterUpdateOp : public OpKernel {
 public:
  explicit ResourceScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        c, /* do_lock */ true, /* sparse */ true, {0});

    Tensor params;
    OP_REQUIRES_OK(
        c, GetInputTensorFromVariable<Device, T>(
               c, 0, /* lock_held unused */ true, /* sparse */ true, &params));

    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(
        c, N_big <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("indices has too many elements for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", N_big, " > ",
                                std::numeric_limits<Index>::max()));
    const Index N = static_cast<Index>(N_big);
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    if (N > 0) {
      auto indices_flat = indices.flat<Index>();
      auto params_flat = params.flat_outer_dims<T>();

      Tensor out_fp32;
      OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<float>::value,
                                         params.shape(), &out_fp32));
      auto out_fp32_flat = out_fp32.flat_outer_dims<float>();

      if (TensorShapeUtils::IsScalar(updates.shape())) {
        const auto update = updates.scalar<T>();

        functor::ScatterScalarFunctor<Device, T, Index, op> functor;
        const Index bad_i = functor(c, c->eigen_device<Device>(), params_flat,
                                    update, indices_flat, out_fp32_flat);
        OP_REQUIRES(c, bad_i < 0,
                    errors::InvalidArgument(
                        "indices", SliceDebugString(indices.shape(), bad_i),
                        " = ", indices_flat(bad_i), " is not in [0, ",
                        params.dim_size(0), ")"));
      } else {
        int64 num_updates = updates.NumElements();
        OP_REQUIRES(c, num_updates % N == 0,
                    errors::InvalidArgument(
                        "shape of indices (", indices.shape().DebugString(),
                        ") is not compatible with the shape of updates (",
                        updates.shape().DebugString(), ")"));
        auto updates_flat = updates.shaped<T, 2>({N, num_updates / N});

        functor::ScatterFunctor<Device, T, Index, op> functor;
        const Index bad_i =
            functor(c, c->template eigen_device<Device>(), params_flat,
                    updates_flat, indices_flat, out_fp32_flat);
        OP_REQUIRES(c, bad_i < 0,
                    errors::InvalidArgument(
                        "indices", SliceDebugString(indices.shape(), bad_i),
                        " = ", indices_flat(bad_i), " is not in [0, ",
                        params.dim_size(0), ")"));
      }
    }
  }
};

#define REGISTER_SCATTER_KERNEL_INDEX(type, index_type, dev, name, op) \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(name)                                                       \
          .Device(DEVICE_##dev)                                        \
          .HostMemory("resource")                                      \
          .TypeConstraint<type>("dtype")                               \
          .TypeConstraint<index_type>("Tindices"),                     \
      ResourceScatterUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_KERNEL_INDEX(type, int64, dev, name, op);

#define REGISTER_SCATTER_ARITHMETIC(type, dev)                \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterAdd",    \
                          scatter_op::UpdateOp::ADD);         \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterSub",    \
                          scatter_op::UpdateOp::SUB);         \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterMul",    \
                          scatter_op::UpdateOp::MUL);         \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterDiv",    \
                          scatter_op::UpdateOp::DIV);         \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterUpdate", \
                          scatter_op::UpdateOp::ASSIGN);
#define REGISTER_SCATTER_MINMAX(type, dev)                 \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterMin", \
                          scatter_op::UpdateOp::MIN);      \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterMax", \
                          scatter_op::UpdateOp::MAX);

#define REGISTER_SCATTER_ARITHMETIC_GPU(type) \
  REGISTER_SCATTER_ARITHMETIC(type, GPU);

TF_CALL_int32(REGISTER_SCATTER_ARITHMETIC_GPU);
TF_CALL_int64(REGISTER_SCATTER_ARITHMETIC_GPU);
TF_CALL_float(REGISTER_SCATTER_ARITHMETIC_GPU);
TF_CALL_bfloat16(REGISTER_SCATTER_ARITHMETIC_GPU);
TF_CALL_half(REGISTER_SCATTER_ARITHMETIC_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SCATTER_ARITHMETIC_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#define REGISTER_SCATTER_MINMAX_GPU(type) REGISTER_SCATTER_MINMAX(type, GPU);

TF_CALL_int32(REGISTER_SCATTER_MINMAX_GPU);
TF_CALL_float(REGISTER_SCATTER_MINMAX_GPU);
TF_CALL_bfloat16(REGISTER_SCATTER_MINMAX_GPU);
TF_CALL_half(REGISTER_SCATTER_MINMAX_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SCATTER_MINMAX_GPU);
#endif  // ITEX_ENABLE_DOUBLE

REGISTER_KERNEL_BUILDER(Name("ResourceScatterUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .TypeConstraint<bool>("dtype")
                            .TypeConstraint<int32>("Tindices"),
                        ResourceScatterUpdateOp<GPUDevice, bool, int32,
                                                scatter_op::UpdateOp::ASSIGN>);
REGISTER_KERNEL_BUILDER(Name("ResourceScatterUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .TypeConstraint<bool>("dtype")
                            .TypeConstraint<int64>("Tindices"),
                        ResourceScatterUpdateOp<GPUDevice, bool, int64,
                                                scatter_op::UpdateOp::ASSIGN>)

#undef REGISTER_SCATTER_ARITHMETIC
#undef REGISTER_SCATTER_MINMAX
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX

}  // namespace itex
