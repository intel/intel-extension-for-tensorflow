/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/kernels/gpu/scatter_functor.h"
#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/util.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

// Check whether updates.shape = indices.shape + params.shape[1:]
static bool ValidShapes(const Tensor& params, const Tensor& updates,
                        const Tensor& indices) {
  if (updates.dims() == 0) return true;
  if (updates.dims() != indices.dims() + params.dims() - 1) return false;
  for (int d = 0; d < indices.dims(); d++) {
    if (updates.dim_size(d) != indices.dim_size(d)) {
      return false;
    }
  }
  for (int d = 1; d < params.dims(); d++) {
    if (params.dim_size(d) != updates.dim_size(d - 1 + indices.dims())) {
      return false;
    }
  }
  return true;
}

static void DoValidationChecking(OpKernelContext* c, const Tensor& params,
                                 const Tensor& indices, const Tensor& updates) {
  OP_REQUIRES(c, params.IsInitialized(),
              errors::FailedPrecondition("Null ref for params"));
  OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
              errors::InvalidArgument("params must be at least 1-D, got shape ",
                                      params.shape().DebugString()));
  OP_REQUIRES(
      c, ValidShapes(params, updates, indices),
      errors::InvalidArgument("Must have updates.shape = indices.shape + "
                              "params.shape[1:] or updates.shape = [], got ",
                              "updates.shape ", updates.shape().DebugString(),
                              ", indices.shape ", indices.shape().DebugString(),
                              ", params.shape ", params.shape().DebugString()));
}

// TODO(itex): Remove out_fp32 memcpy when DPCPP atomic operators
// support bf16/fp16 datatype.
template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
class ScatterUpdateOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here.  Should we have the framework do some sort of
  //   integer promotion automatically, or should that be something
  //   that users have to do explicitly with a conversion operator
  //   in the graph?
  explicit ScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* c) override {
    // Hold mutex while we apply updates
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        c, /* do_lock */ use_exclusive_lock_, /* sparse unused*/ true, {0});
    DoCompute(c);
  }

 private:
  bool use_exclusive_lock_;

  void DoCompute(OpKernelContext* c) {
    Tensor params = c->mutable_input(0, use_exclusive_lock_);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    DoValidationChecking(c, params, indices, updates);
    if (!c->status().ok()) return;

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(
        c, N_big <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("indices has too many elements for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", N_big, " > ",
                                std::numeric_limits<Index>::max()));
    const Index N = static_cast<Index>(indices.NumElements());
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    // We always return the input ref.
    c->forward_ref_input_to_ref_output(0, 0);

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
        auto updates_flat =
            updates.shaped<T, 2>({N, updates.NumElements() / N});

        functor::ScatterFunctor<Device, T, Index, op> functor;
        const Index bad_i = functor(c, c->eigen_device<Device>(), params_flat,
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
  REGISTER_KERNEL_BUILDER(Name(name)                                   \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          ScatterUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_KERNEL_INDEX(type, int64, dev, name, op);

#define REGISTER_SCATTER_ARITHMETIC(type, dev)                                 \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterAdd", scatter_op::UpdateOp::ADD); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterDiv", scatter_op::UpdateOp::DIV); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterMul", scatter_op::UpdateOp::MUL); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterSub", scatter_op::UpdateOp::SUB);

#define REGISTER_SCATTER_MINMAX(type, dev)                                     \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterMin", scatter_op::UpdateOp::MIN); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterMax", scatter_op::UpdateOp::MAX);

#define REGISTER_SCATTER_UPDATE(type, dev)            \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterUpdate", \
                          scatter_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ARITHMETIC_GPU(type) \
  REGISTER_SCATTER_ARITHMETIC(type, GPU);

#define REGISTER_SCATTER_MINMAX_GPU(type) REGISTER_SCATTER_MINMAX(type, GPU);

#define REGISTER_SCATTER_UPDATE_GPU(type) REGISTER_SCATTER_UPDATE(type, GPU);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ARITHMETIC_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_MINMAX_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_UPDATE_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SCATTER_ARITHMETIC_GPU);
TF_CALL_double(REGISTER_SCATTER_MINMAX_GPU);
TF_CALL_double(REGISTER_SCATTER_UPDATE_GPU);
#endif

#undef REGISTER_SCATTER_ARITHMETIC
#undef REGISTER_SCATTER_ARITHMETIC_GPU
#undef REGISTER_SCATTER_MINMAX
#undef REGISTER_SCATTER_MINMAX_GPU
#undef REGISTER_SCATTER_UPDATE
#undef REGISTER_SCATTER_UPDATE_GPU
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX
}  // namespace itex
