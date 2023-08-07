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
#include "itex/core/kernels/gpu/variable_ops.h"

#include "itex/core/utils/errors.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
namespace itex {

typedef Eigen::GpuDevice GPUDevice;

void TemporaryVariableAllocator(TF_OpKernelContext* tf_ctx, TF_Tensor* tf_dest,
                                TF_DataType dtype, const int64_t* dims,
                                int num_dims, TF_Status* tf_status) {
  OpKernelContext ctx(tf_ctx);
  Tensor dest;
  Status s;

  gtl::ArraySlice<const int64_t> dimarray(
      reinterpret_cast<const int64_t*>(dims), num_dims);

  TensorShape shape(dimarray);

  OP_REQUIRES_OK(&ctx,
                 ctx.allocate_temp(static_cast<DataType>(dtype), shape, &dest));

  TF_TensorBitcastFrom(dest.GetTFTensor(), dtype, tf_dest, dims, num_dims,
                       tf_status);
  s = StatusFromTF_Status(tf_status);
  OP_REQUIRES_OK(&ctx, s);

  TF_SetStatus(tf_status, TF_OK, "");
}

template <typename Device, typename T>
class TemporaryVariableOp : public OpKernel {
 public:
  explicit TemporaryVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
    // Variable name defaults to op name if not specified explicitly.
    if (var_name_.empty()) var_name_ = name();
  }

  void Compute(OpKernelContext* ctx) {
    TF_Status* tf_status = TF_NewStatus();
    TF_StringView c_var_name{var_name_.c_str(), var_name_.length()};

    TF_TemporaryVariable(ctx->Get(), static_cast<TF_DataType>(dtype_),
                         shape_.dim_sizes().data(), shape_.dims(), &c_var_name,
                         &TemporaryVariableAllocator, tf_status);

    ITEX_CHECK_EQ(TF_OK, TF_GetCode(tf_status))
        << " Error while calling TF_TemporaryVariable";

    TF_DeleteStatus(tf_status);
  }

 private:
  TensorShape shape_;
  DataType dtype_;
  string var_name_;
};

class DestroyTemporaryVariableOp : public OpKernel {
 public:
  explicit DestroyTemporaryVariableOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
    OP_REQUIRES(ctx, !var_name_.empty(),
                errors::InvalidArgument("Missing var_name attribute"));
  }

  void Compute(OpKernelContext* ctx) override {
    TF_StringView c_var_name{var_name_.c_str(), var_name_.length()};
    TF_Status* tf_status = TF_NewStatus();
    TF_DestroyTemporaryVariable(ctx->Get(), 0, &c_var_name, tf_status);
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(tf_status))
        << " Error while calling TF_DestroyTemporaryVariable";
    TF_DeleteStatus(tf_status);
  }

 private:
  string var_name_;
};

#define REGISTER_GPU_KERNELS(type)                               \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")              \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("dtype"),    \
                          TemporaryVariableOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")       \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T"),        \
                          DestroyTemporaryVariableOp);
TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_uint32(REGISTER_GPU_KERNELS);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);

}  // namespace itex
