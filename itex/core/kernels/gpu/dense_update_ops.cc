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

#include "itex/core/kernels/gpu/dense_update_functor.h"
#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AssignOp : public OpKernel {
 public:
  explicit AssignOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_shape", &validate_shape_));
    // TODO(itex): Enable following check after we have C api to check ref type
    // OP_REQUIRES(context, IsRefType(context->input_dtype(0)),
    //            errors::InvalidArgument("lhs input needs to be a ref type"));
    if (!context
             ->GetAttr("_grappler_relax_allocator_constraints",
                       &relax_constraints_)
             .ok()) {
      relax_constraints_ = false;
    }
  }

  void Compute(OpKernelContext* context) override {
    constexpr int input_ref_index = 0;
    constexpr int output_ref_index = 0;
    constexpr int value_index = 1;

    OP_REQUIRES_OK(context,
                   AssignRefVariableHelper<Device, T>(
                       context, input_ref_index, output_ref_index, value_index,
                       use_exclusive_lock_, validate_shape_));
  }

  bool use_exclusive_lock_;
  bool validate_shape_;
  bool relax_constraints_;
};

template <typename Device, typename T, DenseUpdateType OP>
class DenseUpdateOp : public OpKernel {
 public:
  explicit DenseUpdateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &this->use_locking_));
  }

  void DoUpdate(OpKernelContext* ctx) {
    Tensor& Tparams = ctx->mutable_input(0, this->use_locking_);
    const Tensor& Tupdate = ctx->input(1);
    OP_REQUIRES(ctx, Tparams.IsInitialized(),
                errors::InvalidArgument(
                    "Attempting to use uninitialized parameters: input 0"));
    OP_REQUIRES(
        ctx, Tparams.IsSameSize(Tupdate),
        errors::InvalidArgument("Parameters and update must be the same size"));

    functor::DenseUpdate<GPUDevice, T, OP> update_functor;
    update_functor(ctx->eigen_gpu_device(), Tparams.flat<T>(),
                   Tupdate.flat<T>());
  }

  void Compute(OpKernelContext* context) override {
    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        context, /* do_lock */ this->use_locking_, /* sparse unused*/ false,
        {0});
    DoUpdate(context);
  }

 private:
  bool use_locking_;
};

// Only register 'Assign' on GPU for the subset of types also supported by
// 'Variable' (see variable_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Assign").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      AssignOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int8(REGISTER_GPU_KERNELS);
TF_CALL_int32(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_uint32(REGISTER_GPU_KERNELS);
TF_CALL_complex64(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
TF_CALL_complex128(REGISTER_GPU_KERNELS);
#endif

#define REGISTER_GPU_DENSE_UPDATE(type)                               \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<GPUDevice, type, DenseUpdateType::ADD>)           \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignSub").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<GPUDevice, type, DenseUpdateType::SUB>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_DENSE_UPDATE);
TF_CALL_int32(REGISTER_GPU_DENSE_UPDATE);
TF_CALL_int64(REGISTER_GPU_DENSE_UPDATE);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_DENSE_UPDATE);
#endif
#undef REGISTER_GPU_DENSE_UPDATE
}  // end namespace itex
