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

#include "itex/core/kernels/gpu/sparse_xent_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class SparseSoftmaxXentWithLogitsOp : public OpKernel {
 public:
  explicit SparseSoftmaxXentWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits = context->input(0);
    const Tensor& labels = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits.shape()),
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(labels.shape()),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels.shape().DebugString()));
    OP_REQUIRES(context, logits.dim_size(0) == labels.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same first dimension, "
                    "got logits shape ",
                    logits.shape().DebugString(), " and labels shape ",
                    labels.shape().DebugString()));
    OP_REQUIRES(context, logits.dim_size(1) > 0,
                errors::InvalidArgument(
                    "Must have at least one class, but got logits shape ",
                    logits.shape().DebugString()));

    Tensor softmax_temp;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          logits.shape(), &softmax_temp));

    Tensor* loss_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 0, labels.shape(), &loss_out));
    Tensor* back_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 1, logits.shape(), &back_out));

    if (logits.dim_size(0) > 0) {
      functor::SparseXentFunctor<Device, T, Index> functor_sparse_xent;
      OP_REQUIRES_OK(context, functor_sparse_xent(context->eigen_gpu_device(),
                                                  logits, labels, softmax_temp,
                                                  loss_out, back_out));
    }
  }
};

#define REGISTER(Dev, T, Index)                   \
  REGISTER_KERNEL_BUILDER(                        \
      Name("SparseSoftmaxCrossEntropyWithLogits") \
          .Device(DEVICE_##Dev)                   \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Index>("Tlabels"),      \
      SparseSoftmaxXentWithLogitsOp<Dev##Device, T, Index>);

REGISTER(GPU, float, int32)
REGISTER(GPU, float, int64)
REGISTER(GPU, Eigen::half, int32)
REGISTER(GPU, Eigen::half, int64)
REGISTER(GPU, Eigen::bfloat16, int32)
REGISTER(GPU, Eigen::bfloat16, int64)
#undef REGISTER

}  // namespace itex
