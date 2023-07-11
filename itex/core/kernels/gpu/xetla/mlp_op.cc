/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/xetla/mlp_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class FusedDenseBiasAddGeluOp : public OpKernel {
 public:
  explicit FusedDenseBiasAddGeluOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& feature = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& bias = context->input(2);

    OP_REQUIRES(
        context, feature.dims() == 2,
        errors::InvalidArgument(
            "expexted feature's dimension to be 2, but got ", feature.dims()));

    OP_REQUIRES(
        context, weights.dims() == 2,
        errors::InvalidArgument(
            "expexted weights's dimension to be 2, but got ", weights.dims()));

    OP_REQUIRES(context, feature.dim_size(1) == weights.dim_size(0),
                errors::InvalidArgument(
                    "feature.dim_size[1] must equal to weights.dim_size[0], ",
                    "but got feature.dim_size[1] =", feature.dim_size(1),
                    " and weights.dim_size[0] =", weights.dim_size(0)));

    int m = feature.dim_size(0);
    int n = weights.dim_size(1);
    Tensor* output = nullptr;
    Tensor* workspace = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({m, n}), &output));
    OP_REQUIRES_OK(
        context, context->allocate_output(1, TensorShape({m, n}), &workspace));

    functor::FusedDenseBiasAddGeluFunctor<Device, T>()(
        context, feature, weights, bias, output, workspace);
  }
};

#define REGISTER_GPU_FUNC(type)                           \
  REGISTER_KERNEL_BUILDER(Name("FusedDenseBiasAddGelu")   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T"), \
                          FusedDenseBiasAddGeluOp<GPUDevice, type>);

REGISTER_GPU_FUNC(Eigen::bfloat16);
REGISTER_GPU_FUNC(Eigen::half);
#undef REGISTER_GPU_FUNC

}  // namespace itex
