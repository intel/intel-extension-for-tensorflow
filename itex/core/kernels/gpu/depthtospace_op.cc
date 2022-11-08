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

#include "itex/core/kernels/gpu/depthtospace_op.h"

#include <memory>
#include <string>
#include <utility>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
class DepthToSpaceOp : public OpKernel {
 public:
  explicit DepthToSpaceOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));
    OP_REQUIRES(context, block_size_ > 1,
                errors::InvalidArgument("Block size should be > 1, but was: ",
                                        block_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int dims = input.dims();

    constexpr int kVect = 1;
    constexpr int kDims = 4;
    OP_REQUIRES(context, kDims == dims,
                errors::InvalidArgument("Input rank should be: ", kDims,
                                        " instead of: ", dims));

    constexpr int kNumSpatialDims = 2;
    const int batch_size =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'N'));
    const int input_height =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'H'));
    const int input_width =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'W'));
    const int input_depth =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'C')) *
        kVect;

    const int block_size_sq = block_size_ * block_size_;

    // The depth must be divisible by block_size_ * block_size_
    OP_REQUIRES(
        context, input_depth % block_size_sq == 0,
        errors::InvalidArgument("Input depth dimension ", input_depth,
                                " should be divisible by: ", block_size_sq));

    const int output_depth = input_depth / block_size_sq;
    const int output_width = input_width * block_size_;
    const int output_height = input_height * block_size_;

    // Allocate output tensor.
    Tensor* outputs_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       ShapeFromFormat(data_format_, batch_size, output_height,
                                       output_width, output_depth),
                       &outputs_tensor));
    auto Tinput = input.tensor<T, kDims>();
    auto Toutput = outputs_tensor->tensor<T, kDims>();

    if (data_format_ == FORMAT_NCHW) {
      functor::DepthToSpaceOpFunctor<GPUDevice, T, FORMAT_NCHW> functor;
      functor(context->eigen_device<GPUDevice>(), Tinput, block_size_, Toutput);
    } else {
      functor::DepthToSpaceOpFunctor<GPUDevice, T, FORMAT_NHWC> functor;
      functor(context->eigen_device<GPUDevice>(), Tinput, block_size_, Toutput);
    }
  }

 private:
  int block_size_;
  TensorFormat data_format_;
};

#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("DepthToSpace").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DepthToSpaceOp<type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

}  // end namespace itex
