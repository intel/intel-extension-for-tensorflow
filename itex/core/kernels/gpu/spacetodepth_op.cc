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

#include "itex/core/kernels/gpu/spacetodepth_op.h"

#include <memory>
#include <string>
#include <utility>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/platform_types.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace {
template <typename T>
struct RawType {
  using type = T;
};

template <>
struct RawType<qint8> {
  // spacetodepth_op_gpu.cu.cc does not instantiate SpaceToDepthOpFunctor for
  // int8, so we map qint8 to uint8. Instantiating int8 could slow down
  // compilation and the code generated is almost the same as for uint8.
  using type = uint8;
};
}  // namespace

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SpaceToDepthOp : public OpKernel {
 public:
  explicit SpaceToDepthOp(OpKernelConstruction* context) : OpKernel(context) {
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

    const bool is_int8x4 = (data_format_ == FORMAT_NCHW_VECT_C);
    const int vect = is_int8x4 ? 4 : 1;
    if (is_int8x4) {
      OP_REQUIRES(
          context, dims == 5,
          errors::InvalidArgument("Input rank should be 5 instead of ", dims));
    } else {
      OP_REQUIRES(
          context, dims == 4,
          errors::InvalidArgument("Input rank should be 4 instead of ", dims));
    }

    constexpr int kNumSpatialDims = 2;
    const int batch_size =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'N'));
    const int height =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'H'));
    const int width =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'W'));
    const int input_depth =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'C')) *
        vect;

    // Both width and height must be divisible by block_size.
    OP_REQUIRES(context,
                (width % block_size_) == 0 && (height % block_size_) == 0,
                errors::InvalidArgument(
                    "Image width ", width, " and height ", height,
                    " should be divisible by block_size: ", block_size_));

    // The 'spatial' block of size block_size_ X block_size_ will be moved
    // to depth.
    const int output_depth = input_depth * block_size_ * block_size_;
    const int output_width = width / block_size_;
    const int output_height = height / block_size_;

    // Allocate output tensor.
    Tensor* outputs_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       ShapeFromFormat(data_format_, batch_size, output_height,
                                       output_width, output_depth),
                       &outputs_tensor));

    using RT = typename RawType<T>::type;
    if (data_format_ == FORMAT_NCHW_VECT_C) {
      // NCHW_VECT_C with 4 x qint8 can be treated as NCHW int32.
      auto Tinput_v = input.template reinterpret_last_dimension<int32, 4>();
      auto Toutput_v = outputs_tensor->reinterpret_last_dimension<int32, 4>();
      functor::SpaceToDepthOpFunctor<Device, int32, FORMAT_NCHW> functor;
      functor(context->eigen_gpu_device(), Tinput_v, block_size_, Toutput_v);
    } else if (data_format_ == FORMAT_NCHW) {
      ITEX_CHECK((std::is_same<T, RT>::value));
      functor::SpaceToDepthOpFunctor<Device, RT, FORMAT_NCHW> functor;
      functor(context->eigen_gpu_device(), input.tensor<RT, 4>(), block_size_,
              outputs_tensor->tensor<RT, 4>());
    } else {
      ITEX_CHECK((std::is_same<T, RT>::value));
      functor::SpaceToDepthOpFunctor<Device, RT, FORMAT_NHWC> functor;
      functor(context->eigen_gpu_device(), input.tensor<RT, 4>(), block_size_,
              outputs_tensor->tensor<RT, 4>());
    }
  };

 private:
  int block_size_;
  TensorFormat data_format_;
};

#define REGISTER(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SpaceToDepth").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      SpaceToDepthOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER);
TF_CALL_qint8(REGISTER);
TF_CALL_uint8(REGISTER);
#undef REGISTER
}  // namespace itex
