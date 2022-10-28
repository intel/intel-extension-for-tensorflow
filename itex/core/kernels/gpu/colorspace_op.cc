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

#include "itex/core/kernels/gpu/colorspace_op.h"

#include <algorithm>
#include <cmath>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class RGBToHSVOp : public OpKernel {
 public:
  explicit RGBToHSVOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, channels == 3,
                errors::FailedPrecondition(
                    "input must have 3 channels but input only has ", channels,
                    " channels."));

    // Create the output Tensor with the same dimensions as the input Tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    // Make a canonical image, maintaining the last (channel) dimension, while
    // flattening all others do give the functor easy to work with data.
    typename TTypes<T, 2>::ConstTensor input_data = input.flat_inner_dims<T>();
    typename TTypes<T, 2>::Tensor output_data = output->flat_inner_dims<T>();

    Tensor trange;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value,
                                        TensorShape({input_data.dimension(0)}),
                                        &trange));

    typename TTypes<T, 1>::Tensor range(trange.tensor<T, 1>());

    functor::RGBToHSV<Device, T>()(context->eigen_device<Device>(), input_data,
                                   range, output_data);
  }
};

template <typename Device, typename T>
class HSVToRGBOp : public OpKernel {
 public:
  explicit HSVToRGBOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, channels == 3,
                errors::FailedPrecondition(
                    "input must have 3 channels but input only has ", channels,
                    " channels."));

    // Create the output Tensor with the same dimensions as the input Tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    typename TTypes<T, 2>::ConstTensor input_data = input.flat_inner_dims<T>();
    typename TTypes<T, 2>::Tensor output_data = output->flat_inner_dims<T>();

    functor::HSVToRGB<Device, T>()(context->eigen_device<Device>(), input_data,
                                   output_data);
  }
};

namespace functor {
#define DECLARE_GPU(T)                                               \
  template <>                                                        \
  void RGBToHSV<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, TTypes<T, 2>::ConstTensor input_data,      \
      TTypes<T, 1>::Tensor range, TTypes<T, 2>::Tensor output_data); \
  extern template struct RGBToHSV<GPUDevice, T>;                     \
  template <>                                                        \
  void HSVToRGB<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, TTypes<T, 2>::ConstTensor input_data,      \
      TTypes<T, 2>::Tensor output_data);                             \
  extern template struct HSVToRGB<GPUDevice, T>;
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DECLARE_GPU);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_float(DECLARE_GPU);
TF_CALL_half(DECLARE_GPU);
TF_CALL_bfloat16(DECLARE_GPU);
}  // namespace functor

#define REGISTER_GPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("RGBToHSV").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RGBToHSVOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("HSVToRGB").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      HSVToRGBOp<GPUDevice, T>);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_float(REGISTER_GPU);
TF_CALL_half(REGISTER_GPU);
TF_CALL_bfloat16(REGISTER_GPU);

}  // namespace itex
