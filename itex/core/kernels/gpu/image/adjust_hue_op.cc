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

#include "itex/core/kernels/gpu/image/adjust_hue_op.h"

#include <memory>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/mirror_pad_mode.h"
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

class AdjustHueOpBase : public OpKernel {
 protected:
  explicit AdjustHueOpBase(OpKernelConstruction* context) : OpKernel(context) {}

  struct ComputeOptions {
    const Tensor* input;
    const Tensor* delta;
    Tensor* output;
    int64 channel_count;
  };

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;

 public:
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& delta = context->input(1);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(delta.shape()),
                errors::InvalidArgument("delta must be scalar: ",
                                        delta.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(
        context, channels == 3,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));

    if (input.NumElements() > 0) {
      const int64 channel_count = input.NumElements() / channels;
      ComputeOptions options;
      options.input = &input;
      options.delta = &delta;
      options.output = output;
      options.channel_count = channel_count;
      DoCompute(context, options);
    }
  }
};

template <class Device, typename T>
class AdjustHueOp;

template <typename T>
class AdjustHueOp<GPUDevice, T> : public AdjustHueOpBase {
 public:
  explicit AdjustHueOp(OpKernelConstruction* context)
      : AdjustHueOpBase(context) {}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const Tensor* input = options.input;
    const Tensor* delta = options.delta;
    Tensor* output = options.output;
    const int64 number_of_elements = input->NumElements();
    GPUDevice device = context->eigen_gpu_device();
    const auto stream = device.stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    if (number_of_elements > 0) {
      const T* input_data = input->flat<T>().data();
      const float* delta_h = delta->flat<float>().data();
      T* const output_data = output->flat<T>().data();
      functor::AdjustHueGPU<T>()(&device, number_of_elements, input_data,
                                 delta_h, output_data);
    }
  }
};

#define REGISTER_GPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AdjustHue").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustHueOp<GPUDevice, T>);

REGISTER_GPU(float)
REGISTER_GPU(Eigen::half)
REGISTER_GPU(Eigen::bfloat16)

#undef REGISTER_GPU

}  // namespace itex
