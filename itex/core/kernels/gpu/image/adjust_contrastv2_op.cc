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

#include "itex/core/kernels/gpu/image/adjust_contrastv2_op.h"

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

class AdjustContrastOpV2Base : public OpKernel {
 protected:
  explicit AdjustContrastOpV2Base(OpKernelConstruction* context)
      : OpKernel(context) {}

  struct ComputeOptions {
    const Tensor* input = nullptr;
    const Tensor* factor = nullptr;
    Tensor* output = nullptr;
    int64 batch = 0;
    int64 height = 0;
    int64 width = 0;
    int64 channels = 0;
  };

 public:
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& factor = context->input(1);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    const int64 height = input.dim_size(input.dims() - 3);
    const int64 width = input.dim_size(input.dims() - 2);
    const int64 channels = input.dim_size(input.dims() - 1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(factor.shape()),
                errors::InvalidArgument("contrast_factor must be scalar: ",
                                        factor.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    if (input.NumElements() > 0) {
      const int64 batch = input.NumElements() / (height * width * channels);
      ComputeOptions options;
      options.input = &input;
      options.factor = &factor;
      options.output = output;
      options.batch = batch;
      options.height = height;
      options.width = width;
      options.channels = channels;
      DoCompute(context, options);
    }
  }

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;
};

template <typename Device, typename T>
class AdjustContrastOpv2;

template <typename T>
class AdjustContrastOpv2<GPUDevice, T> : public AdjustContrastOpV2Base {
 public:
  explicit AdjustContrastOpv2(OpKernelConstruction* context)
      : AdjustContrastOpV2Base(context) {}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const int64 shape[4] = {options.batch, options.height, options.width,
                            options.channels};
    functor::AdjustContrastv2<GPUDevice, T>()(
        context->eigen_device<GPUDevice>(), options.input->shaped<T, 4>(shape),
        options.factor->scalar<float>(), options.output->shaped<T, 4>(shape));
  }
};

#define REGISTER_GPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("AdjustContrastv2").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustContrastOpv2<GPUDevice, T>);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_GPU(double)
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_INTEGRAL_TYPES(REGISTER_GPU);
REGISTER_GPU(float)
REGISTER_GPU(Eigen::half)
REGISTER_GPU(Eigen::bfloat16)

#undef REGISTER_GPU

}  // namespace itex
