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

#include "itex/core/kernels/common/softmax_op.h"

#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"
#include "itex/core/kernels/gpu/softmax_op_functor.h"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;
template <typename Device, typename T>
class LogSoftmaxOp : public OpKernel {
 public:
  ~LogSoftmaxOp() {}

  explicit LogSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in_ = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(logits_in_.shape()),
                errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                        logits_in_.shape().DebugString()));
    auto logits_in = logits_in_.flat_inner_dims<T>();
    Tensor* logsoftmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, logits_in_.shape(), &logsoftmax_out));

    if (logits_in_.NumElements() > 0 &&
        logits_in_.NumElements() < std::numeric_limits<int32>::max()) {
      SoftmaxFunctor<Device, T> functor;
      functor(context->eigen_gpu_device(), logits_in, logsoftmax_out, true);
    } else {
      ITEX_LOG(ERROR)
          << "Num of Elements exceeds the max value of int32, please use int64";
      return;
    }
  }
};

template <typename Device, typename T>
class AddV2WithSoftmaxOp : public OpKernel {
 public:
  ~AddV2WithSoftmaxOp() {}

  explicit AddV2WithSoftmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in_tensor = context->input(0);
    const Tensor& adder_tensor = context->input(1);

    TensorShape output_shape =
        logits_in_tensor.NumElements() > adder_tensor.NumElements()
            ? logits_in_tensor.shape()
            : adder_tensor.shape();

    OP_REQUIRES(
        context, TensorShapeUtils::IsVectorOrHigher(logits_in_tensor.shape()),
        errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                logits_in_tensor.shape().DebugString()));

    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, output_shape, &softmax_out));

    if (logits_in_tensor.NumElements() > 0 &&
        logits_in_tensor.NumElements() < std::numeric_limits<int32>::max()) {
      AddV2WithSoftmaxFunctor<Device, T> functor;
      // For AddV2 node in TF graph, a pass will change the order of a, b
      // so we need to verify which one is truely att_mask
      if (logits_in_tensor.NumElements() > adder_tensor.NumElements()) {
        functor(context->eigen_gpu_device(), logits_in_tensor, adder_tensor,
                softmax_out, false);
      } else {
        functor(context->eigen_gpu_device(), adder_tensor, logits_in_tensor,
                softmax_out, false);
      }
    } else {
      ITEX_LOG(ERROR)
          << "Num of Elements exceeds the max value of int32, please use int64";
      return;
    }
  }
};

#define REGISTER_SOFTMAX(type)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Softmax").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftmaxOp<GPUDevice, type>);

TF_CALL_float(REGISTER_SOFTMAX);
TF_CALL_bfloat16(REGISTER_SOFTMAX);
TF_CALL_half(REGISTER_SOFTMAX);

#define REGISTER_LOGSOFTMAX(type)                                      \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LogSoftmax").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      LogSoftmaxOp<GPUDevice, type>);
TF_CALL_float(REGISTER_LOGSOFTMAX);
TF_CALL_bfloat16(REGISTER_LOGSOFTMAX);
TF_CALL_half(REGISTER_LOGSOFTMAX);

#define REGISTER_ADDV2WITHSOFTMAX(type)                      \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAddV2WithSoftmax") \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("T"),    \
                          AddV2WithSoftmaxOp<GPUDevice, type>);
TF_CALL_float(REGISTER_ADDV2WITHSOFTMAX);
TF_CALL_bfloat16(REGISTER_ADDV2WITHSOFTMAX);
TF_CALL_half(REGISTER_ADDV2WITHSOFTMAX);

}  // namespace itex
