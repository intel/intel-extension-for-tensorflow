/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/kernels/common/conv_grad_ops.h"
#include "itex/core/utils/register_types.h"

namespace itex {

#define REGISTER_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(Name("_ITEXConv2DBackpropInput")                 \
                              .Device(DEVICE_CPU)                          \
                              .HostMemory("input_sizes")                   \
                              .TypeConstraint<T>("T"),                     \
                          ConvBackpropInputOp<CPUDevice, T>)               \
  REGISTER_KERNEL_BUILDER(Name("_ITEXConv3DBackpropInput")                 \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T"),                     \
                          ConvBackpropInputOp<CPUDevice, T, false>);       \
  REGISTER_KERNEL_BUILDER(Name("_ITEXConv3DBackpropInputV2")               \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .HostMemory("input_sizes"),                  \
                          ConvBackpropInputOp<CPUDevice, T>);              \
  REGISTER_KERNEL_BUILDER(Name("_ITEXConv2DBackpropInputWithSlice")        \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .HostMemory("begin")                         \
                              .HostMemory("size")                          \
                              .HostMemory("input_sizes"),                  \
                          ConvBackpropInputOp<CPUDevice, T, false, true>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXConv3DBackpropInputV2WithSlice")      \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .HostMemory("begin")                         \
                              .HostMemory("size")                          \
                              .HostMemory("input_sizes"),                  \
                          ConvBackpropInputOp<CPUDevice, T, false, true>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

#define REGISTER_DEPTHWISE_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(Name("_ITEXDepthwiseConv2dNativeBackpropInput") \
                              .Device(DEVICE_CPU)                         \
                              .HostMemory("input_sizes")                  \
                              .TypeConstraint<T>("T"),                    \
                          ConvBackpropInputOp<CPUDevice, T, true>)

TF_CALL_CPU_NUMBER_TYPES(REGISTER_DEPTHWISE_KERNEL);
#undef REGISTER_KERNEL
#undef REGISTER_DEPTHWISE_KERNEL

// Register the intermediate kernel since graph won't be rewritten if nodes
// number < 4 (TF2.10 and before).
// Set priority to avoid duplicate registry error since TF2.11 already
// registered it.
#define REGISTER_INTERMEDIATE_CPU_OP(T)                 \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInputV2") \
                              .Device(DEVICE_CPU)       \
                              .TypeConstraint<T>("T")   \
                              .Priority(CPU_PRIORITY),  \
                          ConvBackpropInputOp<CPUDevice, T>);

TF_CALL_bfloat16(REGISTER_INTERMEDIATE_CPU_OP);
TF_CALL_half(REGISTER_INTERMEDIATE_CPU_OP);
#undef REGISTER_INTERMEDIATE_CPU_OP

}  // namespace itex
