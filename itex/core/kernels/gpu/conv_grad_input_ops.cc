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

#define REGISTER_KERNEL(T)                                                   \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_GPU)                            \
                              .HostMemory("input_sizes")                     \
                              .TypeConstraint<T>("T"),                       \
                          ConvBackpropInputOp<GPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Conv3DBackpropInput").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvBackpropInputOp<GPUDevice, T, false>);                             \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInputV2")                      \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T")                        \
                              .HostMemory("input_sizes"),                    \
                          ConvBackpropInputOp<GPUDevice, T>);                \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInputWithSlice")               \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T")                        \
                              .HostMemory("begin")                           \
                              .HostMemory("size")                            \
                              .HostMemory("input_sizes"),                    \
                          ConvBackpropInputOp<GPUDevice, T, false, true>);   \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInputV2WithSlice")             \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T")                        \
                              .HostMemory("begin")                           \
                              .HostMemory("size")                            \
                              .HostMemory("input_sizes"),                    \
                          ConvBackpropInputOp<GPUDevice, T, false, true>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);

#define REGISTER_BACKPROP_DOUBLE_KERNEL(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_GPU)                            \
                              .HostMemory("input_sizes")                     \
                              .TypeConstraint<T>("T"),                       \
                          ConvBackpropInputOp<GPUDevice, T>);                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Conv3DBackpropInput").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvBackpropInputOp<GPUDevice, T, false>);                             \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInputV2")                      \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<T>("T")                        \
                              .HostMemory("input_sizes"),                    \
                          ConvBackpropInputOp<GPUDevice, T>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_BACKPROP_DOUBLE_KERNEL);
#endif

#define REGISTER_DEPTHWISE_KERNEL(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNativeBackpropInput") \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("input_sizes")             \
                              .TypeConstraint<T>("T"),               \
                          ConvBackpropInputOp<GPUDevice, T, true>)

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_DEPTHWISE_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_DEPTHWISE_KERNEL);
#endif
#undef REGISTER_KERNEL
#undef REGISTER_DEPTHWISE_KERNEL
#undef REGISTER_BACKPROP_DOUBLE_KERNEL

}  // namespace itex
