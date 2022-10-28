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

#define REGISTER_KERNEL(T)                                                    \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")                        \
                              .Device(DEVICE_GPU)                             \
                              .HostMemory("filter_sizes")                     \
                              .TypeConstraint<T>("T"),                        \
                          ConvBackpropFilterOp<GPUDevice, T, false, false>);  \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilterWithBias")                \
                              .Device(DEVICE_GPU)                             \
                              .HostMemory("filter_sizes")                     \
                              .TypeConstraint<T>("T"),                        \
                          ConvBackpropFilterOp<GPUDevice, T, false, true>);   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvBackpropFilterOp<GPUDevice, T, false>);                             \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("filter_sizes"),                    \
                          ConvBackpropFilterOp<GPUDevice, T>);                \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterWithBias")                \
                              .Device(DEVICE_GPU)                             \
                              .HostMemory("filter_sizes")                     \
                              .TypeConstraint<T>("T"),                        \
                          ConvBackpropFilterOp<GPUDevice, T, false, true>);   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ITEXPadWithConv2DBackpropFilter")                                \
          .Device(DEVICE_GPU)                                                 \
          .HostMemory("filter_sizes")                                         \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<int32>("Tpaddings")                                 \
          .HostMemory("paddings"),                                            \
      ConvBackpropFilterOp<GPUDevice, T, false, false, true>);                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ITEXPadWithConv2DBackpropFilterWithBias")                        \
          .Device(DEVICE_GPU)                                                 \
          .HostMemory("filter_sizes")                                         \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<int32>("Tpaddings")                                 \
          .HostMemory("paddings"),                                            \
      ConvBackpropFilterOp<GPUDevice, T, false, true, true>);                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ITEXPadWithConv3DBackpropFilter")                                \
          .Device(DEVICE_GPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<int32>("Tpaddings")                                 \
          .HostMemory("paddings"),                                            \
      ConvBackpropFilterOp<GPUDevice, T, false, false, true>);                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ITEXPadWithConv3DBackpropFilterV2")                              \
          .Device(DEVICE_GPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<int32>("Tpaddings")                                 \
          .HostMemory("paddings")                                             \
          .HostMemory("filter_sizes"),                                        \
      ConvBackpropFilterOp<GPUDevice, T, false, false, true>);                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ITEXPadWithConv3DBackpropFilterWithBias")                        \
          .Device(DEVICE_GPU)                                                 \
          .HostMemory("filter_sizes")                                         \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<int32>("Tpaddings")                                 \
          .HostMemory("paddings"),                                            \
      ConvBackpropFilterOp<GPUDevice, T, false, true, true>);

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);

#define REGISTER_BACKPROP_DOUBLE_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")                        \
                              .Device(DEVICE_GPU)                             \
                              .HostMemory("filter_sizes")                     \
                              .TypeConstraint<T>("T"),                        \
                          ConvBackpropFilterOp<GPUDevice, T, false, false>);  \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvBackpropFilterOp<GPUDevice, T, false>);                             \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("filter_sizes"),                    \
                          ConvBackpropFilterOp<GPUDevice, T>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_BACKPROP_DOUBLE_KERNEL);
#endif

#define REGISTER_DEPTHWISE_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNativeBackpropFilter") \
                              .Device(DEVICE_GPU)                     \
                              .HostMemory("filter_sizes")             \
                              .TypeConstraint<T>("T"),                \
                          ConvBackpropFilterOp<GPUDevice, T, true, false>)

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_DEPTHWISE_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_DEPTHWISE_KERNEL);
#endif
#undef REGISTER_KERNEL
#undef REGISTER_DEPTHWISE_KERNEL
#undef REGISTER_BACKPROP_DOUBLE_KERNEL

}  // namespace itex
