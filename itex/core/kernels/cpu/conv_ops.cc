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

#include "itex/core/kernels/common/conv_ops.h"

#include "itex/core/utils/register_types.h"

namespace itex {

#define REGISTER_CPU_CONV2D(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),           \
      ConvOpBase<CPUDevice, T, T, T, T, T>);                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXFusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),      \
      FusedConvOp<CPUDevice, T, T, T, T, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("_ITEXDepthwiseConv2dNative")                   \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          ConvOpBase<CPUDevice, T, T, T, T, T, false, true>);  \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedConv2DWithSum")                      \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          FusedConvOp<CPUDevice, T, T, T, T, T>);              \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedDepthwiseConv2dNative")              \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          FusedConvOp<CPUDevice, T, T, T, T, T, false, true>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithConv2D")                           \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          ConvOpBase<CPUDevice, T, T, T, T, T, true>);         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithFusedConv2D")                      \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          FusedConvOp<CPUDevice, T, T, T, T, T, true>);        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_FusedConv2DWithSum").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      FusedConvOp<CPUDevice, T, T, T, T, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("_PadWithConv2D")                               \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          ConvOpBase<CPUDevice, T, T, T, T, T, true>);         \
  REGISTER_KERNEL_BUILDER(Name("_PadWithFusedConv2D")                          \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          FusedConvOp<CPUDevice, T, T, T, T, T, true>);
// TODO(itex): remove registration of intermediate kernels. Remapper should
// directly generate _ITEXFusedxxx.
TF_CALL_CPU_NUMBER_TYPES(REGISTER_CPU_CONV2D);
#undef REGISTER_CPU_CONV2D

#define REGISTER_CPU_CONV3D(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ITEXConv3D").Device(DEVICE_CPU).TypeConstraint<T>("T"),      \
      ConvOpBase<CPUDevice, T, T, T, T, T>);                              \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ITEXFusedConv3D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedConvOp<CPUDevice, T, T, T, T, T>);                             \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithConv3D")                      \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<int32>("Tpaddings")         \
                              .HostMemory("paddings"),                    \
                          ConvOpBase<CPUDevice, T, T, T, T, T, true>);    \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithFusedConv3D")                 \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<int32>("Tpaddings")         \
                              .HostMemory("paddings"),                    \
                          FusedConvOp<CPUDevice, T, T, T, T, T, true>);   \
  REGISTER_KERNEL_BUILDER(Name("_PadWithConv3D")                          \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<int32>("Tpaddings")         \
                              .HostMemory("paddings"),                    \
                          ConvOpBase<CPUDevice, T, T, T, T, T, true>);    \
  REGISTER_KERNEL_BUILDER(Name("_PadWithFusedConv3D")                     \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<int32>("Tpaddings")         \
                              .HostMemory("paddings"),                    \
                          FusedConvOp<CPUDevice, T, T, T, T, T, true>);
// TODO(itex): remove registration of intermediate kernels. Remapper should
// directly generate _ITEXFusedxxx.
TF_CALL_CPU_NUMBER_TYPES(REGISTER_CPU_CONV3D);
#undef REGISTER_CPU_CONV3D

}  // namespace itex
