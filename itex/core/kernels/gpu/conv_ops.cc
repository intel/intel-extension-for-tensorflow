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

#include "itex/core/kernels/common/no_ops.h"
#include "itex/core/utils/register_types.h"

namespace itex {

#define REGISTER_GPU_CONV2D(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Conv2D").Device(DEVICE_GPU).TypeConstraint<T>("T"),                \
      ConvOpBase<GPUDevice, T, T, T, T, T>)                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DepthwiseConv2dNative").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvOpBase<GPUDevice, T, T, T, T, T, false, true>)                       \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXFusedConv2D").Device(DEVICE_GPU).TypeConstraint<T>("T"),      \
      FusedConvOp<GPUDevice, T, T, T, T, T>)                                   \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedDepthwiseConv2dNative")              \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          FusedConvOp<GPUDevice, T, T, T, T, T, false, true>)  \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedConv2DWithSum")                      \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          FusedConvOp<GPUDevice, T, T, T, T, T>)               \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithConv2D")                           \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          ConvOpBase<GPUDevice, T, T, T, T, T, true>)          \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithFusedConv2D")                      \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          FusedConvOp<GPUDevice, T, T, T, T, T, true>)         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithDepthwiseConv2dNative")            \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          FusedConvOp<GPUDevice, T, T, T, T, T, true, true>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_CONV2D);
#undef REGISTER_GPU_CONV2D

#define REGISTER_GPU_CONV3D(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ITEXFusedConv3D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FusedConvOp<GPUDevice, T, T, T, T, T>)                              \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithConv3D")                      \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<int32>("Tpaddings")         \
                              .HostMemory("paddings"),                    \
                          ConvOpBase<GPUDevice, T, T, T, T, T, true>)     \
  REGISTER_KERNEL_BUILDER(Name("_ITEXPadWithFusedConv3D")                 \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<int32>("Tpaddings")         \
                              .HostMemory("paddings"),                    \
                          FusedConvOp<GPUDevice, T, T, T, T, T, true>)    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Conv3D").Device(DEVICE_GPU).TypeConstraint<T>("T"),           \
      ConvOpBase<GPUDevice, T, T, T, T, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_CONV3D);
#undef REGISTER_GPU_CONV3D

#define REGISTER_GPU_CONV_DOUBLE(T)                             \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv2D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvOpBase<GPUDevice, T, T, T, T, T>)                     \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvOpBase<GPUDevice, T, T, T, T, T>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_CONV_DOUBLE);
#endif
#undef REGISTER_GPU_CONV_DOUBLE

#define REGISTER_GPU_FUSED_QUANTIZEDCONV(T)                            \
  REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizeV2WithQuantizedConv2D")   \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<float>("Tinput")         \
                              .TypeConstraint<qint8>("Tfilter")        \
                              .TypeConstraint<T>("Tbias")              \
                              .TypeConstraint<quint8>("out_type")      \
                              .HostMemory("min_input")                 \
                              .HostMemory("max_input")                 \
                              .HostMemory("min_filter")                \
                              .HostMemory("max_filter")                \
                              .HostMemory("min_freezed_output")        \
                              .HostMemory("max_freezed_output")        \
                              .HostMemory("min_output")                \
                              .HostMemory("max_output"),               \
                          NoImplementOp);                              \
  REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizedConv2DWithDequantize")   \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<qint8>("Tinput")         \
                              .TypeConstraint<qint8>("Tfilter")        \
                              .TypeConstraint<T>("Tbias")              \
                              .TypeConstraint<float>("out_type")       \
                              .HostMemory("min_input")                 \
                              .HostMemory("max_input")                 \
                              .HostMemory("min_filter")                \
                              .HostMemory("max_filter")                \
                              .HostMemory("min_freezed_output")        \
                              .HostMemory("max_freezed_output"),       \
                          NoImplementOp);                              \
  REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizedConv2DWithCast")         \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<qint8>("Tinput")         \
                              .TypeConstraint<qint8>("Tfilter")        \
                              .TypeConstraint<T>("Tbias")              \
                              .TypeConstraint<Eigen::half>("out_type") \
                              .HostMemory("min_input")                 \
                              .HostMemory("max_input")                 \
                              .HostMemory("min_filter")                \
                              .HostMemory("max_filter")                \
                              .HostMemory("min_freezed_output")        \
                              .HostMemory("max_freezed_output"),       \
                          NoImplementOp);                              \
  REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizedConv2DWithDequantize")   \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<quint8>("Tinput")        \
                              .TypeConstraint<qint8>("Tfilter")        \
                              .TypeConstraint<T>("Tbias")              \
                              .TypeConstraint<float>("out_type")       \
                              .HostMemory("min_input")                 \
                              .HostMemory("max_input")                 \
                              .HostMemory("min_filter")                \
                              .HostMemory("max_filter")                \
                              .HostMemory("min_freezed_output")        \
                              .HostMemory("max_freezed_output"),       \
                          NoImplementOp);                              \
  REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizedConv2DWithCast")         \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<quint8>("Tinput")        \
                              .TypeConstraint<qint8>("Tfilter")        \
                              .TypeConstraint<T>("Tbias")              \
                              .TypeConstraint<Eigen::half>("out_type") \
                              .HostMemory("min_input")                 \
                              .HostMemory("max_input")                 \
                              .HostMemory("min_filter")                \
                              .HostMemory("max_filter")                \
                              .HostMemory("min_freezed_output")        \
                              .HostMemory("max_freezed_output"),       \
                          NoImplementOp);
TF_CALL_qint32(REGISTER_GPU_FUSED_QUANTIZEDCONV);
TF_CALL_float(REGISTER_GPU_FUSED_QUANTIZEDCONV);
#undef REGISTER_GPU_FUSED_QUANTIZEDCONV

}  // namespace itex
