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

#include "itex/core/kernels/common/no_ops.h"
#include "itex/core/kernels/onednn/block/conv_ops_impl.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {
#ifndef INTEL_CPU_ONLY
// FP32 Kernel
// TODO(itex): use new macro HostMemoryList to avoid so long host memory
// declaration
#define REGISTER_KERNEL(T)                                                    \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv2D")                               \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("input_meta")                       \
                              .HostMemory("filter_meta")                      \
                              .HostMemory("output_meta"),                     \
                          OneDnnConvOp<GPUDevice, T, T, T, T, T>);            \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedConv2D")                          \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("input_meta")                       \
                              .HostMemory("filter_meta")                      \
                              .HostMemory("args_meta")                        \
                              .HostMemory("output_meta"),                     \
                          OneDnnFusedConvOp<GPUDevice, T, T, T, T, T>);       \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnPadWithConv2D")                        \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings")             \
                              .HostMemory("paddings")                         \
                              .HostMemory("input_meta")                       \
                              .HostMemory("filter_meta")                      \
                              .HostMemory("output_meta"),                     \
                          OneDnnConvOp<GPUDevice, T, T, T, T, T, true>);      \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnPadWithFusedConv2D")                   \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings")             \
                              .HostMemory("paddings")                         \
                              .HostMemory("input_meta")                       \
                              .HostMemory("filter_meta")                      \
                              .HostMemory("args_meta")                        \
                              .HostMemory("output_meta"),                     \
                          OneDnnFusedConvOp<GPUDevice, T, T, T, T, T, true>); \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv3D")                               \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("input_meta")                       \
                              .HostMemory("filter_meta")                      \
                              .HostMemory("output_meta"),                     \
                          OneDnnConvOp<GPUDevice, T, T, T, T, T>);            \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedConv3D")                          \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("input_meta")                       \
                              .HostMemory("filter_meta")                      \
                              .HostMemory("args_meta")                        \
                              .HostMemory("output_meta"),                     \
                          OneDnnFusedConvOp<GPUDevice, T, T, T, T, T>);       \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnPadWithConv3D")                        \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings")             \
                              .HostMemory("paddings")                         \
                              .HostMemory("input_meta")                       \
                              .HostMemory("filter_meta")                      \
                              .HostMemory("output_meta"),                     \
                          OneDnnConvOp<GPUDevice, T, T, T, T, T, true>);      \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnPadWithFusedConv3D")                   \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings")             \
                              .HostMemory("paddings")                         \
                              .HostMemory("input_meta")                       \
                              .HostMemory("filter_meta")                      \
                              .HostMemory("args_meta")                        \
                              .HostMemory("output_meta"),                     \
                          OneDnnFusedConvOp<GPUDevice, T, T, T, T, T, true>); \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnDepthwiseConv2dNative")                                    \
          .Device(DEVICE_GPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .HostMemory("input_meta")                                           \
          .HostMemory("filter_meta")                                          \
          .HostMemory("output_meta"),                                         \
      OneDnnConvOp<GPUDevice, T, T, T, T, T, false, false, true>);            \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnFusedDepthwiseConv2dNative")                               \
          .Device(DEVICE_GPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .HostMemory("input_meta")                                           \
          .HostMemory("filter_meta")                                          \
          .HostMemory("args_meta")                                            \
          .HostMemory("output_meta"),                                         \
      OneDnnFusedConvOp<GPUDevice, T, T, T, T, T, false, false, true>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

// INT8 kernels
// TODO(itex): use new macro HostMemoryList to avoid so long host memory
// declaration
REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2D")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .HostMemory("min_input")
                            .HostMemory("max_input")
                            .HostMemory("min_filter")
                            .HostMemory("max_filter")
                            .HostMemory("input_meta")
                            .HostMemory("filter_meta")
                            .HostMemory("min_input_meta")
                            .HostMemory("max_input_meta")
                            .HostMemory("min_filter_meta")
                            .HostMemory("max_filter_meta")
                            .HostMemory("min_output")
                            .HostMemory("max_output")
                            .HostMemory("output_meta")
                            .HostMemory("min_output_meta")
                            .HostMemory("max_output_meta"),
                        OneDnnQuantizedConvOp<GPUDevice, quint8, float, qint32,
                                              qint32, false, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2D")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .HostMemory("min_input")
                            .HostMemory("max_input")
                            .HostMemory("min_filter")
                            .HostMemory("max_filter")
                            .HostMemory("input_meta")
                            .HostMemory("filter_meta")
                            .HostMemory("min_input_meta")
                            .HostMemory("max_input_meta")
                            .HostMemory("min_filter_meta")
                            .HostMemory("max_filter_meta")
                            .HostMemory("min_output")
                            .HostMemory("max_output")
                            .HostMemory("output_meta")
                            .HostMemory("min_output_meta")
                            .HostMemory("max_output_meta"),
                        OneDnnQuantizedConvOp<GPUDevice, qint8, float, qint32,
                                              qint32, false, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DAndRequantize")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint8>("out_type")
                            .HostMemory("min_input")
                            .HostMemory("max_input")
                            .HostMemory("min_filter")
                            .HostMemory("max_filter")
                            .HostMemory("min_freezed_output")
                            .HostMemory("max_freezed_output")
                            .HostMemory("input_meta")
                            .HostMemory("filter_meta")
                            .HostMemory("min_input_meta")
                            .HostMemory("max_input_meta")
                            .HostMemory("min_filter_meta")
                            .HostMemory("max_filter_meta")
                            .HostMemory("min_freezed_output_meta")
                            .HostMemory("max_freezed_output_meta")
                            .HostMemory("min_output")
                            .HostMemory("max_output")
                            .HostMemory("output_meta")
                            .HostMemory("min_output_meta")
                            .HostMemory("max_output_meta"),
                        OneDnnQuantizedConvOp<GPUDevice, quint8, float, qint8,
                                              qint8, false, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvOp<GPUDevice, qint8, float, qint8, qint8, false, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBias")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .HostMemory("min_input")
                            .HostMemory("max_input")
                            .HostMemory("min_filter")
                            .HostMemory("max_filter")
                            .HostMemory("input_meta")
                            .HostMemory("filter_meta")
                            .HostMemory("bias_meta")
                            .HostMemory("min_input_meta")
                            .HostMemory("max_input_meta")
                            .HostMemory("min_filter_meta")
                            .HostMemory("max_filter_meta")
                            .HostMemory("min_output")
                            .HostMemory("max_output")
                            .HostMemory("output_meta")
                            .HostMemory("min_output_meta")
                            .HostMemory("max_output_meta"),
                        OneDnnQuantizedConvOp<GPUDevice, quint8, float, qint32,
                                              qint32, true, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBias")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .HostMemory("min_input")
                            .HostMemory("max_input")
                            .HostMemory("min_filter")
                            .HostMemory("max_filter")
                            .HostMemory("input_meta")
                            .HostMemory("filter_meta")
                            .HostMemory("bias_meta")
                            .HostMemory("min_input_meta")
                            .HostMemory("max_input_meta")
                            .HostMemory("min_filter_meta")
                            .HostMemory("max_filter_meta")
                            .HostMemory("min_output")
                            .HostMemory("max_output")
                            .HostMemory("output_meta")
                            .HostMemory("min_output_meta")
                            .HostMemory("max_output_meta"),
                        OneDnnQuantizedConvOp<GPUDevice, qint8, float, qint32,
                                              qint32, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<qint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvOp<GPUDevice, quint8, float, qint8, qint8, true, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBiasAndRequantize")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("Tbias")
                            .TypeConstraint<qint8>("out_type")
                            .HostMemory("min_input")
                            .HostMemory("max_input")
                            .HostMemory("min_filter")
                            .HostMemory("max_filter")
                            .HostMemory("min_freezed_output")
                            .HostMemory("max_freezed_output")
                            .HostMemory("input_meta")
                            .HostMemory("filter_meta")
                            .HostMemory("bias_meta")
                            .HostMemory("min_input_meta")
                            .HostMemory("max_input_meta")
                            .HostMemory("min_filter_meta")
                            .HostMemory("max_filter_meta")
                            .HostMemory("min_freezed_output_meta")
                            .HostMemory("max_freezed_output_meta")
                            .HostMemory("min_output")
                            .HostMemory("max_output")
                            .HostMemory("output_meta")
                            .HostMemory("min_output_meta")
                            .HostMemory("max_output_meta"),
                        OneDnnQuantizedConvOp<GPUDevice, quint8, qint32, qint8,
                                              qint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<qint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvOp<GPUDevice, qint8, float, qint8, qint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<qint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvOp<GPUDevice, qint8, qint32, qint8, qint8, true, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBiasAndRelu")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .HostMemory("min_input")
                            .HostMemory("max_input")
                            .HostMemory("min_filter")
                            .HostMemory("max_filter")
                            .HostMemory("input_meta")
                            .HostMemory("filter_meta")
                            .HostMemory("bias_meta")
                            .HostMemory("min_input_meta")
                            .HostMemory("max_input_meta")
                            .HostMemory("min_filter_meta")
                            .HostMemory("max_filter_meta")
                            .HostMemory("min_output")
                            .HostMemory("max_output")
                            .HostMemory("output_meta")
                            .HostMemory("min_output_meta")
                            .HostMemory("max_output_meta"),
                        OneDnnQuantizedConvReluOp<GPUDevice, quint8, float,
                                                  qint32, qint32, true, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBiasAndRelu")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .HostMemory("min_input")
                            .HostMemory("max_input")
                            .HostMemory("min_filter")
                            .HostMemory("max_filter")
                            .HostMemory("input_meta")
                            .HostMemory("filter_meta")
                            .HostMemory("bias_meta")
                            .HostMemory("min_input_meta")
                            .HostMemory("max_input_meta")
                            .HostMemory("min_filter_meta")
                            .HostMemory("max_filter_meta")
                            .HostMemory("min_output")
                            .HostMemory("max_output")
                            .HostMemory("output_meta")
                            .HostMemory("min_output_meta")
                            .HostMemory("max_output_meta"),
                        OneDnnQuantizedConvReluOp<GPUDevice, qint8, float,
                                                  qint32, qint32, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvReluOp<GPUDevice, quint8, float, quint8, quint8, true,
                              false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvReluOp<GPUDevice, quint8, qint32, quint8, quint8, true,
                              false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvReluOp<GPUDevice, qint8, float, quint8, quint8, true,
                              false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvOp<GPUDevice, qint8, qint32, quint8, quint8, true,
                          false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSumAndRelu")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("summand_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvSumReluOp<GPUDevice, quint8, float, qint32, qint32, true,
                                 false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("min_summand")
        .HostMemory("max_summand")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("summand_meta")
        .HostMemory("min_summand_meta")
        .HostMemory("max_summand_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvSumReluOp<GPUDevice, quint8, float, quint8, quint8, true,
                                 false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("min_summand")
        .HostMemory("max_summand")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("summand_meta")
        .HostMemory("min_summand_meta")
        .HostMemory("max_summand_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvSumReluOp<GPUDevice, quint8, qint32, quint8, quint8,
                                 true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("min_summand")
        .HostMemory("max_summand")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("summand_meta")
        .HostMemory("min_summand_meta")
        .HostMemory("max_summand_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvSumReluOp<GPUDevice, quint8, float, quint8, qint8, true,
                                 false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("min_summand")
        .HostMemory("max_summand")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("summand_meta")
        .HostMemory("min_summand_meta")
        .HostMemory("max_summand_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizedConvSumReluOp<GPUDevice, quint8, qint32, quint8, qint8, true,
                                 false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizeV2WithQuantizedConv2D")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")  // quantizeV2 min input
        .HostMemory("max_input")  // quantizeV2 max input
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizeV2WithQuantizedConv2DOp<GPUDevice, float, float, quint8,
                                          quint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizeV2WithQuantizedConv2D")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .HostMemory("min_input")  // quantizeV2 min input
        .HostMemory("max_input")  // quantizeV2 max input
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("min_output")
        .HostMemory("max_output")
        .HostMemory("output_meta")
        .HostMemory("min_output_meta")
        .HostMemory("max_output_meta"),
    OneDnnQuantizeV2WithQuantizedConv2DOp<GPUDevice, float, qint32, quint8,
                                          quint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithDequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<float>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("output_meta"),
    OneDnnQuantizedConv2DWithDequantizeOp<GPUDevice, qint8, float, float, qint8,
                                          true, false>);
REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithDequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<float>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("output_meta"),
    OneDnnQuantizedConv2DWithDequantizeOp<GPUDevice, quint8, float, float,
                                          qint8, true, false>);
REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithDequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<float>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("output_meta"),
    OneDnnQuantizedConv2DWithDequantizeOp<GPUDevice, qint8, qint32, float,
                                          qint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithDequantize")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<float>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("output_meta"),
    OneDnnQuantizedConv2DWithDequantizeOp<GPUDevice, quint8, qint32, float,
                                          qint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithCast")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<Eigen::half>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("output_meta"),
    OneDnnQuantizedConv2DWithDequantizeOp<GPUDevice, qint8, float, Eigen::half,
                                          qint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithCast")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<Eigen::half>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("output_meta"),
    OneDnnQuantizedConv2DWithDequantizeOp<GPUDevice, quint8, float, Eigen::half,
                                          qint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithCast")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<Eigen::half>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("output_meta"),
    OneDnnQuantizedConv2DWithDequantizeOp<GPUDevice, qint8, qint32, Eigen::half,
                                          qint8, true, false>);
REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithCast")
        .Device(DEVICE_GPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<Eigen::half>("out_type")
        .HostMemory("min_input")
        .HostMemory("max_input")
        .HostMemory("min_filter")
        .HostMemory("max_filter")
        .HostMemory("min_freezed_output")
        .HostMemory("max_freezed_output")
        .HostMemory("input_meta")
        .HostMemory("filter_meta")
        .HostMemory("bias_meta")
        .HostMemory("min_input_meta")
        .HostMemory("max_input_meta")
        .HostMemory("min_filter_meta")
        .HostMemory("max_filter_meta")
        .HostMemory("min_freezed_output_meta")
        .HostMemory("max_freezed_output_meta")
        .HostMemory("output_meta"),
    OneDnnQuantizedConv2DWithDequantizeOp<GPUDevice, quint8, qint32,
                                          Eigen::half, qint8, true, false>);
#endif

}  // namespace itex
