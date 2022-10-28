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

#include "itex/core/kernels/onednn/block/conv_ops_impl.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {
#ifdef INTEL_CPU_ONLY
// FP32 Kernel
#define REGISTER_KERNEL(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),        \
      OneDnnConvOp<CPUDevice, T, T, T, T, T>);                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnFusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      OneDnnFusedConvOp<CPUDevice, T, T, T, T, T>);                           \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnPadWithConv2D")                        \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings"),            \
                          OneDnnConvOp<CPUDevice, T, T, T, T, T, true>);      \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnPadWithFusedConv2D")                   \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings"),            \
                          OneDnnFusedConvOp<CPUDevice, T, T, T, T, T, true>); \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnConv3D").Device(DEVICE_CPU).TypeConstraint<T>("T"),        \
      OneDnnConvOp<CPUDevice, T, T, T, T, T>);                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnFusedConv3D").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      OneDnnFusedConvOp<CPUDevice, T, T, T, T, T>);                           \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnPadWithConv3D")                        \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings"),            \
                          OneDnnConvOp<CPUDevice, T, T, T, T, T, true>);      \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnPadWithFusedConv3D")                   \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings"),            \
                          OneDnnFusedConvOp<CPUDevice, T, T, T, T, T, true>); \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnDepthwiseConv2dNative")                                    \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T"),                                            \
      OneDnnConvOp<CPUDevice, T, T, T, T, T, false, false, true>);            \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnFusedDepthwiseConv2dNative")                               \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T"),                                            \
      OneDnnFusedConvOp<CPUDevice, T, T, T, T, T, false, false, true>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

// INT8 kernels
// TODO(itex): QuantizedConv2D** CPU kernel registrations are all removed,
// since TF Proper has already made the registration even by building without
// --config=mkl. Check whether the registration in TF proper covers all the
// datatype.

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2D")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        OneDnnQuantizedConvOp<CPUDevice, quint8, float, qint32,
                                              qint32, false, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2D")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        OneDnnQuantizedConvOp<CPUDevice, qint8, float, qint32,
                                              qint32, false, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint8>("out_type"),
                        OneDnnQuantizedConvOp<CPUDevice, quint8, float, qint8,
                                              qint8, false, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint8>("out_type"),
    OneDnnQuantizedConvOp<CPUDevice, qint8, float, qint8, qint8, false, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBias")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        OneDnnQuantizedConvOp<CPUDevice, quint8, float, qint32,
                                              qint32, true, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBias")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        OneDnnQuantizedConvOp<CPUDevice, qint8, float, qint32,
                                              qint32, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<qint8>("out_type"),
    OneDnnQuantizedConvOp<CPUDevice, quint8, float, qint8, qint8, true, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBiasAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("Tbias")
                            .TypeConstraint<qint8>("out_type"),
                        OneDnnQuantizedConvOp<CPUDevice, quint8, qint32, qint8,
                                              qint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<qint8>("out_type"),
    OneDnnQuantizedConvOp<CPUDevice, qint8, float, qint8, qint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<qint8>("out_type"),
    OneDnnQuantizedConvOp<CPUDevice, qint8, qint32, qint8, qint8, true, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBiasAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        OneDnnQuantizedConvReluOp<CPUDevice, quint8, float,
                                                  qint32, qint32, true, false>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConv2DWithBiasAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        OneDnnQuantizedConvReluOp<CPUDevice, qint8, float,
                                                  qint32, qint32, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizedConvReluOp<CPUDevice, quint8, float, quint8, quint8, true,
                              false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizedConvReluOp<CPUDevice, quint8, qint32, quint8, quint8, true,
                              false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizedConvReluOp<CPUDevice, qint8, float, quint8, quint8, true,
                              false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizedConvReluOp<CPUDevice, qint8, qint32, quint8, quint8, true,
                              false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSumAndRelu")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("out_type"),
    OneDnnQuantizedConvSumReluOp<CPUDevice, quint8, float, qint32, qint32, true,
                                 false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizedConvSumReluOp<CPUDevice, quint8, float, quint8, quint8, true,
                                 false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizedConvSumReluOp<CPUDevice, quint8, qint32, quint8, quint8,
                                 true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizedConvSumReluOp<CPUDevice, quint8, float, quint8, qint8, true,
                                 false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizedConvSumReluOp<CPUDevice, quint8, qint32, quint8, qint8, true,
                                 false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizeV2WithQuantizedConv2D")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizeV2WithQuantizedConv2DOp<CPUDevice, float, float, quint8,
                                          quint8, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizeV2WithQuantizedConv2D")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    OneDnnQuantizeV2WithQuantizedConv2DOp<CPUDevice, float, qint32, quint8,
                                          quint8, true, false>);

#endif

}  // namespace itex
