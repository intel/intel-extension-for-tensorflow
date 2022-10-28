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

namespace itex {
#ifndef INTEL_CPU_ONLY
REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2D")
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
                                              qint32, false, true>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2D")
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
                                              qint32, false, true>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2DWithBias")
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
                                              qint32, true, true>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedDepthwiseConv2DWithBias")
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
    OneDnnQuantizedConvOp<GPUDevice, qint8, float, qint32, qint32, true, true>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2DWithBiasAndRelu")
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
                                                  qint32, qint32, true, true>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2DWithBiasAndRelu")
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
                                                  qint32, qint32, true, true>);
// Tbias -> float
REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
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
                              true>);

// Tbias -> qint32
REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
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
                              true>);

#else

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2D")
                            .Device(DEVICE_CPU)
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
                        OneDnnQuantizedConvOp<CPUDevice, quint8, float, qint32,
                                              qint32, false, true>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2D")
                            .Device(DEVICE_CPU)
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
                        OneDnnQuantizedConvOp<CPUDevice, qint8, float, qint32,
                                              qint32, false, true>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2DWithBias")
                            .Device(DEVICE_CPU)
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
                        OneDnnQuantizedConvOp<CPUDevice, quint8, float, qint32,
                                              qint32, true, true>);

REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedDepthwiseConv2DWithBias")
        .Device(DEVICE_CPU)
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
    OneDnnQuantizedConvOp<CPUDevice, qint8, float, qint32, qint32, true, true>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2DWithBiasAndRelu")
                            .Device(DEVICE_CPU)
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
                        OneDnnQuantizedConvReluOp<CPUDevice, quint8, float,
                                                  qint32, qint32, true, true>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedDepthwiseConv2DWithBiasAndRelu")
                            .Device(DEVICE_CPU)
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
                        OneDnnQuantizedConvReluOp<CPUDevice, qint8, float,
                                                  qint32, qint32, true, true>);
// Tbias -> float
REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
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
    OneDnnQuantizedConvReluOp<CPUDevice, quint8, float, quint8, quint8, true,
                              true>);

// Tbias -> qint32
REGISTER_KERNEL_BUILDER(
    Name("_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
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
    OneDnnQuantizedConvReluOp<CPUDevice, quint8, qint32, quint8, quint8, true,
                              true>);

#endif  // INTEL_CPU_ONLY
}  // namespace itex
