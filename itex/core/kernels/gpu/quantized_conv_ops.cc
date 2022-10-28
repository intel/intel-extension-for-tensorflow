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

#include "itex/core/kernels/common/quantized_conv_ops.h"

#include "itex/core/kernels/common/no_ops.h"

namespace itex {

#define REGISTER_KERNEL(op, kernel, input_type, bias_type, output_type,     \
                        summand_type, is_depthwise, legacy_fused_ops,       \
                        num_fused_ops)                                      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(op)                                                              \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<input_type>("Tinput")                             \
          .TypeConstraint<qint8>("Tfilter") BIAS_TYPE_CONSTRAINT(bias_type) \
              SUMMAND_TYPE_CONSTRAINT(summand_type)                         \
          .TypeConstraint<output_type>("out_type") HOSTMEMORYLIST,          \
      kernel TEMPLATE_ARGS(GPUDevice, input_type, bias_type, output_type,   \
                           summand_type, is_depthwise, legacy_fused_ops,    \
                           num_fused_ops));

#define REGISTER_KERNEL_ALL_INPUT_TYPES(op, kernel, bias_type, output_type, \
                                        summand_type, is_depthwise,         \
                                        legacy_fused_ops, num_fused_ops)    \
  REGISTER_KERNEL(op, kernel, qint8, bias_type, output_type, summand_type,  \
                  is_depthwise, legacy_fused_ops, num_fused_ops);           \
  REGISTER_KERNEL(op, kernel, quint8, bias_type, output_type, summand_type, \
                  is_depthwise, legacy_fused_ops, num_fused_ops);

#define REGISTER_KERNEL_ALL_BIAS_TYPES(op, kernel, input_type, output_type,  \
                                       summand_type, is_depthwise,           \
                                       legacy_fused_ops, num_fused_ops)      \
  REGISTER_KERNEL(op, kernel, input_type, qint32, output_type, summand_type, \
                  is_depthwise, legacy_fused_ops, num_fused_ops);            \
  REGISTER_KERNEL(op, kernel, input_type, float, output_type, summand_type,  \
                  is_depthwise, legacy_fused_ops, num_fused_ops);

#define REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES(                          \
    op, kernel, output_type, summand_type, is_depthwise, legacy_fused_ops, \
    num_fused_ops)                                                         \
  REGISTER_KERNEL_ALL_INPUT_TYPES(op, kernel, qint32, output_type,         \
                                  summand_type, is_depthwise,              \
                                  legacy_fused_ops, num_fused_ops);        \
  REGISTER_KERNEL_ALL_INPUT_TYPES(op, kernel, float, output_type,          \
                                  summand_type, is_depthwise,              \
                                  legacy_fused_ops, num_fused_ops);

// Conv INT8 old API
#define TEMPLATE_ARGS(GPUDevice, input_type, bias_type, output_type, \
                      summand_type, is_depthwise, legacy_fused_ops,  \
                      num_fused_ops)                                 \
<GPUDevice, input_type, bias_type, output_type, summand_type, is_depthwise, \
  legacy_fused_ops, num_fused_ops>
#define BIAS_TYPE_CONSTRAINT(bias_type)
#define SUMMAND_TYPE_CONSTRAINT(summand_type)
#define HOSTMEMORYLIST                                                   \
  .HostMemoryList4("min_input", "max_input", "min_filter", "max_filter") \
      .HostMemoryList2("min_output", "max_output")
REGISTER_KERNEL_ALL_INPUT_TYPES("QuantizedConv2D", LegacyQuantizedConvOpBase,
                                float, qint32, qint32, false,
                                quantized_fusions::none, 0);
REGISTER_KERNEL_ALL_INPUT_TYPES("QuantizedConv2DPerChannel",
                                LegacyQuantizedConvOpBase, float, qint32,
                                qint32, false, quantized_fusions::none, 0);
REGISTER_KERNEL_ALL_INPUT_TYPES("QuantizedConv2DWithBias",
                                LegacyQuantizedConvOpBase, float, qint32,
                                qint32, false, quantized_fusions::bias, 1);
REGISTER_KERNEL_ALL_INPUT_TYPES("QuantizedConv2DWithBiasAndRelu",
                                LegacyQuantizedConvOpBase, float, qint32,
                                qint32, false, quantized_fusions::bias_relu, 2);
REGISTER_KERNEL("QuantizedConv2DWithBiasSumAndRelu", LegacyQuantizedConvOpBase,
                quint8, float, qint32, qint32, false,
                quantized_fusions::bias_sum_relu, 3);
REGISTER_KERNEL("QuantizedConv2DAndRelu", LegacyQuantizedConvOpBase, quint8,
                float, qint32, qint32, false, quantized_fusions::relu, 1);
REGISTER_KERNEL("QuantizedDepthwiseConv2D", LegacyQuantizedConvOpBase, quint8,
                float, qint32, qint32, true, quantized_fusions::none, 0);
REGISTER_KERNEL("QuantizedDepthwiseConv2DWithBias", LegacyQuantizedConvOpBase,
                quint8, float, qint32, qint32, true, quantized_fusions::bias,
                1);
REGISTER_KERNEL("QuantizedDepthwiseConv2DWithBiasAndRelu",
                LegacyQuantizedConvOpBase, quint8, float, qint32, qint32, true,
                quantized_fusions::bias_relu, 2);
#undef HOSTMEMORYLIST

#define HOSTMEMORYLIST                                                   \
  .HostMemoryList6("min_input", "max_input", "min_filter", "max_filter", \
                   "min_freezed_output", "max_freezed_output")           \
      .HostMemoryList2("min_output", "max_output")
REGISTER_KERNEL_ALL_INPUT_TYPES("QuantizedConv2DAndRequantize",
                                LegacyQuantizedConvOpBase, float, qint8, qint8,
                                false, quantized_fusions::requantize, 1);
REGISTER_KERNEL("QuantizedConv2DAndReluAndRequantize",
                LegacyQuantizedConvOpBase, quint8, float, quint8, quint8, false,
                quantized_fusions::relu_requantize, 2);
#undef HOSTMEMORYLIST
#undef SUMMAND_TYPE_CONSTRAINT
#undef BIAS_TYPE_CONSTRAINT

#define BIAS_TYPE_CONSTRAINT(bias_type) .TypeConstraint<bias_type>("Tbias")
#define SUMMAND_TYPE_CONSTRAINT(summand_type)
#define HOSTMEMORYLIST                                                   \
  .HostMemoryList6("min_input", "max_input", "min_filter", "max_filter", \
                   "min_freezed_output", "max_freezed_output")           \
      .HostMemoryList2("min_output", "max_output")
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("QuantizedConv2DWithBiasAndRequantize",
                                         LegacyQuantizedConvOpBase, qint8,
                                         qint8, false,
                                         quantized_fusions::bias_requantize, 2);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES(
    "QuantizedConv2DWithBiasAndReluAndRequantize", LegacyQuantizedConvOpBase,
    quint8, quint8, false, quantized_fusions::bias_relu_requantize, 3);
REGISTER_KERNEL_ALL_BIAS_TYPES(
    "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
    LegacyQuantizedConvOpBase, quint8, quint8, quint8, true,
    quantized_fusions::bias_relu_requantize, 3);
#undef HOSTMEMORYLIST
#undef SUMMAND_TYPE_CONSTRAINT

#define HOSTMEMORYLIST                                                        \
  .HostMemoryList8("min_input", "max_input", "min_filter", "max_filter",      \
                   "min_freezed_output", "max_freezed_output", "min_summand", \
                   "max_summand")                                             \
      .HostMemoryList2("min_output", "max_output")
#define SUMMAND_TYPE_CONSTRAINT(summand_type) \
  .TypeConstraint<summand_type>("Tsummand")
REGISTER_KERNEL_ALL_BIAS_TYPES("QuantizedConv2DWithBiasSumAndReluAndRequantize",
                               LegacyQuantizedConvOpBase, quint8, quint8,
                               quint8, false,
                               quantized_fusions::bias_sum_relu_requantize, 4);
REGISTER_KERNEL_ALL_BIAS_TYPES(
    "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
    LegacyQuantizedConvOpBase, quint8, quint8, qint8, false,
    quantized_fusions::bias_sum_relu_requantize, 4);

#undef HOSTMEMORYLIST
#undef SUMMAND_TYPE_CONSTRAINT
#undef BIAS_TYPE_CONSTRAINT
#undef TEMPLATE_ARGS

// Conv INT8 new API
#define TEMPLATE_ARGS(GPUDevice, input_type, bias_type, output_type, \
                      summand_type, is_depthwise, legacy_fused_ops,  \
                      num_fused_ops)                                 \
<GPUDevice, input_type, bias_type, output_type, summand_type, is_depthwise, \
  legacy_fused_ops, num_fused_ops>
#define BIAS_TYPE_CONSTRAINT(bias_type) .TypeConstraint<bias_type>("Tbias")
#define SUMMAND_TYPE_CONSTRAINT(summand_type) \
  .TypeConstraint<summand_type>("Tsummand")
#define HOSTMEMORYLIST .HostMemoryList2("host_inputs", "host_outputs")

// TODO(itex): invesitgate why here define summand macro null
// Register QuantizedConv Raw ops
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv2D",
                                         LegacyQuantizedConvOpBase, qint8,
                                         qint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv2D",
                                         LegacyQuantizedConvOpBase, quint8,
                                         qint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv2D",
                                         LegacyQuantizedConvOpBase, quint8,
                                         quint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv2D",
                                         LegacyQuantizedConvOpBase, qint8,
                                         quint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv2D",
                                         LegacyQuantizedConvOpBase,
                                         Eigen::bfloat16, Eigen::bfloat16,
                                         false, quantized_fusions::none, -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv2D",
                                         LegacyQuantizedConvOpBase, float,
                                         float, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedDepthwiseConv2D",
                                         LegacyQuantizedConvOpBase, qint8,
                                         qint8, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedDepthwiseConv2D",
                                         LegacyQuantizedConvOpBase, quint8,
                                         qint8, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedDepthwiseConv2D",
                                         LegacyQuantizedConvOpBase, quint8,
                                         quint8, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedDepthwiseConv2D",
                                         LegacyQuantizedConvOpBase, qint8,
                                         quint8, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedDepthwiseConv2D",
                                         LegacyQuantizedConvOpBase,
                                         Eigen::bfloat16, Eigen::bfloat16, true,
                                         quantized_fusions::none, -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedDepthwiseConv2D",
                                         LegacyQuantizedConvOpBase, float,
                                         float, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv3D",
                                         LegacyQuantizedConvOpBase, qint8,
                                         qint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv3D",
                                         LegacyQuantizedConvOpBase, quint8,
                                         qint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv3D",
                                         LegacyQuantizedConvOpBase, quint8,
                                         quint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv3D",
                                         LegacyQuantizedConvOpBase, qint8,
                                         quint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv3D",
                                         LegacyQuantizedConvOpBase,
                                         Eigen::bfloat16, Eigen::bfloat16,
                                         false, quantized_fusions::none, -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv3D",
                                         LegacyQuantizedConvOpBase, float,
                                         float, false, quantized_fusions::none,
                                         -1);

REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv2D",
                                         LegacyQuantizedConvOpBase, qint32,
                                         qint32, false, quantized_fusions::none,
                                         -1)
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedDepthwiseConv2D",
                                         LegacyQuantizedConvOpBase, qint32,
                                         qint32, true, quantized_fusions::none,
                                         -1)
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_QuantizedConv3D",
                                         LegacyQuantizedConvOpBase, qint32,
                                         qint32, false, quantized_fusions::none,
                                         -1)

// Register QuantizedConv native ops
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv2DV2",
                                         LegacyQuantizedConvOpBase, qint8,
                                         qint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv2DV2",
                                         LegacyQuantizedConvOpBase, quint8,
                                         qint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv2DV2",
                                         LegacyQuantizedConvOpBase, quint8,
                                         quint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv2DV2",
                                         LegacyQuantizedConvOpBase, qint8,
                                         quint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv2DV2",
                                         LegacyQuantizedConvOpBase,
                                         Eigen::bfloat16, Eigen::bfloat16,
                                         false, quantized_fusions::none, -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv2DV2",
                                         LegacyQuantizedConvOpBase, float,
                                         float, false, quantized_fusions::none,
                                         -1);

REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedDepthwiseConv2DV2",
                                         LegacyQuantizedConvOpBase, qint8,
                                         qint8, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedDepthwiseConv2DV2",
                                         LegacyQuantizedConvOpBase, quint8,
                                         qint8, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedDepthwiseConv2DV2",
                                         LegacyQuantizedConvOpBase, quint8,
                                         quint8, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedDepthwiseConv2DV2",
                                         LegacyQuantizedConvOpBase, qint8,
                                         quint8, true, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedDepthwiseConv2DV2",
                                         LegacyQuantizedConvOpBase,
                                         Eigen::bfloat16, Eigen::bfloat16, true,
                                         quantized_fusions::none, -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedDepthwiseConv2DV2",
                                         LegacyQuantizedConvOpBase, float,
                                         float, true, quantized_fusions::none,
                                         -1);

REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv3DV2",
                                         LegacyQuantizedConvOpBase, qint8,
                                         qint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv3DV2",
                                         LegacyQuantizedConvOpBase, quint8,
                                         qint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv3DV2",
                                         LegacyQuantizedConvOpBase, quint8,
                                         quint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv3DV2",
                                         LegacyQuantizedConvOpBase, qint8,
                                         quint8, false, quantized_fusions::none,
                                         -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv3DV2",
                                         LegacyQuantizedConvOpBase,
                                         Eigen::bfloat16, Eigen::bfloat16,
                                         false, quantized_fusions::none, -1);
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv3DV2",
                                         LegacyQuantizedConvOpBase, float,
                                         float, false, quantized_fusions::none,
                                         -1);

REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv2DV2",
                                         LegacyQuantizedConvOpBase, qint32,
                                         qint32, false, quantized_fusions::none,
                                         -1)
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedDepthwiseConv2DV2",
                                         LegacyQuantizedConvOpBase, qint32,
                                         qint32, true, quantized_fusions::none,
                                         -1)
REGISTER_KERNEL_ALL_INPUT_AND_BIAS_TYPES("_ITEXQuantizedConv3DV2",
                                         LegacyQuantizedConvOpBase, qint32,
                                         qint32, false, quantized_fusions::none,
                                         -1);

#undef HOSTMEMORYLIST
#undef SUMMAND_TYPE_CONSTRAINT
#undef BIAS_TYPE_CONSTRAINT
#undef TEMPLATE_ARGS

}  // namespace itex
