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

#include "itex/core/kernels/common/no_ops.h"
#include "itex/core/kernels/common/quantized_matmul_common.h"
#include "itex/core/kernels/onednn/block/quantized_ops.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/quantization_util.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

#define REGISTER_NATIVE_KERNEL(op, kernel, bias_type, output_type)     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(op)                                                         \
          .Device(DEVICE_GPU)                                          \
          .TypeConstraint<quint8>("T1")                                \
          .TypeConstraint<qint8>("T2") BIAS_TYPE_CONSTRAINT(bias_type) \
          .TypeConstraint<output_type>("Toutput") HOSTMEMORYLIST,      \
      kernel TEMPLATE_ARGS(GPUDevice, quint8, qint8, bias_type, output_type));

#define REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(op, kernel, output_type) \
  REGISTER_NATIVE_KERNEL(op, kernel, float, output_type)               \
  REGISTER_NATIVE_KERNEL(op, kernel, qint32, output_type);

// Concrete Native MatMul INT8 kernel implementation
#define TEMPLATE_ARGS(Device, quint8, qint8, bias_type, output_type) \
<Device, quint8, qint8, bias_type, output_type>

#define BIAS_TYPE_CONSTRAINT(bias_type)
#define HOSTMEMORYLIST                                 \
  .HostMemoryList4("min_a", "max_a", "min_b", "max_b") \
      .HostMemoryList2("min_out", "max_out")
REGISTER_NATIVE_KERNEL("QuantizedMatMulWithBiasAndRelu", QuantizedMatMulReluOp,
                       float, qint32);
#undef HOSTMEMORYLIST
#undef BIAS_TYPE_CONSTRAINT

#define BIAS_TYPE_CONSTRAINT(bias_type) .TypeConstraint<bias_type>("Tbias")
#define HOSTMEMORYLIST                                 \
  .HostMemoryList4("min_a", "max_a", "min_b", "max_b") \
      .HostMemoryList2("min_out", "max_out")
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES("QuantizedMatMulWithBias",
                                      QuantizedMatMulOp, qint32);
#undef HOSTMEMORYLIST

#define HOSTMEMORYLIST                                                       \
  .HostMemoryList6("min_a", "max_a", "min_b", "max_b", "min_freezed_output", \
                   "max_freezed_output")                                     \
      .HostMemoryList2("min_out", "max_out")
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "QuantizedMatMulWithBiasAndReluAndRequantize", QuantizedMatMulReluOp,
    quint8);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES("QuantizedMatMulWithBiasAndRequantize",
                                      QuantizedMatMulOp, quint8);
#undef HOSTMEMORYLIST

#define HOSTMEMORYLIST                                                       \
  .HostMemoryList6("min_a", "max_a", "min_b", "max_b", "min_freezed_output", \
                   "max_freezed_output")
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES("QuantizedMatMulWithBiasAndDequantize",
                                      QuantizedMatMulOp, float);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES("QuantizedMatMulWithBiasAndDequantize",
                                      QuantizedMatMulOp, Eigen::bfloat16);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES("QuantizedMatMulWithBiasAndDequantize",
                                      QuantizedMatMulOp, Eigen::half);

// _ITEXQuantizedMatMulWithBiasAndDequantize is used to replace
// QuantizedMatMulWithBiasAndDequantize in frozen graph. Since TF Proper
// QuantizedMatMulWithBiasAndDequantize op doesn't alloc bf16 output, while
// spr-base allows.
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "_ITEXQuantizedMatMulWithBiasAndDequantize", QuantizedMatMulOp, float);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "_ITEXQuantizedMatMulWithBiasAndDequantize", QuantizedMatMulOp,
    Eigen::bfloat16);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "_ITEXQuantizedMatMulWithBiasAndDequantize", QuantizedMatMulOp,
    Eigen::half);
#undef HOSTMEMORYLIST
#undef BIAS_TYPE_CONSTRAINT

#undef TEMPLATE_ARGS
#undef REGISTER_NATIVE_KERNEL

#define REGISTER_NATIVE_KERNEL(op, kernel, input_type, args_type, output_type) \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(op)                                                                 \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<input_type>("T1")                                    \
          .TypeConstraint<qint8>("T2")                                         \
          .TypeConstraint<args_type>("Targs")                                  \
          .TypeConstraint<output_type>("Toutput") HOSTMEMORYLIST,              \
      kernel TEMPLATE_ARGS(GPUDevice, input_type, args_type, output_type));

#define REGISTER_NATIVE_KERNEL_ALL_INPUT_TYPES(op, kernel, args_type, \
                                               output_type)           \
  REGISTER_NATIVE_KERNEL(op, kernel, qint8, args_type, output_type);  \
  REGISTER_NATIVE_KERNEL(op, kernel, quint8, args_type, output_type);

#define REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(op, kernel,        \
                                                        output_type)       \
  REGISTER_NATIVE_KERNEL_ALL_INPUT_TYPES(op, kernel, float, output_type);  \
  REGISTER_NATIVE_KERNEL_ALL_INPUT_TYPES(op, kernel, Eigen::bfloat16,      \
                                         output_type);                     \
  REGISTER_NATIVE_KERNEL_ALL_INPUT_TYPES(op, kernel, Eigen::half,          \
                                         output_type);                     \
  REGISTER_NATIVE_KERNEL_ALL_INPUT_TYPES(op, kernel, qint32, output_type); \
  REGISTER_NATIVE_KERNEL_ALL_INPUT_TYPES(op, kernel, quint8, output_type); \
  REGISTER_NATIVE_KERNEL_ALL_INPUT_TYPES(op, kernel, qint8, output_type);

// Concrete Native MatMul INT8 V1 (deprecated) kernel implementation
#define TEMPLATE_ARGS(Device, input_type, args_type, output_type) \
<Device, input_type, qint8, args_type, output_type>

#define HOSTMEMORYLIST                                 \
  .HostMemoryList4("min_a", "max_a", "min_b", "max_b") \
      .HostMemoryList2("min_product", "max_product")
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedFusedMatMul",
                                                QuantizedFusedMatMulOp, qint32);
#undef HOSTMEMORYLIST

#define HOSTMEMORYLIST .HostMemoryList4("min_a", "max_a", "min_b", "max_b")
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp, float);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp,
    Eigen::bfloat16);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp, Eigen::half);
#undef HOSTMEMORYLIST

#define HOSTMEMORYLIST                                                       \
  .HostMemoryList6("min_a", "max_a", "min_b", "max_b", "min_freezed_output", \
                   "max_freezed_output")                                     \
      .HostMemoryList2("min_product", "max_product")
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndRequantize", QuantizedFusedMatMulOp, quint8);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndRequantize", QuantizedFusedMatMulOp, qint8);
#undef HOSTMEMORYLIST
#undef TEMPLATE_ARGS

#undef REGISTER_NATIVE_KERNEL

#define REGISTER_NATIVE_KERNEL(op, kernel, input_type, args_type, output_type) \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(op)                                                                 \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<input_type>("T1")                                    \
          .TypeConstraint<qint8>("T2")                                         \
          .TypeConstraint<args_type>("Tbias")                                  \
          .TypeConstraint<output_type>("Tout") HOSTMEMORYLIST,                 \
      kernel TEMPLATE_ARGS(GPUDevice, input_type, args_type, output_type));

// Concrete Native MatMul INT8 V2 (latest) kernel implementation
#define TEMPLATE_ARGS(Device, input_type, args_type, output_type) \
<Device, input_type, qint8, args_type, output_type>

#define HOSTMEMORYLIST .HostMemoryList2("host_inputs", "host_outputs")
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                qint32);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                float);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                Eigen::bfloat16);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                qint8);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                quint8);
#undef HOSTMEMORYLIST
#undef TEMPLATE_ARGS

#undef REGISTER_NATIVE_KERNEL

}  // namespace itex
