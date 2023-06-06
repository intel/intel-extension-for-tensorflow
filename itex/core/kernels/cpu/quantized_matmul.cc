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

#include "itex/core/kernels/common/quantized_matmul_common.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

#define REGISTER_NATIVE_KERNEL(op, kernel, bias_type, output_type)     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(op)                                                         \
          .Device(DEVICE_CPU)                                          \
          .TypeConstraint<quint8>("T1")                                \
          .TypeConstraint<qint8>("T2") BIAS_TYPE_CONSTRAINT(bias_type) \
          .TypeConstraint<output_type>("Toutput"),                     \
      kernel TEMPLATE_ARGS(CPUDevice, quint8, qint8, bias_type, output_type));

#define REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(op, kernel, output_type) \
  REGISTER_NATIVE_KERNEL(op, kernel, float, output_type)               \
  REGISTER_NATIVE_KERNEL(op, kernel, qint32, output_type);

// Concrete OneDnn MatMul INT8 kernel implementation
#define TEMPLATE_ARGS(Device, quint8, qint8, bias_type, output_type) \
<Device, quint8, qint8, bias_type, output_type>

#define BIAS_TYPE_CONSTRAINT(bias_type)
REGISTER_NATIVE_KERNEL("_ITEXQuantizedMatMulWithBiasAndRelu",
                       QuantizedMatMulReluOp, float, qint32);
#undef BIAS_TYPE_CONSTRAINT

#define BIAS_TYPE_CONSTRAINT(bias_type) .TypeConstraint<bias_type>("Tbias")
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES("_ITEXQuantizedMatMulWithBias",
                                      QuantizedMatMulOp, qint32);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "_ITEXQuantizedMatMulWithBiasAndReluAndRequantize", QuantizedMatMulReluOp,
    quint8);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "_ITEXQuantizedMatMulWithBiasAndRequantize", QuantizedMatMulOp, quint8);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "_ITEXQuantizedMatMulWithBiasAndDequantize", QuantizedMatMulOp, float);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "_ITEXQuantizedMatMulWithBiasAndDequantize", QuantizedMatMulOp,
    Eigen::bfloat16);
REGISTER_NATIVE_KERNEL_ALL_BIAS_TYPES(
    "_ITEXQuantizedMatMulWithBiasAndDequantize", QuantizedMatMulOp,
    Eigen::half);
#undef BIAS_TYPE_CONSTRAINT

#undef TEMPLATE_ARGS
#undef REGISTER_NATIVE_KERNEL

#define REGISTER_NATIVE_KERNEL(op, kernel, input_type, args_type, output_type) \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(op)                                                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<input_type>("T1")                                    \
          .TypeConstraint<qint8>("T2")                                         \
          .TypeConstraint<args_type>("Targs")                                  \
          .TypeConstraint<output_type>("Toutput"),                             \
      kernel TEMPLATE_ARGS(CPUDevice, input_type, args_type, output_type));

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

// Native rewrite ops
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_ITEXQuantizedFusedMatMul",
                                                QuantizedFusedMatMulOp, qint32);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_ITEXQuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp, float);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_ITEXQuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp,
    Eigen::bfloat16);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_ITEXQuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp,
    Eigen::half);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_ITEXQuantizedFusedMatMulAndRequantize", QuantizedFusedMatMulOp, quint8);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_ITEXQuantizedFusedMatMulAndRequantize", QuantizedFusedMatMulOp, qint8);

// TF original ops
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedFusedMatMul",
                                                QuantizedFusedMatMulOp, qint32);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp, float);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp,
    Eigen::bfloat16);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndDequantize", QuantizedFusedMatMulOp, Eigen::half);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndRequantize", QuantizedFusedMatMulOp, quint8);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES(
    "_QuantizedFusedMatMulAndRequantize", QuantizedFusedMatMulOp, qint8);

#undef TEMPLATE_ARGS

#undef REGISTER_NATIVE_KERNEL

#define REGISTER_NATIVE_KERNEL(op, kernel, input_type, args_type, output_type) \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(op)                                                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<input_type>("T1")                                    \
          .TypeConstraint<qint8>("T2")                                         \
          .TypeConstraint<args_type>("Tbias")                                  \
          .TypeConstraint<output_type>("Tout"),                                \
      kernel TEMPLATE_ARGS(CPUDevice, input_type, args_type, output_type));

// Concrete Native MatMul INT8 V2 (latest) kernel implementation
#define TEMPLATE_ARGS(Device, input_type, args_type, output_type) \
<Device, input_type, qint8, args_type, output_type>

// Native rewrite ops
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_ITEXQuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                qint32);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_ITEXQuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                float);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_ITEXQuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                Eigen::bfloat16);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_ITEXQuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                Eigen::half);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_ITEXQuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                quint8);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_ITEXQuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                qint8);

// TF original ops
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
                                                Eigen::half);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                quint8);
REGISTER_NATIVE_KERNEL_ALL_INPUT_AND_ARGS_TYPES("_QuantizedMatMul",
                                                QuantizedFusedMatMulV2Op,
                                                qint8);

#undef TEMPLATE_ARGS
#undef REGISTER_NATIVE_KERNEL

}  // namespace itex
