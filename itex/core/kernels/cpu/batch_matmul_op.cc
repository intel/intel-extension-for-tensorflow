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

#include "itex/core/kernels/common/batch_matmul_op.h"

namespace itex {

#define REGISTER_BATCH_MATMUL_CPU(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXBatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      BatchMatMulOp<CPUDevice, TYPE, TYPE, TYPE>);                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXBatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      BatchMatMulOp<CPUDevice, TYPE, TYPE, TYPE>);                             \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchMatMulV2")                      \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<TYPE>("T"),                      \
                          BatchMatMulOp<CPUDevice, TYPE, TYPE, TYPE>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_BATCH_MATMUL_CPU);
#undef REGISTER_MATMUL_CPU

// Register the intermediate kernel since graph won't be rewritten if nodes
// number < 4.
#define REGISTER_INTERMEDIATE_CPU_OP(TYPE)                \
  REGISTER_KERNEL_BUILDER(Name("_FusedBatchMatMulV2")     \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          BatchMatMulOp<CPUDevice, TYPE, TYPE, TYPE>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_INTERMEDIATE_CPU_OP);
#undef REGISTER_INTERMEDIATE_CPU_OP

// BatchMatMul INT8 kernel registration
#define REGISTER_NATIVE_KERNEL(op, kernel, lhs_type, rhs_type, output_type,   \
                               is_v2, output_type_name)                       \
  REGISTER_KERNEL_BUILDER(Name(op)                                            \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<lhs_type>("T1")                 \
                              .TypeConstraint<rhs_type>("T2")                 \
                              .TypeConstraint<output_type>(output_type_name), \
                          kernel TEMPLATE_ARGS(CPUDevice, lhs_type, rhs_type, \
                                               output_type, is_v2));

#define REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, output_type, \
                                                 is_v2, output_type_name) \
  REGISTER_NATIVE_KERNEL(op, kernel, qint8, qint8, output_type, is_v2,    \
                         output_type_name);

#define REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES(op, kernel, is_v2,             \
                                                output_type_name)              \
  REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, float, is_v2,           \
                                           output_type_name);                  \
  REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, Eigen::bfloat16, is_v2, \
                                           output_type_name);

// Concrete Native BatchMatMul INT8 V1 API (deprecated) kernel implementation
#define TEMPLATE_ARGS(Device, lhs_type, rhs_type, output_type, is_v2) \
<Device, lhs_type, rhs_type, output_type, is_v2>

REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES("_QuantizedBatchMatMulV2AndDequantize",
                                        QuantizedBatchMatMulV2Op, false,
                                        "Toutput");
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES(
    "_QuantizedFusedBatchMatMulV2AndDequantize", QuantizedBatchMatMulV2Op,
    false, "Toutput");
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES(
    "_ITEXQuantizedBatchMatMulV2AndDequantize", QuantizedBatchMatMulV2Op, false,
    "Toutput");
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES(
    "_ITEXQuantizedFusedBatchMatMulV2AndDequantize", QuantizedBatchMatMulV2Op,
    false, "Toutput");

// Concrete Native BatchMatMul INT8 V2 API (latest) kernel implementation
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES("_QuantizedBatchMatMul",
                                        QuantizedBatchMatMulV2Op, true, "Tout");
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES("_ITEXQuantizedBatchMatMul",
                                        QuantizedBatchMatMulV2Op, true, "Tout");
#undef TEMPLATE_ARGS

#undef REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES
#undef REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES
#undef REGISTER_NATIVE_KERNEL

}  // namespace itex
