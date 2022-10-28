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

#include "itex/core/kernels/common/matmul_op.h"

#include "itex/core/kernels/common/batch_matmul_op.h"
#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

using dnnl::memory;

#define REGISTER_MATMUL_GPU(TYPE)                                            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("BatchMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),      \
      BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE>);                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("BatchMatMulV2").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),    \
      BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE>);                           \
  REGISTER_KERNEL_BUILDER(Name("_FusedBatchMatMulV2")                        \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<TYPE>("T"),                    \
                          BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE>);       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),           \
      MatMulOp<GPUDevice, TYPE, TYPE, TYPE>)                                 \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXFusedMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      MatMulOp<GPUDevice, TYPE, TYPE, TYPE>)                                 \
  REGISTER_KERNEL_BUILDER(Name("_FusedMatMulWithSum")                        \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<TYPE>("T"),                    \
                          MatMulOp<GPUDevice, TYPE, TYPE, TYPE>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_MATMUL_GPU);
#undef REGISTER_MATMUL_GPU

#define REGISTER_MATMUL_GRAD_GPU(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_FusedMatMulGrad").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      FusedMatMulGradOp<GPUDevice, TYPE, TYPE>)
TF_CALL_float(REGISTER_MATMUL_GRAD_GPU);
TF_CALL_bfloat16(REGISTER_MATMUL_GRAD_GPU);
#undef REGISTER_MATMUL_GRAD_GPU

#define REGISTER_BF32MATMUL_GPU(TYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("_ITEXAccMatMul")                    \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, float>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMul")               \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, float>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulWithSum")        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, float>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_BF32MATMUL_GPU);
#undef REGISTER_BF32MATMUL_GPU

#define REGISTER_BF32MATMUL_GPU(TYPE)                                         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXAccMatMul")                              \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, Eigen::bfloat16>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMul")                         \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, Eigen::bfloat16>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulGrad")                     \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tgrad"),                \
                          FusedMatMulGradOp<GPUDevice, TYPE, float>);         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulWithSum")                  \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, Eigen::bfloat16>);

TF_CALL_bfloat16(REGISTER_BF32MATMUL_GPU);
#undef REGISTER_BF32MATMUL_GPU

// Concrete Native BatchMatMul INT8 V1 API (deprecated) kernel implementation
#define REGISTER_NATIVE_KERNEL(op, kernel, lhs_type, rhs_type, output_type,   \
                               is_v2, output_type_name)                       \
  REGISTER_KERNEL_BUILDER(Name(op)                                            \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<lhs_type>("T1")                 \
                              .TypeConstraint<rhs_type>("T2")                 \
                              .TypeConstraint<output_type>(output_type_name)  \
                                  HOSTMEMORYLIST,                             \
                          kernel TEMPLATE_ARGS(GPUDevice, lhs_type, rhs_type, \
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
                                           output_type_name);                  \
  REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, Eigen::half, is_v2,     \
                                           output_type_name);

// Concrete Native BatchMatMul INT8 V2 API (latest) kernel implementation
#define TEMPLATE_ARGS(Device, lhs_type, rhs_type, output_type, is_v2) \
<Device, lhs_type, rhs_type, output_type, is_v2>
#define HOSTMEMORYLIST .HostMemoryList4("min_x", "max_x", "min_y", "max_y")
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES("_QuantizedBatchMatMulV2AndDequantize",
                                        QuantizedBatchMatMulV2Op, false,
                                        "Toutput");
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES(
    "_QuantizedFusedBatchMatMulV2AndDequantize", QuantizedBatchMatMulV2Op,
    false, "Toutput");
#undef HOSTMEMORYLIST

#define HOSTMEMORYLIST .HostMemoryList2("host_inputs", "host_outputs")
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES("_QuantizedBatchMatMul",
                                        QuantizedBatchMatMulV2Op, true, "Tout");
#undef HOSTMEMORYLIST

#undef TEMPLATE_ARGS

#undef REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES
#undef REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES
#undef REGISTER_NATIVE_KERNEL

}  // namespace itex
