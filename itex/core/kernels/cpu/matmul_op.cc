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

#include "itex/core/kernels/common/matmul_op.h"

namespace itex {

#define REGISTER_MATMUL_CPU(TYPE)                                            \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedMatMulWithSum")                    \
                              .Device(DEVICE_CPU)                            \
                              .TypeConstraint<TYPE>("T"),                    \
                          MatMulOp<CPUDevice, TYPE, TYPE, TYPE>);            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXFusedMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      MatMulOp<CPUDevice, TYPE, TYPE, TYPE>);                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),      \
      MatMulOp<CPUDevice, TYPE, TYPE, TYPE>);                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_FusedMatMulGrad").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      FusedMatMulGradOp<CPUDevice, TYPE, TYPE>);                             \
  REGISTER_KERNEL_BUILDER(Name("_FusedMatMulWithSum")                        \
                              .Device(DEVICE_CPU)                            \
                              .TypeConstraint<TYPE>("T"),                    \
                          MatMulOp<CPUDevice, TYPE, TYPE, TYPE>);
// TODO(itex): remove registration of intermediate kernels. Remapper should
// directly generate _ITEXFusedxxx.
TF_CALL_CPU_NUMBER_TYPES(REGISTER_MATMUL_CPU);
#undef REGISTER_MATMUL_CPU

#define REGISTER_BF32MATMUL_CPU(TYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("_ITEXAccMatMul")                    \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<CPUDevice, TYPE, float, float>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMul")               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<CPUDevice, TYPE, float, float>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulWithSum")        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<CPUDevice, TYPE, float, float>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_BF32MATMUL_CPU);
#undef REGISTER_BF32MATMUL_CPU

#define REGISTER_BF32MATMUL_CPU(TYPE)                                         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXAccMatMul")                              \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<CPUDevice, TYPE, float, Eigen::bfloat16>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMul")                         \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<CPUDevice, TYPE, float, Eigen::bfloat16>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulGrad")                     \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tgrad"),                \
                          FusedMatMulGradOp<CPUDevice, TYPE, float>);         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulWithSum")                  \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<CPUDevice, TYPE, float, Eigen::bfloat16>);

TF_CALL_bfloat16(REGISTER_BF32MATMUL_CPU);
#undef REGISTER_BF32MATMUL_CPU

}  // namespace itex
