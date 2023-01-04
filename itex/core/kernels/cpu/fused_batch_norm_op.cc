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

#include "itex/core/kernels/common/fused_batch_norm_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::ThreadPoolDevice CPUDevice;

#define REGISTER_FUSED_BATCHNORM_CPU(T)                                      \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXFusedBatchNorm").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedBatchNormOp<CPUDevice, T, float, false, false>)

TF_CALL_CPU_NUMBER_TYPES(REGISTER_FUSED_BATCHNORM_CPU)
#undef REGISTER_FUSED_BATCHNORM_CPU

#define REGISTER_FUSED_BATCHNORM_CPU(T, U)                                 \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchNormV2")                    \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<U>("U"),                     \
                          FusedBatchNormOp<CPUDevice, T, U, false, false>) \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchNormV3")                    \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<U>("U"),                     \
                          FusedBatchNormOp<CPUDevice, T, U, true, false>)  \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchNormEx")                    \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<U>("U"),                     \
                          FusedBatchNormOp<CPUDevice, T, U, true, true>)   \
  REGISTER_KERNEL_BUILDER(Name("_FusedBatchNormEx")                        \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<U>("U"),                     \
                          FusedBatchNormOp<CPUDevice, T, U, true, true>)

REGISTER_FUSED_BATCHNORM_CPU(float, float);
REGISTER_FUSED_BATCHNORM_CPU(Eigen::bfloat16, float);
#undef REGISTER_FUSED_BATCHNORM_CPU

#define REGISTER_FUSED_BATCHNORM_GRAD_CPU(T, U)                                \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchNormGrad")                      \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          FusedBatchNormGradOp<CPUDevice, T, U, false, false>) \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchNormGradV2")                    \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<U>("U"),                         \
                          FusedBatchNormGradOp<CPUDevice, T, U, false, false>) \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchNormGradV3")                    \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<U>("U"),                         \
                          FusedBatchNormGradOp<CPUDevice, T, U, true, false>)  \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchNormExGrad")                    \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<U>("U"),                         \
                          FusedBatchNormGradOp<CPUDevice, T, U, true, true>)

REGISTER_FUSED_BATCHNORM_GRAD_CPU(float, float);
REGISTER_FUSED_BATCHNORM_GRAD_CPU(Eigen::bfloat16, float);
#undef REGISTER_FUSED_BATCHNORM_GRAD_CPU

REGISTER_KERNEL_BUILDER(
    Name("_QuantizedFusedBatchNorm")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("T")
        .TypeConstraint<float>("U")
        .TypeConstraint<qint8>("Tout"),
    QuantizedFusedBatchNormOp<CPUDevice, qint8, float, false, true>);

REGISTER_KERNEL_BUILDER(
    Name("_ITEXQuantizedFusedBatchNorm")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("T")
        .TypeConstraint<float>("U")
        .TypeConstraint<qint8>("Tout"),
    QuantizedFusedBatchNormOp<CPUDevice, qint8, float, false, true>);

}  // namespace itex
