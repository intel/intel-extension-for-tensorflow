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

#include "itex/core/kernels/common/layer_norm_op.h"

namespace itex {

#define REGISTER_LAYERNORM_CPU(T, U)                     \
  REGISTER_KERNEL_BUILDER(Name("_ITEXLayerNorm")         \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<T>("T")    \
                              .TypeConstraint<U>("U"),   \
                          LayerNormOp<CPUDevice, T, U>); \
  REGISTER_KERNEL_BUILDER(Name("LayerNorm")              \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<T>("T")    \
                              .TypeConstraint<U>("U"),   \
                          LayerNormOp<CPUDevice, T, U>);
REGISTER_LAYERNORM_CPU(float, float);
REGISTER_LAYERNORM_CPU(Eigen::bfloat16, float);
#undef REGISTER_LAYERNORM_CPU

#define REGISTER_LAYERNORM_GRAD_CPU(T, U)                    \
  REGISTER_KERNEL_BUILDER(Name("_ITEXLayerNormGrad")         \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<U>("U"),       \
                          LayerNormGradOp<CPUDevice, T, U>); \
  REGISTER_KERNEL_BUILDER(Name("LayerNormGrad")              \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<U>("U"),       \
                          LayerNormGradOp<CPUDevice, T, U>);
REGISTER_LAYERNORM_GRAD_CPU(float, float);
REGISTER_LAYERNORM_GRAD_CPU(Eigen::bfloat16, float);
#undef REGISTER_LAYERNORM_GRAD_CPU

#define REGISTER_MKLLAYERNORM_CPU(T, U)                                    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXMklLayerNorm").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LayerNormOp<CPUDevice, T, U, true>);
REGISTER_MKLLAYERNORM_CPU(float, float);
REGISTER_MKLLAYERNORM_CPU(Eigen::bfloat16, Eigen::bfloat16);
#undef REGISTER_MKLLAYERNORM_CPU

}  // namespace itex
