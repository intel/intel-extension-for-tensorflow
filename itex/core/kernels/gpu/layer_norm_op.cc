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
#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"
namespace itex {

#define REGISTER_LAYERNORM_GPU(T, U)                   \
  REGISTER_KERNEL_BUILDER(Name("LayerNorm")            \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          LayerNormOp<GPUDevice, T, U>);
REGISTER_LAYERNORM_GPU(float, float);
REGISTER_LAYERNORM_GPU(Eigen::bfloat16, float);
REGISTER_LAYERNORM_GPU(Eigen::half, float);
#undef REGISTER_LAYERNORM_GPU

#define REGISTER_LAYERNORM_GRAD_GPU(T, U)              \
  REGISTER_KERNEL_BUILDER(Name("LayerNormGrad")        \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          LayerNormGradOp<GPUDevice, T, U>);
REGISTER_LAYERNORM_GRAD_GPU(float, float);
REGISTER_LAYERNORM_GRAD_GPU(Eigen::bfloat16, float);
#undef REGISTER_LAYERNORM_GRAD_CPU

#define REGISTER_MKLLAYERNORM_GPU(T, U)                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_MklLayerNorm").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LayerNormOp<GPUDevice, T, U, true>);
REGISTER_MKLLAYERNORM_GPU(float, float);
REGISTER_MKLLAYERNORM_GPU(Eigen::bfloat16, Eigen::bfloat16);
// queue->fill<half> is not supported
// REGISTER_MKLLAYERNORM_GPU(Eigen::half, Eigen::half);
#undef REGISTER_MKLLAYERNORM_GPU

}  // namespace itex
