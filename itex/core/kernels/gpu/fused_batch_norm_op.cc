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

#include "itex/core/kernels/common/fused_batch_norm_op.h"

#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"
#include "itex/core/kernels/gpu/custom_fused_batch_norm_functor.h"
#include "itex/core/kernels/gpu/custom_fused_batch_norm_op.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

using FbnActivationMode = functor::FusedBatchNormActivationMode;

#define REGISTER_FUSED_BATCHNORM_GPU(T)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("FusedBatchNorm").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FusedBatchNormOp<GPUDevice, T, float, false, false>)

REGISTER_FUSED_BATCHNORM_GPU(float);
REGISTER_FUSED_BATCHNORM_GPU(Eigen::bfloat16);
REGISTER_FUSED_BATCHNORM_GPU(Eigen::half);
#undef REGISTER_FUSED_BATCHNORM_GPU

#define REGISTER_FUSED_BATCHNORM_GPU(T, U)                   \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("FusedBatchNormV2")                               \
          .Device(DEVICE_GPU)                                \
          .TypeConstraint<T>("T")                            \
          .TypeConstraint<U>("U"),                           \
      CustomFusedBatchNormOp<GPUDevice, T, U, false, false>) \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("FusedBatchNormV3")                               \
          .Device(DEVICE_GPU)                                \
          .TypeConstraint<T>("T")                            \
          .TypeConstraint<U>("U"),                           \
      CustomFusedBatchNormOp<GPUDevice, T, U, true, false>)  \
  REGISTER_KERNEL_BUILDER(Name("_FusedBatchNormEx")          \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<U>("U"),       \
                          CustomFusedBatchNormOp<GPUDevice, T, U, true, true>)

REGISTER_FUSED_BATCHNORM_GPU(float, float);
REGISTER_FUSED_BATCHNORM_GPU(Eigen::bfloat16, float);
REGISTER_FUSED_BATCHNORM_GPU(Eigen::half, float);
#undef REGISTER_FUSED_BATCHNORM_GPU

#define REGISTER_FUSED_BATCHNORM_GRAD_GPU(T, U)                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("FusedBatchNormGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CustomFusedBatchNormGradOp<GPUDevice, T, U, false, false>)            \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("FusedBatchNormGradV2")                                          \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T")                                           \
          .TypeConstraint<U>("U"),                                          \
      CustomFusedBatchNormGradOp<GPUDevice, T, U, false, false>)            \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("FusedBatchNormGradV3")                                          \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T")                                           \
          .TypeConstraint<U>("U"),                                          \
      CustomFusedBatchNormGradOp<GPUDevice, T, U, true, false>)             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedBatchNormExGrad")                                         \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T")                                           \
          .TypeConstraint<U>("U"),                                          \
      CustomFusedBatchNormGradOp<GPUDevice, T, U, true, true>)

REGISTER_FUSED_BATCHNORM_GRAD_GPU(float, float);
REGISTER_FUSED_BATCHNORM_GRAD_GPU(Eigen::bfloat16, float);
#undef REGISTER_FUSED_BATCHNORM_GRAD_GPU

REGISTER_KERNEL_BUILDER(
    Name("_QuantizedFusedBatchNorm")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("T")
        .TypeConstraint<float>("U")
        .TypeConstraint<qint8>("Tout"),
    QuantizedFusedBatchNormOp<GPUDevice, qint8, float, false, true>);

}  // namespace itex
