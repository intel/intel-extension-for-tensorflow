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

#include "itex/core/kernels/gpu/relu_op.h"

#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"

namespace itex {
// Registration of the forward kernels.
#define REGISTER_GPU_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Relu").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      ReluOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Elu").Device(DEVICE_GPU).TypeConstraint<type>("T"),        \
      EluOp<GPUDevice, type>);                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Relu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      Relu6Op<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LeakyRelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      LeakyReluOp<GPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Gelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      GeluOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ITEXGelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      GeluOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_ITEXMish").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      MishOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Selu").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      SeluOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_ITEXSwish").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SwishOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#define REGISTER_GPU_KERNELS(type)                               \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Selu").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SeluOp<GPUDevice, type>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

#define REGISTER_ELU_GPU_KERNELS(type)                          \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Elu").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      EluOp<GPUDevice, type>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ELU_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_ELU_GPU_KERNELS

#define REGISTER_GRAD_GPU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      ReluGradOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("EluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      EluGradOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("GeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      GeluGradOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ITEXGeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      GeluGradOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      SeluGradOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SwishGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      SwishGradOp<GPUDevice, type>);

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_GRAD_GPU_KERNELS);
#undef REGISTER_GRAD_GPU_KERNELS

#define REGISTER_GPU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SeluGradOp<GPUDevice, type>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

#define REGISTER_ELUGRAD_GPU_KERNELS(type)                          \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("EluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      EluGradOp<GPUDevice, type>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ELUGRAD_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_ELUGRAD_GPU_KERNELS

}  // namespace itex
