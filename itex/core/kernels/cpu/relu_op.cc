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

#include "itex/core/kernels/common/relu_op.h"

namespace itex {

// Registration of the forward kernels.
#define REGISTER_CPU_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXElu").Device(DEVICE_CPU).TypeConstraint<type>("T"),       \
      EluOp<CPUDevice, type>);                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXRelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluOp<CPUDevice, type>);                                            \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXRelu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6Op<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXLeakyRelu").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LeakyReluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXGelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      GeluOp<CPUDevice, type>);                                            \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXSwish").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SwishOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXMish").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      MishOp<CPUDevice, type>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#define REGISTER_GRAD_CPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXEluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),       \
      EluGradOp<CPUDevice, type>);                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluGradOp<CPUDevice, type>);                                            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXRelu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXLeakyReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ITEXGeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      GeluGradOp<CPUDevice, type>);                                            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SwishGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      SwishGradOp<CPUDevice, type>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_GRAD_CPU_KERNELS);
#undef REGISTER_GRAD_CPU_KERNELS

// Custom ops cannot be registered as no op and then rewrite into _OneDnn or
// _ITEX. Graph optimization won't be run if nodes < 4.
#define REGISTER_GELU_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Gelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      GeluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("GeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      GeluGradOp<CPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Swish").Device(DEVICE_CPU).TypeConstraint<type>("T"),    \
      SwishOp<CPUDevice, type>);                                     \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Mish").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      MishOp<CPUDevice, type>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_GELU_KERNELS);
#undef REGISTER_GELU_KERNELS
}  // namespace itex
