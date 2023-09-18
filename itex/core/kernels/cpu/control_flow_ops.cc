/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/common/control_flow_ops.h"

namespace itex {

void EnterOp::Compute(OpKernelContext* context) {
  context->set_output(0, context->input(0));
}

#define REGISTER_CPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Enter").Device(DEVICE_CPU).TypeConstraint<type>("T"), EnterOp)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_KERNEL);

#undef REGISTER_CPU_KERNEL

void ExitOp::Compute(OpKernelContext* context) {
  context->set_output(0, context->input(0));
}

#define REGISTER_CPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Exit").Device(DEVICE_CPU).TypeConstraint<type>("T"), EnterOp)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_KERNEL);

#undef REGISTER_CPU_KERNEL

void NextIterationOp::Compute(OpKernelContext* context) {
  context->set_output(0, context->input(0));
}

#define REGISTER_CPU_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("NextIteration").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      EnterOp)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_KERNEL);

#undef REGISTER_CPU_KERNEL

}  // namespace itex
