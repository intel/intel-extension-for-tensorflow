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

#include "itex/core/kernels/common/softmax_op.h"

namespace itex {
#define REGISTER_KERNEL(TYPE)                                            \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ITEXSoftmax").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      SoftmaxOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace itex
