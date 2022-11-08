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

#include "itex/core/kernels/common/transpose_op.h"

#include "itex/core/utils/register_types.h"
#include "itex/core/utils/register_types_traits.h"

namespace itex {

#define REGISTER(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ITEXTranspose").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TransposeOp<CPUDevice, T>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER);

#undef REGISTER

#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizedTranspose") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          QuantizedTransposeOp<CPUDevice, TYPE>);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace itex
