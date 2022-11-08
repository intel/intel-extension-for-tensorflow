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

#include "itex/core/kernels/common/transpose_op.h"

#include "itex/core/utils/register_types.h"

namespace itex {

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeOp<GPUDevice, T, true>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_complex64(REGISTER);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER);
TF_CALL_complex128(REGISTER);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER

#define REGISTER_KERNEL(TYPE)                                            \
  REGISTER_KERNEL_BUILDER(Name("_QuantizedTranspose")                    \
                              .Device(DEVICE_GPU)                        \
                              .HostMemoryList3("perm", "min_x", "max_x") \
                              .HostMemoryList2("min_y", "max_y")         \
                              .TypeConstraint<TYPE>("T"),                \
                          QuantizedTransposeOp<GPUDevice, TYPE>);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace itex
