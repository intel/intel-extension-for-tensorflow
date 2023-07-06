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

#include "itex/core/kernels/common/cwise_ops_common.h"

namespace itex {
REGISTER5(BinaryOp, GPU, "Equal", functor::equal_to, int64, bool, int16, uint8,
          int8);

REGISTER4(BinaryOp, GPU, "Equal", functor::equal_to, float, Eigen::half,
          Eigen::bfloat16, complex64);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER2(BinaryOp, GPU, "Equal", functor::equal_to, double, complex128);
#endif  // ITEX_ENABLE_DOUBLE

REGISTER3(BinaryOp, GPU, "_ITEXEqualWithCast", functor::equal_to_with_cast,
          float, Eigen::half, Eigen::bfloat16);

REGISTER_KERNEL_BUILDER(Name("Equal")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::equal_to<int32>>);
}  // namespace itex
