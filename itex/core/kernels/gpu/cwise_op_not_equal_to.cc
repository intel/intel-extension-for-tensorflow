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
REGISTER2(BinaryOp, GPU, "NotEqual", functor::not_equal_to, itex::uint8, bool);
REGISTER3(BinaryOp, GPU, "NotEqual", functor::not_equal_to, float, Eigen::half,
          Eigen::bfloat16);

REGISTER2(BinaryOp, GPU, "NotEqual", functor::not_equal_to, itex::int64,
          itex::complex64);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("NotEqual")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::not_equal_to<int32>>);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER2(BinaryOp, GPU, "NotEqual", functor::not_equal_to, double,
          itex::complex128);
#endif  // ITEX_ENABLE_DOUBLE

REGISTER3(BinaryOp, GPU, "_NotEqualWithCast", functor::not_equal_to_with_cast,
          float, Eigen::half, Eigen::bfloat16);
}  // namespace itex
