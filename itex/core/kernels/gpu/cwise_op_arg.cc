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
#define REGISTER_COMPLEX(D, R, C)                         \
  REGISTER_KERNEL_BUILDER(Name("Angle")                   \
                              .Device(DEVICE_##D)         \
                              .TypeConstraint<C>("T")     \
                              .TypeConstraint<R>("Tout"), \
                          UnaryOp<D##Device, functor::get_angle<C>>);

REGISTER_COMPLEX(GPU, float, complex64);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_COMPLEX(GPU, double, complex128);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_COMPLEX
}  // namespace itex
