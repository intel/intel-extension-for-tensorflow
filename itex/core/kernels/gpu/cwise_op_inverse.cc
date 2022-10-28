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
#include "itex/core/kernels/common/cwise_ops_gradients.h"

namespace itex {
REGISTER4(UnaryOp, GPU, "Inv", functor::inverse, Eigen::half, float,
          Eigen::bfloat16, itex::int64);
REGISTER3(BinaryOp, GPU, "InvGrad", functor::inverse_grad, Eigen::half,
          Eigen::bfloat16, float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER(UnaryOp, GPU, "Inv", functor::inverse, double);
REGISTER(BinaryOp, GPU, "InvGrad", functor::inverse_grad, double);
#endif  // ITEX_ENABLE_DOUBLE
}  // namespace itex
