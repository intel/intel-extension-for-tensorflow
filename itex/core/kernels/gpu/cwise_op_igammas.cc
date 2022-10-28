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
REGISTER2(BinaryOp, GPU, "Igamma", functor::igamma, float, Eigen::bfloat16);
REGISTER2(BinaryOp, GPU, "IgammaGradA", functor::igamma_grad_a, float,
          Eigen::bfloat16);
REGISTER2(BinaryOp, GPU, "Igammac", functor::igammac, float, Eigen::bfloat16);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER(BinaryOp, GPU, "Igamma", functor::igamma, double);
REGISTER(BinaryOp, GPU, "IgammaGradA", functor::igamma_grad_a, double);
REGISTER(BinaryOp, GPU, "Igammac", functor::igammac, double);
#endif  // ITEX_ENABLE_DOUBLE
}  // namespace itex
