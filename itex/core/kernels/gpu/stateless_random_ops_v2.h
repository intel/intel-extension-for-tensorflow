/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_OPS_V2_H_
#define ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_OPS_V2_H_

#include "itex/core/kernels/common/random_ops_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

inline Status CheckKeyCounterShape(Algorithm const& alg,
                                   TensorShape const& key_shape,
                                   TensorShape const& counter_shape) {
  if (!(key_shape.dims() == 1 && key_shape.dim_size(0) == RNG_KEY_SIZE)) {
    return errors::InvalidArgument(
        "key must have shape [", RNG_KEY_SIZE, "], not ",
        key_shape.DebugString(),
        ". (Note that batched keys are not supported yet.)");
  }
  auto counter_size = GetCounterSize(alg);
  if (!(counter_shape.dims() == 1 &&
        counter_shape.dim_size(0) >= counter_size)) {
    return errors::InvalidArgument(
        "counter must be a vector with length at least ", counter_size,
        "; got shape: ", counter_shape.DebugString(),
        ". (Note that batched counters are not supported yet.)");
  }
  return Status::OK();
}

}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_OPS_V2_H_
