/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_OPS_H_
#define ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_OPS_H_

#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/plugin_tensor.h"

namespace itex {

// Generates a key and counter that can be used to seed a PhiloxRandom,
// generator, based on the seed value in `seed_t`.
//
// REQUIRES: `seed_t` must be a length-2 vector of type DT_INT{32,64}.
// `out_key` and `out_counter` must be non-null.
Status GenerateKey(Tensor seed_t, random::PhiloxRandom::Key* out_key,
                   random::PhiloxRandom::ResultType* out_counter);

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_OPS_H_
