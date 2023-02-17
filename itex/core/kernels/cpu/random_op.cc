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

#include "itex/core/kernels/common/random_op.h"

#include "itex/core/kernels/cpu/random_op_cpu.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/register_types_traits.h"
#include "itex/core/utils/types.h"

namespace itex {

#define REGISTER_RANDOM_KERNEL(TYPE)      \
  REGISTER_KERNEL_BUILDER(                \
      Name("_ITEXRandomUniform")          \
          .Device(DEVICE_CPU)             \
          .TypeConstraint<int32>("T")     \
          .TypeConstraint<TYPE>("dtype"), \
      PCGRandomOp<CPUDevice,              \
                  random::UniformDistribution<random::PCGRandom, TYPE>>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_RANDOM_KERNEL);
#undef REGISTER_RANDOM_KERNEL

}  // namespace itex
