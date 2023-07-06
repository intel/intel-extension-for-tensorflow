/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/random_op_gpu.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/register_types_traits.h"
#include "itex/core/utils/types.h"

namespace itex {

#define REGISTER_RANDOM_KERNEL(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomUniform")                                                    \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<GPUDevice, random::UniformDistribution<                   \
                                    random::PhiloxRandom, TYPE>>);             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomStandardNormal")                                             \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<GPUDevice,                                                \
                     random::NormalDistribution<random::PhiloxRandom, TYPE>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TruncatedNormal")                                                  \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<                                                          \
          GPUDevice,                                                           \
          random::TruncatedNormalDistribution<                                 \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE>>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_RANDOM_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_RANDOM_KERNEL);
#endif

#define REGISTER_FULL_INT(IntType)           \
  template struct functor::FillPhiloxRandom< \
      GPUDevice,                             \
      random::UniformFullIntDistribution<random::PhiloxRandom, IntType>>

#define REGISTER_INT(IntType)                                                 \
  REGISTER_FULL_INT(IntType);                                                 \
  template struct functor::FillPhiloxRandom<                                  \
      GPUDevice, random::UniformDistribution<random::PhiloxRandom, IntType>>; \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")                            \
                              .Device(DEVICE_GPU)                             \
                              .HostMemory("shape")                            \
                              .HostMemory("minval")                           \
                              .HostMemory("maxval")                           \
                              .TypeConstraint<int32>("T")                     \
                              .TypeConstraint<IntType>("Tout"),               \
                          RandomUniformIntOp<GPUDevice, IntType>);

TF_CALL_int32(REGISTER_INT);
TF_CALL_int64(REGISTER_INT);

}  // namespace itex
