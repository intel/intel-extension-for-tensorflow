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

#include "itex/core/kernels/common/quantized_reshape_op.h"

#include "itex/core/utils/register_types.h"

namespace itex {

#define REGISTER_GPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("QuantizedReshape")        \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shape")        \
                              .HostMemory("input_min")    \
                              .HostMemory("input_max")    \
                              .HostMemory("output_min")   \
                              .HostMemory("output_max")   \
                              .TypeConstraint<type>("T"), \
                          QuantizedReshapeOp)

REGISTER_GPU_KERNEL(quint8);
REGISTER_GPU_KERNEL(qint8);
REGISTER_GPU_KERNEL(qint32);

#undef REGISTER_GPU_KERNEL

}  // namespace itex
