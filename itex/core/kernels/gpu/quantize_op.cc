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

#include "itex/core/kernels/common/quantize_op.h"

#include "itex/core/utils/register_types.h"

namespace itex {

#define REGISTER_KERNEL(dst_type)                            \
  REGISTER_KERNEL_BUILDER(Name("QuantizeV2")                 \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<dst_type>("T") \
                              .HostMemory("min_range")       \
                              .HostMemory("max_range")       \
                              .HostMemory("output_min")      \
                              .HostMemory("output_max"),     \
                          QuantizeV2Op<GPUDevice, dst_type, float>);
REGISTER_KERNEL(qint8);
REGISTER_KERNEL(quint8);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(src_type, dst_type)                      \
  REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizeV2")                \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<src_type>("dtype") \
                              .TypeConstraint<dst_type>("T")     \
                              .HostMemory("min_range")           \
                              .HostMemory("max_range")           \
                              .HostMemory("output_min")          \
                              .HostMemory("output_max"),         \
                          QuantizeV2Op<GPUDevice, dst_type, src_type>);
REGISTER_KERNEL(float, qint8);
REGISTER_KERNEL(float, quint8);
REGISTER_KERNEL(Eigen::bfloat16, qint8);
REGISTER_KERNEL(Eigen::bfloat16, quint8);
#undef REGISTER_KERNEL

}  // namespace itex
