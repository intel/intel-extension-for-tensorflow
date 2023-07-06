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

#include "itex/core/kernels/common/quantize_op.h"

#include "itex/core/utils/register_types.h"

namespace itex {

#define REGISTER_KERNEL(src_type, dst_type)                      \
  REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizeV2")                \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<src_type>("dtype") \
                              .TypeConstraint<dst_type>("T"),    \
                          QuantizeV2Op<CPUDevice, dst_type, src_type>);

REGISTER_KERNEL(float, qint8);
REGISTER_KERNEL(float, quint8);
REGISTER_KERNEL(Eigen::bfloat16, qint8);
REGISTER_KERNEL(Eigen::bfloat16, quint8);
REGISTER_KERNEL(Eigen::half, qint8);
REGISTER_KERNEL(Eigen::half, quint8);
#undef REGISTER_KERNEL

}  // namespace itex
