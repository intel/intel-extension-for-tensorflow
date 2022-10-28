/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_COMMON_FUSED_BATCH_NORM_FUNCTOR_H_
#define ITEX_CORE_KERNELS_COMMON_FUSED_BATCH_NORM_FUNCTOR_H_

#include <string>

#include "itex/core/utils/op_kernel.h"

namespace itex {

template <class T, class U, bool reserved_space, bool is_batch_norm_ex>
class VarAdjust;

template <class T, class U, bool reserved_space, bool is_batch_norm_ex>
class VarAdjustMinus;

namespace functor {
// FusedBatchNormEx op supports side inputs and activations:
// (1) batch_norm + activation
// (2) batch norm + side input + activation
enum class FusedBatchNormActivationMode { kIdentity, kRelu, kReluGrad };

string ToString(FusedBatchNormActivationMode);

Status ParseActivationMode(OpKernelConstruction*,
                           FusedBatchNormActivationMode*);
}  // namespace functor

using FbnActivationMode = functor::FusedBatchNormActivationMode;

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_FUSED_BATCH_NORM_FUNCTOR_H_
