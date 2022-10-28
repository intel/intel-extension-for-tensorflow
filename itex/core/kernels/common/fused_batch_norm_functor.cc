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

#include "itex/core/kernels/common/fused_batch_norm_functor.h"
#include "itex/core/utils/op_requires.h"

namespace itex {
namespace functor {

Status ParseActivationMode(OpKernelConstruction* context,
                           FusedBatchNormActivationMode* activation_mode) {
  string activation_mode_str;
  if (context->HasAttr("activation_mode"))
    TF_RETURN_IF_ERROR(
        context->GetAttr("activation_mode", &activation_mode_str));

  if (activation_mode_str == "Identity") {
    *activation_mode = FusedBatchNormActivationMode::kIdentity;
    return Status::OK();
  }
  if (activation_mode_str == "Relu") {
    *activation_mode = FusedBatchNormActivationMode::kRelu;
    return Status::OK();
  }
  if (activation_mode_str == "ReluGrad") {
    *activation_mode = FusedBatchNormActivationMode::kReluGrad;
    return Status::OK();
  }
  return errors::InvalidArgument("Unsupported activation mode: ",
                                 activation_mode_str);
}
}  // namespace functor
}  // namespace itex
