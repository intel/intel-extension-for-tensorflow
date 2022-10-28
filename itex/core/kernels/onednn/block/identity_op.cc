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

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {
template <typename Device, typename T>
class OneDnnIdentityOp : public OpKernel {
 public:
  ~OneDnnIdentityOp() {}
  explicit OneDnnIdentityOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const size_t kSrcIdx = 0;
    const size_t kOutputIdx = 0;
    OneDnnShape src_onednn_shape;
    GetOneDnnShape(context, kSrcIdx, &src_onednn_shape);
    context->set_output(kOutputIdx, context->input(kSrcIdx));
    ForwardMetaData(context, kSrcIdx, kOutputIdx, src_onednn_shape);
  }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_IDENTITY(T)                           \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnIdentity")      \
                              .Device(DEVICE_GPU)      \
                              .HostMemory("x_meta")    \
                              .HostMemory("y_meta")    \
                              .TypeConstraint<T>("T"), \
                          OneDnnIdentityOp<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_IDENTITY);
#undef REGISTER_IDENTITY
#else
#define REGISTER_IDENTITY(T)                                             \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_OneDnnIdentity").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OneDnnIdentityOp<CPUDevice, T>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_IDENTITY);
#undef REGISTER_IDENTITY
#endif
}  // namespace itex
