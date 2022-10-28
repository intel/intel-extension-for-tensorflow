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

#ifndef ITEX_CORE_KERNELS_COMMON_NO_OPS_H_
#define ITEX_CORE_KERNELS_COMMON_NO_OPS_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

class NoOp : public OpKernel {
 public:
  explicit NoOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

class NoImplementOp : public OpKernel {
 public:
  explicit NoImplementOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, false,
                errors::Unimplemented(this->name(), " ", this->type(),
                                      " op is not implemented"));
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_NO_OPS_H_
