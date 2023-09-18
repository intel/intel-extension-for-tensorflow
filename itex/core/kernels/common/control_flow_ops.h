/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_COMMON_CONTROL_FLOW_OPS_H_
#define ITEX_CORE_KERNELS_COMMON_CONTROL_FLOW_OPS_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"

namespace itex {

// An enter op has one input and one output. It creates or finds
// the child frame that is uniquely identified by the frame_name,
// and makes its input available to the child frame.
class EnterOp : public OpKernel {
 public:
  explicit EnterOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() { return false; }
  ~EnterOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(EnterOp);
};

// An exit op has one input and one output. It exits the current
// frame to its parent frame, and makes its input available to the
// parent frame.
class ExitOp : public OpKernel {
 public:
  explicit ExitOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() { return false; }
  ~ExitOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(ExitOp);
};

// A next_iteration op has one input and one output. It makes its input
// available to the next iteration.
class NextIterationOp : public OpKernel {
 public:
  explicit NextIterationOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() { return false; }
  ~NextIterationOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(NextIterationOp);
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_CONTROL_FLOW_OPS_H_
