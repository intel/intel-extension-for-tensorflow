/* Copyright (c) 2021-2023 Intel Corporation
Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#include "itex/core/kernels/gpu/test_ops.h"

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/bcast.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
class SleepGpuOp : public OpKernel {
 public:
  explicit SleepGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    sleep_kernel(ctx->input(0).scalar<int>()());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SleepOp").Device(DEVICE_GPU).HostMemory("sleep_seconds"), SleepGpuOp);

}  // namespace itex
