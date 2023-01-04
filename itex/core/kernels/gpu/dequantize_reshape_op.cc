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

#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/quantization_util.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class DequantizeReshapeOp : public OpKernel {
 public:
  explicit DequantizeReshapeOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    string error_msg = "This regsitered is a no Op, should never be executed !";
    OP_REQUIRES_OK(context, errors::Aborted("Operation received an exception:",
                                            error_msg));
  }
};

#define REGISTER_KERNEL(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedDequantizeWithReshape") \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<TYPE>("T")          \
                              .HostMemory("min_range")            \
                              .HostMemory("max_range")            \
                              .HostMemory("shape"),               \
                          DequantizeReshapeOp<GPUDevice, TYPE>);

TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#undef REGISTER_KERNEL
}  // namespace itex
