/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/snapshot_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename Device, typename Scalar>
class SnapshotOp : public OpKernel {
 public:
  explicit SnapshotOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    // Try to use buffer forwarding to avoid an explicit copy.
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (!output->SharesBufferWith(input)) {
      functor::Snapshot<Device, Scalar> functor;
      functor(context->eigen_gpu_device(), input.flat<Scalar>(),
              output->flat<Scalar>());
    }
  }
};

typedef Eigen::GpuDevice GPUDevice;
#define REGISTER_KERNEL(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Snapshot").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      SnapshotOp<GPUDevice, TYPE>)

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL);
TF_CALL_complex128(REGISTER_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_complex64(REGISTER_KERNEL);

TF_CALL_INTEGRAL_TYPES(REGISTER_KERNEL);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
TF_CALL_bool(REGISTER_KERNEL);
#undef REGISTER_KERNEL
}  // namespace itex
