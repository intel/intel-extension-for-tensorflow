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

#include "itex/core/kernels/gpu/broadcast_to_op.h"

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/bcast.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

#define INSTANTIATE_GPU_KERNEL(Type) \
  template struct functor::BroadcastTo<GPUDevice, Type>;
INSTANTIATE_GPU_KERNEL(float);
INSTANTIATE_GPU_KERNEL(Eigen::half);
INSTANTIATE_GPU_KERNEL(bool);
INSTANTIATE_GPU_KERNEL(int64);
INSTANTIATE_GPU_KERNEL(int32);
#undef INSTANTIATE_GPU_KERNEL

template <typename Device, typename T>
class BroadcastToOp : public OpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const TensorShape& input_shape = input_tensor.shape();
    const Tensor& shape_tensor = context->input(1);

    TensorShape output_shape;
    OP_REQUIRES_OK(context, MakeShape(shape_tensor, &output_shape));

    // Handle copy.
    if (output_shape == input_shape) {
      context->set_output(0, input_tensor);
      return;
    }

    OP_REQUIRES(context, input_shape.dims() <= output_shape.dims(),
                errors::InvalidArgument(
                    "Rank of input (", input_shape.dims(),
                    ") must be no greater than rank of output shape (",
                    output_shape.dims(), ")."));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    // Handle empty case.
    if (output_shape.num_elements() == 0) {
      return;
    }

    // Handle broadcast from Scalar.
    const Device& device = context->eigen_device<Device>();
    if (input_shape.dims() == 0) {
      functor::FillFunctor<Device, T>()(device, output_tensor->flat<T>(),
                                        input_tensor.scalar<T>());
      return;
    }

    BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(output_shape),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(context, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", input_shape.DebugString(), " vs. ",
                    output_shape.DebugString()));
    OP_REQUIRES(context, BCast::ToShape(bcast.output_shape()) == output_shape,
                errors::InvalidArgument("Unable to broadcast tensor of shape ",
                                        input_shape, " to tensor of shape ",
                                        output_shape));

    functor::BroadcastTo<Device, T>()(device, context, *output_tensor,
                                      output_shape, input_tensor, input_shape,
                                      bcast);
  }
};

#define REGISTER_KERNEL(type)                            \
  REGISTER_KERNEL_BUILDER(Name("BroadcastTo")            \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("shape"),      \
                          BroadcastToOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
TF_CALL_complex64(REGISTER_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL);
TF_CALL_complex128(REGISTER_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNEL

REGISTER_KERNEL_BUILDER(Name("BroadcastTo")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("shape")
                            .HostMemory("output"),
                        BroadcastToOp<CPUDevice, int32>);
}  // namespace itex
