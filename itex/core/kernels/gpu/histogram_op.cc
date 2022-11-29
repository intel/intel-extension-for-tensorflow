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

#include "itex/core/kernels/gpu/histogram_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/platform_types.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tout>
class HistogramFixedWidthOp : public OpKernel {
 public:
  explicit HistogramFixedWidthOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& values_tensor = ctx->input(0);
    const Tensor& value_range_tensor = ctx->input(1);
    const Tensor& nbins_tensor = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(value_range_tensor.shape()),
                errors::InvalidArgument("value_range should be a vector."));
    OP_REQUIRES(ctx, (value_range_tensor.shape().num_elements() == 2),
                errors::InvalidArgument(
                    "value_range should be a vector of 2 elements."));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(nbins_tensor.shape()),
                errors::InvalidArgument("nbins should be a scalar."));

    const auto values = values_tensor.flat<T>();
    const auto value_range = value_range_tensor.flat<T>();
    const auto nbins = nbins_tensor.scalar<int32>()();

    OP_REQUIRES(
        ctx, (value_range(0) < value_range(1)),
        errors::InvalidArgument("value_range should satisfy value_range[0] < "
                                "value_range[1], but got '[",
                                value_range(0), ", ", value_range(1), "]'"));
    OP_REQUIRES(
        ctx, (nbins > 0),
        errors::InvalidArgument("nbins should be a positive number, but got '",
                                nbins, "'"));

    Tensor* out_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({nbins}), &out_tensor));
    auto out = out_tensor->flat<Tout>();

    OP_REQUIRES_OK(
        ctx, functor::HistogramFixedWidthFunctor<Device, T, Tout>::Compute(
                 ctx, values, value_range, nbins, out));
  }
};

#define REGISTER_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("HistogramFixedWidth")          \
                              .Device(DEVICE_GPU)              \
                              .HostMemory("value_range")       \
                              .HostMemory("nbins")             \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int32>("dtype"), \
                          HistogramFixedWidthOp<GPUDevice, type, int32>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS);
TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_int64(REGISTER_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNELS);
#endif

#undef REGISTER_KERNELS

}  // namespace itex
