/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/parameterized_truncated_normal_op.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "itex/core/utils/lib/random/guarded_philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ParameterizedTruncatedNormalOp : public OpKernel {
  // Reshape batches so each batch is this size if possible.
  static const int32 kDesiredBatchSize = 100;

 public:
  explicit ParameterizedTruncatedNormalOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    const Tensor& means_tensor = ctx->input(1);
    const Tensor& stddevs_tensor = ctx->input(2);
    const Tensor& minvals_tensor = ctx->input(3);
    const Tensor& maxvals_tensor = ctx->input(4);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape_tensor.shape()),
        errors::InvalidArgument("Input shape should be a vector, got shape: ",
                                shape_tensor.shape().DebugString()));
    int32 num_batches = shape_tensor.flat<int32>()(0);

    int32 samples_per_batch = 1;
    const int32 num_dims = shape_tensor.dim_size(0);
    for (int32 i = 1; i < num_dims; i++) {
      samples_per_batch *= shape_tensor.flat<int32>()(i);
    }
    const int32 num_elements = num_batches * samples_per_batch;

    // Allocate the output before fudging num_batches and samples_per_batch.
    auto shape_vec = shape_tensor.flat<int32>();
    TensorShape tensor_shape;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                            shape_vec.data(), shape_vec.size(), &tensor_shape));
    Tensor* samples_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor_shape, &samples_tensor));

    // Parameters must be 0-d or 1-d.
    OP_REQUIRES(ctx, means_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input means should be a scalar or vector, got shape: ",
                    means_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, stddevs_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input stddevs should be a scalar or vector, got shape: ",
                    stddevs_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, minvals_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input minvals should be a scalar or vector, got shape: ",
                    minvals_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, maxvals_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input maxvals should be a scalar or vector, got shape: ",
                    maxvals_tensor.shape().DebugString()));

    if ((means_tensor.dims() == 0 || means_tensor.dim_size(0) == 1) &&
        (stddevs_tensor.dims() == 0 || stddevs_tensor.dim_size(0) == 1) &&
        minvals_tensor.dims() == 0 && maxvals_tensor.dims() == 0) {
      // All batches have the same parameters, so we can update the batch size
      // to a reasonable value to improve parallelism (ensure enough batches,
      // and no very small batches which have high overhead).
      int32 size = num_batches * samples_per_batch;
      int32 adjusted_samples = kDesiredBatchSize;
      // Ensure adjusted_batches * adjusted_samples >= size.
      int32 adjusted_batches = Eigen::divup(size, adjusted_samples);
      num_batches = adjusted_batches;
      samples_per_batch = adjusted_samples;
    } else {
      // Parameters must be broadcastable to the shape [num_batches].
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(means_tensor.shape()) ||
              means_tensor.dim_size(0) == 1 ||
              means_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input means should have length 1 or shape[0], got shape: ",
              means_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(stddevs_tensor.shape()) ||
              stddevs_tensor.dim_size(0) == 1 ||
              stddevs_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input stddevs should have length 1 or shape[0], got shape: ",
              stddevs_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(minvals_tensor.shape()) ||
              minvals_tensor.dim_size(0) == 1 ||
              minvals_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input minvals should have length 1 or shape[0], got shape: ",
              minvals_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(maxvals_tensor.shape()) ||
              maxvals_tensor.dim_size(0) == 1 ||
              maxvals_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input maxvals should have length 1 or shape[0], got shape: ",
              maxvals_tensor.shape().DebugString()));
    }

    auto truncFunctor = functor::TruncatedNormalFunctor<Device, T>();
    // Each worker has the fudge factor for samples_per_batch, so use it here.
    random::PhiloxRandom rng =
        generator_.ReserveSamples128(num_batches * 2 * functor::kMaxIterations *
                                     (samples_per_batch + 3) / 4);
    truncFunctor(ctx, ctx->eigen_device<Device>(), num_batches,
                 samples_per_batch, num_elements, means_tensor.flat<T>(),
                 stddevs_tensor.flat<T>(), minvals_tensor.flat<T>(),
                 maxvals_tensor.flat<T>(), rng, samples_tensor->flat<T>());
  }

 private:
  GuardedPhiloxRandom generator_;

  TF_DISALLOW_COPY_AND_ASSIGN(ParameterizedTruncatedNormalOp);
};

#define REGISTER(TYPE)                                         \
  REGISTER_KERNEL_BUILDER(Name("ParameterizedTruncatedNormal") \
                              .Device(DEVICE_GPU)              \
                              .HostMemory("shape")             \
                              .TypeConstraint<TYPE>("dtype"),  \
                          ParameterizedTruncatedNormalOp<GPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_bfloat16(REGISTER);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER);
#endif

#undef REGISTER

};  // namespace itex
