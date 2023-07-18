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

#include "itex/core/kernels/gpu/multinomial_op.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "itex/core/kernels/gpu/stateless_random_ops.h"
#include "itex/core/utils/lib/random/guarded_philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename T, typename OutputType>
struct MultinomialFunctor {
  void operator()(OpKernelContext* ctx, const Device& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat noises,
                  typename TTypes<float>::Flat scores,
                  typename TTypes<float>::Flat scratch, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<OutputType>::Matrix output);
};

extern template struct MultinomialFunctor<GPUDevice, Eigen::half, int32>;
extern template struct MultinomialFunctor<GPUDevice, Eigen::bfloat16, int32>;
extern template struct MultinomialFunctor<GPUDevice, float, int32>;
extern template struct MultinomialFunctor<GPUDevice, int32, int32>;
extern template struct MultinomialFunctor<GPUDevice, int64, int32>;

extern template struct MultinomialFunctor<GPUDevice, Eigen::half, int64>;
extern template struct MultinomialFunctor<GPUDevice, Eigen::bfloat16, int64>;
extern template struct MultinomialFunctor<GPUDevice, float, int64>;
extern template struct MultinomialFunctor<GPUDevice, int32, int64>;
extern template struct MultinomialFunctor<GPUDevice, int64, int64>;

#ifdef ITEX_ENABLE_DOUBLE
extern template struct MultinomialFunctor<GPUDevice, double, int32>;
extern template struct MultinomialFunctor<GPUDevice, double, int64>;
#endif  // ITEX_ENABLE_DOUBLE

}  // namespace functor

namespace {

// Samples from a multinomial distribution.
template <typename Device, typename T, typename OutputType>
class MultinomialOp : public OpKernel {
 public:
  explicit MultinomialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void DoCompute(OpKernelContext* ctx, const Tensor& logits_t,
                 const Tensor& num_samples_t, GuardedPhiloxRandom* generator) {
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_t.shape()),
                errors::InvalidArgument("logits should be a matrix, got shape ",
                                        logits_t.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(num_samples_t.shape()),
        errors::InvalidArgument("num_samples should be a scalar, got shape ",
                                num_samples_t.shape().DebugString()));

    const int num_samples = num_samples_t.scalar<int>()();
    OP_REQUIRES(ctx, num_samples >= 0,
                errors::InvalidArgument(
                    "num_samples should be nonnegative, got ", num_samples));

    for (int i = 0; i < 2; i++) {
      const int64 dim = logits_t.dim_size(i);
      OP_REQUIRES(ctx, static_cast<int>(dim) == dim,
                  errors::InvalidArgument(
                      "logits.shape = ", logits_t.shape().DebugString(),
                      " too large for int"));
    }
    const int batch_size = static_cast<int>(logits_t.dim_size(0));
    const int num_classes = static_cast<int>(logits_t.dim_size(1));
    OP_REQUIRES(ctx, num_classes > 0,
                errors::InvalidArgument("num_classes should be positive, got ",
                                        num_classes));

    Tensor* samples_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, num_samples}),
                                  &samples_t));

    // Execute kernel only for nonempty output; otherwise Eigen crashes on GPU.
    if (samples_t->NumElements() > 0) {
      Tensor noises, scores, scratch;  // Scratch space only used for GPU.
      if (std::is_same<Device, GPUDevice>::value) {
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                DT_FLOAT, TensorShape({batch_size, num_samples, num_classes}),
                &noises));
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                DT_FLOAT, TensorShape({batch_size, num_samples, num_classes}),
                &scores));
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(DT_FLOAT, TensorShape({batch_size, num_samples}),
                               &scratch));
      }

      int num_samples_ceil_4 = (num_samples + 3) / 4 * 4;
      auto rng =
          generator->ReserveRandomOutputs(batch_size * num_samples_ceil_4, 256);

      functor::MultinomialFunctor<Device, T, OutputType>()(
          ctx, ctx->eigen_device<Device>(), logits_t.matrix<T>(),
          noises.flat<float>(), scores.flat<float>(), scratch.flat<float>(),
          batch_size, num_classes, num_samples, rng,
          samples_t->matrix<OutputType>());
    }
  }
};

template <typename Device, typename T, typename OutputType>
class StatefulMultinomialOp : public MultinomialOp<Device, T, OutputType> {
 public:
  explicit StatefulMultinomialOp(OpKernelConstruction* ctx)
      : MultinomialOp<Device, T, OutputType>(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& logits_t = ctx->input(0);
    const Tensor& num_samples_t = ctx->input(1);
    this->DoCompute(ctx, logits_t, num_samples_t, &generator_);
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER(TYPE)                                                      \
  REGISTER_KERNEL_BUILDER(Name("Multinomial")                               \
                              .Device(DEVICE_GPU)                           \
                              .HostMemory("num_samples")                    \
                              .TypeConstraint<TYPE>("T")                    \
                              .TypeConstraint<itex::int32>("output_dtype"), \
                          StatefulMultinomialOp<GPUDevice, TYPE, int32>)    \
  REGISTER_KERNEL_BUILDER(Name("Multinomial")                               \
                              .Device(DEVICE_GPU)                           \
                              .HostMemory("num_samples")                    \
                              .TypeConstraint<TYPE>("T")                    \
                              .TypeConstraint<itex::int64>("output_dtype"), \
                          StatefulMultinomialOp<GPUDevice, TYPE, int64>)

TF_CALL_half(REGISTER);
TF_CALL_bfloat16(REGISTER);
TF_CALL_float(REGISTER);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER

template <typename Device, typename T, typename OutputType>
class StatelessMultinomialOp : public MultinomialOp<Device, T, OutputType> {
 public:
  explicit StatelessMultinomialOp(OpKernelConstruction* ctx)
      : MultinomialOp<Device, T, OutputType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& logits_t = ctx->input(0);
    const Tensor& num_samples_t = ctx->input(1);

    const Tensor& seed_t = ctx->input(2);
    OP_REQUIRES(ctx, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key, &counter));

    GuardedPhiloxRandom generator;
    generator.Init(counter, key);

    this->DoCompute(ctx, logits_t, num_samples_t, &generator);
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER(TYPE)                                                      \
  REGISTER_KERNEL_BUILDER(Name("StatelessMultinomial")                      \
                              .Device(DEVICE_GPU)                           \
                              .HostMemory("num_samples")                    \
                              .HostMemory("seed")                           \
                              .TypeConstraint<TYPE>("T")                    \
                              .TypeConstraint<itex::int32>("output_dtype"), \
                          StatelessMultinomialOp<GPUDevice, TYPE, int32>)   \
  REGISTER_KERNEL_BUILDER(Name("StatelessMultinomial")                      \
                              .Device(DEVICE_GPU)                           \
                              .HostMemory("num_samples")                    \
                              .HostMemory("seed")                           \
                              .TypeConstraint<TYPE>("T")                    \
                              .TypeConstraint<itex::int64>("output_dtype"), \
                          StatelessMultinomialOp<GPUDevice, TYPE, int64>)

TF_CALL_half(REGISTER);
TF_CALL_bfloat16(REGISTER);
TF_CALL_float(REGISTER);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER

}  // end namespace

}  // end namespace itex
