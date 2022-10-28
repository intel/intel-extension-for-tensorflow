/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/stateless_random_gamma_op_v2.h"

#include "itex/core/kernels/gpu/random_op_gpu.h"
#include "itex/core/kernels/gpu/stateless_random_ops.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "tensorflow/c/kernels.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
static constexpr int kReservedSamplesPerOutput = 256;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

typedef random::NormalDistribution<random::PhiloxRandom, double> Normal;
typedef random::UniformDistribution<random::PhiloxRandom, double> Uniform;

template <typename T>
struct StatelessRandomGammaKernel {
  StatelessRandomGammaKernel(const T* alpha_flat, int64_t num_samples,
                             int64_t num_alphas, int64_t samples_per_alpha,
                             const random::PhiloxRandom& random,
                             T* samples_flat)
      : alpha_flat(alpha_flat),
        num_samples(num_samples),
        num_alphas(num_alphas),
        samples_per_alpha(samples_per_alpha),
        random(random),
        samples_flat(samples_flat) {}
  void operator()(sycl::item<1> item) const {
    auto output_idx = item.get_id(0);
    int64 alpha_idx = output_idx / samples_per_alpha;
    int64 sample_idx = output_idx % samples_per_alpha;

    Normal normal;
    Uniform uniform;
    RandomSampleBuffer<Normal> normal_buffer(&normal);
    RandomSampleBuffer<Uniform> uniform_buffer(&uniform);

    const double alpha = static_cast<double>(alpha_flat[alpha_idx]);
    if (alpha == 1.0) {
      // Sample from an exponential distribution.
      // As we want data stable regardless of sharding, we skip on a
      // per-sample basis.
      random::PhiloxRandom gen = random;
      gen.Skip(kReservedSamplesPerOutput * output_idx);
      double u = uniform(&gen)[Uniform::kResultElementCount - 1];
      const double res = -log1p(-u);
      // We use alpha_idx + sample_idx * num_alphas instead of output_idx
      // to generate numbers in the right order (CPU and GPU kernels
      // must generate numbers in the same order).
      samples_flat[alpha_idx + sample_idx * num_alphas] = static_cast<T>(res);
    } else {  // if alpha != 1.0
      // Transformation-rejection from pairs of uniform and normal random
      // variables. http://dl.acm.org/citation.cfm?id=358414
      //
      // The algorithm has an acceptance rate of ~95% for small alpha
      // (~1), and higher accept rates for higher alpha, so runtime is
      // O(NumAlphas * NumSamples * k) with k ~ 1 / 0.95.
      //
      // For alpha<1, we add one to d=alpha-1/3, and multiply the final
      // result by uniform()^(1/alpha)
      const bool alpha_less_than_one = alpha < 1.0;
      const double d = alpha + (alpha_less_than_one ? 2.0 / 3 : -1.0 / 3);
      const double c = 1.0 / 3 / sqrt(d);

      // Since each sample may use a variable number of normal/uniform
      // samples, and we want data stable regardless of sharding, we skip
      // on a per-sample basis.
      random::PhiloxRandom gen = random;
      gen.Skip(kReservedSamplesPerOutput * output_idx);

      // To prevent overwriting SampleBuffer's underlying array with
      // zeros (in tensorflow::random::Array constructor), we just mark
      // the buffer as empty instead of initializing a new SampleBuffer
      // object here. The next call to operator() will fill the buffer
      // with new numbers.
      normal_buffer.Clear();
      uniform_buffer.Clear();

      // Keep trying until we don't reject a sample. In practice, we will
      // only reject ~5% at worst, for low alpha near 1.
      while (true) {
        const double x = normal_buffer(&gen);
        double v = 1 + c * x;
        if (v <= 0) {
          continue;
        }
        v = v * v * v;
        double u = uniform_buffer(&gen);
        // The first option in the if is a "squeeze" short-circuit to
        // dodge the two logs. Magic constant sourced from the paper
        // linked above. Upward of .91 of the area covered by the log
        // inequality is covered by the squeeze as well (larger coverage
        // for smaller values of alpha).
        if ((u < 1 - 0.0331 * (x * x) * (x * x)) ||
            (log(u) < 0.5 * x * x + d * (1 - v + log(v)))) {
          double res = d * v;
          if (alpha_less_than_one) {
            double b = uniform_buffer(&gen);
            res *= pow(b, 1 / alpha);
          }
          // We use alpha_idx + sample_idx * num_alphas instead of
          // output_idx to generate numbers in the right order (CPU and
          // GPU kernels must generate numbers in the same order).
          samples_flat[alpha_idx + sample_idx * num_alphas] =
              static_cast<T>(res);
          break;
        }
      }  // while: true
    }    // if (alpha == 1.0)
  }

 private:
  const T* alpha_flat;
  int64_t num_samples;
  int64_t num_alphas;
  int64_t samples_per_alpha;
  const random::PhiloxRandom random;
  T* samples_flat;
};

template <typename Device, typename T>
void FillKernel(OpKernelContext* ctx, const T* alpha_flat, int64 num_samples,
                int64 num_alphas, int64 samples_per_alpha,
                const random::PhiloxRandom& random, T* samples_flat) {
  using Eigen::numext::exp;
  using Eigen::numext::log;
  using Eigen::numext::log1p;
  using Eigen::numext::pow;

  auto* dpcpp_stream = ctx->GetDeviceStream();
  OP_REQUIRES(ctx, dpcpp_stream != nullptr,
              errors::Internal("No GPU stream available."));
  auto total_items =
      dpcpp_stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  total_items = total_items > num_samples ? num_samples : total_items;
  dpcpp_stream->submit([&](sycl::handler& cgh) {
    StatelessRandomGammaKernel<T> task(alpha_flat, num_samples, num_alphas,
                                       samples_per_alpha, random, samples_flat);
    cgh.parallel_for<StatelessRandomGammaKernel<T>>(sycl::range<1>(total_items),
                                                    task);
  });
}

}  // namespace functor

template <typename Device, typename T>
class StatelessRandomGammaOp : public OpKernel {
 public:
  explicit StatelessRandomGammaOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Sanitize input
    const Tensor& shape_t = context->input(0);
    const Tensor& seed_t = context->input(1);
    TensorShape shape;
    OP_REQUIRES_OK(context, MakeShape(shape_t, &shape));
    OP_REQUIRES(context, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) return;

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(context, GenerateKey(seed_t, &key, &counter));

    // Fill in the random numbers
    Fill(context, random::PhiloxRandom(counter, key), output);
  }

 private:
  void Fill(OpKernelContext* ctx, random::PhiloxRandom random, Tensor* output) {
    const Tensor& alpha_t = ctx->input(2);

    TensorShape samples_shape = output->shape();
    OP_REQUIRES(ctx, TensorShapeUtils::EndsWith(samples_shape, alpha_t.shape()),
                errors::InvalidArgument(
                    "Shape passed in must end with broadcasted shape."));

    const int64 num_alphas = alpha_t.NumElements();
    OP_REQUIRES(ctx, num_alphas > 0,
                errors::InvalidArgument(
                    "Input alpha should have non-zero element count, got: ",
                    num_alphas));

    const int64 num_samples = samples_shape.num_elements();
    const int64 samples_per_alpha = num_samples / num_alphas;
    const T* alpha_flat = alpha_t.flat<T>().data();
    auto samples_flat = output->flat<T>().data();

    functor::FillKernel<Device, T>(ctx, alpha_flat, num_samples, num_alphas,
                                   samples_per_alpha, random, samples_flat);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomGammaOp);
};

// Register GPU kernels for stateless gamma op.
#define REGISTER_DPCPP(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomGammaV2")      \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("shape")            \
                              .HostMemory("seed")             \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatelessRandomGammaOp<GPUDevice, TYPE>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_DPCPP);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_DPCPP);
#endif

#undef REGISTER_DPCPP

}  // namespace itex
