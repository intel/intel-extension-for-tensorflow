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

#include "itex/core/kernels/gpu/stateless_random_gamma_op_v3.h"

#include "itex/core/kernels/common/random_ops_util.h"
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
struct StatelessRandomGammaV3Kernel {
  StatelessRandomGammaV3Kernel(const T* alpha_flat, const int64_t num_samples,
                               const int64_t num_alphas,
                               const int64_t samples_per_alpha,
                               const uint64* key, const uint64* counter,
                               T* samples_flat)
      : alpha_flat(alpha_flat),
        num_samples(num_samples),
        num_alphas(num_alphas),
        samples_per_alpha(samples_per_alpha),
        key(key),
        counter(counter),
        samples_flat(samples_flat) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_samples) return;

    random::PhiloxRandom random_;
    if (key != nullptr && counter != nullptr) {
      random_ = GetPhiloxRandomFromCounterKeyMem(counter, key);
    }

    auto output_idx = id;
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
      random::PhiloxRandom gen = random_;
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
      random::PhiloxRandom gen = random_;
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
  const int64_t num_samples;
  const int64_t num_alphas;
  const int64_t samples_per_alpha;
  const uint64* key;
  const uint64* counter;
  T* samples_flat;
};

template <typename Device, typename T>
void FillKernel(OpKernelContext* ctx, const T* alpha_flat, int64 num_samples,
                int64 num_alphas, int64 samples_per_alpha, const uint64* key,
                const uint64* counter, T* samples_flat) {
  using Eigen::numext::exp;
  using Eigen::numext::log;
  using Eigen::numext::log1p;
  using Eigen::numext::pow;

  auto* ITEX_GPU_stream = ctx->GetDeviceStream();
  OP_REQUIRES(ctx, ITEX_GPU_stream != nullptr,
              errors::Internal("No GPU stream available."));
  auto work_group_size =
      ITEX_GPU_stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (num_samples + work_group_size - 1) / work_group_size;
  ITEX_GPU_stream->submit([&](sycl::handler& cgh) {
    StatelessRandomGammaV3Kernel<T> task(alpha_flat, num_samples, num_alphas,
                                         samples_per_alpha, key, counter,
                                         samples_flat);
    cgh.parallel_for<StatelessRandomGammaV3Kernel<T>>(
        sycl::nd_range<1>(sycl::range<1>(work_group_size * num_wg),
                          sycl::range<1>(work_group_size)),
        task);
  });
}

}  // namespace functor

// A stateless-random-gamma kernel that takes shape, key, counter and algorithm
// as the first 4 inputs.
template <typename Device, typename T>
class StatelessRandomGammaOpWithKeyCounter : public OpKernel {
 public:
  explicit StatelessRandomGammaOpWithKeyCounter(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& key_t = ctx->input(1);
    const Tensor& counter_t = ctx->input(2);
    const Tensor& alg_t = ctx->input(3);

    int alg_id;
    OP_REQUIRES_OK(ctx, GetScalar(alg_t, 3, &alg_id));
    Algorithm alg = Algorithm(alg_id);
    if (alg == RNG_ALG_AUTO_SELECT) {
      alg = RNG_ALG_PHILOX;
    }

    OP_REQUIRES_OK(ctx,
                   CheckKeyCounterShape(alg, key_t.shape(), counter_t.shape()));

    TensorShape shape;
    OP_REQUIRES_OK(ctx, MakeShape(ctx->input(0), &shape));

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) {
      return;
    }

    // Fill in the random numbers
    Fill(ctx, alg, key_t, counter_t, output);
  }

 protected:
  void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
            const Tensor& counter, Tensor* output) {
    int alpha_input_idx = 4;
    const Tensor& alpha_t = ctx->input(alpha_input_idx);
    TensorShape samples_shape = output->shape();

    OP_REQUIRES(ctx, TensorShapeUtils::EndsWith(samples_shape, alpha_t.shape()),
                errors::InvalidArgument(
                    "Shape passed in must end with broadcasted shape."));

    const int64_t num_alphas = alpha_t.NumElements();
    OP_REQUIRES(ctx, num_alphas > 0,
                errors::InvalidArgument(
                    "Input alpha should have non-zero element count, got: ",
                    num_alphas));

    const int64_t num_samples = samples_shape.num_elements();
    const int64_t samples_per_alpha = num_samples / num_alphas;

    const auto alpha_flat = alpha_t.flat<T>().data();
    auto samples_flat = output->flat<T>().data();

    if (alg == RNG_ALG_PHILOX) {
      auto key_data = key.flat<uint64>().data();
      auto counter_data = counter.flat<uint64>().data();
      functor::FillKernel<Device, T>(ctx, alpha_flat, num_samples, num_alphas,
                                     samples_per_alpha, key_data, counter_data,
                                     samples_flat);
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

#define REGISTER_GAMMA_GPU(TYPE)          \
  REGISTER_KERNEL_BUILDER(                \
      Name("StatelessRandomGammaV3")      \
          .Device(DEVICE_GPU)             \
          .HostMemory("shape")            \
          .HostMemory("alg")              \
          .TypeConstraint<TYPE>("dtype"), \
      StatelessRandomGammaOpWithKeyCounter<GPUDevice, TYPE>)

TF_CALL_half(REGISTER_GAMMA_GPU);
TF_CALL_bfloat16(REGISTER_GAMMA_GPU);
TF_CALL_float(REGISTER_GAMMA_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GAMMA_GPU);
#endif

#undef REGISTER_GAMMA_GPU

}  // namespace itex
