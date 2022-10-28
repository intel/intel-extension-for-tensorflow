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

#ifndef ITEX_CORE_KERNELS_GPU_PARAMETERIZED_TRUNCATED_NORMAL_OP_H_
#define ITEX_CORE_KERNELS_GPU_PARAMETERIZED_TRUNCATED_NORMAL_OP_H_

#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/tensor_types.h"

#if defined(_MSC_VER) && !defined(__clang__)
// msvc does not support unroll. One could try the loop pragma but we need to
// take a closer look if this generates better code in this case. For now let
// the compiler take care of it.
#define UNROLL
#else
#define UNROLL _Pragma("unroll")
#endif

namespace itex {

class OpKernelContext;

namespace functor {

static constexpr int kMaxIterations = 1000;

typedef Eigen::GpuDevice GPUDevice;

// Sample a truncated normal random variable, with mean, stddev, minval, and
// maxval parameters for each batch. Uses two rejection sampling algorithms
// described in http://rd.springer.com/article/10.1007/BF00143942 and a randn
// rejection sampler when most of the normal is inside the bounds.
//
// Either minval may be -infinity, or maxval may be +infinity. If the interval
// (minval, maxval) is empty, the result is NaN.
template <typename T>
struct TruncatedNormalKernel {
  TruncatedNormalKernel(random::PhiloxRandom gen_, T* data_, int64 num_batches_,
                        int64 samples_per_batch_, int64 num_elements_,
                        const T* means_, bool single_mean_, const T* stddevs_,
                        bool single_stddev_, const T* minvals_,
                        bool single_minval_, const T* maxvals_,
                        bool single_maxval_, int64 kMaxIterations_)
      : gen(gen_),
        data(data_),
        num_batches(num_batches_),
        samples_per_batch(samples_per_batch_),
        num_elements(num_elements_),
        means(means_),
        single_mean(single_mean_),
        stddevs(stddevs_),
        single_stddev(single_stddev_),
        minvals(minvals_),
        single_minval(single_minval_),
        maxvals(maxvals_),
        single_maxval(single_maxval_),
        kMaxIterations(kMaxIterations_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int32 max_samples_per_item = 2 * kMaxIterations;

    const int32 initial_offset = item.get_global_id(0);
    if (initial_offset >= num_elements) return;

    gen.Skip(max_samples_per_item * initial_offset);
    typedef random::UniformDistribution<random::PhiloxRandom, T> Uniform;
    typedef random::NormalDistribution<random::PhiloxRandom, T> Normal;
    Uniform dist;
    Normal normal_dist;
    const int kDistSize = Uniform::kResultElementCount;
    const T quietNaN = Eigen::NumTraits<T>::quiet_NaN();
    // The randn rejection sampling is used when the mean and at least
    // this many standard deviations are inside the bounds. The uniform
    // proposal samplers become less efficient as the bounds are further
    // from the mean, the reverse is true for the randn sampler. This
    // number was chosen by empirical benchmarking. If modified, the
    // benchmarks in parameterized_truncated_normal_op_test should also be
    // changed.
    const T kStdDevsInsideBoundsToUseRandnSampler = T(1.7);

    const int32 samples_between_processed_elements =
        max_samples_per_item * (item.get_global_range()[0]);

    // Track how many more samples we need to skip before we process the
    // next element.
    int32 remaining_samples = samples_between_processed_elements;

    const int64 batch_id = initial_offset / samples_per_batch;
    T mean = means[single_mean ? 0 : batch_id];
    const T input_stddev = stddevs[single_stddev ? 0 : batch_id];
    T minval = minvals[single_minval ? 0 : batch_id];
    T maxval = maxvals[single_maxval ? 0 : batch_id];

    // Flip the distribution if we can make the lower bound positive.
    T stddev;
    if (sycl::isinf(static_cast<float>(minval)) || maxval < mean) {
      T temp = minval;
      minval = maxval;
      maxval = temp;
      stddev = -input_stddev;
    } else {
      stddev = input_stddev;
    }

    // Calculate normalized samples, then scale them.
    const T normMin = (minval - mean) / stddev;
    const T normMax = (maxval - mean) / stddev;

    // Determine the method to use.
    const T sqrtFactor = Eigen::numext::sqrt((normMin * normMin) + T(4));
    const T cutoff =
        T(2) *
        Eigen::numext::exp(T(0.5) + (normMin * (normMin - sqrtFactor)) / T(4)) /
        (normMin + sqrtFactor);
    const T diff = normMax - normMin;
    const T two = T(2.0);

    // Validate the normalized min and max, because the originals may have
    // been flipped already.
    if (!(input_stddev > T(0) && normMin < normMax &&
          (sycl::isfinite(static_cast<float>(normMin)) ||
           sycl::isfinite(static_cast<float>(normMax))))) {
      data[initial_offset] = quietNaN;
    } else if (((normMin < -kStdDevsInsideBoundsToUseRandnSampler) &&
                (normMax >= T(0.))) ||
               ((normMax > kStdDevsInsideBoundsToUseRandnSampler) &&
                (normMin <= T(0.)))) {
      int numIterations = 0;
      while (numIterations < kMaxIterations) {
        const auto randn = normal_dist(&gen);
        remaining_samples -= gen.kResultElementCount;
        UNROLL for (int i = 0; i < kDistSize; i++) {
          if ((randn[i] >= normMin) && randn[i] <= normMax) {
            data[initial_offset] = randn[i] * stddev + mean;
            numIterations = kMaxIterations;
            break;
          } else if (numIterations + 1 == kMaxIterations) {
            // If we did not successfully sample after all these
            // iterations something is wrong. Output a nan.
            data[initial_offset] = quietNaN;
            numIterations = kMaxIterations;
            break;
          } else {
            numIterations++;
          }
        }
      }
    } else if (diff < cutoff) {
      // Sample from a uniform distribution on [normMin, normMax].

      // Vectorized intermediate calculations for uniform rejection
      // sampling. We always generate at most 4 samples.
      Eigen::array<T, 4> z;
      Eigen::array<T, 4> g;

      const T plusFactor = (normMin < T(0)) ? T(0) : T(normMin * normMin);

      int numIterations = 0;
      while (numIterations < kMaxIterations) {
        const auto rand = dist(&gen);
        remaining_samples -= gen.kResultElementCount;
        UNROLL for (int i = 0; i < kDistSize; i++) {
          z[i] = rand[i] * diff + normMin;
        }
        UNROLL for (int i = 0; i < kDistSize; i++) {
          g[i] = (plusFactor - z[i] * z[i]) / two;
        }

        const auto u = dist(&gen);
        remaining_samples -= gen.kResultElementCount;
        UNROLL for (int i = 0; i < kDistSize; i++) {
          bool accept = u[i] <= Eigen::numext::exp(g[i]);
          if (accept) {
            // Accept the sample z.
            data[initial_offset] = z[i] * stddev + mean;
            // Break out of the nested loop by updating numIterations.
            numIterations = kMaxIterations;
            break;
          } else if (numIterations + 1 >= kMaxIterations) {
            data[initial_offset] = quietNaN;
            numIterations = kMaxIterations;
            break;
          } else {
            numIterations++;
          }
        }
      }
    } else {
      // Sample from an exponential distribution with alpha maximizing
      // acceptance probability, initial_offset by normMin from the
      // origin. Accept only if less than normMax.
      const T alpha =
          (normMin + Eigen::numext::sqrt((normMin * normMin) + T(4))) / T(2);
      int numIterations = 0;
      while (numIterations < kMaxIterations) {
        auto rand = dist(&gen);
        remaining_samples -= gen.kResultElementCount;
        UNROLL for (int i = 0; i < kDistSize; i += 2) {
          const T z = -Eigen::numext::log(rand[i]) / alpha + normMin;
          const T x = normMin < alpha ? alpha - z : normMin - alpha;
          const T g = Eigen::numext::exp(-x * x / two);
          const T u = rand[i + 1];
          bool accept = (u <= g && z < normMax);
          if (accept) {
            data[initial_offset] = z * stddev + mean;
            // Break out of the nested loop by updating numIterations.
            numIterations = kMaxIterations;
            break;
          } else if (numIterations + 1 >= kMaxIterations) {
            data[initial_offset] = quietNaN;
            numIterations = kMaxIterations;
            break;
          } else {
            numIterations++;
          }
        }
      }
    }

    gen.Skip(remaining_samples);
  }

 private:
  random::PhiloxRandom gen;
  T* data;
  int64 num_batches;
  int64 samples_per_batch;
  int64 num_elements;
  const T* means;
  bool single_mean;
  const T* stddevs;
  bool single_stddev;
  const T* minvals;
  bool single_minval;
  const T* maxvals;
  bool single_maxval;
  int64 kMaxIterations;
};

template <typename Device, typename T>
struct TruncatedNormalFunctor;
// Partial specialization for GPU
template <typename T>
struct TruncatedNormalFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d, int64 num_batches,
                  int64 samples_per_batch, int64 num_elements,
                  typename TTypes<T>::ConstFlat means,
                  typename TTypes<T>::ConstFlat stddevs,
                  typename TTypes<T>::ConstFlat minvals,
                  typename TTypes<T>::ConstFlat maxvals,
                  const random::PhiloxRandom& gen,
                  typename TTypes<T>::Flat output) {
    auto stream = d.stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = num_elements;
    auto num_work_groups =
        (num_work_items + work_group_size - 1) / work_group_size;

    stream->submit([&](sycl::handler& cgh) {
      TruncatedNormalKernel<T> task(
          gen, output.data(), num_batches, samples_per_batch, num_elements,
          means.data(), means.dimension(0) == 1, stddevs.data(),
          stddevs.dimension(0) == 1, minvals.data(), minvals.dimension(0) == 1,
          maxvals.data(), maxvals.dimension(0) == 1, kMaxIterations);
      cgh.parallel_for<TruncatedNormalKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_work_groups * work_group_size),
                            sycl::range<1>(work_group_size)),
          task);
    });
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_PARAMETERIZED_TRUNCATED_NORMAL_OP_H_
