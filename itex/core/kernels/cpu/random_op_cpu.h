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

#ifndef ITEX_CORE_KERNELS_CPU_RANDOM_OP_CPU_H_
#define ITEX_CORE_KERNELS_CPU_RANDOM_OP_CPU_H_

#include <algorithm>
#include <cmath>
#include <memory>

#include "itex/core/kernels/common/random_ops_util.h"
#include "itex/core/utils/lib/random/guarded_philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/lib/random/simple_philox.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types_traits.h"

namespace itex {

namespace functor {
using random::PhiloxRandom;
using random::SingleSampleAdapter;

// The default implementation of the functor, which should never be invoked
// But we still need to provide implementation for now for the linker to work,
// since we do not support all the distributions yet.
template <typename Device, class Distribution>
struct FillPhiloxRandom {
  typedef typename Distribution::ResultElementType T;
  void operator()(OpKernelContext* ctx, const Device&, random::PhiloxRandom gen,
                  T* data, int64 size, Distribution dist,
                  const uint64* key = nullptr,
                  const uint64* counter = nullptr) {
    OP_REQUIRES(
        ctx, false,
        errors::Internal(
            "Default `FillPhiloxRandom` implementation should not be executed. "
            "The cause of this error is probably that `FillPhiloxRandom` does "
            "not support this device or random distribution yet."));
  }
};

template <class Generator, typename RealType, class Distribution>
class DistributionVec {
 public:
  explicit DistributionVec(Distribution* dist) { this->dist = dist; }

  typename Distribution::ResultType operator()(Generator* gen) {
    return (*dist)(gen);
  }

  void VecCopy(RealType* data, int64 length) {}

 private:
  Distribution* dist;
};

template <>
class DistributionVec<
    random::PhiloxRandom, Eigen::bfloat16,
    random::UniformDistribution<random::PhiloxRandom, Eigen::bfloat16>> {
 public:
  typedef random::UniformDistribution<random::PhiloxRandom, Eigen::bfloat16>
      Distribution;

  explicit DistributionVec(Distribution* dist) { this->dist = dist; }

  typename Distribution::ResultType operator()(random::PhiloxRandom* gen) {
    typename random::PhiloxRandom::ResultType sample = (*gen)();
    typename Distribution::ResultType result;

    for (int i = 0; i < Distribution::kResultElementCount; ++i) {
      result[i] = random::InternalUint16ToBfloat16(sample[i]);
    }

    return result;
  }

  void VecCopy(Eigen::bfloat16* data, int64 length) {
    // The mantissa has an implicit leading 1, so the above code creates a value
    // in [1, 2). The minus will not cause a rounding that makes the result 1.
    // Instead it will just be close to 1.
    auto result_t = typename TTypes<Eigen::bfloat16>::Tensor(data, length);
    result_t = result_t - Eigen::bfloat16(1.0);
  }

 private:
  Distribution* dist;
};

template <>
class DistributionVec<
    random::PhiloxRandom, float,
    random::UniformDistribution<random::PhiloxRandom, float>> {
 public:
  typedef random::UniformDistribution<random::PhiloxRandom, float> Distribution;

  explicit DistributionVec(Distribution* dist) { this->dist = dist; }

  typename Distribution::ResultType operator()(random::PhiloxRandom* gen) {
    typename random::PhiloxRandom::ResultType sample = (*gen)();
    typename Distribution::ResultType result;

    for (int i = 0; i < Distribution::kResultElementCount; ++i) {
      result[i] = random::InternalUint32ToFloat(sample[i]);
    }

    return result;
  }

  void VecCopy(float* data, int64 length) {
    auto result_t = typename TTypes<float>::Tensor(data, length);
    result_t = result_t - 1.0f;
  }

 private:
  Distribution* dist;
};

// A class to fill a specified range of random groups
template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  static void Run(random::PhiloxRandom gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    gen.Skip(start_group);
    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    DistributionVec<random::PhiloxRandom, T, Distribution> dist_vec(&dist);
    for (int64 index = start_group; index < limit_group_full; ++index) {
      auto samples = dist_vec(&gen);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    int64 remaining_size = 0;
    if (limit_group_full < limit_group) {
      remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist_vec(&gen);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
    dist_vec.VecCopy(
        data + start_group * kGroupSize,
        (limit_group_full - start_group) * kGroupSize + remaining_size);
  }
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  static constexpr int64 kReservedSamplesPerOutput = 256;

  static void Run(random::PhiloxRandom base_gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    static const int kGeneratorSkipPerOutputGroup =
        kGroupSize * kReservedSamplesPerOutput /
        PhiloxRandom::kResultElementCount;

    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    int64 group_index;
    DistributionVec<SingleSampleAdapter<PhiloxRandom>, T, Distribution>
        dist_vec(&dist);
    for (group_index = start_group; group_index < limit_group_full;
         ++group_index) {
      // Reset the generator to the beginning of the output group region
      // This is necessary if we want the results to be independent of order
      // of work
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      auto samples = dist_vec(&single_samples);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    int64 remaining_size = 0;
    if (limit_group_full < limit_group) {
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist_vec(&single_samples);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
    dist_vec.VecCopy(
        data + start_group * kGroupSize,
        (limit_group_full - start_group) * kGroupSize + remaining_size);
  }
};

// Declares the partially CPU-specialized functor struct.
//
// NOTE: Due to inlining done by the compiler, you may need to add
// explicit instantiation of the functor in random_op.cc.  See example
// functor::FillPhiloxRandom<CPUDevice, random::UniformDistribution>.
//
// This functor can take the PhiloxRandom input from either device memory `key`
// and `counter` or a stack value `gen`. If both `key` and `counter` are not
// nullptr, they provide the input; otherwise `gen` provides the input.
template <class Distribution>
struct FillPhiloxRandom<CPUDevice, Distribution> {
  // Partial specialization for CPU to fill the entire region with randoms
  // It splits the work into several tasks and run them in parallel
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64 size,
                  Distribution dist, const uint64* key = nullptr,
                  const uint64* counter = nullptr) {
    const int kGroupSize = Distribution::kResultElementCount;

    int64 total_group_count = (size + kGroupSize - 1) / kGroupSize;

    const int kGroupCost = kGroupSize * (random::PhiloxRandom::kElementCost +
                                         Distribution::kElementCost);

    if (key != nullptr && counter != nullptr) {
      gen = GetPhiloxRandomFromCounterKeyMem(counter, key);
    }

    d.parallelFor(
        total_group_count, Eigen::TensorOpCost(0, 0, kGroupCost),
        [&gen, data, size, dist](Eigen::Index first, Eigen::Index last) {
          FillPhiloxRandomTask<
              Distribution, Distribution::kVariableSamplesPerOutput>::Run(gen,
                                                                          data,
                                                                          size,
                                                                          first,
                                                                          last,
                                                                          dist);
        });
  }
};

}  // namespace functor

}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_CPU_RANDOM_OP_CPU_H_
