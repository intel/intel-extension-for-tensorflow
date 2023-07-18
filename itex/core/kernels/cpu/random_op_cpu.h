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
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types_traits.h"

namespace itex {

namespace functor {
using random::PCGRandom;
using random::PhiloxRandom;
using random::SingleSampleAdapter;

// The default implementation of the functor, which should never be invoked
// But we still need to provide implementation for now for the linker to work,
// since we do not support all the distributions yet.
template <typename Device, class Distribution>
struct FillPhiloxRandom {
  typedef typename Distribution::ResultElementType T;
  void operator()(OpKernelContext* ctx, const Device&, PhiloxRandom gen,
                  T* data, int64 size, Distribution dist,
                  const uint64* key = nullptr,
                  const uint64* counter = nullptr) {
    ITEX_CHECK(false)
        << "Default `FillPhiloxRandom` implementation should not be executed. "
        << "The cause of this error is probably that `FillPhiloxRandom` does "
        << "not support this device or random distribution yet.";
  }
};

template <typename Device, class Distribution>
struct FillPCGRandom {
  typedef typename Distribution::ResultElementType T;
  void operator()(OpKernelContext* ctx, const Device&, PCGRandom gen, T* data,
                  int64 size, Distribution dist) {
    ITEX_CHECK(false)
        << "Default `FillPCGRandom` implementation should not be executed. "
        << "The cause of this error is probably that `FillPCGRandom` does "
        << "not support this device or random distribution yet.";
  }
};

template <class Generator, typename RealType>
class DistributionVec {
 public:
  typedef random::UniformDistribution<Generator, RealType> Distribution;

  explicit DistributionVec(Distribution* dist, const RealType* cmp_data) {
    this->dist = dist;
    if (cmp_data != nullptr) {
      // The mantissa has an implicit leading 1, so the generator creates a
      // value in [1, 2). The addition is to align the dropout rate.
      real_thr = cmp_data[0] + RealType(1.0);
      has_fused_cmp = true;
    } else {
      has_fused_cmp = false;
    }
  }

  typename Distribution::ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    typename Distribution::ResultType result;

    for (int i = 0; i < Distribution::kResultElementCount; ++i) {
      result[i] = Distribution::Converter(sample[i]);
    }

    // PhiloxRandom returns a 128-bit random bits each invocation, which cannot
    // directly use AVX2/AVX512. Thus post vectorized ops will be executed after
    // all random bits are generated. For other generators, such as PCGRandom
    // (1024*8-bit), it is better to execute post vectorized ops immediately
    // after each invocation.
    InnerVecPost(&result[0], Distribution::kResultElementCount);

    return result;
  }

  // Skip the outer VecPost for PCGRandom.
  void VecPost(RealType* data, int64 length) {
    if (std::is_same<Generator, PCGRandom>::value) return;

    VecPostImpl(data, length);
  }

 private:
  void VecPostImpl(RealType* data, int64 length) {
    auto result_t = typename TTypes<RealType>::Tensor(data, length);

    if (has_fused_cmp) {
      // `result_t >= real_thr` is mathematically equivalent to
      // `(result_t - 1.0) >= dropout_rate`, but the vectorized minus is
      // replaced by a scalar addition.
      result_t = (result_t >= real_thr).template cast<RealType>();
    } else {
      // Take away implicit leading 1 to get a value in [0, 1) for pure
      // RandomUniform. The minus will not cause a rounding that makes the
      // result 1. Instead it will just be close to 1.
      result_t = result_t - RealType(1.0);
    }
  }

  // Currently, only PCGRandom is supported.
  void InnerVecPost(RealType* data, int64 length) {
    if (!std::is_same<Generator, PCGRandom>::value) return;

    VecPostImpl(data, length);
  }

  Distribution* dist;
  bool has_fused_cmp;
  RealType real_thr;
};

// A class to fill a specified range of random groups
template <class Generator, class Distribution, bool VariableSamplesPerOutput>
struct FillRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Generator, class Distribution>
struct FillRandomTask<Generator, Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  static void Run(Generator gen, T* data, int64 size, const T* cmp_data,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    // Decide skip strides according to different kResultElementCount:
    // * `1 = (4 + 3) / 4` for normal Distribution.
    // * `1 = (2 + 3) / 4` for double/int64 Distribution.
    // * `4 = (16 + 3) / 4` for vectorized float/bfloat16 Distribution.
    // * `1 = (1024 + 1023) / 1024` for PCG generator.
    const int skip_strides =
        (kGroupSize + gen.kResultElementCount - 1) / gen.kResultElementCount;
    gen.Skip(start_group * skip_strides);
    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    DistributionVec<Generator, T> dist_vec(&dist, cmp_data);
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

    dist_vec.VecPost(
        data + start_group * kGroupSize,
        (limit_group_full - start_group) * kGroupSize + remaining_size);
  }
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Generator, class Distribution>
struct FillRandomTask<Generator, Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  static constexpr int64 kReservedSamplesPerOutput = 256;

  static void Run(Generator base_gen, T* data, int64 size, const T* cmp_data,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    static const int kGeneratorSkipPerOutputGroup =
        kGroupSize * kReservedSamplesPerOutput / Generator::kResultElementCount;

    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    int64 group_index;
    DistributionVec<SingleSampleAdapter<Generator>, T> dist_vec(&dist,
                                                                cmp_data);
    for (group_index = start_group; group_index < limit_group_full;
         ++group_index) {
      // Reset the generator to the beginning of the output group region
      // This is necessary if we want the results to be independent of order
      // of work
      Generator gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<Generator> single_samples(&gen);

      auto samples = dist_vec(&single_samples);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    int64 remaining_size = 0;
    if (limit_group_full < limit_group) {
      Generator gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<Generator> single_samples(&gen);

      remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist_vec(&single_samples);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }

    dist_vec.VecPost(
        data + start_group * kGroupSize,
        (limit_group_full - start_group) * kGroupSize + remaining_size);
  }
};

// Declares the common part for CPU-specialized FillPhiloxRandom/FillPCGRandom.
template <class Generator, class Distribution>
void FillRandom(
    OpKernelContext* ctx, const CPUDevice& d, Generator gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist,
    const typename Distribution::ResultElementType* cmp_data = nullptr) {
  const int kGroupSize = Distribution::kResultElementCount;

  int64 total_group_count = (size + kGroupSize - 1) / kGroupSize;

  const int kGroupCost =
      kGroupSize * (Generator::kElementCost + Distribution::kElementCost);

  d.parallelFor(
      total_group_count, Eigen::TensorOpCost(0, 0, kGroupCost),
      [&gen, data, size, cmp_data, dist](Eigen::Index first,
                                         Eigen::Index last) {
        FillRandomTask<Generator, Distribution,
                       Distribution::kVariableSamplesPerOutput>::Run(gen, data,
                                                                     size,
                                                                     cmp_data,
                                                                     first,
                                                                     last,
                                                                     dist);
      });
}

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
  void operator()(
      OpKernelContext* ctx, const CPUDevice& d, PhiloxRandom gen,
      typename Distribution::ResultElementType* data, int64 size,
      Distribution dist, const uint64* key = nullptr,
      const uint64* counter = nullptr,
      const typename Distribution::ResultElementType* cmp_data = nullptr) {
    if (key != nullptr && counter != nullptr) {
      gen = GetPhiloxRandomFromCounterKeyMem(counter, key);
    }

    FillRandom(ctx, d, gen, data, size, dist, cmp_data);
  }
};

template <class Distribution>
struct FillPCGRandom<CPUDevice, Distribution> {
  void operator()(
      OpKernelContext* ctx, const CPUDevice& d, PCGRandom gen,
      typename Distribution::ResultElementType* data, int64 size,
      Distribution dist,
      const typename Distribution::ResultElementType* cmp_data = nullptr) {
    FillRandom(ctx, d, gen, data, size, dist, cmp_data);
  }
};

}  // namespace functor

}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_CPU_RANDOM_OP_CPU_H_
