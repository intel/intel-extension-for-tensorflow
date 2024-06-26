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

#ifndef ITEX_CORE_UTILS_LIB_RANDOM_PHILOX_RANDOM_H_
#define ITEX_CORE_UTILS_LIB_RANDOM_PHILOX_RANDOM_H_

#include <cmath>
#include <cstdlib>

#include "itex/core/utils/types.h"

// Implement the Philox algorithm to generate random numbers in parallel.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
//   http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

// Function qualifiers that need to work on both CPU and GPU.
// For non-nvcc.
#define PHILOX_DEVICE_FUNC
#define PHILOX_INLINE inline
#define PHILOX_DEVICE_INLINE PHILOX_DEVICE_FUNC PHILOX_INLINE

namespace itex {
namespace random {

// A class that represents an inline array. It can be used on both CPU and GPU,
// and also trivially copyable between CPU and GPU.
// Arguments:
//   T: the array element type;
//   ElementCount: the fixed size of the array;
template <typename T, int ElementCount>
class Array {
 public:
  static constexpr int kElementCount = ElementCount;
  PHILOX_DEVICE_INLINE Array() {
    for (int i = 0; i < ElementCount; ++i) {
      data_[i] = T(0);
    }
  }

  PHILOX_DEVICE_INLINE T& operator[](int index) const {
    return const_cast<T&>(data_[index]);
  }

  PHILOX_DEVICE_INLINE T& operator[](int index) { return data_[index]; }

  size_t size() const { return ElementCount; }

 private:
  T data_[ElementCount];
};

// A class that encapsulates all the states for a random number generator using
// the philox_4x32_10 algorithm. Each invocation returns a 128-bit random bits
// in the form of four uint32.
// There are multiple variants of this algorithm, we picked the 4x32_10 version
// that is most suited for our applications.
// Since this class is meant to be copied between CPU to GPU, it maintains a
// value semantics.
//
// For example: To use this class and populate an array of 1024 randoms on CPU
// with two threads,
//
//  void Fill(PhiloxRandom rnd, uint32* output, int start, int limit) {
//    assert(start % 4 == 0);
//    assert(limit % 4 == 0);
//    rnd.Skip(start / 4);
//    for (int i = start; i < limit; i += 4) {
//      auto sample = rnd();
//      ... copy sample[0..3] to output[i..i+3]
//    }
//  }
//
//  PhiloxRandom rng(seed);
//  PhiloxRandom rng_copy = rng;
//  rng.Skip(1000/4);
//
//  ... schedule Fill(rng_copy, output, 0, 512) in thread 1;
//  ... schedule Fill(rng_copy, output, 512, 1024) in thread 2;
//  ... wait for thread 1 & 2 to finish executing Fill().
//
// NOTE:
// 1. PhiloxRandom is trivially copyable.
// 2. PhiloxRandom is compilable by gcc and nvcc.
class PhiloxRandom {
 public:
  using ResultType = Array<uint32, 4>;
  using ResultElementType = uint32;
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = 4;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 10;
  // The type for the 64-bit key stored in the form of two 32-bit uint
  // that are used in the diffusion process.
  using Key = Array<uint32, 2>;

  PHILOX_DEVICE_INLINE
  PhiloxRandom() {}

  PHILOX_DEVICE_INLINE
  explicit PhiloxRandom(uint64 seed) {
    key_[0] = static_cast<uint32>(seed);
    key_[1] = static_cast<uint32>(seed >> 32);
  }

  PHILOX_DEVICE_INLINE
  explicit PhiloxRandom(uint64 seed_lo, uint64 seed_hi) {
    key_[0] = static_cast<uint32>(seed_lo);
    key_[1] = static_cast<uint32>(seed_lo >> 32);
    counter_[2] = static_cast<uint32>(seed_hi);
    counter_[3] = static_cast<uint32>(seed_hi >> 32);
  }

  PHILOX_DEVICE_INLINE
  PhiloxRandom(ResultType counter, Key key) : counter_(counter), key_(key) {}

  PHILOX_DEVICE_INLINE
  ResultType const& counter() const { return counter_; }

  PHILOX_DEVICE_INLINE
  Key const& key() const { return key_; }

  // Skip the specified number of samples of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE
  void Skip(uint64 count) const {
    const uint32 count_lo = static_cast<uint32>(count);
    uint32 count_hi = static_cast<uint32>(count >> 32);

    counter_[0] += count_lo;
    if (counter_[0] < count_lo) {
      ++count_hi;
    }

    counter_[1] += count_hi;
    if (counter_[1] < count_hi) {
      if (++counter_[2] == 0) {
        ++counter_[3];
      }
    }
  }

  // Returns a group of four random numbers using the underlying Philox
  // algorithm.
  PHILOX_DEVICE_INLINE ResultType operator()() const {
    ResultType counter = counter_;
    Key key = key_;

    // Run the single rounds for ten times. Manually unrolling the loop
    // for better performance.
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);

    SkipOne();

    return counter;
  }

 private:
  // We use the same constants as recommended by the original paper.
  static constexpr uint32 kPhiloxW32A = 0x9E3779B9;
  static constexpr uint32 kPhiloxW32B = 0xBB67AE85;
  static constexpr uint32 kPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32 kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE void SkipOne() const {
    if (++counter_[0] == 0) {
      if (++counter_[1] == 0) {
        if (++counter_[2] == 0) {
          ++counter_[3];
        }
      }
    }
  }

  // Helper function to return the lower and higher 32-bits from two 32-bit
  // integer multiplications.
  // FIXME(itex): directly assign a * b/mul_hi(a, b) to result_low/result_high
  // causes segment fault, fix it in the future
  PHILOX_DEVICE_INLINE
  static void MultiplyHighLow(uint32 a, uint32 b, uint32* result_low,
                              uint32* result_high) {
#ifdef INTEL_CPU_ONLY
    const uint64 product = static_cast<uint64>(a) * b;
    *result_low = static_cast<uint32>(product);
    *result_high = static_cast<uint32>(product >> 32);
#else
    uint32 c = a * b;
    uint32 d = sycl::mul_hi(a, b);
    *result_low = c;
    *result_high = d;
#endif
  }

  // Helper function for a single round of the underlying Philox algorithm.
  PHILOX_DEVICE_INLINE static ResultType ComputeSingleRound(
      const ResultType& counter, const Key& key) {
    uint32 lo0;
    uint32 hi0;
    MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32 lo1;
    uint32 hi1;
    MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

    ResultType result;
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    return result;
  }

  PHILOX_DEVICE_INLINE void RaiseKey(Key* key) const {
    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  ResultType counter_;
  Key key_;
};

class PCGRandom {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = 1024;
  using ResultType = Array<uint32, kResultElementCount>;
  using StateType = Array<uint64, kResultElementCount>;
  using StateElementType = uint64;
  using ResultElementType = uint32;

  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 10;

  PHILOX_DEVICE_INLINE
  PCGRandom() {}

  PHILOX_DEVICE_INLINE
  explicit PCGRandom(uint64 seed) {
    for (int i = 0; i < kResultElementCount; i++) {
      state_[i] = seed;
    }
  }

  PHILOX_DEVICE_INLINE
  explicit PCGRandom(uint64 seed_lo, uint64 seed_hi) {
    for (int i = 0; i < kResultElementCount; i++) {
      state_[i] = seed_lo;
    }
  }

  // Skip the specified number of samples of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE
  void Skip(uint64 count) {
    for (int i = 0; i < kResultElementCount; i++) {
      uint64 cur_inc = i * 2 + 1;
      uint64 cur_mul = 6364136223846793005ULL;
      uint64 acc_mult = 1;
      uint64 acc_plus = 0;
      while (count > 0) {
        if (count & 1) {
          acc_mult *= cur_mul;
          acc_plus = acc_plus * cur_mul + cur_inc;
        }
        cur_inc = (cur_mul + 1) * cur_inc;
        cur_mul *= cur_mul;
        count = count >> 1;
      }
      state_[i] = acc_mult * state_[i] + acc_plus;
    }
  }

  // Returns a group of four random numbers using the underlying Philox
  // algorithm.
  PHILOX_DEVICE_INLINE ResultType operator()() const {
    ResultType result = ComputeSingleRound();
    return result;
  }

 private:
  // We use the same constants as recommended by the original paper.
  // Helper function to skip the next sample of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE void SkipOne() { Skip(1); }

  PHILOX_DEVICE_INLINE
  ResultType ComputeSingleRound() const {
    ResultType xorshifted;
    ResultType rot;
    ResultType result;
    for (int i = 0; i < kResultElementCount; i++) {
      xorshifted[i] = ((state_[i] >> 18u) ^ state_[i]) >> 27u;
      rot[i] = state_[i] >> 59u;
      result[i] =
          (xorshifted[i] >> rot[i]) | (xorshifted[i] << ((-rot[i]) & 31));
      state_[i] = state_[i] * 6364136223846793005ULL + (i * 2 + 1);
    }
    return result;
  }

 private:
  StateType state_;
};

}  // namespace random
}  // namespace itex

#endif  // ITEX_CORE_UTILS_LIB_RANDOM_PHILOX_RANDOM_H_
