/* Copyright (c) 2021 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_LIB_RANDOM_GUARDED_PHILOX_RANDOM_H_
#define ITEX_CORE_UTILS_LIB_RANDOM_GUARDED_PHILOX_RANDOM_H_

#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/mutex.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/types.h"

namespace itex {

// A thread safe wrapper around a Philox generator.  Example usage:
//
//   GuardedRandomPhilox generator;
//   generator.Init(context);
//
//   // In thread safe code
//   const int samples = ...;
//   auto local_generator = generator.ReserveSamples128(samples);
//   for (int i = 0; i < samples; i++)
//     Array<uint32, 4> sample = local_generator();
//     // Use sample
//   }
//
class GuardedPhiloxRandom {
 public:
  // Must call Init to finish initialization
  GuardedPhiloxRandom() : initialized_(false) {}

  // Initialize the generator from attributes "seed" and "seed2".
  // If both seeds are unspecified, use random seeds.
  // Must be called exactly once.
  Status Init(OpKernelConstruction* context);

  // Initialize with given seeds.
  void Init(int64 seed, int64 seed2);
  void Init(random::PhiloxRandom::ResultType counter,
            random::PhiloxRandom::Key key);

  // Reserve a certain number of 128-bit samples.
  // This function is thread safe.  The returned generator is valid for the
  // given number of samples, and can be used without a lock.
  random::PhiloxRandom ReserveSamples128(int64 samples);

  // Reserve a certain number of 32-bit samples.
  random::PhiloxRandom ReserveSamples32(int64 samples) {
    return ReserveSamples128((samples + 3) / 4);
  }

  // Reserve enough random samples in the generator for the given output count.
  random::PhiloxRandom ReserveRandomOutputs(int64 output_count,
                                            int multiplier) {
    int64 conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  mutex mu_;
  random::PhiloxRandom generator_ TF_GUARDED_BY(mu_);
  bool initialized_;

  TF_DISALLOW_COPY_AND_ASSIGN(GuardedPhiloxRandom);
};

class GuardedPCGRandom {
 public:
  // Must call Init to finish initialization
  GuardedPCGRandom() : initialized_(false) {}

  // Initialize the generator from attributes "seed" and "seed2".
  // If both seeds are unspecified, use random seeds.
  // Must be called exactly once.
  Status Init(OpKernelConstruction* context);

  // Initialize with given seeds.
  void Init(int64 seed, int64 seed2);

  // Reserve a certain number of 256-bit samples.
  // This function is thread safe.  The returned generator is valid for the
  // given number of samples, and can be used without a lock.
  random::PCGRandom ReserveSamples256(int64 samples);

  // Reserve a certain number of 32-bit samples.
  random::PCGRandom ReserveSamples32(int64 samples) {
    return ReserveSamples256((samples + 7) / 8);
  }

  // Reserve enough random samples in the generator for the given output count.
  random::PCGRandom ReserveRandomOutputs(int64 output_count, int multiplier) {
    int64 conservative_sample_count = output_count * multiplier;
    return ReserveSamples256(conservative_sample_count);
  }

 private:
  mutex mu_;
  random::PCGRandom generator_ TF_GUARDED_BY(mu_);
  bool initialized_;

  TF_DISALLOW_COPY_AND_ASSIGN(GuardedPCGRandom);
};

}  // namespace itex

#endif  // ITEX_CORE_UTILS_LIB_RANDOM_GUARDED_PHILOX_RANDOM_H_
