/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_COMMON_RANDOM_OPS_UTIL_H_
#define ITEX_CORE_KERNELS_COMMON_RANDOM_OPS_UTIL_H_

#include "itex/core/utils/lib/random/philox_random.h"

namespace itex {

enum Algorithm {
  // The Philox algorithm, as described in paper
  // ['Parallel Random Numbers: As Easy as 1, 2, 3']
  // (https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
  RNG_ALG_PHILOX = 1,
  // The ThreeFry algorithm, as described in paper
  // ['Parallel Random Numbers: As Easy as 1, 2, 3']
  // (https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
  RNG_ALG_THREEFRY = 2,
  // An algorithm auto-selected by the system according to device type.
  RNG_ALG_AUTO_SELECT = 3
};

static constexpr int RNG_KEY_SIZE = 1;
static constexpr int RNG_MAX_COUNTER_SIZE = 2;
// Gets the counter size (in unit of uint64) for a counter-based RNG
// algorithm `alg`. In the case of RNG_ALG_AUTO_SELECT, gets the minimal
// counter size among all algorithms.
inline int GetCounterSize(Algorithm alg) {
  if (alg == RNG_ALG_PHILOX) {
    return 2;
  } else if (alg == RNG_ALG_THREEFRY) {
    return 1;
  }
  return 1;
}

using StateElementType = int64;
using random::PhiloxRandom;

static constexpr int64 PHILOX_MIN_STATE_SIZE =
    (PhiloxRandom::ResultType::kElementCount +
     PhiloxRandom::Key::kElementCount) /
    2;
static constexpr int64 THREEFRY_MIN_STATE_SIZE = 2;

// The following 2 functions use the contract "lower 32 bits for the first
// uint32, higher 32 bits for the second". Note that this is endian-neutral,
// unlike a direct memory copy `memcpy(output, &input, 8)`.
PHILOX_DEVICE_INLINE void Uint64ToUint32s(uint64 input, uint32* output1,
                                          uint32* output2) {
  *output1 = static_cast<uint32>(input);
  *output2 = static_cast<uint32>(input >> 32);
}

PHILOX_DEVICE_INLINE uint64 Uint32sToUint64(uint32 input1, uint32 input2) {
  auto u64_1 = static_cast<uint64>(input1);
  auto u64_2 = static_cast<uint64>(input2);
  return u64_1 | (u64_2 << 32);
}

PHILOX_DEVICE_INLINE PhiloxRandom::ResultType GetCounterFromMem(
    uint64 const* ptr) {
  PhiloxRandom::ResultType counter;
  Uint64ToUint32s(ptr[0], &counter[0], &counter[1]);
  Uint64ToUint32s(ptr[1], &counter[2], &counter[3]);
  return counter;
}

PHILOX_DEVICE_INLINE void WriteCounterToMem(
    PhiloxRandom::ResultType const& counter, uint64* ptr) {
  ptr[0] = Uint32sToUint64(counter[0], counter[1]);
  ptr[1] = Uint32sToUint64(counter[2], counter[3]);
}

PHILOX_DEVICE_INLINE PhiloxRandom::Key GetKeyFromMem(uint64 const* ptr) {
  PhiloxRandom::Key key;
  Uint64ToUint32s(ptr[0], &key[0], &key[1]);
  return key;
}

PHILOX_DEVICE_INLINE void WriteKeyToMem(PhiloxRandom::Key const& key,
                                        uint64* ptr) {
  *ptr = Uint32sToUint64(key[0], key[1]);
}

PHILOX_DEVICE_INLINE PhiloxRandom GetPhiloxRandomFromCounterKeyMem(
    uint64 const* counter_ptr, uint64 const* key_ptr) {
  return PhiloxRandom(GetCounterFromMem(counter_ptr), GetKeyFromMem(key_ptr));
}

// The following 5 functions are made templates to avoid duplicate symbols when
// linking.

// The following 2 functions use the contract "lower 32 bits for the first
// uint32, higher 32 bits for the second". Note that this is endian-neutral,
// unlike a direct memory copy `memcpy(output, &input, 8)`.
PHILOX_DEVICE_INLINE void Int64ToUint32s(int64 input, uint32* output1,
                                         uint32* output2) {
  auto u64 = static_cast<uint64>(input);
  *output1 = static_cast<uint32>(u64);
  *output2 = static_cast<uint32>(u64 >> 32);
}

PHILOX_DEVICE_INLINE int64 Uint32sToInt64(uint32 input1, uint32 input2) {
  auto u64_1 = static_cast<uint64>(input1);
  auto u64_2 = static_cast<uint64>(input2);
  return static_cast<int64>(u64_1 | (u64_2 << 32));
}

PHILOX_DEVICE_INLINE PhiloxRandom
GetPhiloxRandomFromMem(StateElementType const* ptr) {
  PhiloxRandom::ResultType counter;
  PhiloxRandom::Key key;
  Int64ToUint32s(ptr[0], &counter[0], &counter[1]);
  Int64ToUint32s(ptr[1], &counter[2], &counter[3]);
  Int64ToUint32s(ptr[2], &key[0], &key[1]);
  return PhiloxRandom(counter, key);
}

PHILOX_DEVICE_INLINE void WritePhiloxRandomToMem(PhiloxRandom const& philox,
                                                 StateElementType* ptr) {
  PhiloxRandom::ResultType const& counter = philox.counter();
  PhiloxRandom::Key const& key = philox.key();
  ptr[0] = Uint32sToInt64(counter[0], counter[1]);
  ptr[1] = Uint32sToInt64(counter[2], counter[3]);
  ptr[2] = Uint32sToInt64(key[0], key[1]);
}

PHILOX_DEVICE_INLINE void UpdateMemWithPhiloxRandom(PhiloxRandom const& philox,
                                                    int64 output_size,
                                                    StateElementType* ptr) {
  auto new_philox = philox;
  // Multiplier 256 is the same as in `FillPhiloxRandomTask`; do not change
  // it just here.
  auto delta = output_size * 256;
  new_philox.Skip(delta);  // do the actual increasing
  WritePhiloxRandomToMem(new_philox, ptr);
}

}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_RANDOM_OPS_UTIL_H_
