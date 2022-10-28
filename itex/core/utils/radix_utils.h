/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_RADIX_UTILS_H_
#define ITEX_CORE_UTILS_RADIX_UTILS_H_

#include "itex/core/utils/types.h"
// ------------------------------------------------------------------

// Type selection
template <bool IF, typename ThenType, typename ElseType>
struct If {
  using Type = ThenType;
};

template <typename ThenType, typename ElseType>
struct If<false, ThenType, ElseType> {
  using Type = ElseType;
};

// Allows for the treatment of an integral constant
// as a type at compile-time
template <int VAL>
struct Int2Type {
  enum { VALUE = VAL };
};

// Base type for local memory management
template <typename T>
struct BaseStorage {
  enum {
    SIZE = sizeof(T),
  };

  uint8_t storage[SIZE];

  T& Alias() { return reinterpret_cast<T&>(*this); }
};

// Statically determine log2(N), rounded up
template <int N, int VAL = N, int COUNT = 0>
struct Log2 {
  enum { VALUE = Log2<N, (VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
  enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

// Base Unsigned Type
template <size_t TYPE_SIZE>
struct BaseUnsignedT {};
template <>
struct BaseUnsignedT<1> {
  using _type = uint8_t;
};
template <>
struct BaseUnsignedT<2> {
  using _type = uint16_t;
};
template <>
struct BaseUnsignedT<4> {
  using _type = uint32_t;
};
template <>
struct BaseUnsignedT<8> {
  using _type = uint64_t;
};

// NumericTraits
// For unknown/unsupported type we do not have any trait
template <typename T, typename Dummy = void>
struct NumericTraits {};

// For unsigned integrals we use the same type
template <typename T>
struct NumericTraits<T, std::enable_if_t<std::is_integral<T>::value &&
                                         std::is_unsigned<T>::value>> {
  using UnsignedT = T;

  // In ascending order
  static UnsignedT Convert(UnsignedT key, Int2Type<false> /*is_descending*/) {
    return key;
  }
  static UnsignedT ConvertBack(UnsignedT key,
                               Int2Type<false> /*is_descending*/) {
    return key;
  }

  // In descending order
  static UnsignedT Convert(UnsignedT key, Int2Type<true> /*is_descending*/) {
    return ~key;
  }
  static UnsignedT ConvertBack(UnsignedT key,
                               Int2Type<true> /*is_descending*/) {
    return ~key;
  }
};

// For signed integrals
template <typename T>
struct NumericTraits<T, std::enable_if_t<std::is_integral<T>::value &&
                                         std::is_signed<T>::value>> {
  using UnsignedT = typename BaseUnsignedT<sizeof(T)>::_type;

  static constexpr UnsignedT MASK1 = UnsignedT(1) << (sizeof(T) * 8 - 1);
  static constexpr UnsignedT MASK2 = ~MASK1;

  // In ascending order
  static UnsignedT Convert(UnsignedT key, Int2Type<false> /*is_descending*/) {
    return key ^ MASK1;
  }
  static UnsignedT ConvertBack(UnsignedT key,
                               Int2Type<false> /*is_descending*/) {
    return key ^ MASK1;
  }

  // In descending order
  static UnsignedT Convert(UnsignedT key, Int2Type<true> /*is_descending*/) {
    return key ^ MASK2;
  }
  static UnsignedT ConvertBack(UnsignedT key,
                               Int2Type<true> /*is_descending*/) {
    return key ^ MASK2;
  }
};

// For floatings
// TODO(itex): special handling for floating-point -0.0
template <typename T>
struct NumericTraits<
    T, std::enable_if_t<std::is_floating_point<T>::value ||
                        std::is_same<T, Eigen::half>::value ||
                        std::is_same<T, Eigen::bfloat16>::value>> {
  using UnsignedT = typename BaseUnsignedT<sizeof(T)>::_type;

  static constexpr UnsignedT MASK1 = UnsignedT(1) << (sizeof(T) * 8 - 1);
  static constexpr UnsignedT MASK2 = ~MASK1;

  // In ascending order
  static UnsignedT Convert(UnsignedT key, Int2Type<false> /*is_descending*/) {
    UnsignedT mask = (key & MASK1) ? UnsignedT(-1) : MASK1;
    return key ^ mask;
  }
  static UnsignedT ConvertBack(UnsignedT key,
                               Int2Type<false> /*is_descending*/) {
    UnsignedT mask = (key & MASK1) ? MASK1 : UnsignedT(-1);
    return key ^ mask;
  }

  // In descending order
  static UnsignedT Convert(UnsignedT key, Int2Type<true> /*is_descending*/) {
    UnsignedT mask = (key & MASK1) ? UnsignedT(0) : MASK2;
    return key ^ mask;
  }
  static UnsignedT ConvertBack(UnsignedT key,
                               Int2Type<true> /*is_descending*/) {
    UnsignedT mask = (key & MASK1) ? UnsignedT(0) : MASK2;
    return key ^ mask;
  }
};

// RadixExtractor
template <typename KeyT>
class RadixExtractor {
 public:
  using UnsignedT = typename NumericTraits<KeyT>::UnsignedT;

  RadixExtractor(uint32_t begin_bit_, uint32_t num_bits_)
      : begin_bit(begin_bit_), bit_mask((1u << num_bits_) - 1u) {}

  // get the bucket from a certain number of bits
  uint32_t Bucket(UnsignedT key) { return key >> begin_bit & bit_mask; }

 private:
  const uint32_t begin_bit;
  const uint32_t bit_mask;
};
#endif  // ITEX_CORE_UTILS_RADIX_UTILS_H_
