/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// BitCast is an extension of std::bit_cast/absl::bit_cast. Whereas those
// functions require trivially copyable source and destination types, the
// present function template may be specialized for additional types that
// do not satisfy that triviality property, but that have alternative ways
// of accessing their underlying representation.
//
// Concretely, we provide specializations for the "custom floating point types"
// Eigen::half and Eigen::bfloat16. Those types are effectively stored as
// a sequence of bits, but the classes are not trivially copyable.

#ifndef ITEX_CORE_COMPILER_XLA_BIT_CAST_H_
#define ITEX_CORE_COMPILER_XLA_BIT_CAST_H_

#include "absl/base/casts.h"
#include "itex/core/compiler/xla/types.h"
#include "third_party/eigen3/Eigen/Core"

namespace Eigen {
namespace half_impl {
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC numext::uint16_t raw_half_as_uint16(
    const __half_raw& h) {
#if defined(SYCL_DEVICE_ONLY)
  return numext::bit_cast<numext::uint16_t>(h);
#else
  return h.x;
#endif
}
}  // namespace half_impl

namespace numext {
// template <typename Tgt, typename Src>
// EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Tgt bit_cast(const Src& src) {
//   Tgt tgt;
//   const Src staged = src;
//   EIGEN_USING_STD(memcpy)
//   memcpy(&tgt, &staged, sizeof(Tgt));
//   return tgt;
// }

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half
bit_cast<Eigen::half, uint16_t>(const uint16_t& src) {
  return Eigen::half(Eigen::half_impl::raw_uint16_to_half(src));
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint16_t
bit_cast<uint16_t, Eigen::half>(const Eigen::half& src) {
  return Eigen::half_impl::raw_half_as_uint16(src);
}

}  // namespace numext
}  // namespace Eigen

namespace itex_xla {

template <typename T, typename U>
T BitCast(U src) {
  static_assert(sizeof(T) == sizeof(U), "sizes don't match");
  // We would like to check std::is_trivially_copyable here, but there's no
  // reliable implementation of that available to us.
  return absl::bit_cast<T>(src);
}

template <>
inline Eigen::bfloat16 BitCast<Eigen::bfloat16, uint16_t>(uint16_t src) {
  return Eigen::numext::bit_cast<Eigen::bfloat16>(src);
}

template <>
inline uint16_t BitCast<uint16_t, Eigen::bfloat16>(Eigen::bfloat16 src) {
  return Eigen::numext::bit_cast<uint16_t>(src);
}

template <>
inline Eigen::half BitCast<Eigen::half, uint16_t>(uint16_t src) {
  return Eigen::numext::bit_cast<Eigen::half>(src);
}

template <>
inline uint16_t BitCast<uint16_t, Eigen::half>(Eigen::half src) {
  return Eigen::numext::bit_cast<uint16_t>(src);
}

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_BIT_CAST_H_
