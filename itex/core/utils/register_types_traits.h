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

#ifndef ITEX_CORE_UTILS_REGISTER_TYPES_TRAITS_H_
#define ITEX_CORE_UTILS_REGISTER_TYPES_TRAITS_H_

#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace itex {

// Remap POD types by size to equivalent proxy types. This works
// since all we are doing is copying data around.
struct UnusableProxyType;
template <typename Device, int size>
struct proxy_type_pod {
  typedef UnusableProxyType type;
};
#if defined(INTEL_CPU_ONLY) || defined(ITEX_ENABLE_DOUBLE)
template <>
struct proxy_type_pod<CPUDevice, 8> {
  typedef double type;
};
#endif
template <>
struct proxy_type_pod<CPUDevice, 4> {
  typedef float type;
};
template <>
struct proxy_type_pod<CPUDevice, 2> {
  typedef Eigen::half type;
};
template <>
struct proxy_type_pod<CPUDevice, 1> {
  typedef int8 type;
};
#ifdef ITEX_ENABLE_DOUBLE
template <>
struct proxy_type_pod<GPUDevice, 8> {
  typedef double type;
};
#endif
template <>
struct proxy_type_pod<GPUDevice, 4> {
  typedef float type;
};
template <>
struct proxy_type_pod<GPUDevice, 2> {
  typedef Eigen::half type;
};
template <>
struct proxy_type_pod<GPUDevice, 1> {
  typedef int8 type;
};

/// If POD we use proxy_type_pod, otherwise this maps to identity.
template <typename Device, typename T>
struct proxy_type {
  typedef typename std::conditional<
      std::is_arithmetic<T>::value,
      typename proxy_type_pod<Device, sizeof(T)>::type, T>::type type;
  static_assert(sizeof(type) == sizeof(T), "proxy_type_pod is not valid");
};

template <>
struct proxy_type<GPUDevice, ::itex::int64> {
  typedef ::itex::int64 type;
  static_assert(sizeof(type) == sizeof(::itex::int64),
                "proxy_type_pod is not valid");
};

/// The active proxy types
#define TF_CALL_CPU_PROXY_TYPES(m)                                     \
  TF_CALL_int64(m) TF_CALL_int32(m) TF_CALL_uint16(m) TF_CALL_int16(m) \
      TF_CALL_int8(m) TF_CALL_complex128(m)
#define TF_CALL_GPU_PROXY_TYPES(m)                                    \
  TF_CALL_double(m) TF_CALL_float(m) TF_CALL_half(m) TF_CALL_int32(m) \
      TF_CALL_int8(m)
#define TF_CALL_ITEX_GPU_PROXY_TYPES(m) \
  TF_CALL_double(m) TF_CALL_float(m) TF_CALL_int32(m)
}  // namespace itex

#endif  // ITEX_CORE_UTILS_REGISTER_TYPES_TRAITS_H_
