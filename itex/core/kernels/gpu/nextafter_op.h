/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_NEXTAFTER_OP_H_
#define ITEX_CORE_KERNELS_GPU_NEXTAFTER_OP_H_

#include "itex/core/kernels/common/cwise_ops.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename T>
struct nextafter_op {
  EIGEN_EMPTY_STRUCT_CTOR(nextafter_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T operator()(const T& x1,
                                                           const T& x2) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::nextafter(x1, x2);
#else
    return std::nextafter(x1, x2);
#endif  // __SYCL_DEVICE_ONLY__
  }
};

template <>
struct nextafter_op<Eigen::bfloat16> {
  EIGEN_EMPTY_STRUCT_CTOR(nextafter_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::bfloat16 operator()(
      const Eigen::bfloat16& x1, const Eigen::bfloat16& x2) const {
    return static_cast<Eigen::bfloat16>(
        sycl::nextafter(static_cast<float>(x1), static_cast<float>(x2)));
  }
};

template <>
struct nextafter_op<Eigen::half> {
  EIGEN_EMPTY_STRUCT_CTOR(nextafter_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half operator()(
      const Eigen::half& x1, const Eigen::half& x2) const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<Eigen::half>(sycl::nextafter(x1.x, x2.x));
#else
    return static_cast<Eigen::half>(
        std::nextafter(static_cast<float>(x1), static_cast<float>(x2)));
#endif  // __SYCL_DEVICE_ONLY__
  }
};

template <typename T>
struct nextafter : base<T, nextafter_op<T>> {};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_NEXTAFTER_OP_H_
