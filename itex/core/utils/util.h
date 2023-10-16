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

#ifndef ITEX_CORE_UTILS_UTIL_H_
#define ITEX_CORE_UTILS_UTIL_H_

#include <string>
#include <utility>

#include "itex/core/utils/tensor_shape.h"

namespace itex {
std::string SliceDebugString(const TensorShape& shape, const int64 flat);

// Helper to compute 'strides' given a tensor 'shape'. I.e.,
// strides[i] = prod(shape.dim_size[(i+1):])
template <typename T>
gtl::InlinedVector<T, 8> ComputeStride(const TensorShape& shape) {
  const int ndims = shape.dims();
  gtl::InlinedVector<T, 8> strides(ndims);
  T stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<T>(shape.dim_size(i));
  }
  return strides;
}

template <typename T>
inline T DataIndexInit(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T DataIndexInit(T offset, T* x, const T& X, Args&&... args) {
  offset = DataIndexInit(offset, std::forward<Args>(args)...);
  *x = offset % X;
  return offset / X;
}

inline bool DataIndexStep() { return true; }

template <typename T, typename... Args>
inline bool DataIndexStep(T* x, const T& X, Args&&... args) {
  if (DataIndexStep(std::forward<Args>(args)...)) {
    *x = ((*x + 1) == X) ? 0 : (*x + 1);
    return *x == 0;
  }
  return false;
}

}  // namespace itex

#endif  // ITEX_CORE_UTILS_UTIL_H_
