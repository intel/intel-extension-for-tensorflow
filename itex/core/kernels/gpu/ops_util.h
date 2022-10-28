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

#ifndef ITEX_CORE_KERNELS_GPU_OPS_UTIL_H_
#define ITEX_CORE_KERNELS_GPU_OPS_UTIL_H_

// This file contains utilities for various operations.

#include <array>

#include "itex/core/utils/padding.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

// Converts Brain's Padding to Eigen's PaddingType.
Eigen::PaddingType BrainPadding2EigenPadding(Padding padding);

template <typename T>
bool IsInnerDimsSizeAligned(const TensorShape& s) {
  if (s.dims() == 0) return false;
  const int64_t dim0_size = s.dim_size(0);
  if (dim0_size == 0) return false;
#if EIGEN_MAX_ALIGN_BYTES == 0
  return true;
#else
  const int64_t bytes_per_dim0 = (s.num_elements() / dim0_size) * sizeof(T);
  return bytes_per_dim0 % EIGEN_MAX_ALIGN_BYTES == 0;
#endif
}

// Given a shape 's' of a tensor of type T and the `start` and `end` index of a
// dim 0 slice, returns true iff slice is aligned with respect to original
// tensor. Here aligned implies the address is a multiple of
// EIGEN_MAX_ALIGN_BYTES.
template <typename T>
bool IsDim0SliceAligned(const TensorShape& s, int64_t start,
                        int64_t end_or_size) {
  if (s.dims() == 1) {
#if EIGEN_MAX_ALIGN_BYTES == 0
    return true;
#else
    bool start_aligned = (start * sizeof(T)) % EIGEN_MAX_ALIGN_BYTES == 0;
    // End is aligned if either the explicit end index is passed and is a
    // a multiple of EIGEN_MAX_ALIGN_BYTES, or the start index is aligned and
    // the size is aligned. So for convenience we can either pass start and
    // index, or start and size.
    bool end_aligned = (end_or_size * sizeof(T)) % EIGEN_MAX_ALIGN_BYTES == 0;
    return start_aligned && end_aligned;
#endif
  } else {
    return IsInnerDimsSizeAligned<T>(s);
  }
}

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_OPS_UTIL_H_
