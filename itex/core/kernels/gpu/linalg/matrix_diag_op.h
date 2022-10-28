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

#ifndef ITEX_CORE_KERNELS_GPU_LINALG_MATRIX_DIAG_OP_H_
#define ITEX_CORE_KERNELS_GPU_LINALG_MATRIX_DIAG_OP_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

// Reads the diagonal packing alignment.
void ReadAlignment(OpKernelConstruction* context,
                   bool* left_align_superdiagonal,
                   bool* left_align_subdiagonal);

template <typename Device, typename T>
struct MatrixDiagPart {
  EIGEN_ALWAYS_INLINE static void Compute(
      OpKernelContext* context, const Device& device,
      const typename TTypes<T, 3>::ConstTensor& input,
      const typename TTypes<T>::Tensor& output,
      const Eigen::Index lower_diag_index, const Eigen::Index upper_diag_index,
      const Eigen::Index max_diag_len, const T padding_value,
      const bool left_align_superdiagonal, const bool left_align_subdiagonal);
};

template <typename Device, typename T>
struct MatrixDiag {
  EIGEN_ALWAYS_INLINE static void Compute(
      OpKernelContext* context, const Device& device,
      const typename TTypes<T>::ConstTensor& diag,
      const typename TTypes<T, 3>::Tensor& output,
      const Eigen::Index lower_diag_index, const Eigen::Index upper_diag_index,
      const Eigen::Index max_diag_len, const T padding_value,
      const bool left_align_superdiagonal, const bool left_align_subdiagonal);
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_LINALG_MATRIX_DIAG_OP_H_
