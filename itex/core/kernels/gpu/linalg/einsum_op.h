/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_GPU_LINALG_EINSUM_OP_H_
#define ITEX_CORE_KERNELS_GPU_LINALG_EINSUM_OP_H_

#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename Device, typename T, int N>
struct StrideFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.stride(strides);
  }
};

template <typename Device, typename T, int N>
struct InflateFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.inflate(strides);
  }
};

#define DECLARE_GPU_SPECS_NDIM(T, NDIM)                              \
  template struct functor::StrideFunctor<Eigen::GpuDevice, T, NDIM>; \
  template struct functor::InflateFunctor<Eigen::GpuDevice, T, NDIM>;

#define DECLARE_GPU_SPECS(T)    \
  DECLARE_GPU_SPECS_NDIM(T, 1); \
  DECLARE_GPU_SPECS_NDIM(T, 2); \
  DECLARE_GPU_SPECS_NDIM(T, 3); \
  DECLARE_GPU_SPECS_NDIM(T, 4); \
  DECLARE_GPU_SPECS_NDIM(T, 5); \
  DECLARE_GPU_SPECS_NDIM(T, 6);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_NDIM

}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_LINALG_EINSUM_OP_H_
