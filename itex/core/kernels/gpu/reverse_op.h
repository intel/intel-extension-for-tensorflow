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

#ifndef ITEX_CORE_KERNELS_GPU_REVERSE_OP_H_
#define ITEX_CORE_KERNELS_GPU_REVERSE_OP_H_

#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

// Functor used by ReverseOp to do the computations.
template <typename Device, typename T, int Dims>
struct Reverse {
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  const Eigen::array<bool, Dims>& reverse_dims,
                  typename TTypes<T, Dims>::Tensor output,
                  const bool can_use_32bit) {
    if (can_use_32bit) {
      To32Bit(output).device(d) = To32Bit(input).reverse(reverse_dims);
    } else {
      output.device(d) = input.reverse(reverse_dims);
    }
  }
};

template <typename Device, typename T>
struct Reverse<Device, T, 0> {
  void operator()(const Device& d, typename TTypes<T, 0>::ConstTensor input,
                  const Eigen::array<bool, 0>& reverse_dims,
                  typename TTypes<T, 0>::Tensor output,
                  const bool can_use_32bit) {
    // Reversing a scalar is copying it.
    output.device(d) = input;
  }
};

}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_REVERSE_OP_H_
