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

#ifndef ITEX_CORE_KERNELS_GPU_BIAS_OP_H_
#define ITEX_CORE_KERNELS_GPU_BIAS_OP_H_

#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/kernels/gpu/reduction_ops.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

// Functor used by BiasOp to do the computations.
template <typename Device, typename T, int Dims>
struct Bias {
  // Add "bias" to "input", broadcasting it on all dimensions but the last one.
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  typename TTypes<T>::ConstVec bias,
                  typename TTypes<T, Dims>::Tensor output) {
    if (input.size() >= INT_MAX) {
      const Eigen::Index bias_size = bias.dimension(0);
      const Eigen::Index rest_size = input.size() / bias_size;
      Eigen::DSizes<Eigen::Index, 1> one_d(input.size());
      Eigen::DSizes<Eigen::Index, 1> bcast(rest_size);
      output.reshape(one_d).device(d) =
          input.reshape(one_d) + bias.broadcast(bcast);
    } else {
      const int bias_size = bias.dimension(0);
      const int rest_size = input.size() / bias_size;
      Eigen::DSizes<int, 1> one_d(input.size());
      Eigen::DSizes<int, 1> bcast(rest_size);
      To32Bit(output).reshape(one_d).device(d) =
          To32Bit(input).reshape(one_d) + To32Bit(bias).broadcast(bcast);
    }
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_BIAS_OP_H_
