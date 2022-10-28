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

#ifndef ITEX_CORE_KERNELS_GPU_RELU_OP_FUNCTOR_H_
#define ITEX_CORE_KERNELS_GPU_RELU_OP_FUNCTOR_H_
// Functor definition for ReluOp and ReluGradOp, must be compilable by nvcc.

#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {
// Functor used by SeluOp to do the computations.
template <typename Device, typename T>
struct Selu {
  // Computes Selu activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    // features.constant(?)
    const auto scale = static_cast<T>(1.0507009873554804934193349852946);
    const auto scale_alpha = static_cast<T>(1.7580993408473768599402175208123);
    const auto one = static_cast<T>(1);
    const auto zero = static_cast<T>(0);
    activations.device(d) =
        (features < zero)
            .select(scale_alpha * (features.exp() - features.constant(one)),
                    scale * features);
  }
};

// Functor used by SeluGradOp to do the computations.
template <typename Device, typename T>
struct SeluGrad {
  // Computes SeluGrad backprops.
  //
  // gradients: gradients backpropagated to the Selu op.
  // activations: outputs of the Selu op.
  // backprops: gradients to backpropagate to the Selu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor activations,
                  typename TTypes<T>::Tensor backprops) {
    const auto scale = static_cast<T>(1.0507009873554804934193349852946);
    const auto scale_alpha = static_cast<T>(1.7580993408473768599402175208123);
    backprops.device(d) =
        (activations < static_cast<T>(0))
            .select(gradients * (activations + scale_alpha), gradients * scale);
  }
};
}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_RELU_OP_FUNCTOR_H_
