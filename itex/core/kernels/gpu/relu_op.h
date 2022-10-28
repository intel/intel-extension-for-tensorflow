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

#ifndef ITEX_CORE_KERNELS_GPU_RELU_OP_H_
#define ITEX_CORE_KERNELS_GPU_RELU_OP_H_

#include "itex/core/kernels/common/relu_op.h"
#include "itex/core/kernels/gpu/relu_op_functor.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/numeric_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
// Out of line check to save code space (we have this code once, rather
// than once for every NDIMS * NumTypes * Num_different_relu_variants
// functions.
struct ReluHelpers {
  static void ValidateSameSizeHelper(OpKernelContext* context, const Tensor& g,
                                     const Tensor& a) {
    OP_REQUIRES(context, a.IsSameSize(g),
                errors::InvalidArgument("g and a must be the same size"));
  }
  static bool ValidateSameSize(OpKernelContext* context, const Tensor& g,
                               const Tensor& a) {
    ValidateSameSizeHelper(context, g, a);
    return context->status().ok();
  }
};

template <typename Device, typename T>
class SeluOp : public UnaryElementWiseOp<T, SeluOp<Device, T>> {
 public:
  using UnaryElementWiseOp<T, SeluOp<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Selu<Device, T> functor;
    functor(context->eigen_gpu_device(), input.flat<T>(), output->flat<T>());
  }
};

template <typename Device, typename T>
class SeluGradOp : public BinaryElementWiseOp<T, SeluGradOp<Device, T>> {
 public:
  using BinaryElementWiseOp<T, SeluGradOp<Device, T>>::BinaryElementWiseOp;

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (outputs): outputs of the SeluOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }
};

template <typename Device, typename T>
void SeluGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                              const Tensor& g, const Tensor& a,
                                              Tensor* output) {
  if (!ReluHelpers::ValidateSameSize(context, g, a)) return;
  functor::SeluGrad<Device, T> functor;
  functor(context->eigen_gpu_device(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_RELU_OP_H_
