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

#include "itex/core/kernels/gpu/softsign_op.h"

#include "itex/core/utils/errors.h"
#include "itex/core/utils/numeric_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SoftsignOp : public UnaryElementWiseOp<T, SoftsignOp<Device, T>> {
 public:
  explicit SoftsignOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, SoftsignOp<Device, T>>(context) {}

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Softsign<Device, T> functor;
    functor(context->eigen_gpu_device(), input.flat<T>(), output->flat<T>());
  }
};

template <typename Device, typename T>
class SoftsignGradOp
    : public BinaryElementWiseOp<T, SoftsignGradOp<Device, T>> {
 public:
  explicit SoftsignGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, SoftsignGradOp<Device, T>>(context) {}

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to SoftsignOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }
};

template <typename Device, typename T>
void SoftsignGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                                  const Tensor& g,
                                                  const Tensor& a,
                                                  Tensor* output) {
  OP_REQUIRES(context, a.IsSameSize(g),
              errors::InvalidArgument("g and a must be the same size"));
  functor::SoftsignGrad<Device, T> functor;
  functor(context->eigen_gpu_device(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Softsign").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      SoftsignOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SoftsignGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftsignGradOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

}  // namespace itex
