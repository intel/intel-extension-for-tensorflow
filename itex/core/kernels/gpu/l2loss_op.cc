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

#include "itex/core/kernels/gpu/full_reduction_kernels.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct squareHalf {
  inline T operator()(T x) const { return static_cast<T>(0.5) * x * x; }
};

template <typename Device, typename T>
class L2LossOp : public OpKernel {
  explicit L2LossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {}
};

template <typename T>
class L2LossOp<GPUDevice, T> : public OpKernel {
 public:
  explicit L2LossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // The input tensor can be of any number of dimensions, even though it's
    // 2D in most typical applications.
    const Tensor& input = context->input(0);
    // The output is a single number.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    LaunchFullReduction<const T, T, T, sycl::plus<T>, squareHalf<T>>(
        context, input.flat<T>().data(), output->flat<T>().data(), T(0),
        input.flat<T>().size(), sycl::plus<T>(), squareHalf<T>());
  }
};

// specialization for half
template <>
class L2LossOp<GPUDevice, Eigen::half> : public OpKernel {
 public:
  explicit L2LossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // The input tensor can be of any number of dimensions, even though it's
    // 2D in most typical applications.
    const Tensor& input = context->input(0);
    // The output is a single number.
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    // TODO(itex) as Eigen::half will be replaced with sycl::half, we just
    // reinterpret_cast, and this part shall be removed when it's replaced
    typedef Eigen::half T;
    typedef sycl::half BaseT;
    const BaseT* base_input =
        reinterpret_cast<const BaseT*>(input.flat<T>().data());
    BaseT* base_output = reinterpret_cast<BaseT*>(output->flat<T>().data());
    int in_size = input.flat<T>().size();

    LaunchFullReduction<const BaseT, BaseT, float, sycl::plus<float>,
                        squareHalf<BaseT>>(context, base_input, base_output,
                                           0.0f, in_size, sycl::plus<float>(),
                                           squareHalf<BaseT>());
  }
};

template <>
class L2LossOp<GPUDevice, Eigen::bfloat16> : public OpKernel {
 public:
  explicit L2LossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // The input tensor can be of any number of dimensions, even though it's
    // 2D in most typical applications.
    const Tensor& input = context->input(0);
    // The output is a single number.
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    typedef Eigen::bfloat16 T;
    const T* input_ptr = input.flat<T>().data();
    T* output_ptr = output->flat<T>().data();
    int in_size = input.flat<T>().size();

    LaunchFullReduction<const T, T, float, sycl::plus<float>, squareHalf<T>>(
        context, input_ptr, output_ptr, 0.0f, in_size, sycl::plus<float>(),
        squareHalf<T>());
  }
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL_L2LOSS(T)                           \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("L2Loss").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      L2LossOp<GPUDevice, T>);

REGISTER_GPU_KERNEL_L2LOSS(float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_GPU_KERNEL_L2LOSS(double);
#endif
REGISTER_GPU_KERNEL_L2LOSS(Eigen::half);
REGISTER_GPU_KERNEL_L2LOSS(Eigen::bfloat16);
#undef REGISTER_GPU_KERNEL_L2LOSS
};  // namespace itex
