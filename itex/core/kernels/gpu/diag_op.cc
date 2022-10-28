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

#include "itex/core/kernels/gpu/diag_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {
template struct DiagFunctor<GPUDevice, Eigen::half>;
template struct DiagFunctor<GPUDevice, Eigen::bfloat16>;
template struct DiagFunctor<GPUDevice, float>;
template struct DiagFunctor<GPUDevice, double>;
template struct DiagFunctor<GPUDevice, std::complex<float>>;
template struct DiagFunctor<GPUDevice, std::complex<double>>;
template struct DiagFunctor<GPUDevice, int32>;
template struct DiagFunctor<GPUDevice, int64>;

template struct DiagPartFunctor<GPUDevice, Eigen::half>;
template struct DiagPartFunctor<GPUDevice, Eigen::bfloat16>;
template struct DiagPartFunctor<GPUDevice, float>;
template struct DiagPartFunctor<GPUDevice, double>;
template struct DiagPartFunctor<GPUDevice, std::complex<float>>;
template struct DiagPartFunctor<GPUDevice, std::complex<double>>;
template struct DiagPartFunctor<GPUDevice, int32>;
template struct DiagPartFunctor<GPUDevice, int64>;
}  // namespace functor

// Generate the diagonal tensor with the diagonal set to the input tensor.
template <typename Device, typename T>
class DiagOp : public OpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& diagonal = context->input(0);
    const int num_dims = diagonal.dims();
    OP_REQUIRES(
        context, 0 != num_dims,
        errors::InvalidArgument("Input must be at least rank 1, got 0"));
    TensorShape out_shape;
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));
    functor::DiagFunctor<Device, T> diagFunc;
    Status s =
        diagFunc(context, diagonal.NumElements(), diagonal.flat<T>().data(),
                 output_tensor->flat<T>().data());
    OP_REQUIRES_OK(context, s);
  }
};

// Extract the diagonal tensor with the diagonal set to the input tensor.
template <typename Device, typename T>
class DiagPartOp : public OpKernel {
 public:
  explicit DiagPartOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    const int num_dims = tensor.dims();
    const int out_dims = num_dims / 2;
    OP_REQUIRES(
        context, 0 == num_dims % 2,
        errors::InvalidArgument(
            "The rank of the tensor should be even and positive, got shape ",
            tensor.shape().DebugString()));
    for (int i = 0; i < out_dims; i++) {
      OP_REQUIRES(
          context, tensor.dim_size(i) == tensor.dim_size(i + out_dims),
          errors::InvalidArgument("Invalid shape ",
                                  tensor.shape().DebugString(), ": dimensions ",
                                  i, " and ", i + out_dims, " do not match."));
    }

    TensorShape out_shape;
    for (int i = 0; i < out_dims; ++i) {
      out_shape.AddDim(tensor.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    functor::DiagPartFunctor<Device, T> diagPartFunc;
    Status s = diagPartFunc(context, out_shape.num_elements(),
                            tensor.flat<T>().data(), output->flat<T>().data());
    OP_REQUIRES_OK(context, s);
  }
};

#define REGISTER_DIAGOP_GPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Diag").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DiagOp<GPUDevice, T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_DIAGOP_GPU);
TF_CALL_complex64(REGISTER_DIAGOP_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_DIAGOP_GPU);
TF_CALL_complex128(REGISTER_DIAGOP_GPU);
#endif
TF_CALL_int32(REGISTER_DIAGOP_GPU);
TF_CALL_int64(REGISTER_DIAGOP_GPU);
#undef REGISTER_DIAGOP_GPU

#define REGISTER_DIAGPARTOP_GPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("DiagPart").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DiagPartOp<GPUDevice, T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_DIAGPARTOP_GPU);
TF_CALL_complex64(REGISTER_DIAGPARTOP_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_DIAGPARTOP_GPU);
TF_CALL_complex128(REGISTER_DIAGPARTOP_GPU);
#endif
TF_CALL_int32(REGISTER_DIAGPARTOP_GPU);
TF_CALL_int64(REGISTER_DIAGPARTOP_GPU);
#undef REGISTER_DIAGPARTOP_GPU

}  // namespace itex
