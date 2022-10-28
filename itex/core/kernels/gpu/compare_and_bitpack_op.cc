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

#include "itex/core/kernels/gpu/compare_and_bitpack_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CompareAndBitpackOp : public OpKernel {
 public:
  explicit CompareAndBitpackOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& input_t = c->input(0);
    const Tensor& threshold_t = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsScalar(threshold_t.shape()),
        errors::InvalidArgument("Compare must be a scalar, but saw shape: ",
                                threshold_t.shape().DebugString()));
    const TensorShape& input_shape = input_t.shape();
    OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(input_shape),
                errors::InvalidArgument(
                    "Input should be at least a vector, but saw a scalar."));
    OP_REQUIRES(c, input_shape.dim_size(input_shape.dims() - 1) % 8 == 0,
                errors::InvalidArgument(
                    "Inner dimension of input should be "
                    "divisible by ",
                    8, ", but saw shape: ", input_shape.DebugString()));

    TensorShape output_shape = input_shape;
    int rank = input_shape.dims();
    output_shape.set_dim(rank - 1, input_shape.dim_size(rank - 1) / 8);

    Tensor* output_t;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output_t));

    auto input = input_t.flat_inner_dims<T>();
    auto threshold = threshold_t.scalar<T>();
    auto output = output_t->flat_inner_dims<uint8>();

    functor::CompareAndBitpack<Device, T> func;
    func(c, input, threshold, output);
  }
};

#define REGISTER_COMPARE_AND_BITPACK(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("CompareAndBitpack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      CompareAndBitpackOp<GPUDevice, type>);

TF_CALL_half(REGISTER_COMPARE_AND_BITPACK);
TF_CALL_bfloat16(REGISTER_COMPARE_AND_BITPACK);
TF_CALL_float(REGISTER_COMPARE_AND_BITPACK);
TF_CALL_bool(REGISTER_COMPARE_AND_BITPACK);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_COMPARE_AND_BITPACK);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_COMPARE_AND_BITPACK

namespace functor {

#define DECLARE_GPU_SPEC(T)                                      \
  template <>                                                    \
  void CompareAndBitpack<GPUDevice, T>::operator()(              \
      OpKernelContext* c, typename TTypes<T>::ConstMatrix input, \
      typename TTypes<T>::ConstScalar threshold,                 \
      TTypes<uint8>::Matrix output);                             \
  extern template struct CompareAndBitpack<GPUDevice, T>;

TF_CALL_half(DECLARE_GPU_SPEC);
TF_CALL_bfloat16(DECLARE_GPU_SPEC);
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_bool(DECLARE_GPU_SPEC);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DECLARE_GPU_SPEC);
#endif  // ITEX_ENABLE_DOUBLE
#undef DECLARE_GPU_SPEC

}  // namespace functor

}  // namespace itex
