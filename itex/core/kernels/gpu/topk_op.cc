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

#include "itex/core/kernels/gpu/topk_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class TopK : public OpKernel {
 public:
  explicit TopK(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sorted", &sorted_));
    k_ = -1;
  }

  void Compute(OpKernelContext* context) override {
    int k = k_;
    if (context->num_inputs() >= 2) {
      const auto& k_in = context->input(1);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(k_in.shape()),
                  errors::InvalidArgument("k must be scalar, got shape ",
                                          k_in.shape().DebugString()));
      k = k_in.scalar<int32>()();
    }
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
    const auto& input_in = context->input(0);
    OP_REQUIRES(context, input_in.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_in.shape().DebugString()));
    OP_REQUIRES(context, input_in.dim_size(input_in.dims() - 1) >= k,
                errors::InvalidArgument(
                    "input must have at least k columns. Had ",
                    input_in.dim_size(input_in.dims() - 1), ", needed ", k));

    const auto& input = input_in.flat_inner_dims<T>();

    TensorShape output_shape = input_in.shape();
    output_shape.set_dim(input_in.dims() - 1, k);
    Tensor* values_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &values_out));
    Tensor* indices_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &indices_out));

    auto values = values_out->flat_inner_dims<T>();
    auto indices = indices_out->flat_inner_dims<int32>();

    functor::TopKFunctor<Device, T, int32>()(context, input, values, indices,
                                             sorted_, k);
  }

 private:
  int k_;
  bool sorted_;
};

// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPEC(T)                                                  \
  template <>                                                                \
  void TopKFunctor<GPUDevice, T, int32>::operator()(                         \
      OpKernelContext* context, typename TTypes<T, 2>::ConstTensor input,    \
      typename TTypes<T, 2>::Tensor values,                                  \
      typename TTypes<int32, 2>::Tensor indices, bool sorted, int num_topk); \
  extern template struct TopKFunctor<GPUDevice, T, int32>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
TF_CALL_INTEGRAL_TYPES(DECLARE_GPU_SPEC);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DECLARE_GPU_SPEC);
#endif  // ITEX_ENABLE_DOUBLE
#undef DECLARE_GPU_SPEC

}  // namespace functor

#define REGISTER_TOPK_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TopK").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      TopK<GPUDevice, type>)                                     \
  REGISTER_KERNEL_BUILDER(Name("TopKV2")                         \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("k"),                  \
                          TopK<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_TOPK_KERNELS);
TF_CALL_INTEGRAL_TYPES(REGISTER_TOPK_KERNELS);
// TODO(itex): Enable following code after fix the topk kernel bug caused by
// double
/*
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_TOPK_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
*/
#undef REGISTER_TOPK_KERNELS

}  // namespace itex
