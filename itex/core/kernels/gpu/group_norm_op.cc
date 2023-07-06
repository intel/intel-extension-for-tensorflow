/* Copyright (c) 2021-2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/group_norm_op.h"

#include <algorithm>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class GroupNormOp : public OpKernel {
 public:
  explicit GroupNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_groups", &num_groups_));
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(context, context->GetAttr("use_scale", &use_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("use_center", &use_center_));
  }

  void Compute(OpKernelContext* context) override {
    // TODO(itex): support channel first and rank==any
    const Tensor& input = context->input(0);
    const Tensor& gamma = context->input(1);
    const Tensor& beta = context->input(2);

    OP_REQUIRES(context, input.dims() > 3,
                errors::InvalidArgument("input must be at least 3-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, !use_scale_ || gamma.dims() == 1,
                errors::InvalidArgument("gamma must be 1-dimensional",
                                        gamma.shape().DebugString()));
    OP_REQUIRES(context, !use_center_ || beta.dims() == 1,
                errors::InvalidArgument("beta must be 1-dimensional",
                                        beta.shape().DebugString()));

    InputShape shape;
    GetInputShape(context, input, &shape);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    functor::GroupNormFunctor<Device, T>()(
        context, input.flat<T>(), output->template flat<T>(), gamma.vec<T>(),
        beta.vec<T>(), epsilon_, use_scale_, use_center_, shape);
  }

  void GetInputShape(OpKernelContext* context, const Tensor& input,
                     InputShape* shape) {
    const int ndims = input.dims();
    shape->num_batches = input.dim_size(0);
    shape->num_channels = input.dim_size(ndims - 1);
    shape->num_hw =
        input.NumElements() / shape->num_batches / shape->num_channels;
    shape->num_groups = num_groups_;
    shape->chans_per_group = shape->num_channels / num_groups_;
  }

 private:
  int num_groups_;
  bool use_scale_;
  bool use_center_;
  float epsilon_;
};

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                \
  template <>                                                              \
  void GroupNormFunctor<GPUDevice, T>::operator()(                         \
      OpKernelContext* context, typename TTypes<T>::ConstFlat input,       \
      typename TTypes<T>::Flat output, typename TTypes<T>::ConstVec gamma, \
      typename TTypes<T>::ConstVec beta, float epsilon, bool use_scale,    \
      bool use_center, const InputShape& shape);                           \
  extern template struct GroupNormFunctor<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(Eigen::bfloat16);
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ITEXGroupNorm").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      GroupNormOp<GPUDevice, T>);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(Eigen::bfloat16);
#undef REGISTER_GPU_KERNEL

}  // end namespace itex
