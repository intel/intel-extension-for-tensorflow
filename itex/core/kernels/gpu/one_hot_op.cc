/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/one_hot_op.h"

#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/overflow.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

template <typename Device, typename T, typename TI>
class OneHotOp : public OpKernel {
 public:
  explicit OneHotOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& indices = context->input(0);
    const Tensor& depth = context->input(1);
    const Tensor& on_value = context->input(2);
    const Tensor& off_value = context->input(3);
    const TensorShape& indices_shape = indices.shape();

    const int indices_dims = indices_shape.dims();
    const int output_dims = indices_dims + 1;

    // Preliminary validation of sizes.
    OP_REQUIRES(
        context, axis_ == -1 || (axis_ >= 0 && axis_ < output_dims),
        errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                output_dims, ").  But received: ", axis_));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(depth.shape()),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(on_value.shape()),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(off_value.shape()),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value.shape().DebugString()));

    if (axis_ < 0) {
      axis_ += output_dims;
    }

    // The one-hot dimension.
    const int32 depth_v = depth.scalar<int32>()();
    OP_REQUIRES(
        context, depth_v >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth_v));
    OP_REQUIRES(
        context,
        MultiplyWithoutOverflow(indices_shape.num_elements(), depth_v) >= 0,
        errors::InvalidArgument("OneHot result would have shape ",
                                indices_shape.DebugString(), " + [", depth_v,
                                "], which exceeds 2**63 - 1 elements"));

    TensorShape output_shape = indices_shape;
    output_shape.InsertDim(axis_, depth_v);

    auto on_value_t = on_value.scalar<T>();
    auto off_value_t = off_value.scalar<T>();

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() > 0) {
      functor::OneHot<T, TI> onehot_functor;
      onehot_functor.Compute(context, indices, on_value_t, off_value_t, axis_,
                             depth_v, output);
    }
  }

 private:
  int32 axis_;
};

#define REGISTER_ONE_HOT_GPU_INDEX(type, index_type)            \
  REGISTER_KERNEL_BUILDER(Name("OneHot")                        \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<index_type>("TI") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("depth"),             \
                          OneHotOp<GPUDevice, type, index_type>);

#define REGISTER_ONE_HOT_GPU(type)         \
  REGISTER_ONE_HOT_GPU_INDEX(type, uint8); \
  REGISTER_ONE_HOT_GPU_INDEX(type, int32); \
  REGISTER_ONE_HOT_GPU_INDEX(type, int64_t);

TF_CALL_int32(REGISTER_ONE_HOT_GPU);
TF_CALL_int64(REGISTER_ONE_HOT_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_ONE_HOT_GPU);
TF_CALL_complex64(REGISTER_ONE_HOT_GPU);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ONE_HOT_GPU);
TF_CALL_complex128(REGISTER_ONE_HOT_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_ONE_HOT_GPU_INDEX
#undef REGISTER_ONE_HOT_GPU

}  // namespace itex
