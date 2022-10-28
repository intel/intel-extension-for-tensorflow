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

#include "itex/core/kernels/gpu/xent_op.h"

#include "itex/core/utils/bcast.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SoftmaxXentWithLogitsOp : public OpKernel {
 public:
  explicit SoftmaxXentWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);

    TensorShape shape_in = logits_in.shape();
    TensorShape shape_label = labels_in.shape();

    BCast bcast(BCast::FromShape(logits_in.shape()),
                BCast::FromShape(labels_in.shape()));

    if (!logits_in.IsSameSize(labels_in)) {
      OP_REQUIRES(context, bcast.IsValid(),
                  errors::InvalidArgument(
                      "logits and labels must be broadcastable: logits_size=",
                      logits_in.shape().DebugString(),
                      " labels_size=", labels_in.shape().DebugString()));
      shape_in = BCast::ToShape(bcast.output_shape());
    }
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(shape_in),
                errors::InvalidArgument("logits and labels must be either "
                                        "2-dimensional, or broadcasted to be "
                                        "2-dimensional"));
    Tensor softmax_temp;  // temp output of softmax
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   TensorShape({shape_label}),
                                                   &softmax_temp));
    Tensor* loss_out = nullptr;  // loss output, 1D tensor
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({shape_in.dim_size(0)}), &loss_out));
    Tensor* back_out = nullptr;
    // backprop output.
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 1, shape_label, &back_out));
    if (shape_in.dim_size(0) > 0) {
      if (logits_in.IsSameSize(labels_in)) {
        itex::functor::XentFunctor<Device, T> functor_xent;
        OP_REQUIRES_OK(
            context, functor_xent(context->eigen_gpu_device(), logits_in,
                                  labels_in, softmax_temp, loss_out, back_out));

      } else {
        Tensor logits_temp;  // temp tensor for bcast, 2D tensor
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value,
                                    TensorShape({shape_label}), &logits_temp));
        Tensor labels_temp;
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value,
                                    TensorShape({shape_label}), &labels_temp));

        itex::functor::XentFunctorWithEigen<Device, T> functor_xent_with_eigen;
        OP_REQUIRES_OK(context,
                       functor_xent_with_eigen(
                           context->eigen_gpu_device(),
                           BCast::ToIndexArray<2>(bcast.x_bcast()),
                           BCast::ToIndexArray<2>(bcast.y_bcast()),
                           logits_in.template shaped<T, 2>(bcast.x_reshape()),
                           labels_in.template shaped<T, 2>(bcast.y_reshape()),
                           logits_temp.matrix<T>(), labels_temp.matrix<T>(),
                           softmax_temp, loss_out, back_out));
      }
    }
  }
};
#define REGISTER_XENT_OP(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits") \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<T>("T"),          \
                          SoftmaxXentWithLogitsOp<GPUDevice, T>);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_FLOAT_TYPES(REGISTER_XENT_OP);
#else
TF_CALL_GPU_NUMBER_TYPES(REGISTER_XENT_OP);
#endif
#undef REGISTER_XENT_OP

}  // namespace itex
