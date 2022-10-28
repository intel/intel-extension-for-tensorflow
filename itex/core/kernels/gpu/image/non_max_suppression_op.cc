/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/image/non_max_suppression_op.h"

#include <limits>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

// Extracts a scalar of type T from a tensor, with correct type checking.
// This is necessary because several of the kernels here assume
// T == T_threshold.
template <typename T>
T GetScalar(const Tensor& tensor) {
  switch (tensor.dtype()) {
    case DT_FLOAT:
      return static_cast<T>(tensor.scalar<float>()());
    case DT_DOUBLE:
      return static_cast<T>(tensor.scalar<double>()());
    case DT_BFLOAT16:
      return static_cast<T>(tensor.scalar<Eigen::bfloat16>()());
    case DT_HALF:
      return static_cast<T>(tensor.scalar<Eigen::half>()());
    default:
      ITEX_DCHECK(false) << "Unsupported type " << tensor.dtype();
      break;
  }
  return static_cast<T>(0);
}

static inline Status CheckInputs(const Tensor& boxes, const Tensor& scores,
                                 const Tensor& max_output_size,
                                 const Tensor& iou_threshold) {
  if (!TensorShapeUtils::IsScalar(max_output_size.shape())) {
    return errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                   max_output_size.shape().DebugString(),
                                   " (Shape must be rank 0 but is rank ",
                                   max_output_size.dims(), ")");
  }
  if (!TensorShapeUtils::IsScalar(iou_threshold.shape())) {
    return errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                   iou_threshold.shape().DebugString(),
                                   " (Shape must be rank 0 but is rank ",
                                   iou_threshold.dims(), ")");
  }
  const float iou_threshold_val = GetScalar<float>(iou_threshold);
  if (iou_threshold_val < 0 || iou_threshold_val > 1) {
    return errors::InvalidArgument("iou_threshold must be in [0, 1]");
  }
  if (boxes.dims() != 2) {
    return errors::InvalidArgument(
        "boxes must be a rank 2 tensor! (Shape must be rank 2 but is rank ",
        boxes.dims(), ")");
  }
  int num_boxes = boxes.dim_size(0);
  if (boxes.dim_size(1) != 4) {
    return errors::InvalidArgument(
        "boxes must be Nx4 (Dimension must be 4 but is ", boxes.dim_size(1),
        ")");
  }
  if (scores.dims() != 1) {
    return errors::InvalidArgument(
        "scores must be a vector! (Shape must be rank 1 but is rank ",
        scores.dims(), ")");
  }
  if (scores.dim_size(0) != num_boxes) {
    return errors::InvalidArgument(
        "scores has incompatible shape "        // message must be exactly this
        "(Dimensions must be equal, but are ",  // otherwise tests fail!
        num_boxes, " and ", scores.dim_size(0), ")");
  }
  return Status::OK();
}

static inline Status CheckInputs(const Tensor& boxes, const Tensor& scores,
                                 const Tensor& max_output_size,
                                 const Tensor& iou_threshold,
                                 const Tensor& score_threshold) {
  if (!TensorShapeUtils::IsScalar(score_threshold.shape())) {
    return errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                   score_threshold.shape().DebugString(),
                                   " (Shape must be rank 0 but is ", "rank ",
                                   score_threshold.dims(), ")");
  }
  return CheckInputs(boxes, scores, max_output_size, iou_threshold);
}

// ====================// NonMaxSuppressionV2Op //==================== //

template <typename Device>
class NonMaxSuppressionV2Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV2Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    const Tensor& max_output_size = context->input(2);
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES_OK(context,
                   CheckInputs(boxes, scores, max_output_size, iou_threshold));

    Tensor num_saved_outputs_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({}),
                                                   &num_saved_outputs_t));
    int* num_saved_outputs = num_saved_outputs_t.flat<int>().data();
    const int max_output_size_val = max_output_size.scalar<int>()();
    const float iou_threshold_val = GetScalar<float>(iou_threshold);

    functor::NonMaxSuppressionFunctor<Device, false, false> fn;

    fn(context, boxes, scores, num_saved_outputs, max_output_size_val,
       /*pad_to_max_output=*/false, iou_threshold_val,
       /*score_threshold is lowest float if score threshold is disabled*/
       std::numeric_limits<float>::lowest());
  }
};

// ====================// NonMaxSuppressionV3Op //==================== //

template <typename Device>
class NonMaxSuppressionV3Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV3Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& boxes = context->input(0);
    const Tensor& scores = context->input(1);
    const Tensor& max_output_size = context->input(2);
    const Tensor& iou_threshold = context->input(3);
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES_OK(context, CheckInputs(boxes, scores, max_output_size,
                                        iou_threshold, score_threshold));

    Tensor num_saved_outputs_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({}),
                                                   &num_saved_outputs_t));
    int* num_saved_outputs = num_saved_outputs_t.flat<int>().data();
    const int max_output_size_val = max_output_size.scalar<int>()();
    const float iou_threshold_val = GetScalar<float>(iou_threshold);
    float score_threshold_val = GetScalar<float>(score_threshold);

    if (score_threshold_val < 0)
      score_threshold_val = std::numeric_limits<float>::min();

    functor::NonMaxSuppressionFunctor<Device, true, false> fn;
    fn(context, boxes, scores, num_saved_outputs, max_output_size_val,
       /*pad_to_max_output=*/false, iou_threshold_val, score_threshold_val);
  }
};

// ====================// NonMaxSuppressionV4Op //==================== //

template <typename Device>
class NonMaxSuppressionV4Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV4Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_to_max_output_size",
                                             &pad_to_max_output_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& boxes = context->input(0);
    const Tensor& scores = context->input(1);
    const Tensor& max_output_size = context->input(2);
    const Tensor& iou_threshold = context->input(3);
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES_OK(context, CheckInputs(boxes, scores, max_output_size,
                                        iou_threshold, score_threshold));

    Tensor* num_saved_outputs_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &num_saved_outputs_t));
    int* num_saved_outputs = num_saved_outputs_t->flat<int>().data();
    const int max_output_size_val = max_output_size.scalar<int>()();
    const float iou_threshold_val = GetScalar<float>(iou_threshold);
    float score_threshold_val = GetScalar<float>(score_threshold);
    if (score_threshold_val < 0)
      score_threshold_val = std::numeric_limits<float>::min();
    functor::NonMaxSuppressionFunctor<Device, true, true> fn;
    fn(context, boxes, scores, num_saved_outputs, max_output_size_val,
       pad_to_max_output_size_, iou_threshold_val, score_threshold_val);
  }

 private:
  bool pad_to_max_output_size_;
};

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(B1, B2)                                           \
  template <>                                                              \
  void NonMaxSuppressionFunctor<GPUDevice, B1, B2>::operator()(            \
      OpKernelContext* context, const Tensor& boxes, const Tensor& scores, \
      int* num_saved_outputs, const int max_output_size,                   \
      const bool pad_to_max_output, const float iou_threshold,             \
      const float score_threshold);                                        \
  extern template struct NonMaxSuppressionFunctor<GPUDevice, B1, B2>;

DECLARE_GPU_SPEC(false, false);
DECLARE_GPU_SPEC(true, false);
DECLARE_GPU_SPEC(true, true);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV2")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size"),
                        NonMaxSuppressionV2Op<GPUDevice>);

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV3")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size")
                            .HostMemory("score_threshold"),
                        NonMaxSuppressionV3Op<GPUDevice>);

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV4")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size")
                            .HostMemory("score_threshold"),
                        NonMaxSuppressionV4Op<GPUDevice>);
}  // namespace itex
