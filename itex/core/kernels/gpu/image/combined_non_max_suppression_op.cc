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

#include "itex/core/kernels/gpu/image/combined_non_max_suppression_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

static inline void CheckCombinedNmsInputSizes(OpKernelContext* context,
                                              const Tensor& boxes,
                                              const Tensor& scores) {
  OP_REQUIRES(context, boxes.dims() == 4,
              errors::InvalidArgument("boxes must be 4-D",
                                      boxes.shape().DebugString()));
  OP_REQUIRES(context, scores.dims() == 3,
              errors::InvalidArgument("scores must be 3-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(
      context, (boxes.dim_size(0) == scores.dim_size(0)),
      errors::InvalidArgument("boxes and scores must have same batch size"));
  OP_REQUIRES(
      context, scores.dim_size(1) == boxes.dim_size(1),
      errors::InvalidArgument("boxes and scores must have same anchor size"));
  OP_REQUIRES(
      context,
      boxes.dim_size(2) == 1 || boxes.dim_size(2) == scores.dim_size(2),
      errors::InvalidArgument("third dimension of boxes must be either 1 or ",
                              scores.dim_size(2)));
  OP_REQUIRES(context, boxes.dim_size(3) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));
}

static inline void CheckCombinedNmsInputScalar(OpKernelContext* context,
                                               const Tensor& max_output_size,
                                               const Tensor& max_total_size,
                                               const Tensor& iou_threshold,
                                               const Tensor& score_threshold) {
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(max_output_size.shape()),
              errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                      max_output_size.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(max_total_size.shape()),
              errors::InvalidArgument("max_total_size must be 0-D, got shape ",
                                      max_total_size.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
              errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                      iou_threshold.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(score_threshold.shape()),
              errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                      score_threshold.shape().DebugString()));
}

template <typename Device>
class CombinedNonMaxSuppressionOp : public OpKernel {
 public:
  explicit CombinedNonMaxSuppressionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_per_class", &pad_per_class_));
    OP_REQUIRES_OK(context, context->GetAttr("clip_boxes", &clip_boxes_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [batch_size, num_anchors, q, 4]
    const Tensor& inp_boxes = context->input(0);
    // scores: [batch_size, num_anchors, num_classes]
    const Tensor& inp_scores = context->input(1);

    CheckCombinedNmsInputSizes(context, inp_boxes, inp_scores);

    const Tensor& max_output_size = context->input(2);
    const Tensor& max_total_size = context->input(3);
    const Tensor& iou_threshold = context->input(4);
    const Tensor& score_threshold = context->input(5);

    // check scalar
    CheckCombinedNmsInputScalar(context, max_output_size, max_total_size,
                                iou_threshold, score_threshold);

    // max_size_per_class
    const int max_size_per_class = max_output_size.scalar<int>()();
    OP_REQUIRES(context, max_size_per_class > 0,
                errors::InvalidArgument("max_size_per_class must be > 0"));

    // max_total_size_per_batch
    const int max_total_size_per_batch = max_total_size.scalar<int>()();
    OP_REQUIRES(context, max_total_size_per_batch > 0,
                errors::InvalidArgument("max_total_size must be > 0"));
    // Throw warning when `max_total_size` is too large as it may cause OOM.
    if (max_total_size_per_batch > pow(10, 6)) {
      ITEX_LOG(WARNING)
          << "Detected a large value for `max_total_size`. This may "
          << "cause OOM error. (max_total_size: " << max_total_size_per_batch
          << ")";
    }

    // iou_threshold
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));

    // score_threshold
    const float score_threshold_val = score_threshold.scalar<float>()();

    functor::CombinedNonMaxSuppressionFunctor<Device>()(
        context, inp_boxes, inp_scores, max_size_per_class,
        max_total_size_per_batch, iou_threshold_val, score_threshold_val,
        pad_per_class_, clip_boxes_);
  }

 private:
  bool pad_per_class_;
  bool clip_boxes_;
};

REGISTER_KERNEL_BUILDER(Name("CombinedNonMaxSuppression")
                            .Device(DEVICE_GPU)
                            .HostMemory("max_output_size_per_class")
                            .HostMemory("max_total_size")
                            .HostMemory("iou_threshold")
                            .HostMemory("score_threshold"),
                        CombinedNonMaxSuppressionOp<GPUDevice>);

}  // namespace itex
