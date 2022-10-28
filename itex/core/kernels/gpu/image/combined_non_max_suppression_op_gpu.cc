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

#include "itex/core/kernels/gpu/image/combined_non_max_suppression_op_gpu.h"

#include <algorithm>

#include "itex/core/kernels/gpu/image/combined_non_max_suppression_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

static inline void SelectScores(const gpuStream_t& stream, const float* scores,
                                float* candidate_scores,
                                int32* candidate_boxIds,
                                int32* num_valid_candidates, int num_batches,
                                int num_classes, int num_boxes,
                                int num_candidate, float score_threshold,
                                const int max_group_size) {
  constexpr int keys_per_item = 8;
  int group_size = max_group_size;

  if (num_boxes > 2048) {
    if (num_candidate < 1024) {
      group_size = 256;
    } else {
      while (group_size * keys_per_item > 4 * num_candidate) group_size >>= 1;
    }
  } else {
    while (group_size * keys_per_item > 2 * num_boxes) group_size >>= 1;
  }
  group_size = (group_size > 32) ? group_size : 32;

  void (*kernels[])(const gpuStream_t&, const float*, float*, int32*, int32*,
                    int, int, int, int, float) = {
      internal::LaunchTopkScoresKernel<8, 32>,
      internal::LaunchTopkScoresKernel<8, 64>,
      internal::LaunchTopkScoresKernel<8, 128>,
      internal::LaunchTopkScoresKernel<8, 256>,
      internal::LaunchTopkScoresKernel<8, 512>,
      internal::LaunchTopkScoresKernel<8, 1024>,
      internal::LaunchTopkScoresKernel<8, 2048>,
  };

  int index = 0;
  while ((1 << index) * 32 < group_size) ++index;
  kernels[index](stream, scores, candidate_scores, candidate_boxIds,
                 num_valid_candidates, num_batches, num_classes, num_boxes,
                 num_candidate, score_threshold);
}

static inline void SortScores(const gpuStream_t& stream,
                              float* candidate_scores, int32* candidate_boxIds,
                              int32* num_valid_candidates, int num_batches,
                              int num_classes, int num_candidate,
                              const int max_group_size) {
  constexpr int keys_per_item = 8;
  int group_size = max_group_size;

  while (group_size * keys_per_item > 2 * num_candidate) group_size >>= 1;
  group_size = (group_size > 32) ? group_size : 32;

  void (*kernels[])(const gpuStream_t&, float*, int32*, const int32*, int, int,
                    int) = {
      internal::LaunchSortScoresKernel<8, 32>,
      internal::LaunchSortScoresKernel<8, 64>,
      internal::LaunchSortScoresKernel<8, 128>,
      internal::LaunchSortScoresKernel<8, 256>,
      internal::LaunchSortScoresKernel<8, 512>,
      internal::LaunchSortScoresKernel<8, 1024>,
      internal::LaunchSortScoresKernel<8, 2048>,
  };

  int index = 0;
  while ((1 << index) * 32 < group_size) ++index;
  kernels[index](stream, candidate_scores, candidate_boxIds,
                 num_valid_candidates, num_batches, num_classes, num_candidate);
}

static inline void DoNMSReduce(
    const gpuStream_t& stream, const float* boxes,
    const float* candidate_scores, const int32* candidate_boxIds,
    const int32* num_valid_candidates, float* nms_selected_scores,
    int32* nms_selected_boxIds, int32* nms_selected_classIds, int num_batches,
    int num_classes, int num_boxes, int qVal, int num_candidate,
    int max_size_per_class, float iou_threshold, const int max_group_size) {
  int group_size = max_group_size;

  while (group_size > num_candidate) group_size >>= 1;
  group_size = (group_size > 32) ? group_size : 32;

  void (*kernels[])(const gpuStream_t&, const float*, const float*,
                    const int32*, const int32*, float*, int32*, int32*, int,
                    int, int, int, int, int, float) = {
      internal::LaunchNmsPerClassKernel<32>,
      internal::LaunchNmsPerClassKernel<64>,
      internal::LaunchNmsPerClassKernel<128>,
      internal::LaunchNmsPerClassKernel<256>,
      internal::LaunchNmsPerClassKernel<512>,
      internal::LaunchNmsPerClassKernel<1024>,
  };

  int index = 0;
  while ((1 << index) * 32 < group_size) ++index;
  kernels[index](stream, boxes, candidate_scores, candidate_boxIds,
                 num_valid_candidates, nms_selected_scores, nms_selected_boxIds,
                 nms_selected_classIds, num_batches, num_classes, num_boxes,
                 qVal, num_candidate, max_size_per_class, iou_threshold);
}

static inline void MergeScores(
    const gpuStream_t& stream, const float* boxes,
    const float* nms_selected_scores, const int32* nms_selected_boxIds,
    const int32* nms_selected_classIds, float* nmsed_boxes, float* nmsed_scores,
    float* nmsed_classes, int* valid_detections, int num_batches,
    int num_classes, int num_boxes, int qVal, int max_size_per_class,
    int per_batch_size, bool clip_boxes, const int max_group_size) {
  int group_size = max_group_size;

  while (group_size > 2 * per_batch_size) group_size >>= 1;
  group_size = (group_size > 32) ? group_size : 32;

  void (*kernels[])(const gpuStream_t&, const float*, const float*,
                    const int32*, const int32*, float*, float*, float*, int*,
                    int, int, int, int, int, int, bool) = {
      internal::LaunchMergeScoresKernel<32>,
      internal::LaunchMergeScoresKernel<64>,
      internal::LaunchMergeScoresKernel<128>,
      internal::LaunchMergeScoresKernel<256>,
      internal::LaunchMergeScoresKernel<512>,
      internal::LaunchMergeScoresKernel<1024>,
  };

  int index = 0;
  while ((1 << index) * 32 < group_size) ++index;
  kernels[index](stream, boxes, nms_selected_scores, nms_selected_boxIds,
                 nms_selected_classIds, nmsed_boxes, nmsed_scores,
                 nmsed_classes, valid_detections, num_batches, num_classes,
                 num_boxes, qVal, max_size_per_class, per_batch_size,
                 clip_boxes);
}

void CombinedNonMaxSuppressionFunctor<GPUDevice>::operator()(
    OpKernelContext* context, const Tensor& inp_boxes, const Tensor& inp_scores,
    int max_size_per_class, int max_total_size_per_batch, float iou_threshold,
    float score_threshold, bool pad_per_class, bool clip_boxes) {
  const auto& d = context->eigen_gpu_device();
  auto& stream = d.stream();

  const float* boxes = inp_boxes.flat<float>().data();
  const float* scores = inp_scores.flat<float>().data();

  const int num_batches = inp_boxes.dim_size(0);
  const int num_boxes = inp_boxes.dim_size(1);
  const int qVal = inp_boxes.dim_size(2);
  const int num_classes = inp_scores.dim_size(2);

  const int max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();

  // Compute the required number of candidates
  constexpr int SUB_GROUP_SIZE = 16;
  int num_candidate = 0;
  while (num_candidate < 2 * max_size_per_class) {
    num_candidate += SUB_GROUP_SIZE;
  }

  Tensor candidate_scores_t;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DT_FLOAT, TensorShape({num_batches, num_classes, num_candidate}),
          &candidate_scores_t));

  Tensor candidate_boxIds_t;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DT_INT32, TensorShape({num_batches, num_classes, num_candidate}),
          &candidate_boxIds_t));

  Tensor num_valid_candidates_t;
  OP_REQUIRES_OK(context, context->allocate_temp(
                              DT_INT32, TensorShape({num_batches, num_classes}),
                              &num_valid_candidates_t));

  float* candidate_scores = candidate_scores_t.flat<float>().data();
  int32* candidate_boxIds = candidate_boxIds_t.flat<int32>().data();
  int32* num_valid_candidates = num_valid_candidates_t.flat<int32>().data();

  SelectScores(stream, scores, candidate_scores, candidate_boxIds,
               num_valid_candidates, num_batches, num_classes, num_boxes,
               num_candidate, score_threshold, max_group_size);

  SortScores(stream, candidate_scores, candidate_boxIds, num_valid_candidates,
             num_batches, num_classes, num_candidate, max_group_size);

  Tensor nms_selected_scores_t;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DT_FLOAT, TensorShape({num_batches, num_classes, max_size_per_class}),
          &nms_selected_scores_t));

  Tensor nms_selected_boxIds_t;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DT_INT32, TensorShape({num_batches, num_classes, max_size_per_class}),
          &nms_selected_boxIds_t));

  Tensor nms_selected_classIds_t;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DT_INT32, TensorShape({num_batches, num_classes, max_size_per_class}),
          &nms_selected_classIds_t));

  float* nms_selected_scores = nms_selected_scores_t.flat<float>().data();
  int32* nms_selected_boxIds = nms_selected_boxIds_t.flat<int32>().data();
  int32* nms_selected_classIds = nms_selected_classIds_t.flat<int32>().data();

  DoNMSReduce(stream, boxes, candidate_scores, candidate_boxIds,
              num_valid_candidates, nms_selected_scores, nms_selected_boxIds,
              nms_selected_classIds, num_batches, num_classes, num_boxes, qVal,
              num_candidate, max_size_per_class, iou_threshold, max_group_size);

  // allocate output
  int per_batch_size = max_total_size_per_batch;
  if (pad_per_class) {
    per_batch_size =
        std::min(max_total_size_per_batch, max_size_per_class * num_classes);
  }
  Tensor* nmsed_boxes_t = nullptr;
  TensorShape boxes_shape({num_batches, per_batch_size, 4});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, boxes_shape, &nmsed_boxes_t));
  float* nmsed_boxes = nmsed_boxes_t->template flat<float>().data();

  Tensor* nmsed_scores_t = nullptr;
  TensorShape scores_shape({num_batches, per_batch_size});
  OP_REQUIRES_OK(context,
                 context->allocate_output(1, scores_shape, &nmsed_scores_t));
  float* nmsed_scores = nmsed_scores_t->template flat<float>().data();

  Tensor* nmsed_classes_t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(2, scores_shape, &nmsed_classes_t));
  float* nmsed_classes = nmsed_classes_t->template flat<float>().data();

  Tensor* valid_detection_t = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{num_batches},
                                                   &valid_detection_t));
  int* valid_detections = valid_detection_t->template flat<int>().data();

  MergeScores(stream, boxes, nms_selected_scores, nms_selected_boxIds,
              nms_selected_classIds, nmsed_boxes, nmsed_scores, nmsed_classes,
              valid_detections, num_batches, num_classes, num_boxes, qVal,
              max_size_per_class, per_batch_size, clip_boxes, max_group_size);
}

}  // namespace functor
}  // namespace itex
