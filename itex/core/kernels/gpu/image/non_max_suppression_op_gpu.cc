/* Copyright (c) 2022 Intel Corporation

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

#include "itex/core/kernels/gpu/image/non_max_suppression_op_gpu.h"

#include <algorithm>
#include <limits>

#include "itex/core/kernels/gpu/image/non_max_suppression_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
namespace {

template <int KeysPerItem, int SubGroupSize>
void GroupSelectScores(const gpuStream_t& stream, const float* scores,
                       const int num_boxes, const int group_workload,
                       float* output_scores, int* output_boxIds,
                       const int num_candidates, const int num_groups,
                       const int group_size) {
  // Define an array of function pointers
  void (*kernels[])(const gpuStream_t&, const float*, const int, const int,
                    float*, int*, const int, const int) = {
      internal::LaunchGroupTopkKernel<KeysPerItem, 32, SubGroupSize>,
      internal::LaunchGroupTopkKernel<KeysPerItem, 64, SubGroupSize>,
      internal::LaunchGroupTopkKernel<KeysPerItem, 128, SubGroupSize>,
      internal::LaunchGroupTopkKernel<KeysPerItem, 256, SubGroupSize>,
      internal::LaunchGroupTopkKernel<KeysPerItem, 512, SubGroupSize>,
      internal::LaunchGroupTopkKernel<KeysPerItem, 1024, SubGroupSize>,
      internal::LaunchGroupTopkKernel<KeysPerItem, 2048, SubGroupSize>,
  };

  // Select kernel according to the group size
  int index = 0;
  while ((1 << index) * 32 < group_size) ++index;
  kernels[index](stream, scores, num_boxes, group_workload, output_scores,
                 output_boxIds, num_candidates, num_groups);
}

template <bool FilterScore, bool TopkPerformed, int KeysPerItem,
          int SubGroupSize>
void GroupSortScores(const gpuStream_t& stream, const float* input_scores,
                     const int* input_boxIds, const int num_inputs,
                     const float score_threshold, float* candidate_scores,
                     int* candidate_boxIds, int* num_valid_candidates,
                     const int num_candidates, const int max_group_size) {
  // Define an array of function pointers
  void (*kernels[])(const gpuStream_t&, const float*, const int*, const int,
                    const float, float*, int*, int*, const int) = {
      internal::LaunchGroupSortKernel<FilterScore, TopkPerformed, KeysPerItem,
                                      32, SubGroupSize>,
      internal::LaunchGroupSortKernel<FilterScore, TopkPerformed, KeysPerItem,
                                      64, SubGroupSize>,
      internal::LaunchGroupSortKernel<FilterScore, TopkPerformed, KeysPerItem,
                                      128, SubGroupSize>,
      internal::LaunchGroupSortKernel<FilterScore, TopkPerformed, KeysPerItem,
                                      256, SubGroupSize>,
      internal::LaunchGroupSortKernel<FilterScore, TopkPerformed, KeysPerItem,
                                      512, SubGroupSize>,
      internal::LaunchGroupSortKernel<FilterScore, TopkPerformed, KeysPerItem,
                                      1024, SubGroupSize>,
      internal::LaunchGroupSortKernel<FilterScore, TopkPerformed, KeysPerItem,
                                      2048, SubGroupSize>,
  };

  // Get the optimal group size
  int group_size = max_group_size;
  while (group_size * KeysPerItem > 2 * num_inputs) group_size >>= 1;
  group_size = (group_size > 32) ? group_size : 32;

  // Select kernel according to the group size
  int index = 0;
  while ((1 << index) * 32 < group_size) ++index;
  kernels[index](stream, input_scores, input_boxIds, num_inputs,
                 score_threshold, candidate_scores, candidate_boxIds,
                 num_valid_candidates, num_candidates);
}

void DoNMSReduce(const gpuStream_t& stream, const float* boxes,
                 const float* candidate_scores, const int* candidate_boxIds,
                 const int num_valid_candidates, int* selected_boxIds,
                 const int num_boxes, const int num_candidates,
                 const int max_output_size, int* num_saved_outputs,
                 const bool pad_to_max_output, const float iou_threshold,
                 const int max_group_size) {
  // Define an array of function pointers
  void (*kernels[])(const gpuStream_t&, const float*, const float*, const int*,
                    const int, int*, const int, const int, const int, int*,
                    const bool, const float) = {
      internal::LaunchNMSKernel<32>,   internal::LaunchNMSKernel<64>,
      internal::LaunchNMSKernel<128>,  internal::LaunchNMSKernel<256>,
      internal::LaunchNMSKernel<512>,  internal::LaunchNMSKernel<1024>,
      internal::LaunchNMSKernel<2048>,
  };

  // Get the optimal group size
  int group_size = max_group_size;
  while (group_size > 2 * num_candidates) group_size >>= 1;
  group_size = (group_size > 32) ? group_size : 32;

  // Select kernel according to the group size
  int index = 0;
  while ((1 << index) * 32 < group_size) ++index;
  kernels[index](stream, boxes, candidate_scores, candidate_boxIds,
                 num_valid_candidates, selected_boxIds, num_boxes,
                 num_candidates, max_output_size, num_saved_outputs,
                 pad_to_max_output, iou_threshold);
}

// Get sorted scores as candidates by group methods
template <bool FilterScore, int KeysPerItem, int SubGroupSize>
void GroupGetCandidates(OpKernelContext* context, const float* scores,
                        float* candidate_scores, int* candidate_boxIds,
                        int* num_valid_candidates, const int num_boxes,
                        const int num_candidates, const float score_threshold,
                        const int max_group_size) {
  const auto& d = context->eigen_gpu_device();
  auto& stream = d.stream();

  // Max number of scores that can be handled by one work group each time
  const int group_capacity = max_group_size * KeysPerItem;

  // The number of total scores is smaller than group capacity, only sorting
  if (num_boxes < group_capacity) {
    // TopK selection is not performed, TopkPerformed=false
    GroupSortScores<FilterScore, false, KeysPerItem, SubGroupSize>(
        stream, scores, nullptr, num_boxes, score_threshold, candidate_scores,
        candidate_boxIds, num_valid_candidates, num_candidates, max_group_size);
    return;
  }

  // The number of scores is large, so topK selection is performed before
  // sorting. At the topK selection stage, each work group will select
  // num_candidates scores

  // The max number of groups ensures that all the selected scores can
  // be handled by one work group in the next sorting stage
  const int max_num_groups = group_capacity / num_candidates;

  // Number of groups actually needed, should not exceed max_num_groups
  int num_groups = (num_boxes + group_capacity - 1) / group_capacity;
  if (num_groups > max_num_groups) num_groups = max_num_groups;

  // Number of scores handled by each work group
  int group_workload = (num_boxes + num_groups - 1) / num_groups;

  // Allocate temporary storage for selected scores and boxIds
  const int num_temp = num_groups * num_candidates;

  Tensor temp_scores_t;
  OP_REQUIRES_OK(
      context, context->allocate_temp(DataType::DT_FLOAT,
                                      TensorShape({num_temp}), &temp_scores_t));
  Tensor temp_boxIds_t;
  OP_REQUIRES_OK(
      context, context->allocate_temp(DataType::DT_INT32,
                                      TensorShape({num_temp}), &temp_boxIds_t));

  float* ptr_scores = temp_scores_t.flat<float>().data();
  int* ptr_boxIds = temp_boxIds_t.flat<int>().data();

  GroupSelectScores<KeysPerItem, SubGroupSize>(
      stream, scores, num_boxes, group_workload, ptr_scores, ptr_boxIds,
      num_candidates, num_groups, max_group_size);

  // TopK selection is performed, TopkPerformed=true
  GroupSortScores<FilterScore, true, KeysPerItem, SubGroupSize>(
      stream, ptr_scores, ptr_boxIds, num_temp, score_threshold,
      candidate_scores, candidate_boxIds, num_valid_candidates, num_candidates,
      max_group_size);
}

}  // namespace

// ====================// NonMaxSuppressionFunctor //==================== //

template <bool FilterScore, bool ReturnOutputSize>
struct NonMaxSuppressionFunctor<GPUDevice, FilterScore, ReturnOutputSize> {
  void operator()(OpKernelContext* context, const Tensor& boxes,
                  const Tensor& scores, int* num_saved_outputs,
                  const int max_output_size, const bool pad_to_max_output,
                  const float iou_threshold, const float score_threshold) {
    // Define Constants
    // Number of scores loaded into the item's private memory
    constexpr int KeysPerItem = 8;
    constexpr int SubGroupSize = 16;

    const auto& d = context->eigen_gpu_device();
    auto& stream = d.stream();
    const int max_group_size =
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    const int num_boxes = boxes.dim_size(0);

    if (num_boxes == 0 || max_output_size == 0) {
      Tensor* output_indices_t = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({max_output_size}),
                                              &output_indices_t));
      if (max_output_size)
        stream->fill<int>(output_indices_t->flat<int>().data(), 0,
                          max_output_size);

      if (ReturnOutputSize) stream->memset(num_saved_outputs, 0, sizeof(int));
      return;
    }

    // Compute the required number of candidates
    int num_candidates = 0;
    while (num_candidates < 2 * max_output_size) {
      num_candidates += SubGroupSize;
    }
    num_candidates = std::min(num_candidates, num_boxes);

    // Allocate temporary storage for candidates
    Tensor candidate_scores_t;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataType::DT_FLOAT,
                                          TensorShape({num_candidates}),
                                          &candidate_scores_t));
    Tensor candidate_boxIds_t;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataType::DT_INT32,
                                          TensorShape({num_candidates}),
                                          &candidate_boxIds_t));

    const float* original_scores = scores.flat<float>().data();
    float* candidate_scores = candidate_scores_t.flat<float>().data();
    int* candidate_boxIds = candidate_boxIds_t.flat<int>().data();

    // Max number of keys that one work group can handle each time
    const int group_capacity = max_group_size * KeysPerItem;

    OP_REQUIRES(
        context,
        num_candidates * 2 <= group_capacity || num_boxes <= group_capacity,
        errors::Unimplemented(
            "Support for large max_output_size not implemented yet."));

    // Get topK sorted scores as candidates for NMS reduce
    // num_saved_outputs is used here to store the number of valid candidates
    GroupGetCandidates<FilterScore, KeysPerItem, SubGroupSize>(
        context, original_scores, candidate_scores, candidate_boxIds,
        num_saved_outputs, num_boxes, num_candidates, score_threshold,
        max_group_size);

    int num_valid_candidates;
    stream->memcpy(&num_valid_candidates, num_saved_outputs, sizeof(int))
        .wait();

    if (num_valid_candidates == 0) {
      int num_outputs = pad_to_max_output ? max_output_size : 0;

      Tensor* output_indices_t = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({num_outputs}),
                                              &output_indices_t));
      if (num_outputs)
        stream->fill<int>(output_indices_t->flat<int>().data(), 0, num_outputs);

      return;
    }

    // Perform NMS reduce on candidates
    Tensor selected_boxIds_t;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({max_output_size}),
                                &selected_boxIds_t));

    const float* original_boxes = boxes.flat<float>().data();
    int* selected_boxIds = selected_boxIds_t.flat<int>().data();

    DoNMSReduce(stream, original_boxes, candidate_scores, candidate_boxIds,
                num_valid_candidates, selected_boxIds, num_boxes,
                num_candidates, max_output_size, num_saved_outputs,
                pad_to_max_output, iou_threshold, max_group_size);

    // Get outputs from selected indices
    int num_outputs;
    stream->memcpy(&num_outputs, num_saved_outputs, sizeof(int)).wait();

    Tensor* output_indices_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_outputs}),
                                            &output_indices_t));

    int* output_indices = output_indices_t->flat<int>().data();
    stream->memcpy(output_indices, selected_boxIds, num_outputs * sizeof(int));

    return;
  }
};

}  // namespace functor

#define DECLARE_GPU_SPEC(B1, B2) \
  template struct functor::NonMaxSuppressionFunctor<GPUDevice, B1, B2>;

DECLARE_GPU_SPEC(false, false);
DECLARE_GPU_SPEC(true, false);
DECLARE_GPU_SPEC(true, true);

#undef DECLARE_GPU_SPEC

}  // namespace itex
