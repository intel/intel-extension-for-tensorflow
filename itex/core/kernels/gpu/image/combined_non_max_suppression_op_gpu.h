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

#ifndef ITEX_CORE_KERNELS_GPU_IMAGE_COMBINED_NON_MAX_SUPPRESSION_OP_GPU_H_
#define ITEX_CORE_KERNELS_GPU_IMAGE_COMBINED_NON_MAX_SUPPRESSION_OP_GPU_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/group_radix_select.h"
#include "itex/core/utils/group_radix_sort.h"
#include "itex/core/utils/radix_utils.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace functor {
namespace internal {

using LocalAcc = sycl::local_accessor<uint8_t, 1>;

template <int KEYS_PER_ITEM, int GROUP_SIZE, class Selector, int SUB_GROUP_SIZE>
struct TopkScoresKernel {
  TopkScoresKernel(const float* scores, float* candidate_scores,
                   int32* candidate_boxIds, int32* num_valid_candidates,
                   int num_classes, int num_boxes, int num_topk,
                   float score_threshold, LocalAcc scratch)
      : scores_(reinterpret_cast<const uint32*>(scores)),
        candidate_scores_(reinterpret_cast<uint32*>(candidate_scores)),
        candidate_boxIds_(candidate_boxIds),
        num_valid_candidates_(num_valid_candidates),
        num_classes_(num_classes),
        num_boxes_(num_boxes),
        num_topk_(num_topk),
        score_threshold_(*reinterpret_cast<uint32*>(&score_threshold)),
        scratch_(scratch) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(
      sycl::nd_item<2> item) const {
    auto g = item.get_group();
    int batch_id = item.get_group(0);
    int class_id = item.get_group(1);
    int local_id = item.get_local_id(1);

    // set the pointer of local memory and radix selector
    uint8_t* local_mem = ITEXGetLocalAccPointer<uint8_t>(scratch_);
    Selector rselector(g, item.get_sub_group(), local_id, local_mem);

    const uint32* inp_scores = scores_ + batch_id * num_classes_ * num_boxes_;
    uint32 item_scores[KEYS_PER_ITEM] = {0u};
    int32 item_boxIds[KEYS_PER_ITEM] = {0};

    // load data from scores to private memory
    for (int i = 0; i < KEYS_PER_ITEM; ++i) {
      int box_id = local_id * KEYS_PER_ITEM + i;
      if (box_id < num_boxes_) {
        item_scores[i] = inp_scores[box_id * num_classes_ + class_id];
        item_boxIds[i] = box_id;
      }
    }

    // The total number of keys that can be sorted in one work group
    constexpr int CHUNK = KEYS_PER_ITEM * GROUP_SIZE;

    // If num_boxes is greater than CHUNK, the selected scores
    // will be used for the next selecting
    uint32* temp_scores =
        reinterpret_cast<uint32*>(local_mem + Selector::LocalStorage::SIZE);
    int32* temp_boxIds = reinterpret_cast<int32*>(temp_scores + num_topk_);

    int num_selected;
    int num_start = CHUNK;
    while (num_start < num_boxes_) {
      rselector.SelectTopK(item_scores, item_boxIds, temp_scores, temp_boxIds,
                           num_topk_, score_threshold_, &num_selected, 30, 0);
      sycl::group_barrier(g);

      // load selected topk scores from local memory
      for (int i = 0; i < KEYS_PER_ITEM; ++i) {
        int offset = local_id * KEYS_PER_ITEM + i;
        if (offset < num_selected) {
          item_scores[i] = temp_scores[offset];
          item_boxIds[i] = temp_boxIds[offset];
        } else {
          item_scores[i] = 0u;
          int box_id = num_start + offset - num_selected;
          if (box_id < num_boxes_) {
            item_scores[i] = inp_scores[box_id * num_classes_ + class_id];
            item_boxIds[i] = box_id;
          }
        }
      }
      num_start += CHUNK - num_selected;
      sycl::group_barrier(g);
    }

    // pointers of the ouput
    int base_offset = batch_id * num_classes_ + class_id;
    int offset = base_offset * num_topk_;
    uint32* out_scores = candidate_scores_ + offset;
    int32* out_boxIds = candidate_boxIds_ + offset;

    // select topK from the last CHUNK of scores and store in output
    rselector.SelectTopK(item_scores, item_boxIds, out_scores, out_boxIds,
                         num_topk_, score_threshold_, &num_selected, 30, 0);
    if (local_id == 0) num_valid_candidates_[base_offset] = num_selected;
  }

 private:
  const uint32* scores_;
  uint32* candidate_scores_;
  int32* candidate_boxIds_;
  int32* num_valid_candidates_;
  int num_classes_;
  int num_boxes_;
  int num_topk_;
  uint32 score_threshold_;
  LocalAcc scratch_;
};

template <int KEYS_PER_ITEM, int GROUP_SIZE, int SUB_GROUP_SIZE = 16>
void LaunchTopkScoresKernel(const gpuStream_t& stream, const float* scores,
                            float* candidate_scores, int32* candidate_boxIds,
                            int32* num_valid_candidate, int num_batches,
                            int num_classes, int num_boxes, int num_topk,
                            float score_threshold) {
  // Type definitions
  using Rselector =
      GroupRadixPerBitSelector<uint32, KEYS_PER_ITEM, GROUP_SIZE,
                               SUB_GROUP_SIZE, sycl::group<2>, int32>;
  // Compute the required local memory size
  size_t local_memory_size =
      Rselector::LocalStorage::SIZE + num_topk * (sizeof(uint32) + sizeof(int));

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    TopkScoresKernel<KEYS_PER_ITEM, GROUP_SIZE, Rselector, SUB_GROUP_SIZE> task(
        scores, candidate_scores, candidate_boxIds, num_valid_candidate,
        num_classes, num_boxes, num_topk, score_threshold, scratch);
    cgh.parallel_for<
        TopkScoresKernel<KEYS_PER_ITEM, GROUP_SIZE, Rselector, SUB_GROUP_SIZE>>(
        sycl::nd_range<2>(sycl::range<2>(num_batches, num_classes * GROUP_SIZE),
                          sycl::range<2>(1, GROUP_SIZE)),
        task);
  });
}

template <int KEYS_PER_ITEM, int GROUP_SIZE, class Sortor, int SUB_GROUP_SIZE>
struct SortScoresKernel {
  SortScoresKernel(float* candidate_scores, int32* candidate_boxIds,
                   const int32* num_valid_candidates, int num_classes,
                   int num_candidate, LocalAcc scratch)
      : candidate_scores_(reinterpret_cast<uint32*>(candidate_scores)),
        candidate_boxIds_(candidate_boxIds),
        num_valid_candidates_(num_valid_candidates),
        num_classes_(num_classes),
        num_candidate_(num_candidate),
        scratch_(scratch) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(
      sycl::nd_item<2> item) const {
    int batch_id = item.get_group(0);
    int class_id = item.get_group(1);
    int local_id = item.get_local_id(1);

    int base_offset = batch_id * num_classes_ + class_id;
    int offset = base_offset * num_candidate_;

    uint32* ptr_candidate_scores = candidate_scores_ + offset;
    int* ptr_candidate_boxIds = candidate_boxIds_ + offset;
    const int num_valid = num_valid_candidates_[base_offset];

    uint32 item_scores[KEYS_PER_ITEM] = {0u};
    int32 item_boxIds[KEYS_PER_ITEM] = {0};

    // load scores and box_id
    for (int i = 0; i < KEYS_PER_ITEM; ++i) {
      int idx = i + local_id * KEYS_PER_ITEM;
      if (idx < num_valid) {
        item_scores[i] = ptr_candidate_scores[idx];
        item_boxIds[i] = ptr_candidate_boxIds[idx];
      }
    }

    // get the pointer of share local memory
    uint8_t* local_mem = ITEXGetLocalAccPointer<uint8_t>(scratch_);
    // Sorting the scores
    Sortor(item.get_group(), item.get_sub_group(), local_id, local_mem)
        .SortDescending(item_scores, item_boxIds, ptr_candidate_scores,
                        ptr_candidate_boxIds, num_valid, 0, 30);
  }

 private:
  uint32* candidate_scores_;
  int32* candidate_boxIds_;
  const int32* num_valid_candidates_;
  int num_classes_;
  int num_candidate_;
  LocalAcc scratch_;
};

template <int KEYS_PER_ITEM, int GROUP_SIZE, int SUB_GROUP_SIZE = 16>
void LaunchSortScoresKernel(const gpuStream_t& stream, float* candidate_scores,
                            int32* candidate_boxIds,
                            const int32* num_valid_candidates, int num_batches,
                            int num_classes, int num_candidate) {
  // Sortor type definitions
  using Rsortor = GroupRadixSortor<uint32, KEYS_PER_ITEM, GROUP_SIZE,
                                   SUB_GROUP_SIZE, sycl::group<2>, int32>;
  // Compute the required local memory size
  size_t local_memory_size = Rsortor::LocalStorage::SIZE;

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    SortScoresKernel<KEYS_PER_ITEM, GROUP_SIZE, Rsortor, SUB_GROUP_SIZE> task(
        candidate_scores, candidate_boxIds, num_valid_candidates, num_classes,
        num_candidate, scratch);
    cgh.parallel_for<
        SortScoresKernel<KEYS_PER_ITEM, GROUP_SIZE, Rsortor, SUB_GROUP_SIZE>>(
        sycl::nd_range<2>(sycl::range<2>(num_batches, num_classes * GROUP_SIZE),
                          sycl::range<2>(1, GROUP_SIZE)),
        task);
  });
}

using float4 = sycl::vec<float, 4>;

template <typename T>
inline void Swap(T& a, T& b) {
  T temp(a);
  a = b;
  b = temp;
}

// Flip box if necessary
inline void Flipped(float4& box) {  // NOLINT(runtime/references)
  // float4: x(),y(),z(),w()  box: x1,y1,x2,y2
  if (box.x() > box.z()) Swap(box.x(), box.z());
  if (box.y() > box.w()) Swap(box.y(), box.w());
}

// Check whether two boxes have an iou greater than threshold
inline bool OverThreshold(const float4& box_a, const float4& box_b,
                          const float a_area, const float threshold) {
  const float b_area = (box_b.z() - box_b.x()) * (box_b.w() - box_b.y());
  if (a_area == 0.0f || b_area == 0.0f) return false;

  // coord for intersection box
  const float xmin = sycl::fmax(box_a.x(), box_b.x());
  const float ymin = sycl::fmax(box_a.y(), box_b.y());
  const float xmax = sycl::fmin(box_a.z(), box_b.z());
  const float ymax = sycl::fmin(box_a.w(), box_b.w());

  const float width = sycl::fdim(xmax, xmin);
  const float height = sycl::fdim(ymax, ymin);
  const float intersection = width * height;
  return intersection > (a_area + b_area - intersection) * threshold;
}

// NMS reduce
template <int GROUP_SIZE, typename GroupT>
class NMSReduce {
 private:
  struct _LocalStorage {
    uint32 bit_mask[GROUP_SIZE];
  };

  const GroupT& g;
  const int local_id;
  _LocalStorage& local_mem;

  // --------------------------------------------
  // Utility methods
  // --------------------------------------------
  bool CheckMask(const int idx) {
    return local_mem.bit_mask[idx % GROUP_SIZE] >> (idx / GROUP_SIZE) & 1;
  }
  void SetMaskZero(const int idx) {
    local_mem.bit_mask[idx % GROUP_SIZE] &= ~(1 << (idx / GROUP_SIZE));
  }

 public:
  struct LocalStorage : BaseStorage<_LocalStorage> {};

  // Constructor
  NMSReduce(const GroupT& g_, const int local_id_, uint8* local_mem_)
      : g(g_),
        local_id(local_id_),
        local_mem(reinterpret_cast<_LocalStorage&>(*local_mem_)) {}

  template <typename IndexT>
  void DoNms(const float4* boxes, IndexT* out_indices, int* num_accepted,
             int num_elements, int max_output_per_class, float iou_threshold) {
    // initialize mask value
    local_mem.bit_mask[local_id] = 0xFFFFFFFF;
    sycl::group_barrier(g);

    *num_accepted = 0;
    for (int i = 0; i < num_elements; ++i) {
      // if current box is masked by an earlier box, skip it.
      if (!CheckMask(i)) continue;
      // record the selected index
      if (local_id == 0) out_indices[*num_accepted] = i;
      if (++(*num_accepted) >= max_output_per_class) break;

      // get current selected box
      float4 cur_box = boxes[i];
      float cur_area =
          (cur_box.z() - cur_box.x()) * (cur_box.w() - cur_box.y());

      // loop over the left boxes and set corresponding mask value
      for (int idx = local_id + i + 1; idx < num_elements; idx += GROUP_SIZE) {
        float4 target_box = boxes[idx];

        if (OverThreshold(cur_box, target_box, cur_area, iou_threshold)) {
          SetMaskZero(idx);
        }
      }
      sycl::group_barrier(g);
    }
  }
};

template <int GROUP_SIZE, class NmsReducer>
struct NmsPerClassKernel {
  NmsPerClassKernel(const float* boxes, const float* candidate_scores,
                    const int* candidate_boxIds, const int32* valid_candidates,
                    float* nms_selected_scores, int32* nms_selected_boxIds,
                    int32* nms_selected_classIds, int num_classes,
                    int num_boxes, int qVal, int num_candidate,
                    int max_size_per_class, float iou_threshold,
                    LocalAcc scratch)
      : boxes_(reinterpret_cast<const float4*>(boxes)),
        candidate_scores_(candidate_scores),
        candidate_boxIds_(candidate_boxIds),
        valid_candidates_(valid_candidates),
        nms_selected_scores_(nms_selected_scores),
        nms_selected_boxIds_(nms_selected_boxIds),
        nms_selected_classIds_(nms_selected_classIds),
        num_classes_(num_classes),
        num_boxes_(num_boxes),
        qVal_(qVal),
        num_candidate_(num_candidate),
        max_size_per_class_(max_size_per_class),
        iou_threshold_(iou_threshold),
        scratch_(scratch) {}

  void operator()(sycl::nd_item<2> item) const {
    auto g = item.get_group();
    int batch_id = item.get_group(0);
    int class_id = item.get_group(1);
    int local_id = item.get_local_id(1);

    int base_offset = batch_id * num_classes_ + class_id;
    int inp_offset = base_offset * num_candidate_;

    const float* inp_candidate_scores = candidate_scores_ + inp_offset;
    const int32* inp_candidate_boxIds = candidate_boxIds_ + inp_offset;
    const int num_valid = valid_candidates_[base_offset];

    // local memory allocation
    uint8_t* local_mem = ITEXGetLocalAccPointer<uint8_t>(scratch_);

    float4* sorted_boxes =
        reinterpret_cast<float4*>(local_mem + NmsReducer::LocalStorage::SIZE);

    // Load boxes into local memory and compute the areas
    for (int idx = local_id; idx < num_valid; idx += GROUP_SIZE) {
      int box_id = inp_candidate_boxIds[idx];
      int box_offset = (qVal_ == 1) ? batch_id * num_boxes_ + box_id
                                    : batch_id * num_boxes_ * qVal_ +
                                          box_id * qVal_ + class_id;
      float4 cur_box = boxes_[box_offset];
      Flipped(cur_box);
      sorted_boxes[idx] = cur_box;
    }
    sycl::group_barrier(g);

    // Output indices
    uint16* out_indices =
        reinterpret_cast<uint16*>(sorted_boxes + num_candidate_);

    // do NMS reduce
    int num_accepted;
    NmsReducer(g, local_id, local_mem)
        .DoNms(sorted_boxes, out_indices, &num_accepted, num_valid,
               max_size_per_class_, iou_threshold_);

    sycl::group_barrier(g);

    // output NMS selected scores
    int out_offset = base_offset * max_size_per_class_;
    float* out_selected_scores = nms_selected_scores_ + out_offset;
    int32* out_selected_boxIds = nms_selected_boxIds_ + out_offset;
    int32* out_selected_classIds = nms_selected_classIds_ + out_offset;

    for (int idx = local_id; idx < max_size_per_class_; idx += GROUP_SIZE) {
      if (idx < num_accepted) {
        uint16 index = out_indices[idx];
        out_selected_scores[idx] = inp_candidate_scores[index];
        out_selected_boxIds[idx] = inp_candidate_boxIds[index];
        out_selected_classIds[idx] = class_id;
      } else {
        out_selected_scores[idx] = 0.0f;
      }
    }
  }

 private:
  const float4* boxes_;
  const float* candidate_scores_;
  const int32* candidate_boxIds_;
  const int32* valid_candidates_;
  float* nms_selected_scores_;
  int32* nms_selected_boxIds_;
  int32* nms_selected_classIds_;
  int num_classes_;
  int num_boxes_;
  int qVal_;
  int num_candidate_;
  int max_size_per_class_;
  float iou_threshold_;
  LocalAcc scratch_;
};

template <int GROUP_SIZE>
void LaunchNmsPerClassKernel(
    const gpuStream_t& stream, const float* boxes,
    const float* candidate_scores, const int32* candidate_boxIds,
    const int32* num_valid_candidates, float* nms_selected_scores,
    int32* nms_selected_boxIds, int32* nms_selected_classIds, int num_batches,
    int num_classes, int num_boxes, int qVal, int num_candidate,
    int max_size_per_class, float iou_threshold) {
  // NmsReduce type definitions
  using NmsReducer = NMSReduce<GROUP_SIZE, sycl::group<2>>;

  size_t local_memory_size = NmsReducer::LocalStorage::SIZE +
                             num_candidate * sizeof(float4) +
                             max_size_per_class * sizeof(uint16_t);

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    NmsPerClassKernel<GROUP_SIZE, NmsReducer> task(
        boxes, candidate_scores, candidate_boxIds, num_valid_candidates,
        nms_selected_scores, nms_selected_boxIds, nms_selected_classIds,
        num_classes, num_boxes, qVal, num_candidate, max_size_per_class,
        iou_threshold, scratch);
    cgh.parallel_for<NmsPerClassKernel<GROUP_SIZE, NmsReducer>>(
        sycl::nd_range<2>(sycl::range<2>(num_batches, num_classes * GROUP_SIZE),
                          sycl::range<2>(1, GROUP_SIZE)),
        task);
  });
}

struct ScoreT {
  float score;
  int32 index;
};

// Merge two classes of sorted scores
template <int GROUP_SIZE, int SUB_GROUP_SIZE>
class MergeSortedSocres {
 private:
  const int local_id;
  const int max_output;
  int chunk;

  // --------------------------------------------
  // Utility methods
  // --------------------------------------------

  int LowerBound(const ScoreT* sorted_array, const float target) {
    // index: 0 1 2 3 4 5
    // value: 9 7 7 5 3 1  (return index 1 for target 7)
    int left = 0;
    int right = max_output;
    while (left < right) {
      int mid = (left + right) / 2;
      if (sorted_array[mid].score > target) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left;
  }

  int UpperBound(const ScoreT* sorted_array, const float target) {
    // index: 0 1 2 3 4 5
    // value: 9 7 7 5 3 1  (return index 3 for target 7)
    int left = 0;
    int right = max_output;
    while (left < right) {
      int mid = (left + right) / 2;
      if (sorted_array[mid].score >= target) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left;
  }

 public:
  MergeSortedSocres(const int local_id_, const int max_output_)
      : local_id(local_id_), max_output(max_output_) {
    chunk = 0;
    while (chunk < max_output) {
      chunk += SUB_GROUP_SIZE;
    }
  }

  void Merge(const ScoreT* inp1, const ScoreT* inp2, ScoreT* out) {
    for (int idx = local_id; idx < 2 * chunk; idx += GROUP_SIZE) {
      // init
      ScoreT cur_data;
      int new_idx = max_output;  // final index after merging

      if (idx < chunk) {
        int cur_idx = idx;
        if (cur_idx < max_output) {
          cur_data.score = inp1[cur_idx].score;
          cur_data.index = inp1[cur_idx].index;
          new_idx = cur_idx + LowerBound(inp2, cur_data.score);
        }
      } else {
        int cur_idx = idx - chunk;
        if (cur_idx < max_output) {
          cur_data.score = inp2[cur_idx].score;
          cur_data.index = inp2[cur_idx].index;
          new_idx = cur_idx + UpperBound(inp1, cur_data.score);
        }
      }

      if (new_idx < max_output) {
        out[new_idx].score = cur_data.score;
        out[new_idx].index = cur_data.index;
      }
    }
  }
};

template <int GROUP_SIZE, int SUB_GROUP_SIZE = 16>
struct MergeScoresKernel {
  MergeScoresKernel(const float* boxes, const float* nms_selected_scores,
                    const int32* nms_selected_boxIds,
                    const int32* nms_selected_classIds, float* nmsed_boxes,
                    float* nmsed_scores, float* nmsed_classes,
                    int* valid_detections, int num_classes, int num_boxes,
                    int qVal, int max_size_per_class, int per_batch_size,
                    bool clip_boxes, LocalAcc scratch)
      : boxes_(reinterpret_cast<const float4*>(boxes)),
        nms_selected_scores_(nms_selected_scores),
        nms_selected_boxIds_(nms_selected_boxIds),
        nms_selected_classIds_(nms_selected_classIds),
        nmsed_boxes_(reinterpret_cast<float4*>(nmsed_boxes)),
        nmsed_scores_(nmsed_scores),
        nmsed_classes_(nmsed_classes),
        valid_detections_(valid_detections),
        num_classes_(num_classes),
        num_boxes_(num_boxes),
        qVal_(qVal),
        max_size_per_class_(max_size_per_class),
        per_batch_size_(per_batch_size),
        clip_boxes_(clip_boxes),
        scratch_(scratch) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    auto g = item.get_group();
    int batch_id = item.get_group(0);
    int local_id = item.get_local_id(0);

    // pointer of the starting of each batch
    int offset = batch_id * num_classes_ * max_size_per_class_;
    const float* inp_selected_scores = nms_selected_scores_ + offset;
    const int32* inp_selected_boxIds = nms_selected_boxIds_ + offset;
    const int32* inp_selected_classIds = nms_selected_classIds_ + offset;

    // local memory allocation
    uint8_t* local_mem = ITEXGetLocalAccPointer<uint8_t>(scratch_);

    ScoreT* local_mem_in1 = reinterpret_cast<ScoreT*>(local_mem);
    ScoreT* local_mem_in2 = local_mem_in1 + per_batch_size_;
    ScoreT* local_mem_out = local_mem_in2 + per_batch_size_;

    // Load the first class of scores
    for (int idx = local_id; idx < per_batch_size_; idx += GROUP_SIZE) {
      local_mem_in1[idx].score =
          (idx < max_size_per_class_) ? inp_selected_scores[idx] : 0.0f;
      local_mem_in1[idx].index = idx;
    }
    sycl::group_barrier(g);

    // Instantiate
    MergeSortedSocres<GROUP_SIZE, SUB_GROUP_SIZE> mss(local_id,
                                                      per_batch_size_);
    // iterate all the classes
    for (int class_id = 1; class_id < num_classes_; ++class_id) {
      // load the i-th class of scores
      for (int idx = local_id; idx < per_batch_size_; idx += GROUP_SIZE) {
        int index = idx + class_id * max_size_per_class_;
        local_mem_in2[idx].score =
            (idx < max_size_per_class_) ? inp_selected_scores[index] : 0.0f;
        local_mem_in2[idx].index = index;
      }
      sycl::group_barrier(g);

      // merge
      mss.Merge(local_mem_in1, local_mem_in2, local_mem_out);

      Swap(local_mem_in1, local_mem_out);
      sycl::group_barrier(g);
    }

    // TODO(itex): The next line is trying to fix an unknown bug.
    if (local_id == 0) {
      local_mem_in2[0].score = local_mem_in1[0].score;
      local_mem_in2[0].index = local_mem_in1[0].index;
    }

    // get output
    int out_offset = batch_id * per_batch_size_;
    float4* out_nmsed_boxes = nmsed_boxes_ + out_offset;
    float* out_nmsed_scores = nmsed_scores_ + out_offset;
    float* out_nmsed_classes = nmsed_classes_ + out_offset;

    int num_valid = 0;
    for (int i = 0; i < (per_batch_size_ - 1) / GROUP_SIZE + 1; ++i) {
      int idx = local_id + i * GROUP_SIZE;
      int is_valid = 0;
      if (idx < per_batch_size_) {
        float score = local_mem_in1[idx].score;

        if (score > 0.0f) {
          is_valid = 1;
          int index = local_mem_in1[idx].index;
          int class_id = inp_selected_classIds[index];
          int box_id = inp_selected_boxIds[index];

          int box_offset = (qVal_ == 1) ? batch_id * num_boxes_ + box_id
                                        : batch_id * num_boxes_ * qVal_ +
                                              box_id * qVal_ + class_id;
          float4 coord = boxes_[box_offset];

          if (clip_boxes_) {
            const float4 box_min = {0.0, 0.0, 0.0, 0.0};
            const float4 box_max = {1.0, 1.0, 1.0, 1.0};
            coord = sycl::max(sycl::min(coord, box_max), box_min);
          }

          out_nmsed_scores[idx] = score;
          out_nmsed_boxes[idx] = coord;
          out_nmsed_classes[idx] = static_cast<float>(class_id);
        }
      }
      int num_valid_ = sycl::reduce_over_group(g, is_valid, sycl::plus<int>());
      num_valid += num_valid_;
    }
    if (local_id == 0) valid_detections_[batch_id] = num_valid;
  }

 private:
  const float4* boxes_;
  const float* nms_selected_scores_;
  const int32* nms_selected_boxIds_;
  const int32* nms_selected_classIds_;
  float4* nmsed_boxes_;
  float* nmsed_scores_;
  float* nmsed_classes_;
  int* valid_detections_;
  int num_classes_;
  int num_boxes_;
  int qVal_;
  int max_size_per_class_;
  int per_batch_size_;
  bool clip_boxes_;
  LocalAcc scratch_;
};

template <int GROUP_SIZE>
void LaunchMergeScoresKernel(const gpuStream_t& stream, const float* boxes,
                             const float* nms_selected_scores,
                             const int32* nms_selected_boxIds,
                             const int32* nms_selected_classIds,
                             float* nmsed_boxes, float* nmsed_scores,
                             float* nmsed_classes, int* valid_detections,
                             int num_batches, int num_classes, int num_boxes,
                             int qVal, int max_size_per_class,
                             int per_batch_size, bool clip_boxes) {
  // Compute the required local memory size
  size_t local_memory_size = 3 * per_batch_size * sizeof(ScoreT);

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    MergeScoresKernel<GROUP_SIZE> task(
        boxes, nms_selected_scores, nms_selected_boxIds, nms_selected_classIds,
        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, num_classes,
        num_boxes, qVal, max_size_per_class, per_batch_size, clip_boxes,
        scratch);
    cgh.parallel_for<MergeScoresKernel<GROUP_SIZE>>(
        sycl::nd_range<1>(sycl::range<1>(num_batches * GROUP_SIZE),
                          sycl::range<1>(GROUP_SIZE)),
        task);
  });
}

}  // namespace internal
}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_IMAGE_COMBINED_NON_MAX_SUPPRESSION_OP_GPU_H_
