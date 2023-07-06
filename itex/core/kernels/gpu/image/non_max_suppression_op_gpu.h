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

#ifndef ITEX_CORE_KERNELS_GPU_IMAGE_NON_MAX_SUPPRESSION_OP_GPU_H_
#define ITEX_CORE_KERNELS_GPU_IMAGE_NON_MAX_SUPPRESSION_OP_GPU_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/group_radix_select.h"
#include "itex/core/utils/group_radix_sort.h"
#include "itex/core/utils/radix_utils.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace functor {
namespace internal {

using float4 = sycl::vec<float, 4>;
using LocalAcc = sycl::local_accessor<uint8, 1>;

// ------------------------------------------------------------------
// GroupTopkKernel

template <int KeysPerItem, int GroupSize, int SubGroupSize, class Selector>
struct GroupTopkKernel {
  GroupTopkKernel(const float* scores, const int num_boxes,
                  const int group_workload, float* output_scores,
                  int* output_boxIds, const int num_topk, LocalAcc scratch)
      : scores_(reinterpret_cast<const uint32*>(scores)),
        num_boxes_(num_boxes),
        group_workload_(group_workload),
        output_scores_(reinterpret_cast<uint32*>(output_scores)),
        output_boxIds_(output_boxIds),
        num_topk_(num_topk),
        scratch_(scratch) {}

  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<1> item) const {
    auto g = item.get_group();
    int group_id = item.get_group(0);
    int local_id = item.get_local_id(0);

    const int offset = group_id * group_workload_;
    const int actual_group_workload =
        sycl::min(group_workload_, num_boxes_ - offset);

    // set the pointer of local memory and radix selector
    uint8* local_mem = ITEXGetLocalAccPointer<uint8>(scratch_);
    Selector selector(g, item.get_sub_group(), local_id, local_mem);

    uint32 item_scores[KeysPerItem] = {0u};
    int item_boxIds[KeysPerItem] = {0};

    // load data from scores to private memory
    for (int i = 0; i < KeysPerItem; ++i) {
      int idx = local_id * KeysPerItem + i;
      if (idx < actual_group_workload) {
        int box_id = offset + idx;
        item_scores[i] = scores_[box_id];
        item_boxIds[i] = box_id;
      }
    }

    // The total number of keys that can be sorted in one work group
    constexpr int CHUNK = KeysPerItem * GroupSize;

    // If actual_group_workload is greater than CHUNK, the selected scores
    // will be used for the next selecting
    uint32* temp_scores =
        reinterpret_cast<uint32*>(local_mem + Selector::LocalStorage::SIZE);
    int* temp_boxIds = reinterpret_cast<int*>(temp_scores + num_topk_);

    int num_start = CHUNK;
    while (num_start < actual_group_workload) {
      selector.SelectTopK(item_scores, item_boxIds, temp_scores, temp_boxIds,
                          num_topk_, /*begin_bit=*/30, 0);
      sycl::group_barrier(g);

      // load selected topk scores from local memory
      for (int i = 0; i < KeysPerItem; ++i) {
        int idx = local_id * KeysPerItem + i;
        if (idx < num_topk_) {
          item_scores[i] = temp_scores[idx];
          item_boxIds[i] = temp_boxIds[idx];
        } else {
          item_scores[i] = 0u;
          int index = num_start + idx - num_topk_;
          if (index < actual_group_workload) {
            int box_id = offset + index;
            item_scores[i] = scores_[box_id];
            item_boxIds[i] = box_id;
          }
        }
      }
      num_start += CHUNK - num_topk_;
      sycl::group_barrier(g);
    }

    const int output_offset = group_id * num_topk_;
    uint32* ptr_output_scores = output_scores_ + output_offset;
    int* ptr_output_boxIds = output_boxIds_ + output_offset;

    // select topK from the last CHUNK of scores and store in output
    selector.SelectTopK(item_scores, item_boxIds, ptr_output_scores,
                        ptr_output_boxIds, num_topk_, /*begin_bit=*/30, 0);
  }

 private:
  const uint32* scores_;
  const int num_boxes_;
  const int group_workload_;
  uint32* output_scores_;
  int* output_boxIds_;
  const int num_topk_;
  LocalAcc scratch_;
};

template <int KeysPerItem, int GroupSize, int SubGroupSize>
void LaunchGroupTopkKernel(const gpuStream_t& stream, const float* scores,
                           const int num_boxes, const int group_workload,
                           float* output_scores, int* output_boxIds,
                           const int num_candidates, const int num_groups) {
  // Type definitions
  using SelectorT = GroupRadixPerBitSelector<uint32, KeysPerItem, GroupSize,
                                             SubGroupSize, sycl::group<1>, int>;
  using GroupTopK =
      GroupTopkKernel<KeysPerItem, GroupSize, SubGroupSize, SelectorT>;

  // Compute the required local memory size
  size_t local_memory_size = SelectorT::LocalStorage::SIZE +
                             num_candidates * (sizeof(uint32) + sizeof(int));

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    GroupTopK task(scores, num_boxes, group_workload, output_scores,
                   output_boxIds, num_candidates, scratch);
    cgh.parallel_for<GroupTopK>(
        sycl::nd_range<1>(sycl::range<1>(num_groups * GroupSize),
                          sycl::range<1>(GroupSize)),
        task);
  });
}

// ------------------------------------------------------------------
// GroupSortKernel

template <bool FilterScore, bool TopkPerformed, int KeysPerItem, int GroupSize,
          int SubGroupSize, class Sortor, class GroupScanT>
struct GroupSortKernel {
  GroupSortKernel(const float* input_scores, const int* input_boxIds,
                  const int num_inputs, const float score_threshold,
                  float* candidate_scores, int* candidate_boxIds,
                  int* num_valid_candidates, const int num_candidates,
                  LocalAcc scratch)
      : input_scores_(reinterpret_cast<const uint32*>(input_scores)),
        input_boxIds_(input_boxIds),
        num_inputs_(num_inputs),
        score_threshold_(*reinterpret_cast<const uint32*>(&score_threshold)),
        candidate_scores_(reinterpret_cast<uint32*>(candidate_scores)),
        candidate_boxIds_(candidate_boxIds),
        num_valid_candidates_(num_valid_candidates),
        num_candidates_(num_candidates),
        scratch_(scratch) {}

  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<1> item) const {
    auto g = item.get_group();
    auto sg = item.get_sub_group();
    int local_id = item.get_local_id(0);

    uint32 item_scores[KeysPerItem] = {0u};
    int item_boxIds[KeysPerItem] = {0};

    // Load scores and box_id from inputs
    for (int i = 0; i < KeysPerItem; ++i) {
      int idx = i + local_id * KeysPerItem;
      if (idx < num_inputs_) {
        item_scores[i] = input_scores_[idx];
        if (TopkPerformed)
          item_boxIds[i] = input_boxIds_[idx];
        else
          item_boxIds[i] = idx;
      }
    }

    // get the pointer of share local memory
    uint8* local_mem = ITEXGetLocalAccPointer<uint8>(scratch_);

    if (FilterScore) {
      // Need to filter scores, counting the number of valid scores
      int item_valid = 0;
      for (int i = 0; i < KeysPerItem; ++i) {
        if (item_scores[i] > score_threshold_) ++item_valid;
      }

      int item_offset, total_valid;
      GroupScanT(g, sg, local_id, local_mem)
          .ExclusiveSum(item_valid, &item_offset, &total_valid);
      sycl::group_barrier(g);

      int num_valid =
          total_valid > num_candidates_ ? num_candidates_ : total_valid;
      if (local_id == 0) *num_valid_candidates_ = num_valid;

      if (num_valid == 0) return;
    } else {
      // No score is filtered, number of valid scores equals the candidates
      if (local_id == 0) *num_valid_candidates_ = num_candidates_;
    }

    // Sort the scores and output the candidates
    Sortor(g, sg, local_id, local_mem)
        .SortDescending(item_scores, item_boxIds, candidate_scores_,
                        candidate_boxIds_, num_candidates_, 0, 30);
  }

 private:
  const uint32* input_scores_;
  const int* input_boxIds_;
  const int num_inputs_;
  const uint32 score_threshold_;
  uint32* candidate_scores_;
  int* candidate_boxIds_;
  int* num_valid_candidates_;
  const int num_candidates_;
  LocalAcc scratch_;
};

template <bool FilterScore, bool TopkPerformed, int KeysPerItem, int GroupSize,
          int SubGroupSize>
void LaunchGroupSortKernel(const gpuStream_t& stream, const float* input_scores,
                           const int* input_boxIds, const int num_inputs,
                           const float score_threshold, float* candidate_scores,
                           int* candidate_boxIds, int* num_valid_candidates,
                           const int num_candidates) {
  // Type definitions
  using SorterT = GroupRadixSortor<uint32, KeysPerItem, GroupSize, SubGroupSize,
                                   sycl::group<1>, int>;
  using GroupScanT = GroupScan<int, GroupSize, SubGroupSize, sycl::group<1>>;
  using GroupSort =
      GroupSortKernel<FilterScore, TopkPerformed, KeysPerItem, GroupSize,
                      SubGroupSize, SorterT, GroupScanT>;

  // Compute the required local memory size
  size_t local_memory_size =
      SorterT::LocalStorage::SIZE > GroupScanT::LocalStorage::SIZE
          ? SorterT::LocalStorage::SIZE
          : GroupScanT::LocalStorage::SIZE;

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    GroupSort task(input_scores, input_boxIds, num_inputs, score_threshold,
                   candidate_scores, candidate_boxIds, num_valid_candidates,
                   num_candidates, scratch);
    cgh.parallel_for<GroupSort>(
        sycl::nd_range<1>(sycl::range<1>(GroupSize), sycl::range<1>(GroupSize)),
        task);
  });
}

// ------------------------------------------------------------------
// NMSKernel

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
template <int GroupSize, typename GroupT>
class NMSReduce {
 private:
  struct _LocalStorage {
    uint32 bit_mask[GroupSize];
  };

  const GroupT& g;
  const int local_id;
  _LocalStorage& local_mem;

  // --------------------------------------------
  // Utility methods
  // --------------------------------------------
  bool CheckMask(const int idx) {
    return local_mem.bit_mask[idx % GroupSize] >> (idx / GroupSize) & 1;
  }
  void SetMaskZero(const int idx) {
    local_mem.bit_mask[idx % GroupSize] &= ~(1 << (idx / GroupSize));
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
             int num_elements, int max_output_size, float iou_threshold) {
    // initialize mask value
    local_mem.bit_mask[local_id] = 0xFFFFFFFF;
    sycl::group_barrier(g);

    *num_accepted = 0;
    for (int i = 0; i < num_elements; ++i) {
      // if current box is masked by an earlier box, skip it.
      if (!CheckMask(i)) continue;
      // record the selected index
      if (local_id == 0) out_indices[*num_accepted] = i;
      if (++(*num_accepted) >= max_output_size) break;

      // get current selected box
      float4 cur_box = boxes[i];
      float cur_area =
          (cur_box.z() - cur_box.x()) * (cur_box.w() - cur_box.y());

      // loop over the left boxes and set corresponding mask value
      for (int idx = local_id + i + 1; idx < num_elements; idx += GroupSize) {
        float4 target_box = boxes[idx];

        if (OverThreshold(cur_box, target_box, cur_area, iou_threshold)) {
          SetMaskZero(idx);
        }
      }
      sycl::group_barrier(g);
    }
  }
};

template <int GroupSize, class NmsReducer>
struct NMSKernel {
  NMSKernel(const float* boxes, const float* candidate_scores,
            const int* candidate_boxIds, const int num_valid_candidates,
            int* selected_boxIds, const int num_boxes, const int num_candidates,
            const int max_output_size, int* num_outputs,
            const bool pad_to_max_output, const float iou_threshold,
            LocalAcc scratch)
      : boxes_(reinterpret_cast<const float4*>(boxes)),
        candidate_scores_(candidate_scores),
        candidate_boxIds_(candidate_boxIds),
        num_valid_candidates_(num_valid_candidates),
        selected_boxIds_(selected_boxIds),
        num_boxes_(num_boxes),
        num_candidates_(num_candidates),
        max_output_size_(max_output_size),
        num_outputs_(num_outputs),
        pad_to_max_output_(pad_to_max_output),
        iou_threshold_(iou_threshold),
        scratch_(scratch) {}

  void operator()(sycl::nd_item<1> item) const {
    auto g = item.get_group();
    int local_id = item.get_local_id(0);

    // local memory allocation
    uint8* local_mem = ITEXGetLocalAccPointer<uint8>(scratch_);
    float4* sorted_boxes =
        reinterpret_cast<float4*>(local_mem + NmsReducer::LocalStorage::SIZE);

    // Load boxes into local memory and compute the areas
    for (int idx = local_id; idx < num_valid_candidates_; idx += GroupSize) {
      int box_id = candidate_boxIds_[idx];
      float4 cur_box = boxes_[box_id];
      Flipped(cur_box);
      sorted_boxes[idx] = cur_box;
    }
    sycl::group_barrier(g);

    // Output indices
    uint16* out_indices =
        reinterpret_cast<uint16*>(sorted_boxes + num_candidates_);

    // do NMS reduce
    int num_accepted;
    NmsReducer(g, local_id, local_mem)
        .DoNms(sorted_boxes, out_indices, &num_accepted, num_valid_candidates_,
               max_output_size_, iou_threshold_);

    sycl::group_barrier(g);

    // output NMS selected indices
    for (int idx = local_id; idx < max_output_size_; idx += GroupSize) {
      if (idx < num_accepted) {
        uint16 index = out_indices[idx];
        selected_boxIds_[idx] = candidate_boxIds_[index];
      } else {
        selected_boxIds_[idx] = 0;
      }
    }

    if (local_id == 0)
      *num_outputs_ = pad_to_max_output_ ? max_output_size_ : num_accepted;
  }

 private:
  const float4* boxes_;
  const float* candidate_scores_;
  const int* candidate_boxIds_;
  const int num_valid_candidates_;
  int* selected_boxIds_;
  const int num_boxes_;
  const int num_candidates_;
  const int max_output_size_;
  int* num_outputs_;
  const bool pad_to_max_output_;
  const float iou_threshold_;
  LocalAcc scratch_;
};

template <int GroupSize>
void LaunchNMSKernel(const gpuStream_t& stream, const float* boxes,
                     const float* candidate_scores, const int* candidate_boxIds,
                     const int num_valid_candidates, int* selected_boxIds,
                     const int num_boxes, const int num_candidates,
                     const int max_output_size, int* num_outputs,
                     const bool pad_to_max_output, const float iou_threshold) {
  // Type definitions
  using NmsReducer = NMSReduce<GroupSize, sycl::group<1>>;

  size_t local_memory_size = NmsReducer::LocalStorage::SIZE +
                             num_candidates * sizeof(float4) +
                             max_output_size * sizeof(uint16_t);

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    NMSKernel<GroupSize, NmsReducer> task(
        boxes, candidate_scores, candidate_boxIds, num_valid_candidates,
        selected_boxIds, num_boxes, num_candidates, max_output_size,
        num_outputs, pad_to_max_output, iou_threshold, scratch);
    cgh.parallel_for<NMSKernel<GroupSize, NmsReducer>>(
        sycl::nd_range<1>(sycl::range<1>(GroupSize), sycl::range<1>(GroupSize)),
        task);
  });
}

}  // namespace internal
}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_IMAGE_NON_MAX_SUPPRESSION_OP_GPU_H_
