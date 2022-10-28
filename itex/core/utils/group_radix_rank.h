/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_GROUP_RADIX_RANK_H_
#define ITEX_CORE_UTILS_GROUP_RADIX_RANK_H_

#include "itex/core/utils/group_scan.h"
#include "itex/core/utils/radix_utils.h"

namespace itex {
// ------------------------------------------------------------------
// GroupRadixRank
// ------------------------------------------------------------------

template <int GROUP_SIZE, int SUB_GROUP_SIZE, typename GroupT,
          int RADIX_BITS = 4, typename SubGroupT = sycl::sub_group>
class GroupRadixRank {
 private:
  // --------------------------------------------
  // Constants and type definitions
  // --------------------------------------------

  using CounterT = uint16_t;        // type for counters
  using PackedCounterT = uint32_t;  // type for packed counters

  // Constants
  enum {
    RADIX_BUCKETS = 1 << RADIX_BITS,
    NUM_CYCLES = (RADIX_BUCKETS - 1) / SUB_GROUP_SIZE + 1,
    COUNTER_TYPE_SIZE = static_cast<int>(sizeof(CounterT)),
    PACKED_TYPE_SIZE = static_cast<int>(sizeof(PackedCounterT)),
    PACKING_RATIO = PACKED_TYPE_SIZE / COUNTER_TYPE_SIZE,
    LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE,
    LOG_COUNTER_LANES =
        (RADIX_BITS > LOG_PACKING_RATIO) ? RADIX_BITS - LOG_PACKING_RATIO : 0,
    COUNTER_LANES = 1 << LOG_COUNTER_LANES,
  };

  // GroupScan type
  using GroupScanT =
      GroupScan<PackedCounterT, GROUP_SIZE, SUB_GROUP_SIZE, GroupT>;

  // Local memory layout type
  struct _LocalStorage {
    union Detail {
      CounterT counters[COUNTER_LANES * GROUP_SIZE * PACKING_RATIO];
      PackedCounterT packed_counters[GROUP_SIZE][COUNTER_LANES];
    } detail;
    CounterT offsets[2];
    typename GroupScanT::LocalStorage group_scan;
  };

  // --------------------------------------------
  // Work item fields
  // --------------------------------------------

  const GroupT& g;
  const SubGroupT& sg;
  const int local_id;
  const int sg_id;
  _LocalStorage& local_mem;

  // --------------------------------------------
  // Utility methods
  // --------------------------------------------

  // GroupScan prefix callback functor
  struct PrefixCallBack {
    PackedCounterT operator()(PackedCounterT group_sum) {
      PackedCounterT group_prefix = 0;
#pragma unroll
      for (int pack = 1; pack < PACKING_RATIO; ++pack) {
        group_prefix += group_sum << (COUNTER_TYPE_SIZE * 8 * pack);
      }
      return group_prefix;
    }
  };

  uint16_t BucketToPosition(uint32_t bucket, int local_id) {
    uint32_t lane = bucket & (COUNTER_LANES - 1);
    uint32_t sub_lane = bucket >> LOG_COUNTER_LANES;
    return (lane * GROUP_SIZE + local_id) * PACKING_RATIO + sub_lane;
  }

  // Scan counters in local memory
  template <bool IS_SELECT_RANKING = false>
  void ScanCounters() {
    // Upsweep sum
    PackedCounterT* ptr_packed = local_mem.detail.packed_counters[local_id];
    PackedCounterT item_sum = 0;
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      item_sum += ptr_packed[lane];
    }

    // Compute exclusive sum
    PackedCounterT item_exclusive;
    PrefixCallBack prefix_call_back;

    GroupScanT(g, sg, local_id, &local_mem.group_scan)
        .ExclusiveSum(item_sum, &item_exclusive, prefix_call_back);

    // Downsweep exclusive scan
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      PackedCounterT tmp = ptr_packed[lane];
      ptr_packed[lane] = item_exclusive;
      item_exclusive += tmp;
    }

    if (IS_SELECT_RANKING) {
      if (local_id == GROUP_SIZE - 1) {
        local_mem.offsets[1] = static_cast<CounterT>(
            item_exclusive >> (8 * COUNTER_TYPE_SIZE * (PACKING_RATIO - 1)));
      }
    }
    sycl::group_barrier(g);
  }

  // Find the select and active offset for radix select
  void FindOffset(int num_to_select, uint16_t* offset_select,
                  uint16_t* offset_active) {
    // init
    *offset_select = 0;
    *offset_active = 0;
    for (int bucket = 1; bucket < RADIX_BUCKETS; ++bucket) {
      uint16_t position = BucketToPosition(bucket, 0);
      CounterT count = local_mem.detail.counters[position];

      if (count > num_to_select) {
        *offset_active = count;
        break;
      }
      *offset_select = count;
    }
    if (*offset_active == 0) *offset_active = local_mem.offsets[1];
  }

 public:
  // Local memory type
  struct LocalStorage : BaseStorage<_LocalStorage> {};

  // Constructor
  GroupRadixRank(const GroupT& g_, const SubGroupT& sg_, const int local_id_,
                 LocalStorage* local_mem_)
      : g(g_),
        sg(sg_),
        local_id(local_id_),
        sg_id(sg_.get_group_linear_id()),
        local_mem(local_mem_->Alias()) {}

  // --------------------------------------------
  // For radix sort
  template <int KEYS_PER_WORK_ITEM, typename UnsignedT,
            typename RadixExtractorT>
  void RankKeys(UnsignedT (&ukeys)[KEYS_PER_WORK_ITEM],
                uint16_t (&ranks)[KEYS_PER_WORK_ITEM],
                RadixExtractorT* radix_extractor) {
    // For each key, the position of its corresponding counter in local_mem
    uint16_t counter_pos[KEYS_PER_WORK_ITEM];

    // reset local memory counters
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      local_mem.detail.packed_counters[local_id][lane] = 0;
    }
    sycl::group_barrier(g);

#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      uint32_t bucket = radix_extractor->Bucket(ukeys[ikey]);
      counter_pos[ikey] = BucketToPosition(bucket, local_id);

      ranks[ikey] = (local_mem.detail.counters[counter_pos[ikey]])++;
    }
    sycl::group_barrier(g);

    // Scan local memory counters
    ScanCounters();

    // Update ranks of item elements
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      ranks[ikey] += local_mem.detail.counters[counter_pos[ikey]];
    }
  }

  // --------------------------------------------
  // For radix select
  template <int KEYS_PER_WORK_ITEM, typename UnsignedT,
            typename RadixExtractorT>
  void RankKeys(UnsignedT (&ukeys)[KEYS_PER_WORK_ITEM],
                uint16_t (&ranks)[KEYS_PER_WORK_ITEM],
                RadixExtractorT* radix_extractor, uint32_t active_mask,
                int num_to_select, uint16_t* offset_select,
                uint16_t* offset_active) {
    // Set the max rank as the total number of keys in the registers
    constexpr int MAX_RANK = GROUP_SIZE * KEYS_PER_WORK_ITEM;

    // For each key, the position of its corresponding counter in local_mem
    uint16_t counter_pos[KEYS_PER_WORK_ITEM];

    // reset local memory counters
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      local_mem.detail.packed_counters[local_id][lane] = 0;
    }
    sycl::group_barrier(g);

#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      // The inactive key's ranks are set as MAX_RANK
      ranks[ikey] = MAX_RANK;
      if (active_mask >> ikey & 1) {
        uint32_t bucket = radix_extractor->Bucket(ukeys[ikey]);
        counter_pos[ikey] = BucketToPosition(bucket, local_id);

        ranks[ikey] = (local_mem.detail.counters[counter_pos[ikey]])++;
      }
    }
    sycl::group_barrier(g);

    // Scan local memory counters
    ScanCounters<true>();

    // Find the select and active offset
    FindOffset(num_to_select, offset_select, offset_active);

    // Update ranks of item elements
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      if (active_mask >> ikey & 1) {
        ranks[ikey] += local_mem.detail.counters[counter_pos[ikey]];
      }
    }
  }
};
}  // namespace itex
#endif  // ITEX_CORE_UTILS_GROUP_RADIX_RANK_H_
