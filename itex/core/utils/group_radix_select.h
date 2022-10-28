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

#ifndef ITEX_CORE_UTILS_GROUP_RADIX_SELECT_H_
#define ITEX_CORE_UTILS_GROUP_RADIX_SELECT_H_

#include <algorithm>

#include "itex/core/utils/group_radix_rank.h"
#include "itex/core/utils/radix_utils.h"

namespace itex {
// ------------------------------------------------------------------
// GroupRadixSelector
// ------------------------------------------------------------------

template <typename KeyT, int KEYS_PER_WORK_ITEM, int GROUP_SIZE,
          int SUB_GROUP_SIZE, typename GroupT, typename ValueT = KeyT,
          int RADIX_BITS = 4, typename SubGroupT = sycl::sub_group>
class GroupRadixSelector {
 private:
  // --------------------------------------------
  // Constants and type definitions
  // --------------------------------------------

  // Constants
  enum {
    NUM_KEYS = KEYS_PER_WORK_ITEM * GROUP_SIZE,
  };

  // Key traits and unsigned bits type
  using KeyTraits = NumericTraits<KeyT>;
  using UnsignedT = typename KeyTraits::UnsignedT;

  // Rank type
  using GroupRadixRankT =
      GroupRadixRank<GROUP_SIZE, SUB_GROUP_SIZE, GroupT, RADIX_BITS>;
  // Digit extractor type
  using RadixExtractorT = RadixExtractor<KeyT>;

  // Local memory layout type
  union _LocalStorage {
    typename GroupRadixRankT::LocalStorage rank_storage;
    UnsignedT exchange_ukeys[NUM_KEYS];
    ValueT exchange_values[NUM_KEYS];
  };

  // --------------------------------------------
  // Work item fields
  // --------------------------------------------

  const GroupT& g;
  const SubGroupT& sg;
  const int local_id;
  _LocalStorage& local_mem;

  // --------------------------------------------
  // Utility methods
  // --------------------------------------------

  // Convert unsigned keys in descending order
  void ConvertKeys(UnsignedT (&ukeys)[KEYS_PER_WORK_ITEM]) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      ukeys[ikey] = KeyTraits::Convert(ukeys[ikey], Int2Type<true>());
    }
  }

  // --------------------------------------------
  // Exchange unsigned keys
  void ExchangeKeys(UnsignedT (&ukeys)[KEYS_PER_WORK_ITEM],
                    uint16_t (&ranks)[KEYS_PER_WORK_ITEM],
                    uint16_t offset_select, uint16_t offset_active,
                    uint32_t* active_mask) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      if (ranks[ikey] >= offset_select && ranks[ikey] < offset_active) {
        local_mem.exchange_ukeys[ranks[ikey] - offset_select] = ukeys[ikey];
      }
    }
    sycl::group_barrier(g);

    *active_mask = 0u;  // reset active mask
    uint32_t num_active = offset_active - offset_select;

#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      int offset = (local_id * KEYS_PER_WORK_ITEM) + ikey;
      if (offset < num_active) {
        *active_mask |= (1u << ikey);
        ukeys[ikey] = local_mem.exchange_ukeys[offset];
      }
    }
    sycl::group_barrier(g);
  }

  // Exchange values
  void ExchangeValues(ValueT (&values)[KEYS_PER_WORK_ITEM],
                      uint16_t (&ranks)[KEYS_PER_WORK_ITEM],
                      uint16_t offset_select, uint16_t offset_active) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      if (ranks[ikey] >= offset_select && ranks[ikey] < offset_active) {
        local_mem.exchange_values[ranks[ikey] - offset_select] = values[ikey];
      }
    }
    sycl::group_barrier(g);

    uint32_t num_active = offset_active - offset_select;

#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      int offset = (local_id * KEYS_PER_WORK_ITEM) + ikey;
      if (offset < num_active) {
        values[ikey] = local_mem.exchange_values[offset];
      }
    }
    sycl::group_barrier(g);
  }

  // --------------------------------------------
  // Output the selected topk keys
  void WriteOutKeys(UnsignedT (&ukeys)[KEYS_PER_WORK_ITEM], UnsignedT* output,
                    uint16_t (&ranks)[KEYS_PER_WORK_ITEM],
                    uint16_t offset_select, int num_selected) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      if (ranks[ikey] < offset_select) {
        output[ranks[ikey] + num_selected] =
            KeyTraits::ConvertBack(ukeys[ikey], Int2Type<true>());
      }
    }
  }

  // Output values
  void WriteOutValues(ValueT (&values)[KEYS_PER_WORK_ITEM], ValueT* output,
                      uint16_t (&ranks)[KEYS_PER_WORK_ITEM],
                      uint16_t offset_select, int num_selected) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      if (ranks[ikey] < offset_select) {
        output[ranks[ikey] + num_selected] = values[ikey];
      }
    }
  }

  // --------------------------------------------
  // Internal select method
  // --------------------------------------------

  template <int IS_PAIR_SELECT>
  void InternalSelectImpl(KeyT (&keys)[KEYS_PER_WORK_ITEM],
                          ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                          ValueT* out_values, int num_topk, int begin_bit,
                          int end_bit) {
    // data type convert to unsigned
    UnsignedT(&ukeys)[KEYS_PER_WORK_ITEM] =
        reinterpret_cast<UnsignedT(&)[KEYS_PER_WORK_ITEM]>(keys);
    UnsignedT* out_ukeys = reinterpret_cast<UnsignedT*>(out_keys);

    // Convert unsigned keys in descending order
    ConvertKeys(ukeys);

    // Instantiate radix_rank
    GroupRadixRankT radix_rank(g, sg, local_id, &local_mem.rank_storage);

    // initial active mask
    uint32_t active_mask = 0xFFFFFFFF;
    int num_selected = 0;

    // iterate bits from begin to end
    while (true) {
      int pass_bits = sycl::min(RADIX_BITS, begin_bit - end_bit);
      begin_bit -= pass_bits;
      RadixExtractorT radix_extractor(begin_bit, pass_bits);

      // Ranking the unsigned keys
      uint16_t ranks[KEYS_PER_WORK_ITEM];
      uint16_t offset_select, offset_active;
      radix_rank.RankKeys(ukeys, ranks, &radix_extractor, active_mask,
                          num_topk - num_selected, &offset_select,
                          &offset_active);
      sycl::group_barrier(g);

      if (begin_bit == end_bit) offset_select = num_topk - num_selected;
      // Output selected keys
      if (offset_select > 0) {
        WriteOutKeys(ukeys, out_ukeys, ranks, offset_select, num_selected);
        if (IS_PAIR_SELECT)
          WriteOutValues(values, out_values, ranks, offset_select,
                         num_selected);
      }
      num_selected += offset_select;

      // Conditions for terminating the while loop
      if (num_selected == num_topk) break;

      // Exchange active keys and update active_mask
      ExchangeKeys(ukeys, ranks, offset_select, offset_active, &active_mask);
      if (IS_PAIR_SELECT)
        ExchangeValues(values, ranks, offset_select, offset_active);
    }  // end of while loop
  }

 public:
  // Local memory storage type
  struct LocalStorage : BaseStorage<_LocalStorage> {};

  // Constructors
  GroupRadixSelector(const GroupT& g_, const SubGroupT& sg_,
                     const int local_id_, LocalStorage* local_mem_)
      : g(g_), sg(sg_), local_id(local_id_), local_mem(local_mem_->Alias()) {}

  GroupRadixSelector(const GroupT& g_, const SubGroupT& sg_,
                     const int local_id_, uint8_t* local_mem_)
      : g(g_),
        sg(sg_),
        local_id(local_id_),
        local_mem(reinterpret_cast<_LocalStorage&>(*local_mem_)) {}

  // --------------------------------------------
  // Select top-k keys
  void SelectTopK(KeyT (&keys)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                  int num_topk, int begin_bit = sizeof(KeyT) * 8,
                  int end_bit = 0) {
    InternalSelectImpl<false>(keys, keys, out_keys, out_keys, num_topk,
                              begin_bit, end_bit);
  }

  // Select paired top-k keys and values
  void SelectTopK(KeyT (&keys)[KEYS_PER_WORK_ITEM],
                  ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                  ValueT* out_values, int num_topk,
                  int begin_bit = sizeof(KeyT) * 8, int end_bit = 0) {
    InternalSelectImpl<true>(keys, values, out_keys, out_values, num_topk,
                             begin_bit, end_bit);
  }
};

// ------------------------------------------------------------------
// GroupRadixPerBitSelector
// ------------------------------------------------------------------

template <typename KeyT, int KEYS_PER_WORK_ITEM, int GROUP_SIZE,
          int SUB_GROUP_SIZE, typename GroupT, typename ValueT = KeyT,
          typename SubGroupT = sycl::sub_group>
class GroupRadixPerBitSelector {
 private:
  // --------------------------------------------
  // Constants and type definitions
  // --------------------------------------------

  // Key traits and unsigned bits type
  using KeyTraits = NumericTraits<KeyT>;
  using UnsignedT = typename KeyTraits::UnsignedT;

  // GroupScan type
  using GroupScanT = GroupScan<int, GROUP_SIZE, SUB_GROUP_SIZE, GroupT>;

  // Local memory layout type
  struct _LocalStorage {
    typename GroupScanT::LocalStorage group_scan;
  };

  // --------------------------------------------
  // Work item fields
  // --------------------------------------------

  const GroupT& g;
  const SubGroupT& sg;
  const int local_id;
  _LocalStorage& local_mem;

  // --------------------------------------------
  // Utility methods
  // --------------------------------------------

  // Check whether the keys are above the threshold
  bool CheckKeys(KeyT (&keys)[KEYS_PER_WORK_ITEM],
                 ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                 ValueT* out_values, int num_topk, KeyT threshold,
                 int* num_selected) {
    // count the number of valid keys in each work-item
    int item_valid = 0;
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      if (keys[ikey] > threshold) ++item_valid;
    }

    int item_offset, total_valid;
    GroupScanT(g, sg, local_id, &local_mem.group_scan)
        .ExclusiveSum(item_valid, &item_offset, &total_valid);

    if (total_valid == 0) {
      // all keys are invalid, nothing to do
      *num_selected = 0;
      return false;
    } else if (total_valid <= num_topk) {
      // there's no need to select
#pragma unroll
      for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
        if (keys[ikey] > threshold) {
          out_keys[item_offset] = keys[ikey];
          out_values[item_offset] = values[ikey];
          ++item_offset;
        }
      }
      *num_selected = total_valid;
      return false;
    } else {
      *num_selected = num_topk;
      return true;
    }
  }

  // Internal select method
  template <int IS_PAIR_SELECT>
  void InternalSelectImpl(KeyT (&keys)[KEYS_PER_WORK_ITEM],
                          ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                          ValueT* out_values, int num_topk, int begin_bit,
                          int end_bit) {
    // convert data type to unsigned
    UnsignedT(&ukeys)[KEYS_PER_WORK_ITEM] =
        reinterpret_cast<UnsignedT(&)[KEYS_PER_WORK_ITEM]>(keys);
    UnsignedT* out_ukeys = reinterpret_cast<UnsignedT*>(out_keys);

    // Convert unsigned keys in ascending order
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      ukeys[ikey] = KeyTraits::Convert(ukeys[ikey], Int2Type<false>());
    }

    // Instantiate group_scan
    GroupScanT group_scan(g, sg, local_id, &local_mem.group_scan);

    // initialization
    uint32_t bit_mask = 1u << begin_bit;
    uint32_t desired_mask = bit_mask;

    int num_selected = 0;  // num of selected elements
    while (true) {
      // compute the state of each elements
      int item_count = 0;
      uint32_t enabled_mask = 0u;
#pragma unroll
      for (int i = 0; i < KEYS_PER_WORK_ITEM; ++i) {
        int desired = ((ukeys[i] & bit_mask) ^ desired_mask) == 0;
        enabled_mask |= desired << i;
        item_count += desired;
      }

      // do exclusive scan
      int item_offset, group_count;
      group_scan.ExclusiveSum(item_count, num_selected, &item_offset,
                              &group_count);

      // update indexes of selected elements
      if (group_count > num_selected &&
          (group_count <= num_topk || begin_bit == end_bit)) {
#pragma unroll
        for (int i = 0; i < KEYS_PER_WORK_ITEM; ++i) {
          if ((enabled_mask >> i & 1) && (item_offset < num_topk)) {
            out_ukeys[item_offset] =
                KeyTraits::ConvertBack(ukeys[i], Int2Type<false>());
            if (IS_PAIR_SELECT) out_values[item_offset] = values[i];
            ++item_offset;
          }
        }
        num_selected = group_count;
      }

      // Conditions for terminating the while loop
      if (num_selected >= num_topk) {
        return;
      } else if (group_count > num_topk) {
        --begin_bit;
        desired_mask |= 1u << begin_bit;
        bit_mask |= 1u << begin_bit;
      } else {
        desired_mask ^= 1u << begin_bit;
        if (begin_bit > end_bit) {
          --begin_bit;
          desired_mask |= 1u << begin_bit;
          bit_mask |= 1u << begin_bit;
        }
      }
    }  // end of while loop
  }

 public:
  // Local memory storage type
  struct LocalStorage : BaseStorage<_LocalStorage> {};

  // Constructor
  GroupRadixPerBitSelector(const GroupT& g_, const SubGroupT& sg_,
                           const int local_id_, LocalStorage* local_mem_)
      : g(g_), sg(sg_), local_id(local_id_), local_mem(local_mem_->Alias()) {}

  GroupRadixPerBitSelector(const GroupT& g_, const SubGroupT& sg_,
                           const int local_id_, uint8_t* local_mem_)
      : g(g_),
        sg(sg_),
        local_id(local_id_),
        local_mem(reinterpret_cast<_LocalStorage&>(*local_mem_)) {}

  // --------------------------------------------
  // Select top-k keys
  void SelectTopK(KeyT (&keys)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                  int num_topk, int begin_bit = sizeof(KeyT) * 8,
                  int end_bit = 0) {
    InternalSelectImpl<false>(keys, keys, out_keys, out_keys, num_topk,
                              begin_bit - 1, end_bit);
  }

  // Select paired top-k keys and values
  void SelectTopK(KeyT (&keys)[KEYS_PER_WORK_ITEM],
                  ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                  ValueT* out_values, int num_topk,
                  int begin_bit = sizeof(KeyT) * 8, int end_bit = 0) {
    InternalSelectImpl<true>(keys, values, out_keys, out_values, num_topk,
                             begin_bit - 1, end_bit);
  }

  // Select paired top-k with a threshold
  void SelectTopK(KeyT (&keys)[KEYS_PER_WORK_ITEM],
                  ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                  ValueT* out_values, int num_topk, KeyT threshold,
                  int* num_selected, int begin_bit = sizeof(KeyT) * 8,
                  int end_bit = 0) {
    if (CheckKeys(keys, values, out_keys, out_values, num_topk, threshold,
                  num_selected)) {
      InternalSelectImpl<true>(keys, values, out_keys, out_values, num_topk,
                               begin_bit - 1, end_bit);
    }
  }
};
}  // namespace itex
#endif  // ITEX_CORE_UTILS_GROUP_RADIX_SELECT_H_
