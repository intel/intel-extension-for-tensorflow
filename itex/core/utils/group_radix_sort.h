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

#ifndef ITEX_CORE_UTILS_GROUP_RADIX_SORT_H_
#define ITEX_CORE_UTILS_GROUP_RADIX_SORT_H_

#include <algorithm>

#include "itex/core/utils/group_radix_rank.h"
#include "itex/core/utils/radix_utils.h"

namespace itex {
// ------------------------------------------------------------------
// GroupRadixSortor
// ------------------------------------------------------------------

template <typename KeyT, int KEYS_PER_WORK_ITEM, int GROUP_SIZE,
          int SUB_GROUP_SIZE, typename GroupT, typename ValueT = KeyT,
          int RADIX_BITS = 4, typename SubGroupT = sycl::sub_group>
class GroupRadixSortor {
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

  // Convert unsigned keys according to the order
  template <int IS_DESCENDING>
  void ConvertKeys(UnsignedT (&ukeys)[KEYS_PER_WORK_ITEM]) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      ukeys[ikey] = KeyTraits::Convert(ukeys[ikey], Int2Type<IS_DESCENDING>());
    }
  }

  // --------------------------------------------
  // Exchange unsigned keys
  void ExchangeKeys(UnsignedT (&ukeys)[KEYS_PER_WORK_ITEM],
                    uint16_t (&ranks)[KEYS_PER_WORK_ITEM]) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      local_mem.exchange_ukeys[ranks[ikey]] = ukeys[ikey];
    }
    sycl::group_barrier(g);

#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      int offset = (local_id * KEYS_PER_WORK_ITEM) + ikey;
      ukeys[ikey] = local_mem.exchange_ukeys[offset];
    }
    sycl::group_barrier(g);
  }

  // Exchange values
  void ExchangeValues(ValueT (&values)[KEYS_PER_WORK_ITEM],
                      uint16_t (&ranks)[KEYS_PER_WORK_ITEM]) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      local_mem.exchange_values[ranks[ikey]] = values[ikey];
    }
    sycl::group_barrier(g);

#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      int offset = (local_id * KEYS_PER_WORK_ITEM) + ikey;
      values[ikey] = local_mem.exchange_values[offset];
    }
    sycl::group_barrier(g);
  }

  // --------------------------------------------
  // Output unsigned keys according to the order
  template <int IS_DESCENDING>
  void WriteOutKeys(UnsignedT (&ukeys)[KEYS_PER_WORK_ITEM], UnsignedT* output,
                    uint16_t (&ranks)[KEYS_PER_WORK_ITEM], int num_output) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      if (ranks[ikey] < num_output) {
        output[ranks[ikey]] =
            KeyTraits::ConvertBack(ukeys[ikey], Int2Type<IS_DESCENDING>());
      }
    }
  }

  // Output values
  void WriteOutValues(ValueT (&values)[KEYS_PER_WORK_ITEM], ValueT* output,
                      uint16_t (&ranks)[KEYS_PER_WORK_ITEM], int num_output) {
#pragma unroll
    for (int ikey = 0; ikey < KEYS_PER_WORK_ITEM; ++ikey) {
      if (ranks[ikey] < num_output) {
        output[ranks[ikey]] = values[ikey];
      }
    }
  }

  // --------------------------------------------
  // Internal sort method
  // --------------------------------------------

  template <int IS_DESCENDING, int IS_PAIR_SORT>
  void InternalSortImpl(KeyT (&keys)[KEYS_PER_WORK_ITEM],
                        ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                        ValueT* out_values, int num_output, int begin_bit,
                        int end_bit) {
    // Convert keys data type to unsigned
    UnsignedT(&ukeys)[KEYS_PER_WORK_ITEM] =
        reinterpret_cast<UnsignedT(&)[KEYS_PER_WORK_ITEM]>(keys);
    UnsignedT* out_ukeys = reinterpret_cast<UnsignedT*>(out_keys);

    // Convert unsigned keys according to the order
    ConvertKeys<IS_DESCENDING>(ukeys);

    // Instantiate radix_rank
    GroupRadixRankT radix_rank(g, sg, local_id, &local_mem.rank_storage);

    // iterate bits from begin to end
    while (true) {
      int pass_bits = sycl::min(RADIX_BITS, end_bit - begin_bit);
      RadixExtractorT radix_extractor(begin_bit, pass_bits);
      begin_bit += pass_bits;

      // Ranking the unsigned keys
      uint16_t ranks[KEYS_PER_WORK_ITEM];
      radix_rank.RankKeys(ukeys, ranks, &radix_extractor);
      sycl::group_barrier(g);

      // Conditions for terminating the while loop
      if (begin_bit == end_bit) {
        // It is the last iteration, ouput sorted keys and/or values
        WriteOutKeys<IS_DESCENDING>(ukeys, out_ukeys, ranks, num_output);
        if (IS_PAIR_SORT) WriteOutValues(values, out_values, ranks, num_output);
        // terminate the loop
        break;
      }

      // Exchange unsigned keys and/or values
      ExchangeKeys(ukeys, ranks);
      if (IS_PAIR_SORT) ExchangeValues(values, ranks);
    }  // end of while loop
  }

 public:
  // Local memory storage type
  struct LocalStorage : BaseStorage<_LocalStorage> {};

  // Constructors
  GroupRadixSortor(const GroupT& g_, const SubGroupT& sg_, const int local_id_,
                   LocalStorage* local_mem_)
      : g(g_), sg(sg_), local_id(local_id_), local_mem(local_mem_->Alias()) {}

  GroupRadixSortor(const GroupT& g_, const SubGroupT& sg_, const int local_id_,
                   uint8_t* local_mem_)
      : g(g_),
        sg(sg_),
        local_id(local_id_),
        local_mem(reinterpret_cast<_LocalStorage&>(*local_mem_)) {}

  // --------------------------------------------
  // Sort keys in ascending order
  void Sort(KeyT (&keys)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
            int num_output = NUM_KEYS, int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8) {
    InternalSortImpl<false, false>(keys, keys, out_keys, out_keys, num_output,
                                   begin_bit, end_bit);
  }

  // Sort paired keys and values in ascending order
  void Sort(KeyT (&keys)[KEYS_PER_WORK_ITEM],
            ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
            ValueT* out_values, int num_output = NUM_KEYS, int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8) {
    InternalSortImpl<false, true>(keys, values, out_keys, out_values,
                                  num_output, begin_bit, end_bit);
  }

  // Sort keys in descending order
  void SortDescending(KeyT (&keys)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                      int num_output = NUM_KEYS, int begin_bit = 0,
                      int end_bit = sizeof(KeyT) * 8) {
    InternalSortImpl<true, false>(keys, keys, out_keys, out_keys, num_output,
                                  begin_bit, end_bit);
  }

  // Sort paired keys and values in descending order
  void SortDescending(KeyT (&keys)[KEYS_PER_WORK_ITEM],
                      ValueT (&values)[KEYS_PER_WORK_ITEM], KeyT* out_keys,
                      ValueT* out_values, int num_output = NUM_KEYS,
                      int begin_bit = 0, int end_bit = sizeof(KeyT) * 8) {
    InternalSortImpl<true, true>(keys, values, out_keys, out_values, num_output,
                                 begin_bit, end_bit);
  }
};
}  // namespace itex
#endif  // ITEX_CORE_UTILS_GROUP_RADIX_SORT_H_
