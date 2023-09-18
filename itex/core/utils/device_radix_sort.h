/* Copyright (c) 2023 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_DEVICE_RADIX_SORT_H_
#define ITEX_CORE_UTILS_DEVICE_RADIX_SORT_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/group_scan.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/radix_utils.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

///////////////// DeviceRadixSortUpsweep /////////////////
template <typename KeyType, typename ValueType, int32_t SUBGROUP_SIZE,
          int32_t GROUP_SIZE, bool IS_DESCENDING = false,
          int32_t RADIX_BITS = 4, int32_t ELEMS_PER_ITEM = 4>
class DeviceRadixSortUpsweep {
 public:
  using KeyTraits = NumericTraits<KeyType>;
  using KeyTraitsT = typename KeyTraits::UnsignedT;
  using DigitCounter = u_char;
  using PackedCounter = uint32_t;
  int32_t wi_id;
  int32_t wg_id;
  enum {
    SLICE_SIZE = GROUP_SIZE * ELEMS_PER_ITEM,
    PACKING_RATIO = sizeof(PackedCounter) / sizeof(DigitCounter),     // 4
    LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE,                   // 2
    LOG_COUNTER_LANES = std::max(0, RADIX_BITS - LOG_PACKING_RATIO),  // 2
    COUNTER_LANES = 1 << LOG_COUNTER_LANES,                           // 4

    SUBGROUP_NUM = (GROUP_SIZE + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE,
    LANES_PER_SUBGROUP =
        std::max(1, (COUNTER_LANES + SUBGROUP_NUM - 1) / SUBGROUP_NUM),

    RADIX_DIGITS = 1 << RADIX_BITS,
  };

 private:
  const KeyType* keys_in;
  int32_t* counts;
  const int32_t current_bit;
  sycl::nd_item<1> item;

  union LocalStorage {
    DigitCounter digit_counters[COUNTER_LANES][GROUP_SIZE]
                               [PACKING_RATIO];                // [4][512][4]
    PackedCounter packed_counters[COUNTER_LANES][GROUP_SIZE];  // [4][512]
    int32_t group_counters[SUBGROUP_SIZE][RADIX_DIGITS];       // [32][16]
  };

  int32_t local_counters[LANES_PER_SUBGROUP][PACKING_RATIO];  // [1][4]
  LocalStorage& local_storage;

  inline DigitCounter ExtractDigit(const KeyTraitsT key) const {
    return ((key >> current_bit) & ((1 << RADIX_BITS) - 1));
  }

 public:
  inline DeviceRadixSortUpsweep(const KeyType* keys_in, int32_t* counts,
                                const int32_t current_bit,
                                sycl::nd_item<1> item,
                                sycl::local_accessor<unsigned char, 1> slm)
      : keys_in(keys_in),
        counts(counts),
        current_bit(current_bit),
        item(item),
        local_storage(
            reinterpret_cast<LocalStorage&>(*ITEXGetLocalAccPointer(slm))) {
    wi_id = item.get_local_id(0);
    wg_id = item.get_group(0);
  }

  static inline int32_t GetSharedLocalStorageSize() {
    return COUNTER_LANES * GROUP_SIZE * sizeof(PackedCounter);
  }

  inline void ProcessFullSlice(int32_t wg_offset) {
    KeyTraitsT keys[ELEMS_PER_ITEM];
    const KeyTraitsT* block_ptr =
        reinterpret_cast<const KeyTraitsT*>(keys_in + wg_offset);
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ELEM++) {
      keys[ELEM] = KeyTraits::Convert(block_ptr[wi_id + ELEM * GROUP_SIZE],
                                      Int2Type<IS_DESCENDING>());
    }
    item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ELEM++) {
      DigitCounter digit = ExtractDigit(keys[ELEM]);
      auto sub_counter = digit & (PACKING_RATIO - 1);
      auto row_offset = digit >> LOG_PACKING_RATIO;
      local_storage.digit_counters[row_offset][wi_id][sub_counter]++;
    }
  }

  inline void ProcessPartialSlice(int32_t wg_offset, int32_t wg_end) {
    // Process partial slice if necessary using single loads
    wg_offset += wi_id;
    while (wg_offset < wg_end) {
      // Load and bucket key
      const KeyTraitsT key = KeyTraits::Convert(
          reinterpret_cast<const KeyTraitsT*>(keys_in)[wg_offset],
          Int2Type<IS_DESCENDING>());
      DigitCounter digit = ExtractDigit(key);
      auto sub_counter = digit & (PACKING_RATIO - 1);
      auto row_offset = digit >> LOG_PACKING_RATIO;
      local_storage.digit_counters[row_offset][wi_id][sub_counter]++;
      wg_offset += GROUP_SIZE;
    }
  }

  inline void ExtractCounts() {
    int32_t wg_number = item.get_group_range(0);
    int32_t subgroup_id = wi_id / SUBGROUP_SIZE;
    int32_t subgroup_local_id = wi_id % SUBGROUP_SIZE;
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_SUBGROUP; LANE++) {
      int32_t counter_lane = (LANE * SUBGROUP_NUM) + subgroup_id;
      // Place unpacked digit counters in shared memory
      if (counter_lane < COUNTER_LANES) {
        int32_t digit_row = counter_lane << LOG_PACKING_RATIO;
#pragma unroll
        for (int32_t UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
             UNPACKED_COUNTER++) {
          int32_t bin_idx = digit_row + UNPACKED_COUNTER;
          local_storage.group_counters[subgroup_local_id][bin_idx] =
              local_counters[LANE][UNPACKED_COUNTER];
        }
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    if ((RADIX_DIGITS % GROUP_SIZE != 0) && (wi_id < RADIX_DIGITS)) {
      int32_t bin_idx = wi_id;
      int32_t bin_count = 0;
#pragma unroll
      for (int32_t i = 0; i < SUBGROUP_SIZE; ++i)
        bin_count += local_storage.group_counters[i][bin_idx];

      // counts[RADIX_DIGITS][WORKGROUP_NUM]
      counts[(wg_number * bin_idx) + wg_id] = bin_count;
    }
  }

  inline void ResetDigitCounters() {
#pragma unroll
    for (int LANE = 0; LANE < COUNTER_LANES; LANE++)
      local_storage.packed_counters[LANE][wi_id] = 0;
  }

  inline void ResetUnpackedCounters() {
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_SUBGROUP; LANE++) {
#pragma unroll
      for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
           UNPACKED_COUNTER++) {
        local_counters[LANE][UNPACKED_COUNTER] = 0;
      }
    }
  }

  inline void UnpackDigitCounts() {
    int32_t subgroup_id = wi_id / SUBGROUP_SIZE;
    int32_t subgroup_local_id = wi_id % SUBGROUP_SIZE;
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_SUBGROUP; LANE++) {
      const int32_t counter_lane = (LANE * SUBGROUP_NUM) + subgroup_id;
      if (counter_lane < COUNTER_LANES) {
#pragma unroll
        for (int32_t PACKED_COUNTER = 0; PACKED_COUNTER < GROUP_SIZE;
             PACKED_COUNTER += SUBGROUP_SIZE) {
#pragma unroll
          for (int32_t UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
               UNPACKED_COUNTER++) {
            int32_t counter =
                local_storage.digit_counters[counter_lane]
                                            [subgroup_local_id + PACKED_COUNTER]
                                            [UNPACKED_COUNTER];
            local_counters[LANE][UNPACKED_COUNTER] += counter;
          }
        }
      }
    }
  }
};

template <typename KeyType, typename ValueType, int32_t SUBGROUP_SIZE,
          int32_t GROUP_SIZE, bool IS_DESCENDING, int32_t RADIX_BITS,
          int32_t ELEMS_PER_ITEM>
class DeviceRadixSortUpsweepKernel;

template <typename KeyType, typename ValueType, int32_t SUBGROUP_SIZE,
          int32_t GROUP_SIZE, bool IS_DESCENDING, int32_t RADIX_BITS,
          int32_t ELEMS_PER_ITEM>
void DeviceRadixSortUpsweepProcess(sycl::queue* stream, const KeyType* keys_in,
                                   const int32_t sort_size, int32_t* counts,
                                   const int32_t current_bit) {
  using DeviceRadixSortUpsweep_t =
      DeviceRadixSortUpsweep<KeyType, ValueType, SUBGROUP_SIZE, GROUP_SIZE,
                             IS_DESCENDING, RADIX_BITS, ELEMS_PER_ITEM>;

  sycl::device dev = stream->get_device();
  const int32_t workgroup_size = GROUP_SIZE;
  const int32_t slice_size = DeviceRadixSortUpsweep_t::SLICE_SIZE;
  const int32_t physical_work_items =
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
  const int32_t max_workgroup_num = physical_work_items / workgroup_size;
  const int32_t total_slices = (sort_size + slice_size - 1) / slice_size;
  const int32_t workgroup_num = std::min(total_slices, max_workgroup_num);
  const int32_t avg_slices_per_wg = total_slices / workgroup_num;

  // for first several workgroup, they do one more slice.
  const int32_t big_shares = total_slices - (avg_slices_per_wg * workgroup_num);
  const int32_t normal_share_elems = avg_slices_per_wg * slice_size;
  const int32_t normal_base_offset = big_shares * slice_size;
  const int32_t big_share_elems = normal_share_elems + slice_size;

  stream->submit([&](sycl::handler& cgh) {
    auto slm = sycl::local_accessor<unsigned char, 1>(
        DeviceRadixSortUpsweep_t::GetSharedLocalStorageSize(), cgh);
    auto kernel_func = [=](sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
      auto Upsweep =
          DeviceRadixSortUpsweep_t(keys_in, counts, current_bit, item, slm);
      int32_t wg_offset, wg_end;
      if (Upsweep.wg_id < big_shares) {
        // for first several wg, they do one more slice.
        wg_offset = Upsweep.wg_id * big_share_elems;
        wg_end = wg_offset + big_share_elems;
      } else {
        wg_offset = normal_base_offset + Upsweep.wg_id * normal_share_elems;
        wg_end = std::min(sort_size, wg_offset + normal_share_elems);
      }

      // Take RADIX_BITS = 4 as an example, each workitem needs 16 uint32_t
      // RADIX_DIGITS bins named DigitCounters in SLM.
      // To save memory, use 16 u_char (4 uint32_t) as bins, named
      // PackedCounters.

      // ResetDigitCounters
      Upsweep.ResetDigitCounters();
      // ResetUnpackedCounters
      Upsweep.ResetUnpackedCounters();

      // However, u_char is easier to overflow. Hence unroll batches
      // of full slices and perform unpack timely.
      int UNROLL_COUNT = 255 / ELEMS_PER_ITEM;
      int UNROLLED_ELEMENTS = UNROLL_COUNT * slice_size;

      while (wg_offset + UNROLLED_ELEMENTS <= wg_end) {
        for (int i = 0; i < UNROLL_COUNT; ++i) {
          Upsweep.ProcessFullSlice(wg_offset);
          wg_offset += slice_size;
        }

        item.barrier(sycl::access::fence_space::local_space);

        Upsweep.UnpackDigitCounts();

        item.barrier(sycl::access::fence_space::local_space);

        Upsweep.ResetDigitCounters();
      }

      while (wg_offset + slice_size <= wg_end) {
        Upsweep.ProcessFullSlice(wg_offset);
        wg_offset += slice_size;
      }

      Upsweep.ProcessPartialSlice(wg_offset, wg_end);
      item.barrier(sycl::access::fence_space::local_space);
      Upsweep.UnpackDigitCounts();
      item.barrier(sycl::access::fence_space::local_space);
      Upsweep.ExtractCounts();
    };

    cgh.parallel_for<DeviceRadixSortUpsweepKernel<
        KeyType, ValueType, SUBGROUP_SIZE, GROUP_SIZE, IS_DESCENDING,
        RADIX_BITS, ELEMS_PER_ITEM>>(
        sycl::nd_range<1>(sycl::range<1>(workgroup_num * workgroup_size),
                          sycl::range<1>(workgroup_size)),
        kernel_func);
  });
}

///////////////// DeviceRadixSortScanBins /////////////////
template <int32_t SUBGROUP_SIZE, int32_t GROUP_SIZE, int32_t ELEMS_PER_ITEM>
inline void ConsumeSlice(
    int32_t* counts, const int32_t offset,
    int32_t* slice_exclusive_prefix,  // exclusive prefix sum of consumed slices
    sycl::nd_item<1> item, int32_t* slm) {
  // 1. load
  int32_t local_partial[ELEMS_PER_ITEM];
  auto wi_id = item.get_local_id(0);
  auto slice_begin = counts + offset;
#pragma unroll
  for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ELEM++) {
    local_partial[ELEM] = slice_begin[(wi_id * ELEMS_PER_ITEM) + ELEM];
  }
  item.barrier(sycl::access::fence_space::local_space);

  // 2. reduce ELEMS_PER_ITEM elements
  int32_t aggregate = local_partial[0];
#pragma unroll
  for (int i = 1; i < ELEMS_PER_ITEM; ++i) {
    aggregate = aggregate + local_partial[i];
  }

  // 3. group scan get group_exclusive_sum
  auto g = item.get_group();
  auto sg = item.get_sub_group();
  GroupScan<int32_t, GROUP_SIZE, SUBGROUP_SIZE, sycl::group<1>> group_scaner(
      g, sg, wi_id, reinterpret_cast<uint8_t*>(slm));
  int32_t group_exclusive_sum, group_all_sum;
  group_scaner.ExclusiveSum(aggregate, &group_exclusive_sum, &group_all_sum);

  // 4. compute elements' global exclusive_prefix_sum.
  group_exclusive_sum += *slice_exclusive_prefix;
  if (wi_id == 0) *slice_exclusive_prefix += group_all_sum;

  int32_t inclusive = group_exclusive_sum + local_partial[0];
  local_partial[0] = group_exclusive_sum;
  int32_t exclusive = inclusive;
#pragma unroll
  for (int i = 1; i < ELEMS_PER_ITEM; ++i) {
    inclusive = exclusive + local_partial[i];
    local_partial[i] = exclusive;
    exclusive = inclusive;
  }

  // 5. store back to global memory
#pragma unroll
  for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ELEM++) {
    slice_begin[(wi_id * ELEMS_PER_ITEM) + ELEM] = local_partial[ELEM];
  }
}

template <int32_t SUBGROUP_SIZE, int32_t GROUP_SIZE, int32_t ELEMS_PER_ITEM>
inline void ConsumePartialSlice(
    int32_t* counts, const int32_t offset, const int32_t slice_size,
    int32_t* slice_exclusive_prefix,  // exclusive prefix sum of consumed slices
    sycl::nd_item<1> item, int32_t* slm) {
  // 1. load
  int32_t local_partial[ELEMS_PER_ITEM];
  auto wi_id = item.get_local_id(0);
  auto slice_begin = counts + offset;
#pragma unroll
  for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ELEM++) {
    if ((wi_id * ELEMS_PER_ITEM) + ELEM < slice_size) {
      local_partial[ELEM] = slice_begin[(wi_id * ELEMS_PER_ITEM) + ELEM];
    } else {
      local_partial[ELEM] = 0;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  // 2. reduce ELEMS_PER_ITEM elements
  int32_t aggregate = local_partial[0];
#pragma unroll
  for (int i = 1; i < ELEMS_PER_ITEM; ++i) {
    aggregate = aggregate + local_partial[i];
  }

  // 3. group scan get group_exclusive_sum
  auto g = item.get_group();
  auto sg = item.get_sub_group();
  GroupScan<int32_t, GROUP_SIZE, SUBGROUP_SIZE, sycl::group<1>> group_scaner(
      g, sg, wi_id, reinterpret_cast<uint8_t*>(slm));
  int32_t group_exclusive_sum, group_all_sum;
  group_scaner.ExclusiveSum(aggregate, &group_exclusive_sum, &group_all_sum);

  // 4. compute elements' global exclusive_prefix_sum.
  group_exclusive_sum += *slice_exclusive_prefix;
  if (wi_id == 0) *slice_exclusive_prefix += group_all_sum;

  int32_t inclusive = group_exclusive_sum + local_partial[0];
  local_partial[0] = group_exclusive_sum;
  int32_t exclusive = inclusive;
#pragma unroll
  for (int i = 1; i < ELEMS_PER_ITEM; ++i) {
    inclusive = exclusive + local_partial[i];
    local_partial[i] = exclusive;
    exclusive = inclusive;
  }

  // 5. store back to global memory
#pragma unroll
  for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ELEM++) {
    if (wi_id * ELEMS_PER_ITEM + ELEM < slice_size) {
      slice_begin[(wi_id * ELEMS_PER_ITEM) + ELEM] = local_partial[ELEM];
    }
  }
}

template <int32_t SUBGROUP_SIZE, int32_t GROUP_SIZE, int32_t ELEMS_PER_ITEM>
class DeviceRadixSortScanBinsKernel;

template <int32_t SUBGROUP_SIZE, int32_t GROUP_SIZE, int32_t ELEMS_PER_ITEM = 4>
void DeviceRadixSortScanBins(sycl::queue* stream, int32_t* counts,
                             const int count_size) {
  stream->submit([&](sycl::handler& cgh) {
    auto slm =
        sycl::local_accessor<int32_t, 1>(GROUP_SIZE / SUBGROUP_SIZE, cgh);
    auto kernel_func = [=](sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
      int offset = 0;
      int32_t slice_exclusive_prefix = 0;
      const int32_t SLICE_SIZE = ELEMS_PER_ITEM * GROUP_SIZE;
      while (offset + SLICE_SIZE <= count_size) {
        ConsumeSlice<SUBGROUP_SIZE, GROUP_SIZE, ELEMS_PER_ITEM>(
            counts, offset, &slice_exclusive_prefix, item,
            ITEXGetLocalAccPointer(slm));
        offset += SLICE_SIZE;
      }

      if (offset < count_size) {
        ConsumePartialSlice<SUBGROUP_SIZE, GROUP_SIZE, ELEMS_PER_ITEM>(
            counts, offset, count_size - offset, &slice_exclusive_prefix, item,
            ITEXGetLocalAccPointer(slm));
      }
    };

    // only use one workgroup because count_size is small for scan.
    cgh.parallel_for<DeviceRadixSortScanBinsKernel<SUBGROUP_SIZE, GROUP_SIZE,
                                                   ELEMS_PER_ITEM>>(
        sycl::nd_range<1>(sycl::range<1>(GROUP_SIZE),
                          sycl::range<1>(GROUP_SIZE)),
        kernel_func);
  });
}

///////////////// DeviceRadixSortDownsweep /////////////////
template <typename KeyType, typename ValueType, int32_t SUBGROUP_SIZE,
          int32_t GROUP_SIZE, bool IS_DESCENDING = false,
          int32_t RADIX_BITS = 4, int32_t ELEMS_PER_ITEM = 4>
class DeviceRadixSortDownsweep {
 public:
  using KeyTraits = NumericTraits<KeyType>;
  using KeyTraitsT = typename KeyTraits::UnsignedT;
  using DigitCounter = uint16_t;
  using PackedCounter = uint32_t;
  int wi_id;
  int wg_id;
  enum {
    SLICE_SIZE = ELEMS_PER_ITEM * GROUP_SIZE,
    PACKING_RATIO = sizeof(PackedCounter) / sizeof(DigitCounter),     // 2
    LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE,                   // 1
    LOG_COUNTER_LANES = std::max(0, RADIX_BITS - LOG_PACKING_RATIO),  // 3
    COUNTER_LANES = 1 << LOG_COUNTER_LANES,                           // 8

    RADIX_DIGITS = 1 << RADIX_BITS,
    KEY_TRAITS_TYPE_MASK = 1l << ((sizeof(KeyTraitsT) << 3) - 1),
  };

 private:
  const KeyType* keys_in;
  KeyType* keys_out;
  const ValueType* values_in;
  ValueType* values_out;
  int32_t* counts;
  const int32_t sort_size;
  const int32_t current_bit;
  sycl::nd_item<1> item;
  int32_t bin_offset;

  union RankT {
    DigitCounter digit_counters[COUNTER_LANES][GROUP_SIZE]
                               [PACKING_RATIO];                // [8][512][2]
    PackedCounter packed_counters[COUNTER_LANES][GROUP_SIZE];  // [8][512]
    PackedCounter scan_flat[COUNTER_LANES * GROUP_SIZE];
  };

  union LocalStorage {
    RankT rank_storage;
    struct {
      KeyTraitsT exchange_keys[SLICE_SIZE];
      int32_t relative_bin_offsets[RADIX_DIGITS];
    };
    ValueType exchange_values[SLICE_SIZE];
    int32_t exclusive_digit_prefix[RADIX_DIGITS];
  };

  LocalStorage& local_storage;

  inline void LoadKeys(KeyTraitsT (&keys)[ELEMS_PER_ITEM],
                       const int32_t wg_offset, const int32_t valid_items) {
    KeyTraitsT PADDING_KEY;
    PADDING_KEY = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
    PADDING_KEY = PADDING_KEY ^ (PADDING_KEY - 1);

    auto slice_begin = keys_in + wg_offset;
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ELEM++) {
      int offset = (wi_id * ELEMS_PER_ITEM) + ELEM;
      keys[ELEM] =
          (offset < valid_items)
              ? KeyTraits::Convert(
                    reinterpret_cast<const KeyTraitsT*>(slice_begin)[offset],
                    Int2Type<IS_DESCENDING>())
              : PADDING_KEY;
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  inline void LoadValues(ValueType (&values)[ELEMS_PER_ITEM], int32_t wg_offset,
                         const int32_t valid_items) {
    auto slice_begin = values_in + wg_offset;
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ELEM++) {
      int offset = (wi_id * ELEMS_PER_ITEM) + ELEM;
      if (offset < valid_items) values[ELEM] = slice_begin[offset];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  inline DigitCounter ExtractDigit(const KeyTraitsT key) {
    return ((key >> current_bit) & ((1 << RADIX_BITS) - 1));
  }

  inline PackedCounter GroupExclusiveSum(
      PackedCounter (&scan_flat)[COUNTER_LANES * GROUP_SIZE]) {
    // Get exclusive sum in this workitem
    PackedCounter local_exclusive_sum[COUNTER_LANES];
    PackedCounter aggregate = 0;
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      local_exclusive_sum[lane] = aggregate;
      aggregate += scan_flat[wi_id * COUNTER_LANES + lane];
    }

    // GroupScan get group_exclusive_sum
    auto g = item.get_group();
    auto sg = item.get_sub_group();
    GroupScan<PackedCounter, GROUP_SIZE, SUBGROUP_SIZE, sycl::group<1>>
        group_scaner(g, sg, wi_id, reinterpret_cast<uint8_t*>(&scan_flat));
    PackedCounter group_exclusive_sum, group_all_sum;
    group_scaner.ExclusiveSum(aggregate, &group_exclusive_sum, &group_all_sum);

    // Summary exclusive_sum in this workgroup
#pragma unroll
    for (int line = 0; line < COUNTER_LANES; ++line) {
      scan_flat[wi_id * COUNTER_LANES + line] =
          group_exclusive_sum + local_exclusive_sum[line];
    }
    item.barrier(sycl::access::fence_space::local_space);

    return group_all_sum;
  }

  inline void RankKeys(KeyTraitsT (&key)[ELEMS_PER_ITEM],
                       int32_t (&rank)[ELEMS_PER_ITEM],
                       int32_t* exclusive_digit_prefix) {
    DigitCounter* digit_counters[ELEMS_PER_ITEM];

    // Reset counters
#pragma unroll
    for (int ELEM = 0; ELEM < COUNTER_LANES; ++ELEM)
      local_storage.rank_storage.packed_counters[ELEM][wi_id] = 0;
    item.barrier(sycl::access::fence_space::local_space);

    // Put elements to its digit_counters bin
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ++ELEM) {
      auto digit = ExtractDigit(key[ELEM]);
      auto sub_counter = digit >> LOG_COUNTER_LANES;
      auto counter_lane = digit & (COUNTER_LANES - 1);
      digit_counters[ELEM] =
          &local_storage.rank_storage
               .digit_counters[counter_lane][wi_id][sub_counter];
      rank[ELEM] = *digit_counters[ELEM];
      *digit_counters[ELEM] = rank[ELEM] + 1;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Exclusive scan on packed_counters
    PackedCounter group_all_sum =
        GroupExclusiveSum(local_storage.rank_storage.scan_flat);

    // Decode packed data: add prefix sum of lower sub_counters to higher
    // sub_counters'
    PackedCounter c = 0;
#pragma unroll
    for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
      c += group_all_sum << (sizeof(DigitCounter) * 8 * STEP);
    }
#pragma unroll
    for (int LINE = 0; LINE < COUNTER_LANES; LINE++) {
      local_storage.rank_storage.packed_counters[LINE][wi_id] += c;
    }
    item.barrier(sycl::access::fence_space::local_space);
    // Group scan on digit_counters done

    // Finish ranking
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ++ELEM) {
      rank[ELEM] += *digit_counters[ELEM];
    }
    item.barrier(sycl::access::fence_space::local_space);

    int32_t bin_idx = wi_id;
    if ((GROUP_SIZE == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS)) {
      uint32_t counter_lane = (bin_idx & (COUNTER_LANES - 1));
      uint32_t sub_counter = bin_idx >> (LOG_COUNTER_LANES);
      *exclusive_digit_prefix =
          local_storage.rank_storage
              .digit_counters[counter_lane][0][sub_counter];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  template <bool FULL_SLICE>
  void ScatterKeys(KeyTraitsT (&twiddled_keys)[ELEMS_PER_ITEM],
                   int32_t (&relative_bin_offsets)[ELEMS_PER_ITEM],
                   int32_t (&ranks)[ELEMS_PER_ITEM], int32_t valid_items) {
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ++ELEM) {
      local_storage.exchange_keys[ranks[ELEM]] = twiddled_keys[ELEM];
    }
    item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ++ELEM) {
      KeyTraitsT key = local_storage.exchange_keys[wi_id + (ELEM * GROUP_SIZE)];
      auto digit = ExtractDigit(key);
      relative_bin_offsets[ELEM] = local_storage.relative_bin_offsets[digit];

      if (FULL_SLICE ||
          (static_cast<int32_t>(wi_id + (ELEM * GROUP_SIZE)) < valid_items)) {
        reinterpret_cast<KeyTraitsT*>(keys_out)[relative_bin_offsets[ELEM] +
                                                wi_id + (ELEM * GROUP_SIZE)] =
            KeyTraits::ConvertBack(key, Int2Type<IS_DESCENDING>());
      }
    }
  }

  template <bool FULL_SLICE>
  void GatherScatterValues(int32_t (&relative_bin_offsets)[ELEMS_PER_ITEM],
                           int32_t (&ranks)[ELEMS_PER_ITEM], int32_t wg_offset,
                           int32_t valid_items) {
    ValueType values[ELEMS_PER_ITEM];
    LoadValues(values, wg_offset, valid_items);

#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ++ELEM) {
      local_storage.exchange_values[ranks[ELEM]] = values[ELEM];
    }
    item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int ELEM = 0; ELEM < ELEMS_PER_ITEM; ++ELEM) {
      ValueType value =
          local_storage.exchange_values[wi_id + (ELEM * GROUP_SIZE)];

      if (FULL_SLICE ||
          (static_cast<int32_t>(wi_id + (ELEM * GROUP_SIZE)) < valid_items)) {
        values_out[relative_bin_offsets[ELEM] + wi_id + (ELEM * GROUP_SIZE)] =
            value;
      }
    }
  }

  template <bool FULL_SLICE>
  inline void ProcessSlice(int32_t wg_offset,
                           const int32_t valid_items = SLICE_SIZE) {
    KeyTraitsT keys[ELEMS_PER_ITEM];
    int32_t ranks[ELEMS_PER_ITEM];
    int32_t relative_bin_offsets[ELEMS_PER_ITEM];
    LoadKeys(keys, wg_offset, valid_items);

    int32_t exclusive_digit_prefix;
    RankKeys(keys, ranks, &exclusive_digit_prefix);

    // Copy exclusive_digit_prefix to SLM
    if ((GROUP_SIZE == RADIX_DIGITS) || (wi_id < RADIX_DIGITS)) {
      local_storage.exclusive_digit_prefix[wi_id] = exclusive_digit_prefix;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Get inclusive digit prefix
    int32_t inclusive_digit_prefix;
    if ((GROUP_SIZE == RADIX_DIGITS) || (wi_id < RADIX_DIGITS)) {
      // Get inclusive digit prefix from exclusive prefix (lower bins come
      // first)
      inclusive_digit_prefix =
          (wi_id == RADIX_DIGITS - 1)
              ? (GROUP_SIZE * ELEMS_PER_ITEM)
              : local_storage.exclusive_digit_prefix[wi_id + 1];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if ((GROUP_SIZE == RADIX_DIGITS) || (wi_id < RADIX_DIGITS)) {
      bin_offset -= exclusive_digit_prefix;
      local_storage.relative_bin_offsets[wi_id] = bin_offset;
      bin_offset += inclusive_digit_prefix;
    }

    ScatterKeys<FULL_SLICE>(keys, relative_bin_offsets, ranks, valid_items);
    GatherScatterValues<FULL_SLICE>(relative_bin_offsets, ranks, wg_offset,
                                    valid_items);
  }

 public:
  inline DeviceRadixSortDownsweep(const KeyType* keys_in, KeyType* keys_out,
                                  const ValueType* values_in,
                                  ValueType* values_out, int32_t* counts,
                                  const int32_t sort_size,
                                  const int32_t current_bit,
                                  sycl::nd_item<1> item,
                                  sycl::local_accessor<unsigned char, 1> slm)
      : keys_in(keys_in),
        keys_out(keys_out),
        values_in(values_in),
        values_out(values_out),
        counts(counts),
        sort_size(sort_size),
        current_bit(current_bit),
        item(item),
        local_storage(
            reinterpret_cast<LocalStorage&>(*ITEXGetLocalAccPointer(slm))) {
    wi_id = item.get_local_id(0);
    wg_id = item.get_group(0);
    int32_t bin_idx = wi_id;
    if ((GROUP_SIZE == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS)) {
      bin_offset = counts[(item.get_group_range(0) * bin_idx) + wg_id];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  static inline int32_t GetSharedLocalStorageSize() {
    return sizeof(LocalStorage);
  }

  inline void ProcessRegion(int32_t wg_offset, int32_t wg_end) {
#pragma unroll
    while (wg_offset + SLICE_SIZE <= wg_end) {
      ProcessSlice<true>(wg_offset);
      wg_offset += SLICE_SIZE;
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (wg_offset < wg_end) {
      ProcessSlice<false>(wg_offset, wg_end - wg_offset);
    }
  }
};

template <typename KeyType, typename ValueType, int32_t SUBGROUP_SIZE,
          int32_t GROUP_SIZE, bool IS_DESCENDING, int32_t RADIX_BITS,
          int32_t ELEMS_PER_ITEM>
class DeviceRadixSortDownsweepKernel;

template <typename KeyType, typename ValueType, int32_t SUBGROUP_SIZE,
          int32_t GROUP_SIZE, bool IS_DESCENDING, int32_t RADIX_BITS,
          int32_t ELEMS_PER_ITEM>
void DeviceRadixSortDownsweepProcess(sycl::queue* stream,
                                     const KeyType* keys_in,
                                     const ValueType* values_in,
                                     KeyType* keys_out, ValueType* values_out,
                                     const int sort_size, int32_t* counts,
                                     const int32_t current_bit) {
  using DeviceRadixSortDownsweep_t =
      DeviceRadixSortDownsweep<KeyType, ValueType, SUBGROUP_SIZE, GROUP_SIZE,
                               IS_DESCENDING, RADIX_BITS, ELEMS_PER_ITEM>;

  sycl::device dev = stream->get_device();
  const int32_t workgroup_size = GROUP_SIZE;
  const int32_t slice_size = DeviceRadixSortDownsweep_t::SLICE_SIZE;
  const int32_t physical_work_items =
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
  const int32_t max_workgroup_num = physical_work_items / workgroup_size;
  const int32_t total_slices = (sort_size + slice_size - 1) / slice_size;
  const int32_t workgroup_num = std::min(total_slices, max_workgroup_num);
  const int32_t avg_slices_per_wg = total_slices / workgroup_num;

  // for first several workgroup, they do one more slice.
  const int32_t big_shares = total_slices - (avg_slices_per_wg * workgroup_num);
  const int32_t normal_share_elems = avg_slices_per_wg * slice_size;
  const int32_t normal_base_offset = big_shares * slice_size;
  const int32_t big_share_elems = normal_share_elems + slice_size;

  stream->submit([&](sycl::handler& cgh) {
    auto slm = sycl::local_accessor<unsigned char, 1>(
        DeviceRadixSortDownsweep_t::GetSharedLocalStorageSize(), cgh);
    auto kernel_func = [=](sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
      auto Downsweep =
          DeviceRadixSortDownsweep_t(keys_in, keys_out, values_in, values_out,
                                     counts, sort_size, current_bit, item, slm);

      int32_t wg_offset, wg_end;
      if (Downsweep.wg_id < big_shares) {
        // for first several workgroup, they do one more slice.
        wg_offset = Downsweep.wg_id * big_share_elems;
        wg_end = wg_offset + big_share_elems;
      } else {
        wg_offset = normal_base_offset + Downsweep.wg_id * normal_share_elems;
        wg_end = std::min(sort_size, wg_offset + normal_share_elems);
      }
      Downsweep.ProcessRegion(wg_offset, wg_end);
    };

    cgh.parallel_for<DeviceRadixSortDownsweepKernel<
        KeyType, ValueType, SUBGROUP_SIZE, GROUP_SIZE, IS_DESCENDING,
        RADIX_BITS, ELEMS_PER_ITEM>>(
        sycl::nd_range<1>(sycl::range<1>(workgroup_num * workgroup_size),
                          sycl::range<1>(workgroup_size)),
        kernel_func);
  });
}

template <typename KeyType, typename ValueType, int32_t SUBGROUP_SIZE,
          int32_t GROUP_SIZE, bool IS_DESCENDING, int32_t RADIX_BITS,
          int32_t ELEMS_PER_ITEM = 4>
void DeviceRadixSortIterationImpl(
    sycl::queue* stream, const KeyType* keys_in, const ValueType* values_in,
    KeyType* keys_out, ValueType* values_out, const int32_t sort_size,
    int32_t* counts,  // counts[RADIX_DIGITS][WORKGROUP_NUM]
    const int32_t count_size, const int32_t current_bit) {
  // In each iteration, RadixSort would sort keys/values using
  // keys' RADIX_BITS, from current_bit to current_bit + RADIX_BITS,
  // to examine a key belong to which RADIX_DIGITS and put it into
  // corresponding RADIX_DIGITS bin.
  // In a DeviceRadixSort iteration, each workgroup takes a contiguous
  // keys/values segment, and run 3 kernels:

  // 1. Each workgroup summaries the number of each RADIX_DIGITS bin from
  // its segment locally according to current_bit/RADIX_BITS. After this
  // kernel, we know workgroup with wg_id have counts[radix_digit][wg_id]
  // keys in radix_digit bin regarding to current_bit/RADIX_BITS.
  DeviceRadixSortUpsweepProcess<KeyType, ValueType, SUBGROUP_SIZE, GROUP_SIZE,
                                IS_DESCENDING, RADIX_BITS, ELEMS_PER_ITEM>(
      stream, keys_in, sort_size, counts, current_bit);

  // 2. Use a workgroup to perform ExclusiveSumScan over counts.
  // After this kernel, we know elements with radix_digit in workgroup with
  // wg_id should use counts[radix_digit][wg_id] as global offset to store
  // keys_out/values_out.
  DeviceRadixSortScanBins<SUBGROUP_SIZE, GROUP_SIZE, ELEMS_PER_ITEM>(
      stream, counts, count_size);

  // 3. Each workitem summaries the number of each RADIX_DIGITS bin from
  // its segment locally according to current_bit/RADIX_BITS and then do
  // ExclusiveSumScan in its workgroup to get workgroup ExclusiveSum.
  // Then store elements into keys_out/values_out by using
  // counts[radix_digit][wg_id] and workitems' ExclusiveSum.
  DeviceRadixSortDownsweepProcess<KeyType, ValueType, SUBGROUP_SIZE, GROUP_SIZE,
                                  IS_DESCENDING, RADIX_BITS, ELEMS_PER_ITEM>(
      stream, keys_in, values_in, keys_out, values_out, sort_size, counts,
      current_bit);
}

template <typename KeyType, typename ValueType, bool IS_DESCENDING,
          int32_t RADIX_BITS>
void DeviceRadixSortIteration(sycl::queue* stream, const KeyType* keys_in,
                              const ValueType* values_in, KeyType* keys_out,
                              ValueType* values_out, const int32_t sort_size,
                              int32_t* counts, const int32_t count_size,
                              const int32_t current_bit) {
  sycl::device device = stream->get_device();
  const int32_t max_workgroup_size =
      device.get_info<sycl::info::device::max_work_group_size>();
  const int32_t max_subgroup_size =
      device.get_info<sycl::info::device::sub_group_sizes>().back();

#define DISPATCH_RADIX_SORT_IMPI(SG_SIZE, WG_SIZE)                         \
  DeviceRadixSortIterationImpl<KeyType, ValueType, SG_SIZE, WG_SIZE,       \
                               IS_DESCENDING, RADIX_BITS>(                 \
      stream, keys_in, values_in, keys_out, values_out, sort_size, counts, \
      count_size, current_bit);

  if (max_workgroup_size < 512) {
    DISPATCH_RADIX_SORT_IMPI(16, 256)
  } else {
    switch (max_subgroup_size) {
      case 32:
        DISPATCH_RADIX_SORT_IMPI(32, 512)
        break;
      default:
        DISPATCH_RADIX_SORT_IMPI(16, 512)
    }
  }
}

template <typename KeyType, typename ValueType, bool IS_DESCENDING = false,
          int32_t RADIX_BITS = 4>
Status DispatchDeviceRadixSort(OpKernelContext* context, const KeyType* keys_in,
                               const ValueType* vals_in, KeyType* keys_out,
                               ValueType* vals_out, const int sort_size) {
  const int32_t radix_iters = (sizeof(KeyType) * 8) / RADIX_BITS;
  const int32_t radix_digits = 1 << RADIX_BITS;

  const GPUDevice& device = context->eigen_device<GPUDevice>();
  sycl::queue* stream = device.stream();
  sycl::device dev = stream->get_device();
  const int32_t workgroup_size =
      dev.get_info<sycl::info::device::max_work_group_size>() < 512 ? 256 : 512;
  const int32_t physical_work_items =
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
  const int32_t max_workgroup_num = physical_work_items / workgroup_size;
  const int64_t count_size = max_workgroup_num * radix_digits;

  Tensor count_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<int32_t>::value,
                                            TensorShape({count_size}),
                                            &count_tensor));
  int32_t* counts = count_tensor.flat<int32_t>().data();
  stream->memset(counts, 0, count_size * sizeof(int32_t));

  Tensor tmp_key_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<KeyType>::value,
                                            TensorShape({sort_size}),
                                            &tmp_key_tensor));
  KeyType* keys_tmp = tmp_key_tensor.flat<KeyType>().data();

  Tensor tmp_val_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<ValueType>::value,
                                            TensorShape({sort_size}),
                                            &tmp_val_tensor));
  ValueType* vals_tmp = tmp_val_tensor.flat<ValueType>().data();

  int32_t current_bit = 0;
  int32_t radix_iter = 0;
  DeviceRadixSortIteration<KeyType, ValueType, IS_DESCENDING, RADIX_BITS>(
      stream, keys_in, vals_in, keys_tmp, vals_tmp, sort_size, counts,
      count_size, current_bit);
  radix_iter += 1;

  current_bit = radix_iter * RADIX_BITS;
  DeviceRadixSortIteration<KeyType, ValueType, IS_DESCENDING, RADIX_BITS>(
      stream, keys_tmp, vals_tmp, keys_out, vals_out, sort_size, counts,
      count_size, current_bit);
  radix_iter += 1;

  for (; radix_iter < radix_iters; ++radix_iter) {
    current_bit = radix_iter * RADIX_BITS;
    if (radix_iter % 2 == 0) {
      DeviceRadixSortIteration<KeyType, ValueType, IS_DESCENDING, RADIX_BITS>(
          stream, keys_out, vals_out, keys_tmp, vals_tmp, sort_size, counts,
          count_size, current_bit);
    } else {
      DeviceRadixSortIteration<KeyType, ValueType, IS_DESCENDING, RADIX_BITS>(
          stream, keys_tmp, vals_tmp, keys_out, vals_out, sort_size, counts,
          count_size, current_bit);
    }
  }
  return Status::OK();
}

}  // namespace itex
#endif  // ITEX_CORE_UTILS_DEVICE_RADIX_SORT_H_
