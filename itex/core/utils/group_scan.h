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

#ifndef ITEX_CORE_UTILS_GROUP_SCAN_H_
#define ITEX_CORE_UTILS_GROUP_SCAN_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include "itex/core/utils/radix_utils.h"

namespace itex {

// ------------------------------------------------------------------
// GroupScan for sum
// ------------------------------------------------------------------

template <typename T, int GROUP_SIZE, int SUB_GROUP_SIZE, typename GroupT,
          typename SubGroupT = sycl::sub_group>
class GroupScan {
 private:
  // --------------------------------------------
  // Constants and type definitions
  // --------------------------------------------

  static_assert(GROUP_SIZE % SUB_GROUP_SIZE == 0,
                "Work-group size must be a multiple of the sub-group size");

  enum {
    SUB_GROUP_RANGE = GROUP_SIZE / SUB_GROUP_SIZE,
    STEPS = Log2<SUB_GROUP_SIZE>::VALUE,
  };

  // Local memory layout type
  struct _LocalStorage {
    T sg_storage[SUB_GROUP_RANGE];
    T group_storage;
  };

  // --------------------------------------------
  // Work item fields
  // --------------------------------------------

  const GroupT& g;
  const SubGroupT& sg;
  const int local_id;
  const int sg_id;
  const int sg_local_id;
  _LocalStorage& local_mem;

  // --------------------------------------------
  // Utility methods
  // --------------------------------------------

  // Compute inclusive and exclusive sum in each sub_group
  void SubGroupScan(T input, T* inclusive_sum, T* exclusive_sum) {
    *inclusive_sum = input;

#pragma unroll
    for (int i = 0; i < STEPS; ++i) {
      uint32_t offset = 1u << i;
      T tmp = sycl::shift_group_right(sg, *inclusive_sum, offset);
      if (sg_local_id >= offset) (*inclusive_sum) += tmp;
    }

    *exclusive_sum = (*inclusive_sum) - input;
  }

  // Compute the sub_group wide prefix and group-wide sum
  T ComputeSgPrefix(T sg_sum, T init, T* group_sum) {
    // Last work item in each sub_group shares its sg_sum
    if (sg_local_id == SUB_GROUP_SIZE - 1) {
      local_mem.sg_storage[sg_id] = sg_sum;
    }
    sycl::group_barrier(g);

    T sg_prefix;
    *group_sum = init;

#pragma unroll
    for (int i = 0; i < SUB_GROUP_RANGE; ++i) {
      if (sg_id == i) sg_prefix = *group_sum;
      *group_sum += local_mem.sg_storage[i];
    }
    sycl::group_barrier(g);

    return sg_prefix;
  }

 public:
  // Local memory storage type
  struct LocalStorage : BaseStorage<_LocalStorage> {};

  // Constructors
  GroupScan(const GroupT& g_, const SubGroupT& sg_, const int local_id_,
            LocalStorage* local_mem_)
      : g(g_),
        sg(sg_),
        local_id(local_id_),
        sg_id(sg_.get_group_linear_id()),
        sg_local_id(sg_.get_local_linear_id()),
        local_mem(local_mem_->Alias()) {}

  GroupScan(const GroupT& g_, const SubGroupT& sg_, const int local_id_,
            uint8_t* local_mem_)
      : g(g_),
        sg(sg_),
        local_id(local_id_),
        sg_id(sg_.get_group_linear_id()),
        sg_local_id(sg_.get_local_linear_id()),
        local_mem(reinterpret_cast<_LocalStorage&>(*local_mem_)) {}

  // Exclusive Sum
  void ExclusiveSum(T input, T* output) {
    T group_sum;
    ExclusiveSum(input, 0, output, &group_sum);
  }

  // Exclusive Sum
  void ExclusiveSum(T input, T* output, T* group_sum) {
    ExclusiveSum(input, 0, output, group_sum);
  }

  // Exclusive Sum
  void ExclusiveSum(T input, T init, T* output, T* group_sum) {
    T inclusive_sum;
    SubGroupScan(input, &inclusive_sum, output);

    // Compute the sub_group wide prefix and group-wide sum
    T sg_prefix = ComputeSgPrefix(inclusive_sum, init, group_sum);

    // Apply sub_group prefix to each work item
    *output += sg_prefix;
  }

  // ExclusiveSum with a callback operation
  template <typename CallBackOp>
  void ExclusiveSum(T input, T* output,
                    CallBackOp& callback_op) {  // NOLINT(runtime/references)
    T group_sum;
    ExclusiveSum(input, 0, output, &group_sum);

    if (sg_id == 0) {
      T group_prefix = callback_op(group_sum);
      if (sg_local_id == 0) {
        local_mem.group_storage = group_prefix;
        *output = group_prefix;
      }
    }
    sycl::group_barrier(g);

    T group_prefix = local_mem.group_storage;
    if (local_id > 0) *output += group_prefix;
  }
};
}  // namespace itex
#endif  // ITEX_CORE_UTILS_GROUP_SCAN_H_
