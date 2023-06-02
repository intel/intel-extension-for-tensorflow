/* Copyright (c) 2021-2023 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_GPU_UNIQUE_OP_H_
#define ITEX_CORE_KERNELS_GPU_UNIQUE_OP_H_
#include <iterator>
#include <limits>

#include "itex/core/kernels/gpu/topk_op.h"
#include "itex/core/kernels/gpu/unique_op_helpers.h"
#include "itex/core/utils/group_radix_sort.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/radix_utils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace impl {

template <typename T>
using __shared__ = sycl::accessor<T, 1, sycl::access::mode::read_write,
                                  sycl::access::target::local>;

inline int Log2Floor(uint32_t n) {
  if (n == 0) return -1;
  int log = 0;
  uint32_t value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  return log;
}

inline int Log2Ceiling(uint32_t n) {
  int floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

template <typename TIndex>
struct ExtractFirstOccurrenceIndicesKernel {
  ExtractFirstOccurrenceIndicesKernel(int64_t input_size, int64_t uniq_size,
                                      TIndex* sorted_input_inds,
                                      TIndex* sorted_input_unique_ids,
                                      TIndex* unique_input_inds,
                                      TIndex* segment_ends)
      : input_size(input_size),
        uniq_size(uniq_size),
        sorted_input_inds(sorted_input_inds),
        sorted_input_unique_ids(sorted_input_unique_ids),
        unique_input_inds(unique_input_inds),
        segment_ends(segment_ends) {}
  void operator()(sycl::nd_item<1> item) const {
    auto global_id = item.get_global_id(0);
    auto global_range = item.get_global_range(0);
    for (int32_t i = global_id, step = global_range; i < input_size;
         i += step) {
      TIndex sorted_input_unique_id = sorted_input_unique_ids[i];
      if (i == 0 || sorted_input_unique_id != sorted_input_unique_ids[i - 1]) {
        unique_input_inds[sorted_input_unique_id] = sorted_input_inds[i];
        if (segment_ends) {
          if (i == 0) {
            // First thread writes the last element.
            segment_ends[uniq_size - 1] = input_size;
          } else {
            segment_ends[sorted_input_unique_id - 1] = i;
          }
        }
      }
    }
  }

 private:
  int64_t input_size;
  int64_t uniq_size;
  TIndex* sorted_input_inds;
  TIndex* sorted_input_unique_ids;
  TIndex* unique_input_inds;
  TIndex* segment_ends;
};

template <typename TIndex>
Status LaunchExtractFirstOccurrenceIndicesKernel(
    sycl::queue* stream, int64_t input_size, int64_t uniq_size,
    TIndex* sorted_input_inds, TIndex* sorted_input_unique_ids,
    TIndex* unique_input_inds, TIndex* segment_ends,
    sycl::range<1> global_range, sycl::range<1> local_range) {
  stream->submit([&](sycl::handler& cgh) {
    ExtractFirstOccurrenceIndicesKernel<TIndex> task(
        input_size, uniq_size, sorted_input_inds, sorted_input_unique_ids,
        unique_input_inds, segment_ends);
    cgh.parallel_for<ExtractFirstOccurrenceIndicesKernel<TIndex>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

// Scatters the index of the first occurrence of each unique input value to
// unique_input_inds.
// If segment_ends is not nullptr, it is filled with the end index of each
// unique value's range in the sorted input (the last element is always set
// to input_size).
template <typename TIndex>
Status ExtractFirstOccurrenceIndices(sycl::queue* stream, int64_t input_size,
                                     int64_t uniq_size,
                                     TIndex* sorted_input_inds,
                                     TIndex* sorted_input_unique_ids,
                                     TIndex* unique_input_inds,
                                     TIndex* segment_ends) {
  assert(input_size > 0);  // Crash OK
  int workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  sycl::range<1> local_range(workgroup_size);
  const int num_wg = (input_size + workgroup_size - 1) / workgroup_size;
  sycl::range<1> global_range(num_wg * workgroup_size);

  return LaunchExtractFirstOccurrenceIndicesKernel<TIndex>(
      stream, input_size, uniq_size, sorted_input_inds, sorted_input_unique_ids,
      unique_input_inds, segment_ends, global_range, local_range);
}

template <typename T, typename TIndex>
struct GatherOutputsAndInvertPermutationKernel {
  GatherOutputsAndInvertPermutationKernel(int64_t uniq_size, const T* input,
                                          TIndex* sorted_unique_input_inds,
                                          TIndex* sorted_unique_perm,
                                          TIndex* segment_ends, T* output,
                                          TIndex* inv_sorted_unique_perm,
                                          TIndex* count)
      : uniq_size(uniq_size),
        input(input),
        sorted_unique_input_inds(sorted_unique_input_inds),
        sorted_unique_perm(sorted_unique_perm),
        segment_ends(segment_ends),
        output(output),
        inv_sorted_unique_perm(inv_sorted_unique_perm),
        count(count) {}
  void operator()(sycl::nd_item<1> item) const {
    auto global_id = item.get_global_id(0);
    auto global_range = item.get_global_range(0);
    if (global_id >= uniq_size) {
      return;
    }
    for (int32_t i = global_id, step = global_range; i < uniq_size; i += step) {
      output[i] = input[sorted_unique_input_inds[i]];
      auto j = sorted_unique_perm[i];
      inv_sorted_unique_perm[j] = i;
      if (count) {
        TIndex beg = j == 0 ? 0 : segment_ends[j - 1];
        TIndex end = segment_ends[j];
        count[i] = end - beg;
      }
    }
  }

 private:
  int64_t uniq_size;
  const T* input;
  TIndex* sorted_unique_input_inds;
  TIndex* sorted_unique_perm;
  TIndex* segment_ends;
  T* output;
  TIndex* inv_sorted_unique_perm;
  TIndex* count;
};

template <typename T, typename TIndex>
Status LaunchGatherOutputsAndInvertPermutationKernel(
    sycl::queue* stream, int64_t uniq_size, const T* input,
    TIndex* sorted_unique_input_inds, TIndex* sorted_unique_perm,
    TIndex* segment_ends, T* output, TIndex* inv_sorted_unique_perm,
    TIndex* count, sycl::range<1> global_range, sycl::range<1> local_range) {
  stream->submit([&](sycl::handler& cgh) {
    GatherOutputsAndInvertPermutationKernel<T, TIndex> task(
        uniq_size, input, sorted_unique_input_inds, sorted_unique_perm,
        segment_ends, output, inv_sorted_unique_perm, count);
    cgh.parallel_for<GatherOutputsAndInvertPermutationKernel<T, TIndex>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

// Gathers input values using sorted_unique_input_inds, and inverts the
// permutation specified by sorted_unique_perm.
template <typename T, typename TIndex>
Status GatherOutputsAndInvertPermutation(sycl::queue* stream, int64_t uniq_size,
                                         const T* input,
                                         TIndex* sorted_unique_input_inds,
                                         TIndex* sorted_unique_perm,
                                         TIndex* segment_ends, T* output,
                                         TIndex* inv_sorted_unique_perm,
                                         TIndex* count) {
  if (uniq_size == 0) return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  const int workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  sycl::range<1> local_range(workgroup_size);
  const int num_wg = (uniq_size + workgroup_size - 1) / workgroup_size;
  sycl::range<1> global_range(num_wg * workgroup_size);

  return LaunchGatherOutputsAndInvertPermutationKernel<T, TIndex>(
      stream, uniq_size, input, sorted_unique_input_inds, sorted_unique_perm,
      segment_ends, output, inv_sorted_unique_perm, count, global_range,
      local_range);
}

template <typename TIndex>
struct LookupAndScatterUniqueIdsKernel {
  LookupAndScatterUniqueIdsKernel(int64_t input_size, TIndex* sorted_input_inds,
                                  TIndex* sorted_input_unique_ids,
                                  TIndex* inv_sorted_unique_perm, TIndex* idx)
      : input_size(input_size),
        sorted_input_inds(sorted_input_inds),
        sorted_input_unique_ids(sorted_input_unique_ids),
        inv_sorted_unique_perm(inv_sorted_unique_perm),
        idx(idx) {}
  void operator()(sycl::nd_item<1> item) const {
    auto global_id = item.get_global_id(0);
    auto global_range = item.get_global_range(0);
    for (int32_t i = global_id, step = global_range; i < input_size;
         i += step) {
      idx[sorted_input_inds[i]] =
          inv_sorted_unique_perm[sorted_input_unique_ids[i]];
    }
  }

 private:
  int64_t input_size;
  TIndex* sorted_input_inds;
  TIndex* sorted_input_unique_ids;
  TIndex* inv_sorted_unique_perm;
  TIndex* idx;
};

template <typename TIndex>
Status LaunchLookupAndScatterUniqueIdsKernel(
    sycl::queue* stream, int64_t input_size, TIndex* sorted_input_inds,
    TIndex* sorted_input_unique_ids, TIndex* inv_sorted_unique_perm,
    TIndex* idx, sycl::range<1> global_range, sycl::range<1> local_range) {
  stream->submit([&](sycl::handler& cgh) {
    LookupAndScatterUniqueIdsKernel<TIndex> task(input_size, sorted_input_inds,
                                                 sorted_input_unique_ids,
                                                 inv_sorted_unique_perm, idx);
    cgh.parallel_for<LookupAndScatterUniqueIdsKernel<TIndex>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

// Maps the values of sorted_input_unique_ids and scatters them to idx using
// sorted_input_inds.
template <typename TIndex>
Status LookupAndScatterUniqueIds(sycl::queue* stream, int64_t input_size,
                                 TIndex* sorted_input_inds,
                                 TIndex* sorted_input_unique_ids,
                                 TIndex* inv_sorted_unique_perm, TIndex* idx) {
  assert(input_size > 0);  // Crash OK
  const int workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  sycl::range<1> local_range(workgroup_size);
  const int num_wg = (input_size + workgroup_size - 1) / workgroup_size;
  sycl::range<1> global_range(num_wg * workgroup_size);

  return LaunchLookupAndScatterUniqueIdsKernel<TIndex>(
      stream, input_size, sorted_input_inds, sorted_input_unique_ids,
      inv_sorted_unique_perm, idx, global_range, local_range);
}

template <typename T>
struct RangeInitWorkItemKernel {
  RangeInitWorkItemKernel(T start, T delta, T size, T* out)
      : start(start), delta(delta), size(size), out(out) {}
  void operator()(sycl::nd_item<1> item) const {
    auto global_id = item.get_global_id(0);
    auto global_range = item.get_global_range(0);
    for (int32_t i = global_id, step = global_range; i < size; i += step) {
      out[i] = start + i * delta;
    }
  }

 private:
  T start;
  T delta;
  T size;
  T* out;
};

template <typename T>
Status RangeInitWorkItemImpl(sycl::queue* stream, const T start, const T delta,
                             const T size, T* out, sycl::range<1> global_range,
                             sycl::range<1> local_range) {
  stream->submit([&](sycl::handler& cgh) {
    RangeInitWorkItemKernel<T> task(start, delta, size, out);
    cgh.parallel_for<RangeInitWorkItemKernel<T>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });

  return Status::OK();
}

// Initialize out with range start, start + delta, start + 2 * delta, ...
template <typename T>
Status LaunchRangeInitKernel(sycl::queue* stream, const T start, const T delta,
                             const T size, T* out) {
  if (size == 0) return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  const int workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  sycl::range<1> local_range(workgroup_size);
  const int num_wg = (size + workgroup_size - 1) / workgroup_size;
  sycl::range<1> global_range(num_wg * workgroup_size);

  return RangeInitWorkItemImpl<T>(stream, start, delta, size, out, global_range,
                                  local_range);
}

template <typename KeyT, typename ValueT, int KEYS_PER_ITEM, int SUBGROUP_SIZE,
          class Sortor>
struct RadixSortKernel {
  RadixSortKernel(__shared__<uint8_t> scratch, int32_t num_instances,
                  int32_t num_bits, KeyT* d_input, ValueT* d_input_inds_ptr,
                  KeyT* sorted_input_ptr, ValueT* sorted_input_inds_ptr)
      : scratch(scratch),
        num_instances(num_instances),
        num_bits(num_bits),
        d_input(d_input),
        d_input_inds_ptr(d_input_inds_ptr),
        sorted_input_ptr(sorted_input_ptr),
        sorted_input_inds_ptr(sorted_input_inds_ptr) {}
  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    int local_id = item.get_local_id(0);

    KeyT* d_input_iter = d_input + local_id * KEYS_PER_ITEM;
    ValueT* d_input_inds_iter = d_input_inds_ptr + local_id * KEYS_PER_ITEM;

    KeyT item_scores[KEYS_PER_ITEM];
    ValueT item_boxIds[KEYS_PER_ITEM];

#pragma unroll
    for (int i = 0; i < KEYS_PER_ITEM; i++) {
      if (local_id * KEYS_PER_ITEM + i < num_instances) {
        item_scores[i] = d_input_iter[i];
        item_boxIds[i] = d_input_inds_iter[i];
      } else {
        item_scores[i] = std::numeric_limits<KeyT>::max();
      }
    }
    // get the pointer of share local memory
    uint8_t* local_mem = scratch.get_pointer().get();
    // Sorting the scores in ascending order
    Sortor(item.get_group(), item.get_sub_group(), local_id, local_mem)
        .Sort(item_scores, item_boxIds, sorted_input_ptr, sorted_input_inds_ptr,
              num_instances, 0, num_bits);
  }

 private:
  __shared__<uint8_t> scratch;
  int32_t num_instances;
  int32_t num_bits;
  KeyT* d_input;
  ValueT* d_input_inds_ptr;
  KeyT* sorted_input_ptr;
  ValueT* sorted_input_inds_ptr;
};

template <typename KeyT, typename ValueT, int KEYS_PER_ITEM, int SUBGROUP_SIZE,
          class Sortor>
Status LaunchRadixSortKernel(sycl::queue* stream, const int32_t num_instances,
                             KeyT* d_input, ValueT* d_input_inds_ptr,
                             KeyT* sorted_input_ptr,
                             ValueT* sorted_input_inds_ptr,
                             sycl::range<1> global_range,
                             sycl::range<1> local_range,
                             size_t local_memory_size,
                             int num_bits = sizeof(KeyT) * 8) {
  stream->submit([&](sycl::handler& cgh) {
    __shared__<uint8_t> scratch(sycl::range<1>{local_memory_size}, cgh);
    RadixSortKernel<KeyT, ValueT, KEYS_PER_ITEM, SUBGROUP_SIZE, Sortor> task(
        scratch, num_instances, num_bits, d_input, d_input_inds_ptr,
        sorted_input_ptr, sorted_input_inds_ptr);
    cgh.parallel_for<
        RadixSortKernel<KeyT, ValueT, KEYS_PER_ITEM, SUBGROUP_SIZE, Sortor>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });

  return Status::OK();
}

template <typename KeyT, typename ValueT, int KEYS_PER_ITEM, int GROUP_SIZE,
          int SUBGROUP_SIZE>
Status DispatchRadixSort(OpKernelContext* context, const int32_t size,
                         KeyT* keys_in, ValueT* indices_in, KeyT* keys_out,
                         ValueT* indices_out, int num_bits = sizeof(KeyT) * 8) {
  if (size == 0) return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  const GPUDevice& device = context->eigen_device<GPUDevice>();
  sycl::queue* stream = device.stream();

  Tensor tmp_indices_in;
  if (!indices_in) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<ValueT>::value, TensorShape({size}), &tmp_indices_in));
    ValueT* mutable_indices_in = tmp_indices_in.flat<ValueT>().data();
    indices_in = mutable_indices_in;
    // Set indices_in to range only if indices_in is created internally.
    ITEX_CHECK_OK(LaunchRangeInitKernel<ValueT>(stream, ValueT(0), ValueT(1),
                                                ValueT(size), indices_in));
  }

  Tensor tmp_keys_out;
  if (!keys_out) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<KeyT>::value, TensorShape({size}), &tmp_keys_out));
    KeyT* mutable_keys_out = tmp_keys_out.flat<KeyT>().data();
    keys_out = mutable_keys_out;
  }

  if (size <= KEYS_PER_ITEM * GROUP_SIZE) {
    using Rsortor = GroupRadixSortor<
        KeyT, /*key_per_item==*/KEYS_PER_ITEM, /*group_size=*/GROUP_SIZE,
        /*subgroup_size =*/SUBGROUP_SIZE, sycl::group<1>, ValueT>;
    // Compute the required local memory size
    size_t local_memory_size = Rsortor::LocalStorage::SIZE;
    const int32_t num_wg = 1;
    sycl::range<1> global_range(num_wg * GROUP_SIZE);
    sycl::range<1> local_range(GROUP_SIZE);

    return LaunchRadixSortKernel<KeyT, ValueT, KEYS_PER_ITEM, SUBGROUP_SIZE,
                                 Rsortor>(
        stream, size, keys_in, indices_in, keys_out, indices_out, global_range,
        local_range, local_memory_size, num_bits);
  } else {
    // TODO(itex): Kernel is too slow if inputs size is large. We temporary
    // set group size as max value, and plan to optimize the kernel performance
    // in the future.
    int max_group_size =
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    Tensor tmp_keys_buffer;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<KeyT>::value, TensorShape({size}), &tmp_keys_buffer));

    ::itex::functor::DispatchToFallBackRadixSort(
        stream, keys_in, keys_out, tmp_keys_buffer.flat<KeyT>().data(),
        indices_out, indices_in, 1, size, max_group_size);

    return Status::OK();
  }
}

template <typename InputIteratorT, typename OutputIteratorT, typename BinaryOp>
void DispatchScan(
    OpKernelContext* context, const int N, InputIteratorT data,
    OutputIteratorT result,
    const typename std::iterator_traits<InputIteratorT>::value_type init,
    bool is_exclusive, bool is_reverse, BinaryOp binary_op) {
  if (is_exclusive) {
    if (is_reverse)
      _scan_kernel<InputIteratorT, OutputIteratorT, BinaryOp, true, true>(
          data, result, init, binary_op, N, context);
    else
      _scan_kernel<InputIteratorT, OutputIteratorT, BinaryOp, true, false>(
          data, result, init, binary_op, N, context);
  } else {
    if (is_reverse)
      _scan_kernel<InputIteratorT, OutputIteratorT, BinaryOp, false, true>(
          data, result, init, binary_op, N, context);
    else
      _scan_kernel<InputIteratorT, OutputIteratorT, BinaryOp, false, false>(
          data, result, init, binary_op, N, context);
  }
}

}  // namespace impl

// Returns true iff index is at the end of a segment (which is equivalent to the
// beginning of the next segment).
template <typename T>
struct SegmentIndicatorFunctor {
  const T* sorted_input_ptr_;
  SegmentIndicatorFunctor(
      const T* sorted_input_ptr)  // NOLINT(runtime/explicit)
      : sorted_input_ptr_(sorted_input_ptr) {}
  bool operator()(const int32_t& i) const {
    return i > 0 && sorted_input_ptr_[i] != sorted_input_ptr_[i - 1];
  }
};

template <typename IteratorType, typename ConversionOp, typename InputIteratorT,
          typename OffsetT = ptrdiff_t>
class TransformIterator {
 public:
  typedef TransformIterator self_type;
  typedef OffsetT difference_type;
  typedef IteratorType value_type;
  typedef IteratorType* pointer;
  typedef IteratorType reference;
  typedef std::random_access_iterator_tag iterator_category;

 private:
  ConversionOp conversion_op;
  InputIteratorT input_itr;

 public:
  /// Constructor
  TransformIterator(InputIteratorT input_itr, ConversionOp conversion_op)
      : conversion_op(conversion_op), input_itr(input_itr) {}

  reference operator*() const { return conversion_op(*input_itr); }

  template <typename Index>
  reference operator[](Index n) const {
    return conversion_op(input_itr[n]);
  }
};

template <typename IteratorType, typename OffsetT = ptrdiff_t>
class CountIterator {
 public:
  typedef OffsetT difference_type;
  typedef IteratorType value_type;
  typedef IteratorType reference;
  typedef std::random_access_iterator_tag iterator_category;

 private:
  IteratorType val;

 public:
  /// Constructor
  explicit CountIterator(const IteratorType& val) : val(val) {}

  template <typename Index>
  reference operator[](Index n) const {
    return val + (IteratorType)n;
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_UNIQUE_OP_H_
