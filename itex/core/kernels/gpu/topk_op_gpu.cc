/* Copyright (c) 2021-2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/topk_op.h"
#include "itex/core/utils/group_radix_select.h"
#include "itex/core/utils/group_radix_sort.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/radix_utils.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace {

template <typename T, typename IndexT>
struct CopyKernel {
  CopyKernel(size_t total, T* values, const T* input, IndexT* indices,
             int32_t num_cols)
      : total(total),
        values(values),
        input(input),
        indices(indices),
        num_cols(num_cols) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= total) return;
    values[id] = input[id];
    indices[id] = id % num_cols;
  }

 private:
  size_t total;
  T* values;
  const T* input;
  IndexT* indices;
  int32_t num_cols;
};

template <typename T, typename IndexT>
void LaunchCopyKernel(const gpuStream_t& stream, const T* input, T* values,
                      IndexT* indices, int num_rows, int num_cols,
                      int num_topk) {
  const int32 max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();

  auto total = num_rows * num_cols;
  auto num_work_group = (total + max_group_size - 1) / max_group_size;

  stream->submit([&](sycl::handler& cgh) {
    CopyKernel<T, IndexT> task(total, values, input, indices, num_cols);
    cgh.parallel_for<CopyKernel<T, IndexT>>(
        sycl::nd_range<1>(sycl::range<1>(num_work_group * max_group_size),
                          sycl::range<1>(max_group_size)),
        task);
  });
}

using LocalAcc = sycl::local_accessor<uint8_t, 1>;
template <int KEYS_PER_ITEM, int GROUP_SIZE, int SUB_GROUP_SIZE, typename T,
          typename IndexT>
struct RadixTopKKernel {
  RadixTopKKernel(LocalAcc scratch, const T* input, T* values, IndexT* indices,
                  int32_t num_cols, int32_t num_topk)
      : scratch(scratch),
        input(input),
        values(values),
        indices(indices),
        num_cols(num_cols),
        num_topk(num_topk) {}
  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    // Type definitions
    using Rselector =
        GroupRadixPerBitSelector<T, KEYS_PER_ITEM, GROUP_SIZE, SUB_GROUP_SIZE,
                                 sycl::group<1>, IndexT>;

    auto g = item.get_group();
    int row_id = item.get_group(0);
    int local_id = item.get_local_id(0);

    // Set up radix selector
    uint8_t* local_mem = scratch.get_pointer().get();
    Rselector rselector(g, item.get_sub_group(), local_id, local_mem);

    // Pointer of the input
    const T* inp_values = input + row_id * num_cols;

    T item_values[KEYS_PER_ITEM];
    IndexT item_indices[KEYS_PER_ITEM];

    // load data to private memory
    for (int i = 0; i < KEYS_PER_ITEM; ++i) {
      item_values[i] = std::numeric_limits<T>::lowest();
      int id = local_id * KEYS_PER_ITEM + i;
      if (id < num_cols) {
        item_values[i] = inp_values[id];
        item_indices[i] = id;
      }
    }

    // The total number of keys that can be sorted in one work group
    constexpr int CHUNK = KEYS_PER_ITEM * GROUP_SIZE;

    // If num_cols is greater than CHUNK, the selected topk values
    // will be used for the next selecting
    T* temp_values =
        reinterpret_cast<T*>(local_mem + Rselector::LocalStorage::SIZE);
    IndexT* temp_indices = reinterpret_cast<IndexT*>(temp_values + num_topk);

    int num_start = CHUNK;
    while (num_start < num_cols) {
      rselector.SelectTopK(item_values, item_indices, temp_values, temp_indices,
                           num_topk);
      sycl::group_barrier(g);

      // load selected topk values from local memory
      for (int i = 0; i < KEYS_PER_ITEM; ++i) {
        int offset = local_id * KEYS_PER_ITEM + i;
        if (offset < num_topk) {
          item_values[i] = temp_values[offset];
          item_indices[i] = temp_indices[offset];
        } else {
          item_values[i] = std::numeric_limits<T>::lowest();
          int id = num_start + offset - num_topk;
          if (id < num_cols) {
            item_values[i] = inp_values[id];
            item_indices[i] = id;
          }
        }
      }
      num_start += CHUNK - num_topk;
    }

    // pointers of the ouput
    T* out_values = values + row_id * num_topk;
    IndexT* out_indices = indices + row_id * num_topk;

    // select topK from the last CHUNK of scores and store in output
    rselector.SelectTopK(item_values, item_indices, out_values, out_indices,
                         num_topk);
  }

 private:
  LocalAcc scratch;
  const T* input;
  T* values;
  IndexT* indices;
  int32_t num_cols;
  int32_t num_topk;
};

template <int KEYS_PER_ITEM, int GROUP_SIZE, int SUB_GROUP_SIZE = 16,
          typename T, typename IndexT>
void LaunchRadixTopKKernel(const gpuStream_t& stream, const T* input, T* values,
                           IndexT* indices, int num_rows, int num_cols,
                           int num_topk) {
  // Type definitions
  using Rselector =
      GroupRadixPerBitSelector<T, KEYS_PER_ITEM, GROUP_SIZE, SUB_GROUP_SIZE,
                               sycl::group<1>, IndexT>;

  // Compute the required local memory size
  size_t local_memory_size =
      Rselector::LocalStorage::SIZE + num_topk * (sizeof(T) + sizeof(IndexT));
  // submit task
  stream->submit([&](sycl::handler& cgh) {
    // Local Memory Management
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    RadixTopKKernel<KEYS_PER_ITEM, GROUP_SIZE, SUB_GROUP_SIZE, T, IndexT> task(
        scratch, input, values, indices, num_cols, num_topk);
    cgh.parallel_for<
        RadixTopKKernel<KEYS_PER_ITEM, GROUP_SIZE, SUB_GROUP_SIZE, T, IndexT>>(
        sycl::nd_range<1>(sycl::range<1>(num_rows * GROUP_SIZE),
                          sycl::range<1>(GROUP_SIZE)),
        task);
  });
}

template <int KEYS_PER_ITEM, int GROUP_SIZE, int RADIX_BITS, int SUB_GROUP_SIZE,
          typename T, typename IndexT>
struct RadixSortKernel {
  RadixSortKernel(LocalAcc scratch, T* values, IndexT* indices, int num_topk)
      : scratch(scratch),
        values(values),
        indices(indices),
        num_topk(num_topk) {}
  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    // Type definitions
    using Rsortor =
        GroupRadixSortor<T, KEYS_PER_ITEM, GROUP_SIZE, SUB_GROUP_SIZE,
                         sycl::group<1>, IndexT, RADIX_BITS>;
    auto g = item.get_group();
    int row_id = item.get_group(0);
    int local_id = item.get_local_id(0);

    // pointers of the data
    T* ptr_values = values + row_id * num_topk;
    IndexT* ptr_indices = indices + row_id * num_topk;

    T item_values[KEYS_PER_ITEM];
    IndexT item_indices[KEYS_PER_ITEM];

    // load data to private memory
    for (int i = 0; i < KEYS_PER_ITEM; ++i) {
      item_values[i] = std::numeric_limits<T>::lowest();
      int id = local_id * KEYS_PER_ITEM + i;
      if (id < num_topk) {
        item_values[i] = ptr_values[id];
        item_indices[i] = ptr_indices[id];
      }
    }

    // get the pointer of share local memory
    uint8_t* local_mem = scratch.get_pointer().get();

    // Do sorting
    Rsortor(g, item.get_sub_group(), local_id, local_mem)
        .SortDescending(item_values, item_indices, ptr_values, ptr_indices,
                        num_topk);
  }

 private:
  LocalAcc scratch;
  T* values;
  IndexT* indices;
  int num_topk;
};

template <int KEYS_PER_ITEM, int GROUP_SIZE, int RADIX_BITS = 4,
          int SUB_GROUP_SIZE = 16, typename T, typename IndexT>
void LaunchRadixSortKernel(const gpuStream_t& stream, T* values,
                           IndexT* indices, int num_rows, int num_topk) {
  // Type definitions
  using Rsortor = GroupRadixSortor<T, KEYS_PER_ITEM, GROUP_SIZE, SUB_GROUP_SIZE,
                                   sycl::group<1>, IndexT, RADIX_BITS>;
  // Compute the required local memory size
  size_t local_memory_size = Rsortor::LocalStorage::SIZE;

  stream->submit([&](sycl::handler& cgh) {
    // Local Memory Management
    LocalAcc scratch(sycl::range<1>{local_memory_size}, cgh);
    RadixSortKernel<KEYS_PER_ITEM, GROUP_SIZE, RADIX_BITS, SUB_GROUP_SIZE, T,
                    IndexT>
        task(scratch, values, indices, num_topk);
    cgh.parallel_for<RadixSortKernel<KEYS_PER_ITEM, GROUP_SIZE, RADIX_BITS,
                                     SUB_GROUP_SIZE, T, IndexT>>(
        sycl::nd_range<1>(sycl::range<1>(num_rows * GROUP_SIZE),
                          sycl::range<1>(GROUP_SIZE)),
        task);
  });
}

template <typename LocalAcc, typename KeyT, typename ValueT, int RADIX_BITS,
          bool Ascending = true>
struct FallBackKeyValueRadixSort {
  using KeyTraits = NumericTraits<KeyT>;
  using UnsignedT = typename KeyTraits::UnsignedT;
  FallBackKeyValueRadixSort(const KeyT* key_array_, KeyT* key_src_,
                            KeyT* key_dst_, ValueT* value_src_,
                            ValueT* value_dst_, LocalAcc counters_,
                            const int row_, const int col_)
      : key_array(key_array_),
        key_src(key_src_),
        key_dst(key_dst_),
        value_src(value_src_),
        value_dst(value_dst_),
        counters(counters_),
        row(row_),
        col(col_) {}
  inline void operator()(sycl::nd_item<1> item) const {
    constexpr int RADIX_STATUS = 1 << RADIX_BITS;
    auto group_size = item.get_local_range(0);
    auto id = item.get_local_linear_id();
    auto group_id = item.get_group(0);

    const int num_per_items = (col + group_size - 1) / group_size;
    const int begin_bit = 0;
    const int end_bit = sizeof(KeyT) * 8;
    int current_bit = begin_bit;

    const int group_offset = group_id * col;
    const KeyT* group_key = key_array + group_offset;
    KeyT* group_key_src = key_src + group_offset;
    KeyT* group_key_dst = key_dst + group_offset;

    ValueT* group_value_src = value_src + group_offset;
    ValueT* group_value_dst = value_dst + group_offset;

    while (current_bit < end_bit) {
// reset counters[RADIX_STATUS][group_size]
#pragma unroll
      for (int i = 0; i < RADIX_STATUS; ++i) {
        counters[i * group_size + id] = 0;
      }
      item.barrier();

      int pass_bit = std::min(RADIX_BITS, end_bit - current_bit);
      RadixExtractor<KeyT> digit_extractor(current_bit, pass_bit);

      // bins
      for (int i = 0; i < num_per_items; ++i) {
        if (id * num_per_items + i < col) {
          KeyT key;
          if (current_bit == 0)
            key = group_key[id * num_per_items + i];
          else
            key = group_key_src[id * num_per_items + i];
          UnsignedT unsigned_key = *(reinterpret_cast<UnsignedT*>(&key));
          uint32_t bucket = digit_extractor.Bucket(
              KeyTraits::Convert(unsigned_key, Int2Type<!Ascending>()));
          ++counters[bucket * group_size + id];
        }
      }
      item.barrier();

      // scans
      int aggregate = 0;
      for (int i = 0; i < RADIX_STATUS; ++i) {
        aggregate += counters[id * RADIX_STATUS + i];
      }
      int updated_aggregate = sycl::exclusive_scan_over_group(
          item.get_group(), aggregate, sycl::plus<int>());
#pragma unroll
      for (int i = 0; i < RADIX_STATUS; ++i) {
        int before = counters[id * RADIX_STATUS + i];
        counters[id * RADIX_STATUS + i] = updated_aggregate;
        updated_aggregate += before;
      }
      item.barrier();

      // reorder
      for (int i = 0; i < num_per_items; ++i) {
        if (id * num_per_items + i < col) {
          KeyT key;
          ValueT value;
          if (current_bit == 0) {
            key = group_key[id * num_per_items + i];
            value = id * num_per_items + i;
          } else {
            key = group_key_src[id * num_per_items + i];
            value = group_value_src[id * num_per_items + i];
          }
          UnsignedT unsigned_key = *(reinterpret_cast<UnsignedT*>(&key));
          uint32_t bucket = digit_extractor.Bucket(
              KeyTraits::Convert(unsigned_key, Int2Type<!Ascending>()));
          int pos = counters[bucket * group_size + id];
          if (pos < 0 || pos >= col) break;

          group_key_dst[pos] = key;
          group_value_dst[pos] = value;
          ++counters[bucket * group_size + id];
        }
      }
      item.barrier();
      std::swap(group_key_src, group_key_dst);
      std::swap(group_value_src, group_value_dst);
      current_bit += pass_bit;
    }
  }

 public:
  const KeyT* key_array;
  KeyT* key_src;
  KeyT* key_dst;
  ValueT* value_src;
  ValueT* value_dst;
  LocalAcc counters;
  const int row;
  const int col;
};

template <typename KeyT, typename ValueT, int RADIX_BITS, bool Ascending = true>
void LaunchFallBackKeyValueRadixSort(const gpuStream_t& stream,
                                     const KeyT* key_array, KeyT* key_src,
                                     KeyT* key_dst, ValueT* value_src,
                                     ValueT* value_dst, const int row,
                                     const int col, const int group_size) {
  sycl::nd_range<1> range(row * group_size, group_size);
  constexpr int RADIX_STATUS = 1 << RADIX_BITS;
  stream->submit([&](sycl::handler& cgh) {
    typedef sycl::local_accessor<int, 1> LocalAcc;
    LocalAcc counters(RADIX_STATUS * group_size, cgh);
    FallBackKeyValueRadixSort<LocalAcc, KeyT, ValueT, RADIX_BITS, Ascending>
        task(key_array, key_src, key_dst, value_src, value_dst, counters, row,
             col);
    cgh.parallel_for<FallBackKeyValueRadixSort<LocalAcc, KeyT, ValueT,
                                               RADIX_BITS, Ascending>>(range,
                                                                       task);
  });
}
}  // anonymous namespace

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename KeyT, typename ValueT, bool Ascending>
void DispatchToFallBackRadixSort(const gpuStream_t& stream,
                                 const KeyT* key_array, KeyT* key_src,
                                 KeyT* key_dst, ValueT* value_src,
                                 ValueT* value_dst, const int num_rows,
                                 const int num_cols, const int max_group_size) {
  int group_size = max_group_size;

  int RADIX_BITS = 4;
  int RADIX_STATUS = 1 << RADIX_BITS;

  const int slm_size =
      stream->get_device()
          .template get_info<sycl::info::device::local_mem_size>();
  while (RADIX_STATUS * group_size * sizeof(int) >= slm_size &&
         group_size >= 32) {
    group_size >>= 1;
  }
  while (RADIX_STATUS * group_size * sizeof(int) >= slm_size &&
         RADIX_BITS > 1) {
    --RADIX_BITS;
    RADIX_STATUS = 1 << RADIX_BITS;
  }
  if (RADIX_STATUS * group_size * sizeof(int) >= slm_size) {
    std::stringstream ss;
    ss << "Not Supported hardware, as SLM is too small, required "
          "minumum size is "
       << RADIX_STATUS * group_size * sizeof(int)
       << " got hardward slm siz: " << slm_size;
    ITEX_LOG(FATAL) << ss.str();
  }

  switch (RADIX_BITS) {
#define HANDLE_N(NUM)                                                        \
  case (NUM):                                                                \
    LaunchFallBackKeyValueRadixSort<KeyT, ValueT, NUM, Ascending>(           \
        stream, key_array, key_src, key_dst, value_src, value_dst, num_rows, \
        num_cols, group_size);                                               \
    break;
    HANDLE_N(4)
    HANDLE_N(3)
    HANDLE_N(2)
    HANDLE_N(1)
#undef HANDLE_N
  }
}

//  for large k case, directly sort
template <typename T, typename IndexT>
void LaunchLargeKKernel(OpKernelContext* context, const T* input,
                        typename TTypes<T, 2>::Tensor values,
                        typename TTypes<IndexT, 2>::Tensor indices,
                        const int num_topk, const int num_rows,
                        const int num_cols, const int max_group_size) {
  const auto& d = context->eigen_gpu_device();
  auto& stream = d.stream();
  Tensor values_tmp_ping;
  Tensor indices_tmp_ping;
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<T>::value,
                                        TensorShape({num_rows, num_cols}),
                                        &values_tmp_ping));
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<IndexT>::value,
                                        TensorShape({num_rows, num_cols}),
                                        &indices_tmp_ping));
  Tensor values_tmp_pong;
  Tensor indices_tmp_pong;
  T* values_tmp_pong_ptr;
  IndexT* indices_tmp_pong_ptr;
  if (num_topk == num_cols) {
    values_tmp_pong_ptr = values.data();
    indices_tmp_pong_ptr = indices.data();
  } else {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          TensorShape({num_rows, num_cols}),
                                          &values_tmp_pong));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<IndexT>::value,
                                          TensorShape({num_rows, num_cols}),
                                          &indices_tmp_pong));
    values_tmp_pong_ptr = values_tmp_pong.flat<T>().data();
    indices_tmp_pong_ptr = indices_tmp_pong.flat<IndexT>().data();
  }
  T* values_src = values_tmp_pong_ptr;
  IndexT* indices_src = indices_tmp_pong_ptr;
  T* values_dst = values_tmp_ping.flat<T>().data();
  IndexT* indices_dst = indices_tmp_ping.flat<IndexT>().data();

  DispatchToFallBackRadixSort<T, IndexT, false>(
      stream, input, values_src, values_dst, indices_src, indices_dst, num_rows,
      num_cols, max_group_size);

  if (num_topk < num_cols) {
    // Need to copy subsets of sorted_indices and sorted_outputs to
    // indices and outputs.
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, 0};
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, num_topk};
    To32Bit(values).device(d) =
        To32Bit(values_tmp_pong.matrix<T>()).slice(slice_indices, slice_sizes);
    To32Bit(indices).device(d) = To32Bit(indices_tmp_pong.matrix<IndexT>())
                                     .slice(slice_indices, slice_sizes);
  }
}

template <typename T, typename IndexT>
void TopKFunctor<GPUDevice, T, IndexT>::operator()(
    OpKernelContext* context, typename TTypes<T, 2>::ConstTensor input,
    typename TTypes<T, 2>::Tensor values,
    typename TTypes<IndexT, 2>::Tensor indices, bool sorted, int num_topk) {
  const int num_rows = input.dimension(0);
  const int num_cols = input.dimension(1);

  // Nothing to do for top-nothing or over nothing.
  if (num_topk == 0 || num_rows == 0) return;

  const auto& d = context->eigen_gpu_device();
  auto& stream = d.stream();

  const int32 max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();

  constexpr int keys_per_item = 8;
  constexpr int max_keys_per_item = 9;

  bool is_large_k = (num_topk > max_group_size * max_keys_per_item);

  if (is_large_k) {
    if (num_topk == num_cols && !sorted) {
      LaunchCopyKernel(stream, input.data(), values.data(), indices.data(),
                       num_rows, num_cols, num_topk);
    } else {
      LaunchLargeKKernel<T, IndexT>(context, input.data(), values, indices,
                                    num_topk, num_rows, num_cols,
                                    max_group_size);
    }
  } else {
    //-------------------------------------------------------------
    // for non-large k case, first to select topk
    if (num_topk == num_cols) {
      LaunchCopyKernel(stream, input.data(), values.data(), indices.data(),
                       num_rows, num_cols, num_topk);
    } else {
      int group_size = max_group_size;
      if (num_cols > 2048) {
        if (num_topk < 1024) {
          group_size = 256;
        } else {
          while (group_size * keys_per_item > 4 * num_topk) group_size >>= 1;
        }
      } else {
        while (group_size * keys_per_item > 2 * num_cols) group_size >>= 1;
      }
      group_size = (group_size > 32) ? group_size : 32;

      void (*kernels[])(const gpuStream_t&, const T*, T*, IndexT*, int, int,
                        int) = {
          LaunchRadixTopKKernel<keys_per_item, 32>,
          LaunchRadixTopKKernel<keys_per_item, 64>,
          LaunchRadixTopKKernel<keys_per_item, 128>,
          LaunchRadixTopKKernel<keys_per_item, 256>,
          LaunchRadixTopKKernel<keys_per_item, 512>,
          LaunchRadixTopKKernel<keys_per_item, 1024>,
          LaunchRadixTopKKernel<keys_per_item, 2048>,
      };

      int idx = 0;
      while ((1 << idx) * 32 < group_size) ++idx;
      kernels[idx](stream, input.data(), values.data(), indices.data(),
                   num_rows, num_cols, num_topk);
    }

    //-------------------------------------------------------------
    // for non-large k case, then to sort topk
    if (sorted) {
      int group_size = max_group_size;
      while (group_size * keys_per_item > 2 * num_topk) group_size >>= 1;
      group_size = (group_size > 32) ? group_size : 32;

      if (group_size * keys_per_item >= num_topk) {
        int idx = 0;
        while ((1 << idx) * 32 < group_size) ++idx;
        void (*kernels[])(const gpuStream_t&, T*, IndexT*, int, int) = {
            LaunchRadixSortKernel<keys_per_item, 32>,
            LaunchRadixSortKernel<keys_per_item, 64>,
            LaunchRadixSortKernel<keys_per_item, 128>,
            LaunchRadixSortKernel<keys_per_item, 256>,
            LaunchRadixSortKernel<keys_per_item, 512>,
            LaunchRadixSortKernel<keys_per_item, 1024>,
            LaunchRadixSortKernel<keys_per_item, 2048>,
        };
        kernels[idx](stream, values.data(), indices.data(), num_rows, num_topk);
      } else {
        int idx = 0;
        while ((1 << idx) * 256 < group_size) ++idx;
        void (*kernels_for_large_array[])(const gpuStream_t&, T*, IndexT*, int,
                                          int) = {
            LaunchRadixSortKernel<max_keys_per_item, 256>,
            LaunchRadixSortKernel<max_keys_per_item, 512>,
            LaunchRadixSortKernel<max_keys_per_item, 1024>,
            LaunchRadixSortKernel<max_keys_per_item, 2048>,
        };
        kernels_for_large_array[idx](stream, values.data(), indices.data(),
                                     num_rows, num_topk);
      }
    }
  }
}

#define INSTANTIATE_GPU(T)                                                   \
  template struct TopKFunctor<GPUDevice, T, int32>;                          \
  template void DispatchToFallBackRadixSort<T, int32, true>(                 \
      const gpuStream_t& stream, const T* key_array, T* key_src, T* key_dst, \
      int32* value_src, int32* value_dst, const int num_rows,                \
      const int num_cols, const int max_group_size);                         \
  template struct TopKFunctor<GPUDevice, T, int64>;                          \
  template void DispatchToFallBackRadixSort<T, int64, true>(                 \
      const gpuStream_t& stream, const T* key_array, T* key_src, T* key_dst, \
      int64* value_src, int64* value_dst, const int num_rows,                \
      const int num_cols, const int max_group_size);

TF_CALL_GPU_NUMBER_TYPES(INSTANTIATE_GPU);
TF_CALL_INTEGRAL_TYPES(INSTANTIATE_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(INSTANTIATE_GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef INSTANTIATE_GPU

}  // namespace functor
}  // namespace itex
