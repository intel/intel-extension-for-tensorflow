/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_UNIQUE_OP_HELPERS_H_
#define ITEX_CORE_KERNELS_GPU_UNIQUE_OP_HELPERS_H_

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace impl {

template <typename T>
using LocalAcc = sycl::local_accessor<T, 1>;

template <bool IsReverse>
inline int MapReversedIndex(int dim_size, int index);

template <>
inline int MapReversedIndex<true>(int dim_size, int index) {
  return dim_size - 1 - index;
}

template <>
inline int MapReversedIndex<false>(int dim_size, int index) {
  return index;
}

template <typename InputIteratorT, typename OutputIteratorT, typename BinaryOp,
          typename LocalAccessor, int GroupSize, int ElemsPerWorkItem,
          bool IsExclusive, bool IsReverse>
struct GroupScan {
  using T = typename std::iterator_traits<InputIteratorT>::value_type;
  GroupScan(InputIteratorT in_data, OutputIteratorT out_data,
            LocalAccessor local_mem, T init, BinaryOp binary_op, size_t N)
      : in_data_(in_data),
        out_data_(out_data),
        local_mem_(local_mem),
        init_(init),
        binary_op_(binary_op),
        N_(N){};

  void operator()(sycl::nd_item<1> item) const {
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    T* local_mem_ptr = ITEXGetLocalAccPointer<T>(local_mem_);

    // read data from global memory to SLM
    auto end = GroupSize * ElemsPerWorkItem;
#pragma unroll
    for (int i = lid; i < end; i += GroupSize) {
      if (i < N_)
        local_mem_ptr[i] = in_data_[MapReversedIndex<IsReverse>(N_, i)];
      else
        local_mem_ptr[i] = init_;
    }
    sycl::group_barrier(group);

    // reduce
    T prefix = init_;
    const int local_start = lid * ElemsPerWorkItem;
#pragma unroll
    for (int i = 0; i < ElemsPerWorkItem; ++i) {
      prefix = binary_op_(prefix, local_mem_ptr[local_start + i]);
    }

    // scan
    T updated_prefix =
        sycl::exclusive_scan_over_group(group, prefix, binary_op_);

    // down sweep
    if (IsExclusive) {
#pragma unroll
      for (int i = 0; i < ElemsPerWorkItem; ++i) {
        T tmp = local_mem_ptr[local_start + i];
        local_mem_ptr[local_start + i] = updated_prefix;
        updated_prefix = binary_op_(updated_prefix, tmp);
      }
    } else {
#pragma unroll
      for (int i = 0; i < ElemsPerWorkItem; ++i) {
        T tmp = local_mem_ptr[local_start + i];
        updated_prefix = binary_op_(updated_prefix, tmp);
        local_mem_ptr[local_start + i] = updated_prefix;
      }
    }

    sycl::group_barrier(group);

// write  output
#pragma unroll
    for (int i = lid; i < end; i += GroupSize) {
      if (i < N_)
        out_data_[MapReversedIndex<IsReverse>(N_, i)] = local_mem_ptr[i];
    }
  }

 private:
  InputIteratorT in_data_;
  OutputIteratorT out_data_;
  LocalAccessor local_mem_;
  T init_;
  BinaryOp binary_op_;
  size_t N_;
};

template <typename InputIteratorT, typename OutputIteratorT, typename BinaryOp,
          typename LocalAccessor, int GroupSize, int ElemsPerWorkItem,
          bool IsExclusive, bool IsReverse>
struct DeviceScanFirstStep {
  using T = typename std::iterator_traits<InputIteratorT>::value_type;
  DeviceScanFirstStep(InputIteratorT in_data, OutputIteratorT out_data,
                      OutputIteratorT inter_out, LocalAccessor local_mem,
                      T init, BinaryOp binary_op, size_t N)
      : in_data_(in_data),
        out_data_(out_data),
        inter_out_(inter_out),
        local_mem_(local_mem),
        init_(init),
        binary_op_(binary_op),
        N_(N){};

  void operator()(sycl::nd_item<1> item) const {
    auto group_id = item.get_group_linear_id();
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    T* local_mem_ptr = ITEXGetLocalAccPointer<T>(local_mem_);

    // read data from global memory to slm
    auto start = group_id * GroupSize * ElemsPerWorkItem;
    auto end = (group_id + 1) * GroupSize * ElemsPerWorkItem;

#pragma unroll
    for (int i = lid; start + i < end; i += GroupSize) {
      if (start + i < N_)
        local_mem_ptr[i] = in_data_[MapReversedIndex<IsReverse>(N_, start + i)];
      else
        local_mem_ptr[i] = init_;
    }
    sycl::group_barrier(group);

    // reduce
    T prefix = init_;
    const int local_start = lid * ElemsPerWorkItem;
#pragma unroll
    for (int i = 0; i < ElemsPerWorkItem; ++i) {
      prefix = binary_op_(prefix, local_mem_ptr[local_start + i]);
    }

    // scan
    T updated_prefix =
        sycl::exclusive_scan_over_group(group, prefix, binary_op_);

    if (IsExclusive) {
#pragma unroll
      for (int i = 0; i < ElemsPerWorkItem; ++i) {
        T tmp = local_mem_ptr[local_start + i];
        local_mem_ptr[local_start + i] = updated_prefix;
        updated_prefix = binary_op_(updated_prefix, tmp);
      }
    } else {
#pragma unroll
      for (int i = 0; i < ElemsPerWorkItem; ++i) {
        T tmp = local_mem_ptr[local_start + i];
        updated_prefix = binary_op_(updated_prefix, tmp);
        local_mem_ptr[local_start + i] = updated_prefix;
      }
    }
    sycl::group_barrier(group);

// write  output
#pragma unroll
    for (int i = lid; start + i < end; i += GroupSize) {
      if (start + i < N_)
        out_data_[MapReversedIndex<IsReverse>(N_, start + i)] =
            local_mem_ptr[i];
    }
    // write internal output
    if (lid == GroupSize - 1) {
      inter_out_[group_id] = updated_prefix;
    }
  }

 private:
  InputIteratorT in_data_;
  OutputIteratorT out_data_;
  OutputIteratorT inter_out_;
  LocalAccessor local_mem_;
  T init_;
  BinaryOp binary_op_;
  size_t N_;
};

template <typename InputIteratorT, typename OutputIteratorT, typename BinaryOp,
          int GroupSize, int ElemsPerWorkItem, bool IsReverse>
struct DeviceScanSecondStep {
  DeviceScanSecondStep(InputIteratorT in_data, OutputIteratorT out_data,
                       BinaryOp binary_op, size_t N)
      : in_data_(in_data), out_data_(out_data), binary_op_(binary_op), N_(N) {}
  void operator()(sycl::nd_item<1> item) const {
    auto group_id = item.get_group_linear_id();
    auto lid = item.get_local_linear_id();
    auto start = group_id * GroupSize * ElemsPerWorkItem;
    auto end = (group_id + 1) * GroupSize * ElemsPerWorkItem;
    end = end < N_ ? end : N_;

    typename std::iterator_traits<InputIteratorT>::value_type carry =
        in_data_[group_id];
    // #pragma unroll
    for (int i = start + lid; i < end; i += GroupSize) {
      out_data_[MapReversedIndex<IsReverse>(N_, i)] =
          binary_op_(out_data_[MapReversedIndex<IsReverse>(N_, i)], carry);
    }
  }

 private:
  InputIteratorT in_data_;
  OutputIteratorT out_data_;
  BinaryOp binary_op_;
  size_t N_;
};

template <typename InputIteratorT, typename OutputIteratorT, typename BinaryOp,
          bool IsExclusive, bool IsReverse>
void _scan_kernel(
    InputIteratorT data, OutputIteratorT result,
    typename std::iterator_traits<InputIteratorT>::value_type init,
    BinaryOp binary_op, const int N, OpKernelContext* context);

template <typename InputIteratorT, typename OutputIteratorT, typename BinaryOp,
          int GroupSize, int ElemsPerWorkItem, bool IsExclusive, bool IsReverse>
void launch_device_scan(
    InputIteratorT in, OutputIteratorT out,
    typename std::iterator_traits<InputIteratorT>::value_type init,
    BinaryOp binary_op, size_t N, int num_work_group,
    OpKernelContext* context) {
  using T = typename std::iterator_traits<InputIteratorT>::value_type;

  Tensor inter_result_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                 TensorShape({num_work_group}),
                                                 &inter_result_tensor));
  T* inter_result = inter_result_tensor.flat<T>().data();

  sycl::nd_range<1> thread_range(num_work_group * GroupSize, GroupSize);
  int scratch_size = GroupSize * ElemsPerWorkItem;

  auto& stream = (context->eigen_gpu_device()).stream();

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc<T> scratch(scratch_size, cgh);
    DeviceScanFirstStep<InputIteratorT, OutputIteratorT, BinaryOp, LocalAcc<T>,
                        GroupSize, ElemsPerWorkItem, IsExclusive, IsReverse>
        task(in, out, inter_result, scratch, init, binary_op, N);
    cgh.parallel_for<DeviceScanFirstStep<
        InputIteratorT, OutputIteratorT, BinaryOp, LocalAcc<T>, GroupSize,
        ElemsPerWorkItem, IsExclusive, IsReverse>>(thread_range, task);
  });

  _scan_kernel<T*, OutputIteratorT, BinaryOp, true, false>(
      inter_result, inter_result, init, binary_op, num_work_group, context);

  stream->submit([&](sycl::handler& cgh) {
    DeviceScanSecondStep<T*, OutputIteratorT, BinaryOp, GroupSize,
                         ElemsPerWorkItem, IsReverse>
        task(inter_result, out, binary_op, N);
    cgh.parallel_for<DeviceScanSecondStep<
        T*, OutputIteratorT, BinaryOp, GroupSize, ElemsPerWorkItem, IsReverse>>(
        thread_range, task);
  });
}

template <typename InputIteratorT, typename OutputIteratorT, typename BinaryOp,
          int GroupSize, int ElemsPerWorkItem, bool IsExclusive, bool IsReverse>
void launch_group_scan(
    InputIteratorT in, OutputIteratorT out,
    typename std::iterator_traits<InputIteratorT>::value_type init,
    BinaryOp binary_op, size_t N, OpKernelContext* context) {
  using T = typename std::iterator_traits<InputIteratorT>::value_type;
  sycl::nd_range<1> thread_range(GroupSize, GroupSize);
  int scratch_size = GroupSize * ElemsPerWorkItem;

  auto& stream = (context->eigen_gpu_device()).stream();

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc<T> scratch(scratch_size, cgh);
    GroupScan<InputIteratorT, OutputIteratorT, BinaryOp, LocalAcc<T>, GroupSize,
              ElemsPerWorkItem, IsExclusive, IsReverse>
        task(in, out, scratch, init, binary_op, N);
    cgh.parallel_for<
        GroupScan<InputIteratorT, OutputIteratorT, BinaryOp, LocalAcc<T>,
                  GroupSize, ElemsPerWorkItem, IsExclusive, IsReverse>>(
        thread_range, task);
  });
}

template <typename InputIteratorT, typename OutputIteratorT, typename BinaryOp,
          bool IsExclusive, bool IsReverse>
void _scan_kernel(
    InputIteratorT data, OutputIteratorT result,
    typename std::iterator_traits<InputIteratorT>::value_type init,
    BinaryOp binary_op, const int N, OpKernelContext* context) {
  constexpr int MaxWorkGroupSize = 512;
  constexpr int SubGroupSize = 32;
  constexpr int ElemsPerWorkItem = 8;
  constexpr int max_elems_per_work_group = MaxWorkGroupSize * ElemsPerWorkItem;

  if (N <= max_elems_per_work_group) {
    if (N > MaxWorkGroupSize) {
      int n = ElemsPerWorkItem;
      while (MaxWorkGroupSize * n >= 2 * N) {
        n >>= 1;
      }
#define HANDLE_N(NUM)                                                 \
  case NUM:                                                           \
    launch_group_scan<InputIteratorT, OutputIteratorT, BinaryOp,      \
                      MaxWorkGroupSize, NUM, IsExclusive, IsReverse>( \
        data, result, init, binary_op, N, context);                   \
    break;
      switch (n) {
        HANDLE_N(8)
        HANDLE_N(4)
        HANDLE_N(2)
        HANDLE_N(1)
        default:
          std::cerr << "error, should never be called" << std::endl;
          exit(-1);
      }
#undef HANDLE_N
    } else {
      int group_size = MaxWorkGroupSize;
      while (group_size >= 2 * N) {
        group_size >>= 1;
      }
      group_size = group_size > SubGroupSize ? group_size : SubGroupSize;
#define HANDLE_N(NUM)                                                        \
  case NUM:                                                                  \
    launch_group_scan<InputIteratorT, OutputIteratorT, BinaryOp, NUM, 1,     \
                      IsExclusive, IsReverse>(data, result, init, binary_op, \
                                              N, context);                   \
    break;
      switch (group_size) {
        HANDLE_N(1024)
        HANDLE_N(512)
        HANDLE_N(256)
        HANDLE_N(128)
        HANDLE_N(64)
        HANDLE_N(32)
        HANDLE_N(16)
        default:
          std::cerr << "error, should never be called" << std::endl;
          exit(-1);
      }
#undef HANDLE_N
    }
    return;
  }

  const int num_work_group =
      (N + max_elems_per_work_group - 1) / max_elems_per_work_group;
  if (num_work_group < 64) {
    int k = ElemsPerWorkItem;
    int tmp_work_group = num_work_group;
    while (tmp_work_group * 2 < 64 && k > 1) {
      tmp_work_group *= 2;
      k >>= 1;
    }
#define HANDLE_N(NUM)                                                  \
  case NUM:                                                            \
    launch_device_scan<InputIteratorT, OutputIteratorT, BinaryOp,      \
                       MaxWorkGroupSize, NUM, IsExclusive, IsReverse>( \
        data, result, init, binary_op, N, tmp_work_group, context);    \
    break;
    switch (k) {
      HANDLE_N(8)
      HANDLE_N(4)
      HANDLE_N(2)
      HANDLE_N(1)
      default:
        std::cerr << "error, should never be called" << std::endl;
        exit(-1);
    }
#undef HANDLE_N
  } else {
    launch_device_scan<InputIteratorT, OutputIteratorT, BinaryOp,
                       MaxWorkGroupSize, ElemsPerWorkItem, IsExclusive,
                       IsReverse>(data, result, init, binary_op, N,
                                  num_work_group, context);
  }
}

}  // namespace impl

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_UNIQUE_OP_HELPERS_H_
