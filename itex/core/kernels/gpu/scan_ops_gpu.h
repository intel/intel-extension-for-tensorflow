/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_SCAN_OPS_GPU_H_
#define ITEX_CORE_KERNELS_GPU_SCAN_OPS_GPU_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace functor {
namespace internal {

template <typename T>
struct Identity {
  inline T operator()(const T& x) const { return x; }
};

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

template <typename T>
using LocalAcc = sycl::accessor<T, 1, sycl::access::mode::read_write,
                                sycl::access::target::local>;

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, typename LocalAccessor, int GroupSize,
          int ElemsPerWorkItem, bool IsExclusive, bool IsReverse,
          typename InputFunctor, typename OutputFunctor>
struct GroupScan {
  GroupScan(InputT* in_data, OutputT* out_data, LocalAccessor local_mem,
            InitValueT init, BinaryOp binary_op, size_t N, InputFunctor in_func,
            OutputFunctor out_func)
      : in_data_(in_data),
        out_data_(out_data),
        local_mem_(local_mem),
        init_(init),
        binary_op_(binary_op),
        N_(N),
        in_func_(in_func),
        out_func_(out_func) {}

  void operator()(sycl::nd_item<1> item) const {
    // Use InitValue as itermediate computation datatype,
    // as required in https://wg21.link/P0571
    typedef InitValueT T;
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    T* local_mem_ptr = local_mem_.get_pointer().get();

    // read data from global memory to SLM
    auto end = GroupSize * ElemsPerWorkItem;
#pragma unroll
    for (int i = lid; i < end; i += GroupSize) {
      if (i < N_)
        local_mem_ptr[i] =
            T(in_func_(in_data_[MapReversedIndex<IsReverse>(N_, i)]));
      else
        local_mem_ptr[i] = init_;
    }
    item.barrier(sycl::access::fence_space::local_space);

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

    item.barrier(sycl::access::fence_space::local_space);

// write  output
#pragma unroll
    for (int i = lid; i < end; i += GroupSize) {
      if (i < N_)
        out_data_[MapReversedIndex<IsReverse>(N_, i)] =
            OutputT(out_func_(local_mem_ptr[i]));
    }
  }

 private:
  InputT* in_data_;
  OutputT* out_data_;
  LocalAccessor local_mem_;
  InitValueT init_;
  BinaryOp binary_op_;
  size_t N_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
};

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, int GroupSize, int ElemsPerWorkItem,
          bool IsExclusive, bool IsReverse, typename InputFunctor,
          typename OutputFunctor>
void launchGroupScan(OpKernelContext* ctx, InputT* in, OutputT* out,
                     InitValueT init, BinaryOp binary_op, size_t N,
                     InputFunctor in_func, OutputFunctor out_func) {
  sycl::nd_range<1> thread_range(GroupSize, GroupSize);
  int scratch_size = GroupSize * ElemsPerWorkItem;
  auto& stream = (ctx->eigen_gpu_device()).stream();
  stream->submit([&](sycl::handler& cgh) {
    LocalAcc<InitValueT> scratch(scratch_size, cgh);
    GroupScan<InputT, OutputT, InitValueT, BinaryOp, LocalAcc<InitValueT>,
              GroupSize, ElemsPerWorkItem, IsExclusive, IsReverse, InputFunctor,
              OutputFunctor>
        task(in, out, scratch, init, binary_op, N, in_func, out_func);
    cgh.parallel_for<GroupScan<
        InputT, OutputT, InitValueT, BinaryOp, LocalAcc<InitValueT>, GroupSize,
        ElemsPerWorkItem, IsExclusive, IsReverse, InputFunctor, OutputFunctor>>(
        thread_range, task);
  });
}

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, typename LocalAccessor, int GroupSize,
          int ElemsPerWorkItem, bool IsExclusive, bool IsReverse,
          typename InputFunctor>
struct DeviceScanFirstStep {
  DeviceScanFirstStep(InputT* in_data, OutputT* out_data, InitValueT* inter_out,
                      LocalAccessor local_mem, InitValueT init,
                      BinaryOp binary_op, size_t N, InputFunctor func)
      : in_data_(in_data),
        out_data_(out_data),
        inter_out_(inter_out),
        local_mem_(local_mem),
        init_(init),
        binary_op_(binary_op),
        N_(N),
        func_(func) {}
  void operator()(sycl::nd_item<1> item) const {
    typedef InitValueT T;
    auto group_id = item.get_group_linear_id();
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    T* local_mem_ptr = local_mem_.get_pointer().get();

    // read data from global memory to slm
    auto start = group_id * GroupSize * ElemsPerWorkItem;
    auto end = (group_id + 1) * GroupSize * ElemsPerWorkItem;

#pragma unroll
    for (int i = lid; start + i < end; i += GroupSize) {
      if (start + i < N_)
        local_mem_ptr[i] =
            T(func_(in_data_[MapReversedIndex<IsReverse>(N_, start + i)]));
      else
        local_mem_ptr[i] = init_;
    }
    item.barrier(sycl::access::fence_space::local_space);

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
    item.barrier(sycl::access::fence_space::local_space);

// write  output
#pragma unroll
    for (int i = lid; start + i < end; i += GroupSize) {
      if (start + i < N_)
        out_data_[MapReversedIndex<IsReverse>(N_, start + i)] =
            OutputT(local_mem_ptr[i]);
    }
    // write internal output
    if (lid == GroupSize - 1) {
      inter_out_[group_id] = updated_prefix;
    }
  }

 private:
  InputT* in_data_;
  OutputT* out_data_;
  InitValueT* inter_out_;
  LocalAccessor local_mem_;
  InitValueT init_;
  BinaryOp binary_op_;
  size_t N_;
  InputFunctor func_;
};

template <typename InputT, typename OutputT, typename BinaryOp, int GroupSize,
          int ElemsPerWorkItem, bool IsReverse, typename OutputFunctor>
struct DeviceScanSecondStep {
  DeviceScanSecondStep(InputT* in_data, OutputT* out_data, BinaryOp binary_op,
                       size_t N, OutputFunctor func)
      : in_data_(in_data), out_data_(out_data), binary_op_(binary_op), N_(N) {}
  void operator()(sycl::nd_item<1> item) const {
    // Use InitValue as itermediate computation datatype,
    // as required in https://wg21.link/P0571
    typedef InputT T;
    auto group_id = item.get_group_linear_id();
    auto lid = item.get_local_linear_id();
    auto start = group_id * GroupSize * ElemsPerWorkItem;
    auto end = (group_id + 1) * GroupSize * ElemsPerWorkItem;
    end = end < N_ ? end : N_;

    T carry = in_data_[group_id];
    // #pragma unroll
    for (int i = start + lid; i < end; i += GroupSize) {
      out_data_[MapReversedIndex<IsReverse>(N_, i)] = OutputT(func(
          binary_op_(T(out_data_[MapReversedIndex<IsReverse>(N_, i)]), carry)));
    }
  }

 private:
  InputT* in_data_;
  OutputT* out_data_;
  BinaryOp binary_op_;
  size_t N_;
  OutputFunctor func;
};

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, bool IsExclusive, bool IsReverse,
          typename InputFunctor, typename OutputFunctor>
void launchFullScanImpl(OpKernelContext* ctx, InputT* in, OutputT* out,
                        InitValueT init, BinaryOp binary_op, const int N,
                        InputFunctor in_func, OutputFunctor out_func);

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, int GroupSize, int ElemsPerWorkItem,
          bool IsExclusive, bool IsReverse, typename InputFunctor,
          typename OutputFunctor>
void launchDeviceScan(OpKernelContext* ctx, InputT* in, OutputT* out,
                      InitValueT init, BinaryOp binary_op, size_t N,
                      int num_work_group, InputFunctor in_func,
                      OutputFunctor out_func) {
  Tensor inter_result_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<InitValueT>::value,
                                         TensorShape({num_work_group}),
                                         &inter_result_tensor));
  auto inter_result = inter_result_tensor.flat<InitValueT>().data();

  sycl::nd_range<1> thread_range(num_work_group * GroupSize, GroupSize);
  int scratch_size = GroupSize * ElemsPerWorkItem;
  auto& stream = (ctx->eigen_gpu_device()).stream();

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc<InitValueT> scratch(scratch_size, cgh);
    DeviceScanFirstStep<InputT, OutputT, InitValueT, BinaryOp,
                        LocalAcc<InitValueT>, GroupSize, ElemsPerWorkItem,
                        IsExclusive, IsReverse, InputFunctor>
        task(in, out, inter_result, scratch, init, binary_op, N, in_func);
    cgh.parallel_for<DeviceScanFirstStep<
        InputT, OutputT, InitValueT, BinaryOp, LocalAcc<InitValueT>, GroupSize,
        ElemsPerWorkItem, IsExclusive, IsReverse, InputFunctor>>(thread_range,
                                                                 task);
  });

  launchFullScanImpl<InitValueT, InitValueT, InitValueT, BinaryOp, true, false,
                     Identity<InitValueT>, Identity<InitValueT>>(
      ctx, inter_result, inter_result, init, binary_op, num_work_group,
      Identity<InitValueT>(), Identity<InitValueT>());

  stream->submit([&](sycl::handler& cgh) {
    DeviceScanSecondStep<InitValueT, OutputT, BinaryOp, GroupSize,
                         ElemsPerWorkItem, IsReverse, OutputFunctor>
        task(inter_result, out, binary_op, N, out_func);
    cgh.parallel_for<
        DeviceScanSecondStep<InitValueT, OutputT, BinaryOp, GroupSize,
                             ElemsPerWorkItem, IsReverse, OutputFunctor>>(
        thread_range, task);
  });
}

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, bool IsExclusive, bool IsReverse,
          typename InputFunctor, typename OutputFunctor>
void launchFullScanImpl(OpKernelContext* ctx, InputT* in, OutputT* out,
                        InitValueT init, BinaryOp binary_op, const int N,
                        InputFunctor in_func, OutputFunctor out_func) {
  constexpr int MaxWorkGroupSize = 512;
  constexpr int SubGroupSize = 32;
  constexpr int ElemsPerWorkItem = 8;
  constexpr int max_elems_per_work_group = MaxWorkGroupSize * ElemsPerWorkItem;
  // TODO(itex): remove this hard code once compiler can return right number of
  // compute unit
  constexpr int NumSSOnPVC = 64;

  if (N <= max_elems_per_work_group) {
    if (N > MaxWorkGroupSize) {
      int n = ElemsPerWorkItem;
      while (MaxWorkGroupSize * n >= 2 * N) {
        n >>= 1;
      }
#define HANDLE_N(NUM)                                                          \
  case NUM:                                                                    \
    launchGroupScan<InputT, OutputT, InitValueT, BinaryOp, MaxWorkGroupSize,   \
                    NUM, IsExclusive, IsReverse, InputFunctor, OutputFunctor>( \
        ctx, in, out, init, binary_op, N, in_func, out_func);                  \
    break;
      switch (n) {
        HANDLE_N(8)
        HANDLE_N(4)
        HANDLE_N(2)
        HANDLE_N(1)
        default:
          ITEX_LOG(ERROR) << "error, should never be called";
      }
#undef HANDLE_N
    } else {
      int group_size = MaxWorkGroupSize;
      while (group_size >= 2 * N) {
        group_size >>= 1;
      }
      group_size = group_size > SubGroupSize ? group_size : SubGroupSize;
#define HANDLE_N(NUM)                                                     \
  case NUM:                                                               \
    launchGroupScan<InputT, OutputT, InitValueT, BinaryOp, NUM, 1,        \
                    IsExclusive, IsReverse, InputFunctor, OutputFunctor>( \
        ctx, in, out, init, binary_op, N, in_func, out_func);             \
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
          ITEX_LOG(ERROR) << "error, should never be called";
      }
#undef HANDLE_N
    }
    return;
  }

  const int num_work_group =
      (N + max_elems_per_work_group - 1) / max_elems_per_work_group;
  if (num_work_group < NumSSOnPVC) {
    int k = ElemsPerWorkItem;
    int tmp_work_group = num_work_group;
    while (tmp_work_group * 2 < NumSSOnPVC && k > 1) {
      tmp_work_group *= 2;
      k >>= 1;
    }
#define HANDLE_N(NUM)                                                         \
  case NUM:                                                                   \
    launchDeviceScan<InputT, OutputT, InitValueT, BinaryOp, MaxWorkGroupSize, \
                     NUM, IsExclusive, IsReverse, InputFunctor,               \
                     OutputFunctor>(ctx, in, out, init, binary_op, N,         \
                                    tmp_work_group, in_func, out_func);       \
    break;
    switch (k) {
      HANDLE_N(8)
      HANDLE_N(4)
      HANDLE_N(2)
      HANDLE_N(1)
      default:
        ITEX_LOG(ERROR) << "error, should never be called";
    }
#undef HANDLE_N
  } else {
    launchDeviceScan<InputT, OutputT, InitValueT, BinaryOp, MaxWorkGroupSize,
                     ElemsPerWorkItem, IsExclusive, IsReverse, InputFunctor,
                     OutputFunctor>(ctx, in, out, init, binary_op, N,
                                    num_work_group, in_func, out_func);
  }
}
}  // namespace internal

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, typename InputFunctor = internal::Identity<InputT>,
          typename OutputFunctor = internal::Identity<InitValueT>>
void launchFullScan(OpKernelContext* ctx, InputT* in, OutputT* out,
                    InitValueT init, BinaryOp binary_op,
                    const bool is_exclusive, const bool is_reverse, const int N,
                    InputFunctor in_func = internal::Identity<InputT>(),
                    OutputFunctor out_func = internal::Identity<InitValueT>()) {
  if (is_exclusive) {
    if (is_reverse)
      internal::launchFullScanImpl<InputT, OutputT, InitValueT, BinaryOp, true,
                                   true, InputFunctor, OutputFunctor>(
          ctx, in, out, init, binary_op, N, in_func, out_func);
    else
      internal::launchFullScanImpl<InputT, OutputT, InitValueT, BinaryOp, true,
                                   false, InputFunctor, OutputFunctor>(
          ctx, in, out, init, binary_op, N, in_func, out_func);
  } else {
    if (is_reverse)
      internal::launchFullScanImpl<InputT, OutputT, InitValueT, BinaryOp, false,
                                   true, InputFunctor, OutputFunctor>(
          ctx, in, out, init, binary_op, N, in_func, out_func);
    else
      internal::launchFullScanImpl<InputT, OutputT, InitValueT, BinaryOp, false,
                                   false, InputFunctor, OutputFunctor>(
          ctx, in, out, init, binary_op, N, in_func, out_func);
  }
}

}  // namespace functor

namespace functor {

namespace internal {
template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, int GroupSize, int SubGroupSize,
          int ElemsPerWorkItem, bool IsExclusive, bool IsReverse,
          typename InputFunctor, typename OutputFunctor>
struct VanillaPartialScan {
  VanillaPartialScan(InputT* in_data, OutputT* out_data, InitValueT init,
                     BinaryOp binary_op, const int num_outer,
                     const int num_scaned, const int num_inner,
                     InputFunctor in_func, OutputFunctor out_func)
      : in_data_(in_data),
        out_data_(out_data),
        init_(init),
        binary_op_(binary_op),
        num_outer_(num_outer),
        num_scaned_(num_scaned),
        num_inner_(num_inner),
        in_func_(in_func),
        out_func_(out_func) {}

  void operator()(sycl::nd_item<1> item) const {
    // Use InitValue as itermediate computation datatype,
    // as required in https://wg21.link/P0571
    typedef InitValueT T;
    auto group_id = item.get_group_linear_id();
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();

    int group_id_outer = group_id / num_inner_;
    int group_id_inner = group_id - group_id_outer * num_inner_;
    int group_start =
        group_id_outer * num_scaned_ * num_inner_ + group_id_inner;

    // read data from global memory to SLM
    auto group_elems = GroupSize * ElemsPerWorkItem;
    int num_loops = (num_scaned_ + group_elems - 1) / group_elems;
    T carry = init_;
    for (int loop = 0; loop < num_loops; ++loop) {
      T local_data[ElemsPerWorkItem];
#pragma unroll
      for (int i = 0; i < ElemsPerWorkItem; ++i) {
        int start = loop * group_elems + lid * ElemsPerWorkItem + i;
        if (start < num_scaned_)
          local_data[i] =
              T(in_func_(in_data_[group_start + MapReversedIndex<IsReverse>(
                                                    num_scaned_, start) *
                                                    num_inner_]));
        else
          local_data[i] = init_;
      }

      // reduce
      T prefix = init_;
#pragma unroll
      for (int i = 0; i < ElemsPerWorkItem; ++i) {
        prefix = binary_op_(prefix, local_data[i]);
      }

      // scan
      T updated_prefix =
          sycl::exclusive_scan_over_group(group, prefix, binary_op_);
      updated_prefix = binary_op_(updated_prefix, carry);

      // down sweep
      if (IsExclusive) {
#pragma unroll
        for (int i = 0; i < ElemsPerWorkItem; ++i) {
          T tmp = local_data[i];
          local_data[i] = updated_prefix;
          updated_prefix = binary_op_(updated_prefix, tmp);
        }
      } else {
#pragma unroll
        for (int i = 0; i < ElemsPerWorkItem; ++i) {
          T tmp = local_data[i];
          updated_prefix = binary_op_(updated_prefix, tmp);
          local_data[i] = updated_prefix;
        }
      }

      carry = sycl::group_broadcast(group, updated_prefix, GroupSize - 1);

// write  output
#pragma unroll
      for (int i = 0; i < ElemsPerWorkItem; ++i) {
        int start = loop * group_elems + lid * ElemsPerWorkItem + i;
        if (start < num_scaned_)
          out_data_[group_start +
                    MapReversedIndex<IsReverse>(num_scaned_, start) *
                        num_inner_] = OutputT(out_func_(local_data[i]));
      }
    }
  }

 private:
  InputT* in_data_;
  OutputT* out_data_;
  InitValueT init_;
  BinaryOp binary_op_;
  const int num_outer_;
  const int num_scaned_;
  const int num_inner_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
};

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, int GroupSize, int SubGroupSize,
          int ElemsPerWorkItem, bool IsExclusive, bool IsReverse,
          typename InputFunctor, typename OutputFunctor>
void vanillaPartialScanFunc(OpKernelContext* ctx, InputT* in, OutputT* out,
                            InitValueT init, BinaryOp binary_op,
                            const int num_outer, const int num_scaned,
                            const int num_inner, InputFunctor in_func,
                            OutputFunctor out_func) {
  const int num_preserved = num_outer * num_inner;
  sycl::nd_range<1> thread_range(num_preserved * GroupSize, GroupSize);
  auto& stream = (ctx->eigen_gpu_device()).stream();
  stream->submit([&](sycl::handler& cgh) {
    VanillaPartialScan<InputT, OutputT, InitValueT, BinaryOp, GroupSize,
                       SubGroupSize, ElemsPerWorkItem, IsExclusive, IsReverse,
                       InputFunctor, OutputFunctor>
        task(in, out, init, binary_op, num_outer, num_scaned, num_inner,
             in_func, out_func);
    cgh.parallel_for<VanillaPartialScan<
        InputT, OutputT, InitValueT, BinaryOp, GroupSize, SubGroupSize,
        ElemsPerWorkItem, IsExclusive, IsReverse, InputFunctor, OutputFunctor>>(
        thread_range, task);
  });
}

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, bool IsExclusive, bool IsReverse,
          typename InputFunctor, typename OutputFunctor>
void launchVanillaPartialScanKernel(OpKernelContext* ctx, InputT* in_data,
                                    OutputT* out_data, InitValueT init,
                                    BinaryOp binary_op, const int num_outer,
                                    const int num_scaned, const int num_inner,
                                    InputFunctor in_func,
                                    OutputFunctor out_func) {
  constexpr int MaxWorkGroupSize = 512;
  constexpr int SubGroupSize = 32;
  constexpr int ElemsPerWorkItem = 8;
  constexpr int max_elems_per_work_group = MaxWorkGroupSize * ElemsPerWorkItem;
  if (num_scaned >= max_elems_per_work_group) {
    vanillaPartialScanFunc<InputT, OutputT, InitValueT, BinaryOp,
                           MaxWorkGroupSize, SubGroupSize, ElemsPerWorkItem,
                           IsExclusive, IsReverse, InputFunctor, OutputFunctor>(
        ctx, in_data, out_data, init, binary_op, num_outer, num_scaned,
        num_inner, in_func, out_func);
  } else {
    if (num_scaned > MaxWorkGroupSize) {
      int n = ElemsPerWorkItem;
      while (MaxWorkGroupSize * n >= 2 * num_scaned) {
        n >>= 1;
      }
#define HANDLE_N(NUM)                                                        \
  case NUM:                                                                  \
    vanillaPartialScanFunc<InputT, OutputT, InitValueT, BinaryOp,            \
                           MaxWorkGroupSize, SubGroupSize, NUM, IsExclusive, \
                           IsReverse, InputFunctor, OutputFunctor>(          \
        ctx, in_data, out_data, init, binary_op, num_outer, num_scaned,      \
        num_inner, in_func, out_func);                                       \
    break;
      switch (n) {
        HANDLE_N(8)
        HANDLE_N(4)
        HANDLE_N(2)
        HANDLE_N(1)
        default:
          ITEX_LOG(ERROR) << "error, should never be called";
      }
#undef HANDLE_N
    } else {
      int group_size = MaxWorkGroupSize;
      while (group_size >= 2 * num_scaned) {
        group_size >>= 1;
      }
      group_size = group_size > SubGroupSize ? group_size : SubGroupSize;
#define HANDLE_N(NUM)                                                   \
  case NUM:                                                             \
    vanillaPartialScanFunc<InputT, OutputT, InitValueT, BinaryOp, NUM,  \
                           SubGroupSize, 1, IsExclusive, IsReverse,     \
                           InputFunctor, OutputFunctor>(                \
        ctx, in_data, out_data, init, binary_op, num_outer, num_scaned, \
        num_inner, in_func, out_func);                                  \
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
          ITEX_LOG(ERROR) << "error, should never be called";
      }
#undef HANDLE_N
    }
  }
}

// do scan along row axis
template <typename T, typename Item, typename BinaryOp, int SubGroupSize,
          int NumSubGroup>
void slmScan(Item item, T* array, T* carry_array, const T* old_data,
             T* updated_data, const T init, BinaryOp binary_op) {
  constexpr int k = NumSubGroup / SubGroupSize;
  auto sg_group = item.get_sub_group();
  int sg_id = sg_group.get_group_linear_id();
  int lid_in_sg = sg_group.get_local_linear_id();

  array[sg_id + lid_in_sg * NumSubGroup] = *old_data;
  item.barrier(sycl::access::fence_space::local_space);

  // ----------------------------
  // Three stage scan in SLM
  // ----------------------------

  // Frist stage:  sub group scan
  int offset = sg_id * SubGroupSize;
  const T data = array[offset + lid_in_sg];
  T exclusive_sum = sycl::exclusive_scan_over_group(sg_group, data, binary_op);

  // Second stage: compute subgroup prefix
  if (lid_in_sg == SubGroupSize - 1) carry_array[sg_id] = exclusive_sum + data;
  item.barrier(sycl::access::fence_space::local_space);

  if (sg_id == 0) {
    int offset = lid_in_sg * k;
    T carry = init;
    for (int i = 0; i < k; ++i) {
      T tmp = carry_array[offset + i];
      carry_array[offset + i] = carry;
      carry = binary_op(carry, tmp);
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  array[offset + lid_in_sg] = binary_op(exclusive_sum, carry_array[sg_id]);
  item.barrier(sycl::access::fence_space::local_space);

  *updated_data = array[sg_id + lid_in_sg * NumSubGroup];
}

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, typename LocalAccessor, int GroupSize,
          int SubGroupSize, bool IsExclusive, bool IsReverse,
          typename InputFunctor, typename OutputFunctor>
struct OptimizedOuterScan {
  OptimizedOuterScan(InputT* in_data, OutputT* out_data,
                     LocalAccessor local_mem, LocalAccessor local_mem_carry,
                     InitValueT init, BinaryOp binary_op, const int num_outer,
                     const int num_scaned, const int num_inner,
                     const int elems_per_work_item, const int k,
                     InputFunctor in_func, OutputFunctor out_func)
      : in_data_(in_data),
        out_data_(out_data),
        local_mem_(local_mem),
        local_mem_carry_(local_mem_carry),
        init_(init),
        binary_op_(binary_op),
        num_outer_(num_outer),
        num_scaned_(num_scaned),
        num_inner_(num_inner),
        elems_per_work_item_(elems_per_work_item),
        k_(k),
        in_func_(in_func),
        out_func_(out_func) {}

  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<1> item) const {
    // Use InitValue as itermediate computation datatype,
    // as required in https://wg21.link/P0571
    typedef InitValueT T;
    auto group_id = item.get_group_linear_id();
    auto sg_group = item.get_sub_group();

    constexpr int NumSubGroup = GroupSize / SubGroupSize;
    int sg_id = sg_group.get_group_linear_id();
    int lid_in_sg = sg_group.get_local_linear_id();

    int outer_group_id = group_id / (num_inner_ / SubGroupSize);
    int inner_group_id =
        group_id - outer_group_id * (num_inner_ / SubGroupSize);
    int g_offset = outer_group_id * num_scaned_ * num_inner_ +
                   inner_group_id * SubGroupSize;

    // Each SubGroup load data, do reduce, then store data to SLM
    T reduce_sum = init_;
#pragma unroll(4)
    for (int i = 0; i < elems_per_work_item_; ++i) {
      int scan_id = sg_id * elems_per_work_item_ + i;
      if (scan_id >= num_scaned_) break;
      scan_id = MapReversedIndex<IsReverse>(num_scaned_, scan_id);
      T data =
          T(in_func_(in_data_[g_offset + scan_id * num_inner_ + lid_in_sg]));
      reduce_sum = binary_op_(reduce_sum, data);
    }

    T updated_reduce_sum = init_;
    slmScan<T, sycl::nd_item<1>, BinaryOp, SubGroupSize, NumSubGroup>(
        item, local_mem_.get_pointer().get(),
        local_mem_carry_.get_pointer().get(), &reduce_sum, &updated_reduce_sum,
        init_, binary_op_);

    // Each SubGroup do interanl scan and store data to Global Memory
    if (IsExclusive) {
#pragma unroll(4)
      for (int i = 0; i < elems_per_work_item_; ++i) {
        int scan_id = MapReversedIndex<IsReverse>(
            num_scaned_, sg_id * elems_per_work_item_ + i);
        if (scan_id >= num_scaned_) break;
        T old_data =
            T(in_func_(in_data_[g_offset + scan_id * num_inner_ + lid_in_sg]));
        out_data_[g_offset + scan_id * num_inner_ + lid_in_sg] =
            OutputT(out_func_(updated_reduce_sum));
        updated_reduce_sum = binary_op_(updated_reduce_sum, old_data);
      }
    } else {
#pragma unroll(4)
      for (int i = 0; i < elems_per_work_item_; ++i) {
        int scan_id = sg_id * elems_per_work_item_ + i;
        if (scan_id >= num_scaned_) break;
        scan_id = MapReversedIndex<IsReverse>(num_scaned_, scan_id);
        T old_data =
            T(in_func_(in_data_[g_offset + scan_id * num_inner_ + lid_in_sg]));
        updated_reduce_sum = binary_op_(updated_reduce_sum, old_data);
        out_data_[g_offset + scan_id * num_inner_ + lid_in_sg] =
            OutputT(out_func_(updated_reduce_sum));
      }
    }
  }

 private:
  InputT* in_data_;
  OutputT* out_data_;
  LocalAccessor local_mem_;
  LocalAccessor local_mem_carry_;
  InitValueT init_;
  BinaryOp binary_op_;
  const int num_outer_;
  const int num_scaned_;
  const int num_inner_;
  const int elems_per_work_item_;
  const int k_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
};

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, int GroupSize, int SubGroupSize, bool IsExclusive,
          bool IsReverse, typename InputFunctor, typename OutputFunctor>
void optimizedOuterScanKernelFunc(OpKernelContext* ctx, InputT* in,
                                  OutputT* out, InitValueT init,
                                  BinaryOp binary_op, const int num_outer,
                                  const int num_scaned, const int num_inner,
                                  const int elems_per_work_item,
                                  InputFunctor in_func,
                                  OutputFunctor out_func) {
  constexpr int NumSubGroup = GroupSize / SubGroupSize;
  constexpr int k = NumSubGroup / SubGroupSize;
  const int num_preserved = num_outer * num_inner;
  sycl::nd_range<1> thread_range(num_preserved / SubGroupSize * GroupSize,
                                 GroupSize);

  auto& stream = (ctx->eigen_gpu_device()).stream();
  stream->submit([&](sycl::handler& cgh) {
    sycl::accessor<InitValueT, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        scratch(SubGroupSize * NumSubGroup, cgh);
    sycl::accessor<InitValueT, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        scratch_carry(SubGroupSize * k, cgh);
    OptimizedOuterScan<
        InputT, OutputT, InitValueT, BinaryOp,
        sycl::accessor<InitValueT, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>,
        GroupSize, SubGroupSize, IsExclusive, IsReverse, InputFunctor,
        OutputFunctor>
        task(in, out, scratch, scratch_carry, init, binary_op, num_outer,
             num_scaned, num_inner, elems_per_work_item, k, in_func, out_func);
    cgh.parallel_for<OptimizedOuterScan<
        InputT, OutputT, InitValueT, BinaryOp,
        sycl::accessor<InitValueT, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>,
        GroupSize, SubGroupSize, IsExclusive, IsReverse, InputFunctor,
        OutputFunctor>>(thread_range, task);
  });
}

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, int GroupSize, int SubGroupSize, bool IsExclusive,
          bool IsReverse, typename InputFunctor, typename OutputFunctor>
void launchOptimizedPartialScanKernel(OpKernelContext* ctx, InputT* in_data,
                                      OutputT* out_data, InitValueT init,
                                      BinaryOp binary_op, const int num_outer,
                                      const int num_scaned, const int num_inner,
                                      InputFunctor in_func,
                                      OutputFunctor out_func) {
  constexpr int NumSubGroup = GroupSize / SubGroupSize;
  const int elems_per_work_item = (num_scaned + NumSubGroup - 1) / NumSubGroup;

  optimizedOuterScanKernelFunc<InputT, OutputT, InitValueT, BinaryOp, GroupSize,
                               SubGroupSize, IsExclusive, IsReverse,
                               InputFunctor, OutputFunctor>(
      ctx, in_data, out_data, init, binary_op, num_outer, num_scaned, num_inner,
      elems_per_work_item, in_func, out_func);
}

// Always use one workgroup to computele scan along one dimemsion
template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, bool is_exclusive, bool is_reverse,
          typename InputFunctor, typename OutputFunctor>
void launchPartialScanImpl(OpKernelContext* ctx, InputT* in_data,
                           OutputT* out_data, InitValueT init,
                           BinaryOp binary_op, const int num_outer,
                           const int num_scaned, const int num_inner,
                           InputFunctor in_func, OutputFunctor out_func) {
  auto stream = ctx->GetDeviceStream();
  int sub_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::sub_group_sizes>()
          .back();
  int max_work_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();

  if (sub_group_size >= 32 && max_work_group_size >= 1024)
    sub_group_size = 32;
  else if (sub_group_size >= 16 && max_work_group_size >= 256)
    sub_group_size = 16;
  else
    ITEX_LOG(ERROR)
        << "Not implemented for hardware with maximum subgroup size lesser "
           "than 16 or max work group size lesser than 256";

  constexpr int NumSSOnPVC = 64;
  const int num_preserved = num_outer * num_inner;
  if (num_inner % sub_group_size != 0 ||
      num_preserved / sub_group_size < NumSSOnPVC) {
    ITEX_VLOG(3) << "Using vanilla scan kernel";
    launchVanillaPartialScanKernel<InputT, OutputT, InitValueT, BinaryOp,
                                   is_exclusive, is_reverse, InputFunctor,
                                   OutputFunctor>(
        ctx, in_data, out_data, init, binary_op, num_outer, num_scaned,
        num_inner, in_func, out_func);
  } else {
    ITEX_VLOG(3) << "using optimized scan kernel";
    if (sub_group_size == 32)
      launchOptimizedPartialScanKernel<InputT, OutputT, InitValueT, BinaryOp,
                                       1024, 32, is_exclusive, is_reverse,
                                       InputFunctor, OutputFunctor>(
          ctx, in_data, out_data, init, binary_op, num_outer, num_scaned,
          num_inner, in_func, out_func);
    else if (max_work_group_size >= 512)
      launchOptimizedPartialScanKernel<InputT, OutputT, InitValueT, BinaryOp,
                                       512, 16, is_exclusive, is_reverse,
                                       InputFunctor, OutputFunctor>(
          ctx, in_data, out_data, init, binary_op, num_outer, num_scaned,
          num_inner, in_func, out_func);
    else
      launchOptimizedPartialScanKernel<InputT, OutputT, InitValueT, BinaryOp,
                                       256, 16, is_exclusive, is_reverse,
                                       InputFunctor, OutputFunctor>(
          ctx, in_data, out_data, init, binary_op, num_outer, num_scaned,
          num_inner, in_func, out_func);
  }
}
}  // namespace internal

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, typename InputFunctor = internal::Identity<InputT>,
          typename OutputFunctor = internal::Identity<InitValueT>>
void launchPartialScan(
    OpKernelContext* ctx, InputT* in, OutputT* out, InitValueT init,
    BinaryOp binary_op, const bool is_exclusive, const bool is_reverse,
    const int num_outer, const int num_scaned, const int num_inner,
    InputFunctor in_func = internal::Identity<InputT>(),
    OutputFunctor out_func = internal::Identity<InitValueT>()) {
  if (is_exclusive) {
    if (is_reverse)
      internal::launchPartialScanImpl<InputT, OutputT, InitValueT, BinaryOp,
                                      true, true, InputFunctor, OutputFunctor>(
          ctx, in, out, init, binary_op, num_outer, num_scaned, num_inner,
          in_func, out_func);
    else
      internal::launchPartialScanImpl<InputT, OutputT, InitValueT, BinaryOp,
                                      true, false, InputFunctor, OutputFunctor>(
          ctx, in, out, init, binary_op, num_outer, num_scaned, num_inner,
          in_func, out_func);
  } else {
    if (is_reverse)
      internal::launchPartialScanImpl<InputT, OutputT, InitValueT, BinaryOp,
                                      false, true, InputFunctor, OutputFunctor>(
          ctx, in, out, init, binary_op, num_outer, num_scaned, num_inner,
          in_func, out_func);
    else
      internal::launchPartialScanImpl<InputT, OutputT, InitValueT, BinaryOp,
                                      false, false, InputFunctor,
                                      OutputFunctor>(
          ctx, in, out, init, binary_op, num_outer, num_scaned, num_inner,
          in_func, out_func);
  }
}

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SCAN_OPS_GPU_H_
