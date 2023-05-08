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

#include "itex/core/kernels/gpu/argmax_op.h"

#include <algorithm>
#include <limits>
#include <type_traits>

#include "itex/core/kernels/gpu/reduction_utils.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPEC(T)                            \
  template struct ArgMaxFunctor<GPUDevice, T, int64>; \
  template struct ArgMaxFunctor<GPUDevice, T, int32>; \
  template struct ArgMinFunctor<GPUDevice, T, int64>; \
  template struct ArgMinFunctor<GPUDevice, T, int32>;

DEFINE_GPU_SPEC(float);
DEFINE_GPU_SPEC(Eigen::half);
DEFINE_GPU_SPEC(bool);
#ifdef ITEX_ENABLE_DOUBLE
DEFINE_GPU_SPEC(double);
#endif  // ITEX_ENABLE_DOUBLE

#undef DECLARE_GPU_SPEC

template <typename T>
using __slm__ = sycl::accessor<T, 1, sycl::access::mode::read_write,
                               sycl::access::target::local>;

template <typename T>
struct Identity {
  inline T operator()(const T& x) const { return x; }
};

template <typename T>
struct Tuple {
  using ValueType = T;

  T value;
  int index;

  Tuple(T v, int index) : value(v), index(index) {}
  Tuple(Tuple<T> v, int index) : value(v.value), index(v.index) {}
};

template <typename T>
bool operator>(const Tuple<T>& a, const Tuple<T>& b) {
  if (a.value > b.value) {
    return true;
  } else if (a.value == b.value) {
    return a.index < b.index;
  } else {
    return false;
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, Tuple<T>& v) {
  stream << "Tuple(" << v.value << "," << v.index << ")";
  return stream;
}

struct TupleAssign {};
struct ValueAssign {};
struct IndexAssign {};

template <typename ResultType, typename ValueType, typename = void>
struct AssignTrait {
  using Tag = IndexAssign;
};

template <typename ResultType, typename ValueType>
struct AssignTrait<
    ResultType, ValueType,
    typename std::enable_if<std::is_class<ResultType>::value>::type> {
  using Tag = TupleAssign;
};

template <typename ResultType, typename ValueType>
struct AssignTrait<
    ResultType, ValueType,
    typename std::enable_if<std::is_arithmetic<ValueType>::value>::type> {
  using Tag = ValueAssign;
};

template <typename ResultType, typename ValueType>
inline void AssignInternal(ResultType* a, ValueType* b, IndexAssign) {
  *a = b->index;
}

template <typename ResultType, typename ValueType>
inline void AssignInternal(ResultType* a, ValueType* b, TupleAssign) {
  *a = *b;
}

template <typename ResultType, typename ValueType>
inline void AssignInternal(ResultType* a, ValueType* b, ValueAssign) {
  *a = *b;
}

template <typename ResultType, typename ValueType>
inline void Assign(ResultType* a, ValueType* b) {
  AssignInternal(a, b, typename AssignTrait<ResultType, ValueType>::Tag{});
}

namespace internal {
template <typename Group>
void group_barrier(Group group,
                   sycl::memory_scope FenceScope = Group::fence_scope) {
  if (FenceScope == sycl::memory_scope::work_group) {
    uint32_t flags = static_cast<uint32_t>(
        __spv::MemorySemanticsMask::SequentiallyConsistent |
        __spv::MemorySemanticsMask::WorkgroupMemory);
    __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                           flags);
  } else {
    sycl::group_barrier(group, FenceScope);
  }
}

template <typename Group, typename OutputT, typename Op>
void reduce_over_group(Group group, __slm__<OutputT> local_data, int group_size,
                       int local_id, Op op) {
#define REDUCE_HALF(LENGTH)                                            \
  if (group_size >= LENGTH) {                                          \
    if (local_id < LENGTH / 2) {                                       \
      local_data[local_id] =                                           \
          op(local_data[local_id], local_data[local_id + LENGTH / 2]); \
    }                                                                  \
    internal::group_barrier(group);                                    \
  }

  REDUCE_HALF(512);
  REDUCE_HALF(256);
  REDUCE_HALF(128);
  REDUCE_HALF(64);
#undef REDUCE_HALF

  if (local_id == 0) {
#pragma unroll
    for (int i = 1; i < 32; i++) {
      local_data[0] = op(local_data[0], local_data[i]);
    }
  }
}
}  // namespace internal

struct LegacyMode {};
struct VectorMode {};

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, int ITEMS_PER_ITEM>
void ConsumRange(sycl::nd_item<1> item, InputT* in_data,
                 __slm__<InitValueT> local_data, OutputT* out_data,
                 InitValueT init, int in_size, int elems_per_group, BinaryOp op,
                 LegacyMode) {
  auto lid = item.get_local_id(0);
  auto g = item.get_group();
  auto group_id = item.get_group(0);
  int group_size = item.get_local_range(0);

  int group_start = group_id * elems_per_group;
  if (group_start >= in_size) return;

  int group_end = std::min(group_start + elems_per_group, in_size);
  InitValueT sum = init;
  for (int index = group_start + lid; index < group_end; index += group_size) {
    sum = op(sum, InitValueT(in_data[index], index));
  }

  local_data[lid] = sum;
  internal::group_barrier(g);

  internal::reduce_over_group(g, local_data, group_size, lid, op);
  if (lid == 0) {
    Assign(out_data + group_id, &(local_data[0]));
  }
}

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, int ITEMS_PER_ITEM>
void ConsumRange(sycl::nd_item<1> item, InputT* in_data,
                 __slm__<InitValueT> local_data, OutputT* out_data,
                 InitValueT init, int in_size, int elems_per_group, BinaryOp op,
                 VectorMode) {
  enum {
    VEC_LENGTH = 4,
    WORDS = ITEMS_PER_ITEM / VEC_LENGTH,
  };

  auto lid = item.get_local_id(0);
  auto g = item.get_group();
  auto group_id = item.get_group(0);
  auto group_size = item.get_local_range(0);

  using VecT = sycl::vec<InputT, VEC_LENGTH>;
  InputT input_items[ITEMS_PER_ITEM];
  VecT* vec_items = reinterpret_cast<VecT*>(input_items);

  int group_start = group_id * elems_per_group;

  InitValueT aggregate = init;
  int loops = elems_per_group / (group_size * ITEMS_PER_ITEM);
  for (int l = 0; l < loops; ++l) {
    int start_offset = group_start + l * group_size * ITEMS_PER_ITEM;
    VecT* vec_in =
        reinterpret_cast<VecT*>(in_data + start_offset + (lid * VEC_LENGTH));

#pragma unroll
    for (int i = 0; i < WORDS; ++i) {
      vec_items[i] = vec_in[i * group_size];
    }

#pragma unroll
    for (int i = 0; i < WORDS; ++i) {
      for (int j = 0; j < VEC_LENGTH; j++) {
        int index =
            start_offset + (lid * VEC_LENGTH) + i * group_size * VEC_LENGTH + j;
        aggregate =
            op(aggregate, InitValueT(input_items[i * VEC_LENGTH + j], index));
      }
    }
  }

  local_data[lid] = aggregate;
  internal::group_barrier(g);

  internal::reduce_over_group(g, local_data, group_size, lid, op);
  if (lid == 0) {
    Assign(out_data + group_id, &(local_data[0]));
  }
}

template <typename InputT, typename OutputT, typename InitValueT, typename Op,
          int ITEMS_PER_ITEM, typename = void>
struct GroupReduceKernel {
  GroupReduceKernel(InputT* in_data, __slm__<InitValueT> local_data,
                    OutputT* out_data, InitValueT init_val, int in_size,
                    int elems_per_group, Op op)
      : in_data_(in_data),
        local_data_(local_data),
        out_data_(out_data),
        init_val_(init_val),
        in_size_(in_size),
        elems_per_group_(elems_per_group),
        op_(op) {}

  void operator()(sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    if ((group_id + 1) * elems_per_group_ > in_size_) {
      ConsumRange<InputT, OutputT, InitValueT, Op, ITEMS_PER_ITEM>(
          item, in_data_, local_data_, out_data_, init_val_, in_size_,
          elems_per_group_, op_, LegacyMode());
    } else {
      ConsumRange<InputT, OutputT, InitValueT, Op, ITEMS_PER_ITEM>(
          item, in_data_, local_data_, out_data_, init_val_, in_size_,
          elems_per_group_, op_, VectorMode());
    }
  }

 private:
  InputT* in_data_;
  __slm__<InitValueT> local_data_;
  OutputT* out_data_;
  InitValueT init_val_;
  int in_size_;
  int elems_per_group_;
  Op op_;
};

template <typename InputT, typename OutputT, typename InitValueT, typename Op,
          int ITEMS_PER_ITEM>
struct GroupReduceKernel<
    InputT, OutputT, InitValueT, Op, ITEMS_PER_ITEM,
    typename std::enable_if<
        std::is_class<InputT>::value ||
        std::is_same<InputT, Eigen::bfloat16>::value>::type> {
  GroupReduceKernel(InputT* in_data, __slm__<InitValueT> local_data,
                    OutputT* out_data, InitValueT init_val, int in_size,
                    int elems_per_group, Op op)
      : in_data_(in_data),
        local_data_(local_data),
        out_data_(out_data),
        init_val_(init_val),
        in_size_(in_size),
        elems_per_group_(elems_per_group),
        op_(op) {}

  void operator()(sycl::nd_item<1> item) const {
    ConsumRange<InputT, OutputT, InitValueT, Op, ITEMS_PER_ITEM>(
        item, in_data_, local_data_, out_data_, init_val_, in_size_,
        elems_per_group_, op_, LegacyMode());
  }

 private:
  InputT* in_data_;
  __slm__<InitValueT> local_data_;
  OutputT* out_data_;
  InitValueT init_val_;
  int in_size_;
  int elems_per_group_;
  Op op_;
};

template <typename InitValueT>
InitValueT* CreateScratchTensorAndGet(OpKernelContext* context, int length) {
  Tensor scratch_tensor;
  Status status = context->allocate_temp(
      DataTypeToEnum<int8>::value,
      TensorShape({static_cast<int64>(length * sizeof(InitValueT))}),
      &scratch_tensor);
  ITEX_CHECK_OK(status);
  InitValueT* scratch = reinterpret_cast<InitValueT*>(scratch_tensor.data());
  return scratch;
}

template <typename InputT, typename OutputT, typename InitValueT, typename Op>
void LaunchScalarReduction(OpKernelContext* context, InputT* in, OutputT* out,
                           InitValueT init_val, int in_size, Op op) {
  typedef typename std::remove_cv<
      typename std::remove_reference<InputT>::type>::type BaseInputT;

  const GPUDevice& device = context->eigen_device<GPUDevice>();
  sycl::queue* stream = device.stream();

  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  const int group_size = std::min(512, max_group_size);
  const int elems_per_item = 8;
  const int max_elems_per_group = group_size * elems_per_item * 6;

  BaseInputT* in_unqualified = const_cast<BaseInputT*>(in);

  if (in_size <= max_elems_per_group) {
    int elems_per_group = RoundUp(in_size, group_size * elems_per_item);
    sycl::range<1> local(group_size);
    sycl::range<1> global(group_size);
    stream->submit([&](sycl::handler& cgh) {
      __slm__<InitValueT> local_data(group_size, cgh);
      GroupReduceKernel<BaseInputT, OutputT, InitValueT, Op, elems_per_item>
          task(in_unqualified, local_data, out, init_val, in_size,
               elems_per_group, op);
      cgh.parallel_for<GroupReduceKernel<BaseInputT, OutputT, InitValueT, Op,
                                         elems_per_item>>(
          sycl::nd_range<1>(global, local), task);
    });
  } else {
    int num_wg;
    if (in_size <= 65536 * 1024)  //  fit into PVC L3: 204M
      num_wg = std::min(static_cast<int>(group_size),
                        DivUp(in_size, group_size * elems_per_item));
    else
      num_wg = std::min(static_cast<int>(max_elems_per_group),
                        DivUp(in_size, group_size * elems_per_item));

    InitValueT* scratch =
        CreateScratchTensorAndGet<InitValueT>(context, num_wg);

    int elems_per_group =
        RoundUp(DivUp(in_size, num_wg), group_size * elems_per_item);

    sycl::range<1> local(group_size);
    sycl::range<1> global(group_size * num_wg);
    stream->submit([&](sycl::handler& cgh) {
      __slm__<InitValueT> local_data(group_size, cgh);
      GroupReduceKernel<BaseInputT, InitValueT, InitValueT, Op, elems_per_item>
          task(in_unqualified, local_data, scratch, init_val, in_size,
               elems_per_group, op);
      cgh.parallel_for<GroupReduceKernel<BaseInputT, InitValueT, InitValueT, Op,
                                         elems_per_item>>(
          sycl::nd_range<1>(global, local), task);
    });

    local = sycl::range<1>(group_size);
    elems_per_group = RoundUp(num_wg, group_size * elems_per_item);
    stream->submit([&](sycl::handler& cgh) {
      __slm__<InitValueT> local_data(group_size, cgh);
      GroupReduceKernel<InitValueT, OutputT, InitValueT, Op, elems_per_item>
          task(scratch, local_data, out, init_val, num_wg, elems_per_group, op);
      cgh.parallel_for<GroupReduceKernel<InitValueT, OutputT, InitValueT, Op,
                                         elems_per_item>>(
          sycl::nd_range<1>(local, local), task);
    });
  }
}

template <typename T, typename ResultType>
inline typename std::enable_if_t<std::is_class<ResultType>::value, T>
ConvertRowIndex(T a, int x, int y) {
  return a;
}

template <typename T, typename ResultType>
inline typename std::enable_if_t<std::is_arithmetic<ResultType>::value, T>
ConvertRowIndex(T a, int x, int y) {
  int index = a.index;
  index = static_cast<int>(index % y);
  return T(a.value, index);
}

// map one workitem to one row
template <typename T, typename OutputT, typename BinaryOp, typename InitValueT>
struct SimpleRowReduction {
  SimpleRowReduction(T* in_data, OutputT* out_data, int extend_x, int extend_y,
                     InitValueT init, BinaryOp op)
      : in_data_(in_data),
        out_data_(out_data),
        extend_x_(extend_x),
        extend_y_(extend_y),
        init_(init),
        op_(op) {}

  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();
    if (id < extend_x_) {
      int offset = id * extend_y_;
      InitValueT aggregate = init_;
#pragma unroll
      for (int i = 0; i < extend_y_; ++i) {
        InitValueT tmp = InitValueT(in_data_[offset + i], offset + i);
        aggregate = op_(aggregate, tmp);
      }
      InitValueT result =
          ConvertRowIndex<InitValueT, OutputT>(aggregate, extend_x_, extend_y_);
      Assign(out_data_ + id, &result);
    }
  }

 private:
  T* in_data_;
  OutputT* out_data_;
  int extend_x_;
  int extend_y_;
  InitValueT init_;
  BinaryOp op_;
};

template <typename InputT, typename OutputT, typename BinaryOp,
          typename InitValueT>
struct GroupRowReduction {
  GroupRowReduction(InputT* in_data, __slm__<InitValueT> local_data,
                    OutputT* out_data, int extend_x, int extend_y,
                    InitValueT init, BinaryOp op)
      : in_data_(in_data),
        local_data_(local_data),
        out_data_(out_data),
        extend_x_(extend_x),
        extend_y_(extend_y),
        init_(init),
        op_(op) {}

  void operator()(sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    auto group_size = item.get_local_range(0);
    int g_offset = group_id * extend_y_;
    InitValueT aggregate = init_;

    for (int i = lid; i < extend_y_; i += group_size) {
      InitValueT data = InitValueT(in_data_[g_offset + i], g_offset + i);
      aggregate = op_(aggregate, data);
    }

    local_data_[lid] = aggregate;
    internal::group_barrier(group);

    internal::reduce_over_group(group, local_data_, group_size, lid, op_);

    if (lid == 0) {
      InitValueT result = ConvertRowIndex<InitValueT, OutputT>(
          local_data_[0], extend_x_, extend_y_);
      Assign(out_data_ + group_id, &result);
    }
  }

 private:
  InputT* in_data_;
  __slm__<InitValueT> local_data_;
  OutputT* out_data_;
  int extend_x_;
  int extend_y_;
  InitValueT init_;
  BinaryOp op_;
};

// map one row to one subgroup
template <typename InputT, typename OutputT, typename BinaryOp,
          typename InitValueT>
struct SubGroupRowReduction {
  SubGroupRowReduction(InputT* in_data, __slm__<InitValueT> local_data,
                       OutputT* out_data, int extend_x, int extend_y,
                       InitValueT init, BinaryOp op)
      : in_data_(in_data),
        local_data_(local_data),
        out_data_(out_data),
        extend_x_(extend_x),
        extend_y_(extend_y),
        init_(init),
        op_(op) {}
  [[intel::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    auto sg = item.get_sub_group();
    auto group_size = item.get_local_range(0);
    auto subgroup_id = sg.get_group_linear_id();
    auto lane_id = sg.get_local_id();
    auto sub_group_size = sg.get_local_range()[0];

    int x_index = group_id * (group_size / sub_group_size) + subgroup_id;
    int local_start = subgroup_id * sub_group_size;
    if (x_index < extend_x_) {
      InitValueT aggregate = init_;
      int start_offset = x_index * extend_y_;

#pragma unroll
      for (int i = lane_id; i < extend_y_; i += sub_group_size) {
        InitValueT data =
            InitValueT(in_data_[start_offset + i], start_offset + i);
        aggregate = op_(aggregate, data);
      }
      local_data_[local_start + lane_id] = aggregate;
    } else {
      local_data_[local_start + lane_id] = init_;
    }

    sycl::group_barrier(sg, sycl::memory_scope::sub_group);

    if (lane_id == 0 && x_index < extend_x_) {
      for (int i = 1; i < sub_group_size; i++) {
        local_data_[local_start] =
            op_(local_data_[local_start], local_data_[local_start + i]);
      }

      InitValueT result = ConvertRowIndex<InitValueT, OutputT>(
          local_data_[local_start], extend_x_, extend_y_);
      Assign(out_data_ + x_index, &result);
    }
  }

 private:
  InputT* in_data_;
  __slm__<InitValueT> local_data_;
  OutputT* out_data_;
  int extend_x_;
  int extend_y_;
  InitValueT init_;
  BinaryOp op_;
};

template <typename InputT, typename OutputT, typename BinaryOp,
          typename InitValueT>
void LaunchRowReduction(OpKernelContext* context, InputT* in_data,
                        OutputT* out_data, int extend_x, int extend_y,
                        InitValueT init, BinaryOp op) {
  typedef typename std::remove_cv<
      typename std::remove_reference<InputT>::type>::type BaseInputT;

  const GPUDevice& device = context->eigen_device<GPUDevice>();
  sycl::queue* stream = device.stream();
  BaseInputT* in_unqualified = const_cast<BaseInputT*>(in_data);

  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_group_size);

  using SimpleReduction =
      SimpleRowReduction<InputT, OutputT, BinaryOp, InitValueT>;
  using SubGroupReduction =
      SubGroupRowReduction<InputT, OutputT, BinaryOp, InitValueT>;
  using GroupReduction =
      GroupRowReduction<InputT, OutputT, BinaryOp, InitValueT>;

  if (extend_y <= 32) {
    int num_wg = DivUp(extend_x, static_cast<int>(group_size));
    sycl::nd_range<1> range(num_wg * group_size, group_size);

    stream->submit([&](sycl::handler& cgh) {
      SimpleReduction task(in_unqualified, out_data, extend_x, extend_y, init,
                           op);
      cgh.parallel_for<SimpleReduction>(range, task);
    });
  } else if (extend_y <= 1024) {
    constexpr int max_sub_group_size = 32;
    int num_sub_groups_in_group = group_size / max_sub_group_size;
    int num_wg = DivUp(extend_x, num_sub_groups_in_group);

    sycl::range<1> local(group_size);
    sycl::range<1> global(num_wg * group_size);
    stream->submit([&](sycl::handler& cgh) {
      __slm__<InitValueT> local_data(group_size, cgh);
      SubGroupReduction task(in_unqualified, local_data, out_data, extend_x,
                             extend_y, init, op);
      cgh.parallel_for<SubGroupReduction>(sycl::nd_range<1>(global, local),
                                          task);
    });
  } else {
    sycl::range<1> local(group_size);
    sycl::range<1> global(extend_x * local[0]);
    stream->submit([&](sycl::handler& cgh) {
      __slm__<InitValueT> local_data(group_size, cgh);
      GroupReduction task(in_unqualified, local_data, out_data, extend_x,
                          extend_y, init, op);
      cgh.parallel_for<GroupReduction>(sycl::nd_range<1>(global, local), task);
    });
  }
}

template <typename InputT, typename OutputT, typename BinaryOp,
          typename InitValueT>
struct SimpleColReduction {
  SimpleColReduction(InputT* in_data, OutputT* out_data, int extend_x,
                     int extend_y, int extend_z, InitValueT init, BinaryOp op)
      : in_data_(in_data),
        out_data_(out_data),
        extend_x_(extend_x),
        extend_y_(extend_y),
        extend_z_(extend_z),
        init_(init),
        op_(op) {}
  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();
    if (id < extend_x_ * extend_z_) {
      int outer = id / extend_z_;
      int inner = id - outer * extend_z_;

      int in_offset = outer * extend_y_ * extend_z_ + inner;
      InitValueT aggregate = init_;
      for (int i = 0; i < extend_y_; ++i) {
        int offset = in_offset + i * extend_z_;
        InitValueT tmp = InitValueT(in_data_[offset], offset);
        aggregate = op_(aggregate, tmp);
      }

      aggregate.index = aggregate.index / extend_z_;
      // TODO(itex) the mod operation will return 0.
      aggregate.index =
          aggregate.index -
          static_cast<int>(aggregate.index / extend_y_) * extend_y_;
      Assign(out_data_ + id, &aggregate);
    }
  }

 private:
  InputT* in_data_;
  OutputT* out_data_;
  int extend_x_;
  int extend_y_;
  int extend_z_;
  InitValueT init_;
  BinaryOp op_;
};

template <typename T, typename ResultType>
inline typename std::enable_if_t<std::is_class<ResultType>::value, T>
ConvertColIndex(T a, int x, int y, int z) {
  return a;
}

template <typename T, typename ResultType>
inline typename std::enable_if_t<std::is_arithmetic<ResultType>::value, T>
ConvertColIndex(T a, int x, int y, int z) {
  int index = a.index;
  index = static_cast<int>(index / z);
  index = index - static_cast<int>(index / y) * y;

  return T(a.value, index);
}

template <typename InputT, typename OutputT, typename LocalAccessor,
          typename BinaryOp, typename InitValueT, int MaxGroupSize,
          int SubGroupSize>
struct SubGroupColReduction {
  SubGroupColReduction(InputT* in_data, LocalAccessor scratch,
                       OutputT* out_data, int extend_x, int extend_y,
                       int extend_z, int elems_per_item, int num_sub_group,
                       int num_segments_y, InitValueT init, BinaryOp op)
      : in_data_(in_data),
        scratch_(scratch),
        out_data_(out_data),
        extend_x_(extend_x),
        extend_y_(extend_y),
        extend_z_(extend_z),
        elems_per_item_(elems_per_item),
        num_sub_group_(num_sub_group),
        num_segments_y_(num_segments_y),
        init_(init),
        op_(op) {}

  void operator()(sycl::nd_item<3> item) const {
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1);
    int z_group_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_id();

    int x_offset = x_group_id * extend_y_ * extend_z_;

    // each subgroup load data and reduce elems_per_item
    InitValueT aggregate = init_;
    int z_offset = z_group_id * SubGroupSize + lane_id;
    if (z_offset < extend_z_) {
      for (int i = 0; i < elems_per_item_; ++i) {
        int y_idx = y_group_id * num_sub_group_ * elems_per_item_ +
                    num_sub_group_ * i + subgroup_id;
        if (y_idx >= extend_y_) break;
        int offset = x_offset + y_idx * extend_z_ + z_offset;
        InitValueT tmp = InitValueT(in_data_[offset], offset);
        aggregate = op_(aggregate, tmp);
      }
    }
    // each subgroup write result to slm
    scratch_[subgroup_id + lane_id * num_sub_group_] = aggregate;
    item.barrier(sycl::access::fence_space::local_space);

    // slm reduce and write output
    if (subgroup_id == 0) {
      int local_start = lane_id * num_sub_group_;
#pragma unroll
      for (int i = 1; i < MaxGroupSize / SubGroupSize; i++) {
        scratch_[local_start] =
            op_(scratch_[local_start], scratch_[local_start + i]);
      }

      z_offset = z_group_id * SubGroupSize + lane_id;
      if (z_offset < extend_z_) {
        int offset = x_group_id * extend_z_ * num_segments_y_ +
                     y_group_id * extend_z_ + z_offset;

        InitValueT result = ConvertColIndex<InitValueT, OutputT>(
            scratch_[local_start], extend_x_, extend_y_, extend_z_);
        Assign(out_data_ + offset, &result);
      }
    }
  }

 private:
  InputT* in_data_;
  LocalAccessor scratch_;
  OutputT* out_data_;
  int extend_x_;
  int extend_y_;
  int extend_z_;
  int elems_per_item_;
  int num_sub_group_;
  int num_segments_y_;
  InitValueT init_;
  BinaryOp op_;
};

template <typename InputT, typename OutputT, typename BinaryOp,
          typename InitValueT, int MaxGroupSize, int SubGroupSize>
void LaunchSubGroupColReduction(OpKernelContext* ctx, InputT* in_data,
                                OutputT* out_data, int extend_x, int extend_y,
                                int extend_z, InitValueT init, BinaryOp op) {
  enum { MaxSubGroup = MaxGroupSize / SubGroupSize, ElemsPerItem = 32 };

  int elems_per_item = ElemsPerItem;
  int num_sub_group = MaxSubGroup;

  if (extend_y * 2 <= num_sub_group * elems_per_item) {
    while (num_sub_group * elems_per_item >= extend_y * 2 &&
           elems_per_item > 1) {
      elems_per_item >>= 1;
    }
  }
  assert(num_sub_group % SubGroupSize == 0);
  int num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item);
  int num_segments_z = DivUp(extend_z, SubGroupSize);

  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();

  if (num_segments_y > 1) {
    using ColReductionStage1 =
        SubGroupColReduction<InputT, InitValueT, __slm__<InitValueT>, BinaryOp,
                             InitValueT, MaxGroupSize, SubGroupSize>;
    using ColReductionStage2 =
        SubGroupColReduction<InitValueT, OutputT, __slm__<InitValueT>, BinaryOp,
                             InitValueT, MaxGroupSize, SubGroupSize>;

    while (num_segments_y > num_sub_group * ElemsPerItem) {
      elems_per_item <<= 1;
      num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item);
    }
    sycl::range<3> local(1, num_sub_group, SubGroupSize);
    sycl::range<3> global(extend_x, num_segments_y * local[1],
                          num_segments_z * local[2]);

    InitValueT* inter_out = CreateScratchTensorAndGet<InitValueT>(
        ctx, extend_x * num_segments_y * extend_z);

    stream->submit([&](sycl::handler& cgh) {
      __slm__<InitValueT> local_data(num_sub_group * SubGroupSize, cgh);
      ColReductionStage1 task(in_data, local_data, inter_out, extend_x,
                              extend_y, extend_z, elems_per_item, num_sub_group,
                              num_segments_y, init, op);
      cgh.parallel_for<ColReductionStage1>(sycl::nd_range<3>(global, local),
                                           task);
    });

    global = sycl::range<3>{static_cast<size_t>(extend_x), local[1],
                            num_segments_z * local[2]};
    stream->submit([&](sycl::handler& cgh) {
      __slm__<InitValueT> local_data(num_sub_group * SubGroupSize, cgh);
      ColReductionStage2 task(inter_out, local_data, out_data, extend_x,
                              extend_y, extend_z, elems_per_item, num_sub_group,
                              1, init, op);
      cgh.parallel_for<ColReductionStage2>(sycl::nd_range<3>(global, local),
                                           task);
    });
  } else {
    using ColReduction =
        SubGroupColReduction<InputT, OutputT, __slm__<InitValueT>, BinaryOp,
                             InitValueT, MaxGroupSize, SubGroupSize>;
    sycl::range<3> local(1, num_sub_group, SubGroupSize);
    sycl::range<3> global(extend_x, local[1], num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      __slm__<InitValueT> local_data(num_sub_group * SubGroupSize, cgh);
      ColReduction task(in_data, local_data, out_data, extend_x, extend_y,
                        extend_z, elems_per_item, num_sub_group, 1, init, op);
      cgh.parallel_for<ColReduction>(sycl::nd_range<3>(global, local), task);
    });
  }
}

template <typename InputT, typename OutputT, typename BinaryOp,
          typename InitValueT>
void LaunchColReduction(OpKernelContext* context, InputT* in_data,
                        OutputT* out_data, int extend_x, int extend_y,
                        int extend_z, InitValueT init, BinaryOp op) {
  typedef typename std::remove_cv<
      typename std::remove_reference<InputT>::type>::type BaseInputT;

  const GPUDevice& device = context->eigen_device<GPUDevice>();
  sycl::queue* stream = device.stream();
  BaseInputT* in_unqualified = const_cast<BaseInputT*>(in_data);

  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_group_size);

  int elems_per_item = extend_y / (extend_x * extend_z);

  if (elems_per_item < 4) {
    const int out_size = extend_x * extend_z;
    int num_wg = (out_size + group_size - 1) / group_size;
    sycl::nd_range<1> range(num_wg * group_size, group_size);
    stream->submit([&](sycl::handler& cgh) {
      SimpleColReduction<InputT, OutputT, BinaryOp, InitValueT> task(
          in_unqualified, out_data, extend_x, extend_y, extend_z, init, op);
      cgh.parallel_for<
          SimpleColReduction<InputT, OutputT, BinaryOp, InitValueT>>(range,
                                                                     task);
    });
  } else {
    if (max_group_size >= 1024) {
      LaunchSubGroupColReduction<InputT, OutputT, BinaryOp, InitValueT, 1024,
                                 32>(context, in_unqualified, out_data,
                                     extend_x, extend_y, extend_z, init, op);

    } else if (max_group_size >= 256) {
      LaunchSubGroupColReduction<InputT, OutputT, BinaryOp, InitValueT, 256,
                                 16>(context, in_unqualified, out_data,
                                     extend_x, extend_y, extend_z, init, op);
    } else {
      std::stringstream ss;
      ss << "Unsupported col reduce algorithm for group size == "
         << max_group_size << " which is lower than 256";
      ITEX_LOG(FATAL) << ss.str();
    }
  }
}

template <typename T>
struct MaxTupleReducer {
  static Tuple<T> init() {
    return Tuple<T>(std::numeric_limits<T>::lowest(), 0);
  }

  Tuple<T> operator()(const Tuple<T>& x, const Tuple<T>& y) const {
    if (x.value > y.value) {
      return x;
    } else if (x.value == y.value) {
      if (x.index < y.index) {
        return x;
      } else {
        return y;
      }
    } else {
      return y;
    }
  }

  using MapType = T;
};

template <>
struct MaxTupleReducer<Eigen::half> {
  static Tuple<float> init() {
    return Tuple<float>(std::numeric_limits<float>::lowest(), 0);
  }

  Tuple<float> operator()(const Tuple<float>& x, const Tuple<float>& y) const {
    if (x.value > y.value) {
      return x;
    } else if (x.value == y.value) {
      if (x.index < y.index) {
        return x;
      } else {
        return y;
      }
    } else {
      return y;
    }
  }

  using MapType = sycl::half;
};

template <>
struct MaxTupleReducer<Eigen::bfloat16> {
  static Tuple<float> init() {
    return Tuple<float>(std::numeric_limits<float>::lowest(), 0);
  }

  Tuple<float> operator()(const Tuple<float>& x, const Tuple<float>& y) const {
    if (x.value > y.value) {
      return x;
    } else if (x.value == y.value) {
      if (x.index < y.index) {
        return x;
      } else {
        return y;
      }
    } else {
      return y;
    }
  }

  using MapType = Eigen::bfloat16;
};

template <typename T>
struct MinTupleReducer {
  static Tuple<T> init() { return Tuple<T>(std::numeric_limits<T>::max(), 0); }

  Tuple<T> operator()(const Tuple<T>& x, const Tuple<T>& y) const {
    if (x.value < y.value) {
      return x;
    } else if (x.value == y.value) {
      if (x.index < y.index) {
        return x;
      } else {
        return y;
      }
    } else {
      return y;
    }
  }

  using MapType = T;
};

template <>
struct MinTupleReducer<Eigen::half> {
  static Tuple<float> init() {
    return Tuple<float>(std::numeric_limits<float>::max(), 0);
  }

  Tuple<float> operator()(const Tuple<float>& x, const Tuple<float>& y) const {
    if (x.value < y.value) {
      return x;
    } else if (x.value == y.value) {
      if (x.index < y.index) {
        return x;
      } else {
        return y;
      }
    } else {
      return y;
    }
  }

  using MapType = sycl::half;
};

template <>
struct MinTupleReducer<Eigen::bfloat16> {
  static Tuple<float> init() {
    return Tuple<float>(std::numeric_limits<float>::max(), 0);
  }

  Tuple<float> operator()(const Tuple<float>& x, const Tuple<float>& y) const {
    if (x.value < y.value) {
      return x;
    } else if (x.value == y.value) {
      if (x.index < y.index) {
        return x;
      } else {
        return y;
      }
    } else {
      return y;
    }
  }

  using MapType = Eigen::bfloat16;
};

struct ReduceFunctor {
  template <typename InputT, typename OutputT, typename Reducer>
  static void Reduce(OpKernelContext* context, InputT in, OutputT out,
                     const Reducer& reducer, bool reduce_first_axis) {
    using BaseInputT = typename Reducer::MapType;

    int x = in.dimension(0);
    int y = in.rank() >= 2 ? in.dimension(1) : 1;
    int z = in.rank() >= 3 ? in.dimension(2) : 1;

    if (in.rank() == 1 && out.rank() == 0) {
      int in_size = x * y * z;
      LaunchScalarReduction(context,
                            reinterpret_cast<const BaseInputT*>(in.data()),
                            out.data(), reducer.init(), in_size, reducer);
    } else if (in.rank() == 2 && out.rank() == 1 && !reduce_first_axis) {
      LaunchRowReduction(context,
                         reinterpret_cast<const BaseInputT*>(in.data()),
                         out.data(), x, y, reducer.init(), reducer);
    } else if (in.rank() == 2 && out.rank() == 1 && reduce_first_axis) {
      x = 1;
      y = in.dimension(0);
      z = in.dimension(1);
      LaunchColReduction(context,
                         reinterpret_cast<const BaseInputT*>(in.data()),
                         out.data(), x, y, z, reducer.init(), reducer);
    } else if (in.rank() == 3 && out.rank() == 2 && !reduce_first_axis) {
      LaunchColReduction(context,
                         reinterpret_cast<const BaseInputT*>(in.data()),
                         out.data(), x, y, z, reducer.init(), reducer);

    } else {
      std::stringstream ss;
      ss << "Invalid reduction requested: in_rank, out_rank" << in.rank() << " "
         << out.rank();
      ITEX_LOG(FATAL) << ss.str();
    }
  }
};

template <typename Device, typename T, typename Tout>
class ArgMaxOp : public OpKernel {
 public:
  explicit ArgMaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dimension = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dimension.shape()),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension.shape().DebugString()));
    const int32 dim = internal::SubtleMustCopy(dimension.scalar<int32>()());
    const int input_dims = input.dims();
    int axis = dim < 0 ? dim + input_dims : dim;
    OP_REQUIRES(context, FastBoundsCheck(axis, input_dims),
                errors::InvalidArgument("Expected dimension in the range [",
                                        -input_dims, ", ", input_dims,
                                        "), but got ", dim));
    OP_REQUIRES(
        context, input.dim_size(axis) > 0,
        errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                                input.shape().DebugString()));

    TensorShape output_shape;
    const TensorShape& input_shape = input.shape();

    for (int d = 0; d < input_dims - 1; ++d) {
      output_shape.AddDim(input_shape.dim_size((d < axis) ? d : d + 1));
    }

    ReductionHelper helper;
    OP_REQUIRES_OK(context, helper.Simplify(input, dimension, false));
    ITEX_CHECK_GE(helper.ndims(), 0);

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, helper.out_shape(), &output));

    if (output_shape.num_elements() == 0) {
      return;
    }

    const Eigen::GpuDevice& device = context->eigen_gpu_device();
    // when reduced axis' dimension size equal to 1, result must be tensor of
    // zeros.
    if (input_shape.dim_size(axis) == 1) {
      auto out = output->template flat<Tout>();
      out.device(device) = out.constant(Tout(0));
      return;
    }

    MaxTupleReducer<T> reducer;

    if (helper.ndims() == 1 && helper.reduce_first_axis()) {
      ReduceFunctor::Reduce(context, helper.in<T, 1>(input),
                            helper.out<Tout, 0>(output), reducer,
                            helper.reduce_first_axis());
      return;
    } else if (helper.ndims() == 2 && !helper.reduce_first_axis()) {
      ReduceFunctor::Reduce(context, helper.in<T, 2>(input),
                            helper.out<Tout, 1>(output), reducer,
                            helper.reduce_first_axis());
      return;
    } else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
      ReduceFunctor::Reduce(context, helper.in<T, 2>(input),
                            helper.out<Tout, 1>(output), reducer,
                            helper.reduce_first_axis());
      return;
    } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
      ReduceFunctor::Reduce(context, helper.in<T, 3>(input),
                            helper.out<Tout, 2>(output), reducer,
                            helper.reduce_first_axis());
      return;
    }

#define HANDLE_DIM(NDIM)                             \
  case NDIM:                                         \
    ArgMaxFunctor<GPUDevice, T, Tout>::Reduce##NDIM( \
        device, input.tensor<T, NDIM>(), axis,       \
        output->tensor<Tout, NDIM - 1>());           \
    break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Argmax and Argmin only support up "
                                            "to 7 input dimensions, but got ",
                                            input_dims, ". Inputs shape: ",
                                            input.shape().DebugString()));
    }
  }
#undef HANDLE_DIM
};

template <typename Device, typename T, typename Tout>
class ArgMinOp : public OpKernel {
 public:
  explicit ArgMinOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dimension = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dimension.shape()),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension.shape().DebugString()));
    const int32 dim = internal::SubtleMustCopy(dimension.scalar<int32>()());
    const int input_dims = input.dims();
    int axis = dim < 0 ? dim + input_dims : dim;
    OP_REQUIRES(context, FastBoundsCheck(axis, input_dims),
                errors::InvalidArgument("Expected dimension in the range [",
                                        -input_dims, ", ", input_dims,
                                        "), but got ", dim));
    OP_REQUIRES(
        context, input.dim_size(axis) > 0,
        errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                                input.shape().DebugString()));

    TensorShape output_shape;
    const TensorShape& input_shape = input.shape();

    for (int d = 0; d < input_dims - 1; ++d) {
      output_shape.AddDim(input_shape.dim_size((d < axis) ? d : d + 1));
    }

    ReductionHelper helper;
    OP_REQUIRES_OK(context, helper.Simplify(input, dimension, false));
    ITEX_CHECK_GE(helper.ndims(), 0);

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, helper.out_shape(), &output));

    if (output_shape.num_elements() == 0) {
      return;
    }

    const Eigen::GpuDevice& device = context->eigen_gpu_device();
    // when reduced axis' dimension size equal to 1, result must be tensor of
    // zeros.
    if (input_shape.dim_size(axis) == 1) {
      auto out = output->template flat<Tout>();
      out.device(device) = out.constant(Tout(0));
      return;
    }

    MinTupleReducer<T> reducer;

    if (helper.ndims() == 1 && helper.reduce_first_axis()) {
      ReduceFunctor::Reduce(context, helper.in<T, 1>(input),
                            helper.out<Tout, 0>(output), reducer,
                            helper.reduce_first_axis());
      return;
    } else if (helper.ndims() == 2 && !helper.reduce_first_axis()) {
      ReduceFunctor::Reduce(context, helper.in<T, 2>(input),
                            helper.out<Tout, 1>(output), reducer,
                            helper.reduce_first_axis());
      return;
    } else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
      ReduceFunctor::Reduce(context, helper.in<T, 2>(input),
                            helper.out<Tout, 1>(output), reducer,
                            helper.reduce_first_axis());
      return;
    } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
      ReduceFunctor::Reduce(context, helper.in<T, 3>(input),
                            helper.out<Tout, 2>(output), reducer,
                            helper.reduce_first_axis());
      return;
    }

#define HANDLE_DIM(NDIM)                             \
  case NDIM:                                         \
    ArgMinFunctor<GPUDevice, T, Tout>::Reduce##NDIM( \
        device, input.tensor<T, NDIM>(), axis,       \
        output->tensor<Tout, NDIM - 1>());           \
    break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Argmax and Argmin only support up "
                                            "to 7 input dimensions, but got ",
                                            input_dims, ". Inputs shape: ",
                                            input.shape().DebugString()));
    }
  }
#undef HANDLE_DIM
};

#define REGISTER_ARGMAX_GPU(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                              \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int64_t>("output_type") \
                              .TypeConstraint<int32>("Tidx")          \
                              .HostMemory("dimension"),               \
                          ArgMaxOp<GPUDevice, type, int64>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                              \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int32>("output_type")   \
                              .TypeConstraint<int32>("Tidx")          \
                              .HostMemory("dimension"),               \
                          ArgMaxOp<GPUDevice, type, int32>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                              \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int64_t>("output_type") \
                              .TypeConstraint<int32>("Tidx")          \
                              .HostMemory("dimension"),               \
                          ArgMinOp<GPUDevice, type, int64>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                              \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int32>("output_type")   \
                              .TypeConstraint<int32>("Tidx")          \
                              .HostMemory("dimension"),               \
                          ArgMinOp<GPUDevice, type, int32>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ARGMAX_GPU);
TF_CALL_bool(REGISTER_ARGMAX_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_ARGMAX_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_ARGMAX_GPU

}  // namespace itex
