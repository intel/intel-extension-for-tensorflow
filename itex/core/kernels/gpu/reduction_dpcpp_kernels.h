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

#ifndef ITEX_CORE_KERNELS_GPU_REDUCTION_DPCPP_KERNELS_H_
#define ITEX_CORE_KERNELS_GPU_REDUCTION_DPCPP_KERNELS_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {
namespace reduciton_helper {
template <typename T>
using LocalAcc = sycl::accessor<T, 1, sycl::access::mode::read_write,
                                sycl::access::target::local>;

template <typename T>
struct Identity {
  inline T operator()(T x) const { return x; }
};

template <typename InputT, typename OutputT, typename InputFunctor,
          typename OutputFunctor, typename InitValueT, typename Op,
          int ITEMS_PER_ITEM>
struct GroupReduceKernel {
  enum {
    VEC_LENGTH = 4,
    WORDS = ITEMS_PER_ITEM / VEC_LENGTH,
  };

  GroupReduceKernel(InputT* in_data, OutputT* out_data, InputFunctor in_func,
                    OutputFunctor out_func, InitValueT init_val, int in_size,
                    int elems_per_group, Op op)
      : in_data_(in_data),
        out_data_(out_data),
        in_func_(in_func),
        out_func_(out_func),
        init_val_(init_val),
        in_size_(in_size),
        elems_per_group_(elems_per_group),
        op_(op) {}

  void operator()(sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    if ((group_id + 1) * elems_per_group_ > in_size_)
      ConsumRange(item, Int2Type<false>());
    else
      ConsumRange(item, Int2Type<true>());
  }

  void ConsumRange(sycl::nd_item<1> item, Int2Type<false> /*can_vec*/) const {
    auto lid = item.get_local_id(0);
    auto g = item.get_group();
    auto group_id = item.get_group(0);
    int group_size = item.get_local_range(0);

    int group_start = group_id * elems_per_group_;
    int group_end = group_start + elems_per_group_;
    group_end = group_end > in_size_ ? in_size_ : group_end;
    if (group_start >= in_size_) return;

    InitValueT sum = init_val_;
#pragma unroll
    for (int index = group_start + lid; index < group_end;
         index += group_size) {
      sum = op_(sum, InitValueT(in_func_(in_data_[index])));
    }
    InitValueT res = sycl::reduce_over_group(g, sum, op_);
    if (lid == 0) out_data_[group_id] = OutputT(out_func_(res));
  }

  inline void ConsumRange(sycl::nd_item<1> item,
                          Int2Type<true> /*can_vec*/) const {
    auto lid = item.get_local_id(0);
    auto g = item.get_group();
    auto group_id = item.get_group(0);
    auto group_size = item.get_local_range(0);

    typedef sycl::vec<InputT, VEC_LENGTH> VecT;
    InputT input_items[ITEMS_PER_ITEM];
    VecT* vec_items = reinterpret_cast<VecT*>(input_items);

    int group_start = group_id * elems_per_group_;

    InitValueT aggregate = init_val_;
    int loops = elems_per_group_ / (group_size * ITEMS_PER_ITEM);
    for (int l = 0; l < loops; ++l) {
      int start_offset = group_start + l * group_size * ITEMS_PER_ITEM;
      VecT* vec_in =
          reinterpret_cast<VecT*>(in_data_ + start_offset + (lid * VEC_LENGTH));

#pragma unroll
      for (int i = 0; i < WORDS; ++i) {
        vec_items[i] = vec_in[i * group_size];
      }

#pragma unroll
      for (int i = 0; i < ITEMS_PER_ITEM; ++i) {
        aggregate = op_(aggregate, InitValueT(in_func_(input_items[i])));
      }
    }
    InitValueT res = sycl::reduce_over_group(g, aggregate, op_);
    if (lid == 0) out_data_[group_id] = OutputT(out_func_(res));
  }

  InputT* in_data_;
  OutputT* out_data_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
  InitValueT init_val_;
  int in_size_;
  int elems_per_group_;
  Op op_;
};

template <typename OutputT, typename InputFunctor, typename OutputFunctor,
          typename InitValueT, typename Op, int ITEMS_PER_ITEM>
struct GroupReduceKernel<Eigen::bfloat16, OutputT, InputFunctor, OutputFunctor,
                         InitValueT, Op, ITEMS_PER_ITEM> {
  enum {
    VEC_LENGTH = 4,
    WORDS = ITEMS_PER_ITEM / VEC_LENGTH,
  };

  typedef Eigen::bfloat16 InputT;

  GroupReduceKernel(InputT* in_data, OutputT* out_data, InputFunctor in_func,
                    OutputFunctor out_func, InitValueT init_val, int in_size,
                    int elems_per_group, Op op)
      : in_data_(in_data),
        out_data_(out_data),
        in_func_(in_func),
        out_func_(out_func),
        init_val_(init_val),
        in_size_(in_size),
        elems_per_group_(elems_per_group),
        op_(op) {}

  void operator()(sycl::nd_item<1> item) const {
    ConsumRange(item, Int2Type<false>());
  }

  void ConsumRange(sycl::nd_item<1> item, Int2Type<false> /*can_vec*/) const {
    auto lid = item.get_local_id(0);
    auto g = item.get_group();
    auto group_id = item.get_group(0);
    int group_size = item.get_local_range(0);

    int group_start = group_id * elems_per_group_;
    int group_end = group_start + elems_per_group_;
    group_end = group_end > in_size_ ? in_size_ : group_end;
    if (group_start >= in_size_) return;

    InitValueT sum = init_val_;
#pragma unroll
    for (int index = group_start + lid; index < group_end;
         index += group_size) {
      sum = op_(sum, InitValueT(in_func_(in_data_[index])));
    }
    InitValueT res = sycl::reduce_over_group(g, sum, op_);
    if (lid == 0) out_data_[group_id] = OutputT(out_func_(res));
  }

  InputT* in_data_;
  OutputT* out_data_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
  InitValueT init_val_;
  int in_size_;
  int elems_per_group_;
  Op op_;
};

// map one workitem to one row
template <typename InputT, typename OutputT, typename InitValueT,
          typename InputFunctor, typename OutputFunctor, typename BinaryOp>
struct SimpleRowReduction {
  SimpleRowReduction(InputT* in_data, OutputT* out_data, InitValueT init,
                     int extend_x, int extend_y, InputFunctor in_func,
                     OutputFunctor out_func, BinaryOp op)
      : in_data_(in_data),
        out_data_(out_data),
        init_(init),
        extend_x_(extend_x),
        extend_y_(extend_y),
        in_func_(in_func),
        out_func_(out_func),
        op_(op) {}
  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();
    if (id < extend_x_) {
      int offset = id * extend_y_;
      InitValueT aggregate = InitValueT(in_func_(in_data_[offset]));
#pragma unroll
      for (int i = 1; i < extend_y_; ++i) {
        InitValueT tmp = InitValueT(in_func_(in_data_[offset + i]));
        aggregate = op_(aggregate, tmp);
      }
      out_data_[id] = OutputT(out_func_(aggregate));
    }
  }

  InputT* in_data_;
  OutputT* out_data_;
  InitValueT init_;
  int extend_x_;
  int extend_y_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
  BinaryOp op_;
};

// map one row to one workgroup
template <typename InputT, typename OutputT, typename InitValueT,
          typename InputFunctor, typename OutputFunctor, typename BinaryOp>
struct GroupRowReduction {
  GroupRowReduction(InputT* in_data, OutputT* out_data, InitValueT init,
                    int extend_x, int extend_y, InputFunctor in_func,
                    OutputFunctor out_func, BinaryOp op)
      : in_data_(in_data),
        out_data_(out_data),
        init_(init),
        extend_x_(extend_x),
        extend_y_(extend_y),
        in_func_(in_func),
        out_func_(out_func),
        op_(op) {}
  void operator()(sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    int g_offset = group_id * extend_y_;
    int group_size = item.get_local_range(0);
    InitValueT aggregate = init_;
#pragma unroll
    for (int i = lid; i < extend_y_; i += group_size) {
      InitValueT data = InitValueT(in_func_(in_data_[g_offset + i]));
      aggregate = op_(aggregate, data);
    }
    InitValueT group_aggregate = sycl::reduce_over_group(group, aggregate, op_);
    if (lid == 0) out_data_[group_id] = OutputT(out_func_(group_aggregate));
  }

  InputT* in_data_;
  OutputT* out_data_;
  InitValueT init_;
  int extend_x_;
  int extend_y_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
  BinaryOp op_;
};

// map one row to one subgroup
template <typename InputT, typename OutputT, typename InitValueT,
          typename InputFunctor, typename OutputFunctor, typename BinaryOp,
          int SubGroupSize>
struct SubGroupRowReduction {
  SubGroupRowReduction(InputT* in_data, OutputT* out_data, InitValueT init,
                       int extend_x, int extend_y, InputFunctor in_func,
                       OutputFunctor out_func, BinaryOp op)
      : in_data_(in_data),
        out_data_(out_data),
        init_(init),
        extend_x_(extend_x),
        extend_y_(extend_y),
        in_func_(in_func),
        out_func_(out_func),
        op_(op) {}
  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_id();
    int SubGroupPerGroup = item.get_local_range(0) / SubGroupSize;
    int x_index = group_id * SubGroupPerGroup + subgroup_id;
    if (x_index >= extend_x_ * extend_y_) return;

    InitValueT aggregate = init_;
    int start_offset = x_index * extend_y_;
#pragma unroll
    for (int i = lane_id; i < extend_y_; i += SubGroupSize) {
      InitValueT data = InitValueT(in_func_(in_data_[start_offset + i]));
      aggregate = op_(aggregate, data);
    }
    InitValueT sg_aggregate = sycl::reduce_over_group(sg, aggregate, op_);
    if (lane_id == 0) out_data_[x_index] = OutputT(out_func_(sg_aggregate));
  }

  InputT* in_data_;
  OutputT* out_data_;
  InitValueT init_;
  int extend_x_;
  int extend_y_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
  BinaryOp op_;
};

template <typename InputT, typename OutputT, typename InitValueT,
          typename InputFunctor, typename OutputFunctor, typename BinaryOp>
struct SimpleColReduction {
  SimpleColReduction(InputT* in_data, OutputT* out_data, InitValueT init,
                     int extend_x, int extend_y, int extend_z,
                     InputFunctor in_func, OutputFunctor out_func, BinaryOp op)
      : in_data_(in_data),
        out_data_(out_data),
        init_(init),
        extend_x_(extend_x),
        extend_y_(extend_y),
        extend_z_(extend_z),
        in_func_(in_func),
        out_func_(out_func),
        op_(op) {}
  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();
    const int out_size = extend_x_ * extend_z_;
    if (id < out_size) {
      int outer = id / extend_z_;
      int inner = id - outer * extend_z_;

      int in_offset = outer * extend_y_ * extend_z_ + inner;
      InitValueT aggregate = InitValueT(in_func_(in_data_[in_offset]));
#pragma unroll
      for (int i = 1; i < extend_y_; ++i) {
        InitValueT tmp =
            InitValueT(in_func_(in_data_[in_offset + i * extend_z_]));
        aggregate = op_(aggregate, tmp);
      }
      out_data_[id] = OutputT(out_func_(aggregate));
    }
  }

  InputT* in_data_;
  OutputT* out_data_;
  InitValueT init_;
  int extend_x_;
  int extend_y_;
  int extend_z_;

  InputFunctor in_func_;
  OutputFunctor out_func_;
  BinaryOp op_;
};

template <typename InputT, typename OutputT, typename InitValueT,
          typename LocalAcc, typename InputFunctor, typename OutputFunctor,
          typename BinaryOp, int GroupSize, int SubGroupSize>
struct SubGroupColReductionKernel {
  SubGroupColReductionKernel(InputT* in_data, OutputT* out_data,
                             InitValueT init, LocalAcc scratch, int extend_x,
                             int extend_y, int extend_z, int num_segments_y,
                             int elems_per_item, InputFunctor in_func,
                             OutputFunctor out_func, BinaryOp op)
      : in_data_(in_data),
        out_data_(out_data),
        init_(init),
        scratch_(scratch),
        extend_x_(extend_x),
        extend_y_(extend_y),
        extend_z_(extend_z),
        num_segments_y_(num_segments_y),
        elems_per_item_(elems_per_item),
        in_func_(in_func),
        out_func_(out_func),
        op_(op) {}
  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<3> item) const {
    // get start index
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1);
    int z_group_id = item.get_group(2);
    int num_sub_group = GroupSize / SubGroupSize;

    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_id();

    int x_offset = x_group_id * extend_y_ * extend_z_;

    // each subgroup load data and reduce elems_per_item
    InitValueT aggregate = init_;
    int z_offset = z_group_id * SubGroupSize + lane_id;
    if (z_offset < extend_z_) {
      for (int i = 0; i < elems_per_item_; ++i) {
        int y_idx = y_group_id * num_sub_group * elems_per_item_ +
                    num_sub_group * i + subgroup_id;
        if (y_idx >= extend_y_) break;
        int offset = x_offset + y_idx * extend_z_ + z_offset;
        InitValueT tmp = InitValueT(in_func_(in_data_[offset]));
        aggregate = op_(aggregate, tmp);
      }
    }
    // each subgroup write result to slm
    scratch_[subgroup_id + lane_id * num_sub_group] = aggregate;
    item.barrier(sycl::access::fence_space::local_space);

    // slm reduce and write output
    InitValueT value = scratch_[subgroup_id * num_sub_group + lane_id];
    InitValueT aggregate_over_sg = sycl::reduce_over_group(sg, value, op_);
    z_offset = z_group_id * SubGroupSize + subgroup_id;
    if (z_offset < extend_z_ && lane_id == 0) {
      int offset = x_group_id * extend_z_ * num_segments_y_ +
                   y_group_id * extend_z_ + z_group_id * SubGroupSize +
                   subgroup_id;
      out_data_[offset] = OutputT(out_func_(aggregate_over_sg));
    }
  }

  InputT* in_data_;
  OutputT* out_data_;
  InitValueT init_;
  LocalAcc scratch_;
  const int extend_x_;
  const int extend_y_;
  const int extend_z_;
  const int num_segments_y_;
  const int elems_per_item_;
  InputFunctor in_func_;
  OutputFunctor out_func_;
  BinaryOp op_;
};

}  // namespace reduciton_helper

template <typename InputT, typename OutputT, typename InitValueT, typename Op,
          typename InputFunctor = reduciton_helper::Identity<InputT>,
          typename OutputFunctor = reduciton_helper::Identity<InitValueT>>
void LaunchFullReduction(
    OpKernelContext* ctx, InputT* in, OutputT* out, InitValueT init_val,
    int in_size, Op op,
    InputFunctor in_func = reduciton_helper::Identity<InputT>(),
    OutputFunctor out_func = reduciton_helper::Identity<InitValueT>()) {
  enum { Elems_Per_ITEM = 8 };
  typedef typename std::remove_cv<
      typename std::remove_reference<InputT>::type>::type BaseInputT;

  BaseInputT* in_unqualified = const_cast<BaseInputT*>(in);
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  int max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_group_size);
  const int Max_Elems_Per_Group = group_size * Elems_Per_ITEM * 6;

  if (in_size <= Max_Elems_Per_Group) {
    int elems_per_group = RoundUp(in_size, group_size * Elems_Per_ITEM);
    sycl::range<1> local(group_size);
    sycl::range<1> global(group_size);
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::GroupReduceKernel<BaseInputT, OutputT, InputFunctor,
                                          OutputFunctor, InitValueT, Op,
                                          Elems_Per_ITEM>
          task(in_unqualified, out, in_func, out_func, init_val, in_size,
               elems_per_group, op);
      cgh.parallel_for<reduciton_helper::GroupReduceKernel<
          BaseInputT, OutputT, InputFunctor, OutputFunctor, InitValueT, Op,
          Elems_Per_ITEM>>(sycl::nd_range<1>(global, local), task);
    });
  } else {
    int num_wg =
        std::min(group_size, DivUp(in_size, group_size * Elems_Per_ITEM));

    int elems_per_group =
        RoundUp(DivUp(in_size, num_wg), group_size * Elems_Per_ITEM);
    num_wg = DivUp(in_size, elems_per_group);

    Tensor scratch_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<InitValueT>::value,
                                      TensorShape({num_wg}), &scratch_tensor));
    InitValueT* scratch = scratch_tensor.flat<InitValueT>().data();

    sycl::range<1> local(group_size);
    sycl::range<1> global(group_size * num_wg);
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::GroupReduceKernel<
          BaseInputT, InitValueT, InputFunctor,
          reduciton_helper::Identity<InitValueT>, InitValueT, Op,
          Elems_Per_ITEM>
          task(in_unqualified, scratch, in_func,
               reduciton_helper::Identity<InitValueT>(), init_val, in_size,
               elems_per_group, op);
      cgh.parallel_for<reduciton_helper::GroupReduceKernel<
          BaseInputT, InitValueT, InputFunctor,
          reduciton_helper::Identity<InitValueT>, InitValueT, Op,
          Elems_Per_ITEM>>(sycl::nd_range<1>(global, local), task);
    });

    local = sycl::range<1>(group_size);
    elems_per_group = RoundUp(num_wg, group_size * Elems_Per_ITEM);
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::GroupReduceKernel<
          InitValueT, OutputT, reduciton_helper::Identity<InitValueT>,
          OutputFunctor, InitValueT, Op, Elems_Per_ITEM>
          task(scratch, out, reduciton_helper::Identity<InitValueT>(), out_func,
               init_val, num_wg, elems_per_group, op);
      cgh.parallel_for<reduciton_helper::GroupReduceKernel<
          InitValueT, OutputT, reduciton_helper::Identity<InitValueT>,
          OutputFunctor, InitValueT, Op, Elems_Per_ITEM>>(
          sycl::nd_range<1>(local, local), task);
    });
  }
}

template <typename InputT, typename OutputT, typename InitValueT,
          typename InputFunctor, typename OutputFunctor, typename Op,
          int SubGroupSize>
void launchSugGroupRowReduction(OpKernelContext* ctx, InputT* in_data,
                                OutputT* out_data, InitValueT init_val,
                                int extend_x, int extend_y, int group_size,
                                InputFunctor in_func, OutputFunctor out_func,
                                Op op) {
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  int SubGroupPerGroup = group_size / SubGroupSize;
  int num_wg = DivUp(extend_x, SubGroupPerGroup);

  sycl::range<1> local(group_size);
  sycl::range<1> global(num_wg * group_size);
  sycl::nd_range<1> range(global, local);
  stream->submit([&](sycl::handler& cgh) {
    reduciton_helper::SubGroupRowReduction<InputT, OutputT, InitValueT,
                                           InputFunctor, OutputFunctor, Op,
                                           SubGroupSize>
        sg_reduction(in_data, out_data, init_val, extend_x, extend_y, in_func,
                     out_func, op);

    cgh.parallel_for<reduciton_helper::SubGroupRowReduction<
        InputT, OutputT, InitValueT, InputFunctor, OutputFunctor, Op,
        SubGroupSize>>(range, sg_reduction);
  });
}

template <typename InputT, typename OutputT, typename InitValueT, typename Op,
          typename InputFunctor = reduciton_helper::Identity<InputT>,
          typename OutputFunctor = reduciton_helper::Identity<InitValueT>>
void LaunchRowReduction(
    OpKernelContext* ctx, InputT* in_data, OutputT* out_data,
    InitValueT init_val, int extend_x, int extend_y, Op op,
    InputFunctor in_func = reduciton_helper::Identity<InputT>(),
    OutputFunctor out_func = reduciton_helper::Identity<InitValueT>()) {
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_group_size);
  if (extend_y <= 32) {
    int num_wg = DivUp(extend_x, group_size);
    sycl::nd_range<1> range(num_wg * group_size, group_size);
    auto event = stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::SimpleRowReduction<InputT, OutputT, InitValueT,
                                           InputFunctor, OutputFunctor, Op>
          simple_row_reduction(in_data, out_data, init_val, extend_x, extend_y,
                               in_func, out_func, op);
      cgh.parallel_for<reduciton_helper::SimpleRowReduction<
          InputT, OutputT, InitValueT, InputFunctor, OutputFunctor, Op>>(
          range, simple_row_reduction);
    });

  } else if (extend_y <= 1024) {
    int max_sub_group_size =
        (stream->get_device())
            .template get_info<sycl::info::device::sub_group_sizes>()
            .back();
    if (max_sub_group_size == 32) {
      launchSugGroupRowReduction<InputT, OutputT, InitValueT, InputFunctor,
                                 OutputFunctor, Op, 32>(
          ctx, in_data, out_data, init_val, extend_x, extend_y, group_size,
          in_func, out_func, op);
    } else if (max_sub_group_size == 16) {
      launchSugGroupRowReduction<InputT, OutputT, InitValueT, InputFunctor,
                                 OutputFunctor, Op, 16>(
          ctx, in_data, out_data, init_val, extend_x, extend_y, group_size,
          in_func, out_func, op);
    } else {
      std::stringstream ss;
      ss << "Unsupported row reduce algorithm for max sub group size == "
         << max_group_size << " which is lower than 16";
      ITEX_LOG(FATAL) << ss.str();
    }
  } else {
    sycl::range<1> local(group_size);
    sycl::range<1> global(extend_x * local[0]);
    sycl::nd_range<1> range(global, local);
    auto event = stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::GroupRowReduction<InputT, OutputT, InitValueT,
                                          InputFunctor, OutputFunctor, Op>
          g_reduction(in_data, out_data, init_val, extend_x, extend_y, in_func,
                      out_func, op);
      cgh.parallel_for<reduciton_helper::GroupRowReduction<
          InputT, OutputT, InitValueT, InputFunctor, OutputFunctor, Op>>(
          range, g_reduction);
    });
  }
}

template <typename InputT, typename OutputT, typename InitValueT,
          int MaxGroupSize, int SubGroupSize, typename Op,
          typename InputFunctor, typename OutputFunctor>
void SubGroupColReduction(OpKernelContext* ctx, InputT* in_data,
                          OutputT* out_data, int extend_x, int extend_y,
                          int extend_z, InitValueT init, Op op,
                          InputFunctor in_func, OutputFunctor out_func) {
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
    while (num_segments_y > num_sub_group * ElemsPerItem) {
      elems_per_item <<= 1;
      num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item);
    }

    sycl::range<3> local(1, num_sub_group, SubGroupSize);
    sycl::range<3> global(extend_x, num_segments_y * local[1],
                          num_segments_z * local[2]);
    Tensor scratch_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<InitValueT>::value,
                            TensorShape({extend_x * num_segments_y * extend_z}),
                            &scratch_tensor));
    InitValueT* inter_out = scratch_tensor.flat<InitValueT>().data();

    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<InitValueT> scratch(
          num_sub_group * SubGroupSize, cgh);
      reduciton_helper::SubGroupColReductionKernel<
          InputT, InitValueT, InitValueT,
          reduciton_helper::LocalAcc<InitValueT>, InputFunctor,
          reduciton_helper::Identity<InitValueT>, Op, MaxGroupSize,
          SubGroupSize>
          task(in_data, inter_out, init, scratch, extend_x, extend_y, extend_z,
               num_segments_y, elems_per_item, in_func,
               reduciton_helper::Identity<InitValueT>(), op);
      cgh.parallel_for<reduciton_helper::SubGroupColReductionKernel<
          InputT, InitValueT, InitValueT,
          reduciton_helper::LocalAcc<InitValueT>, InputFunctor,
          reduciton_helper::Identity<InitValueT>, Op, MaxGroupSize,
          SubGroupSize>>(sycl::nd_range<3>(global, local), task);
    });

    global = sycl::range<3>{static_cast<size_t>(extend_x), local[1],
                            num_segments_z * local[2]};
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<InitValueT> scratch(
          num_sub_group * SubGroupSize, cgh);
      reduciton_helper::SubGroupColReductionKernel<
          InitValueT, OutputT, InitValueT,
          reduciton_helper::LocalAcc<InitValueT>,
          reduciton_helper::Identity<InitValueT>, OutputFunctor, Op,
          MaxGroupSize, SubGroupSize>
          task(inter_out, out_data, init, scratch, extend_x, num_segments_y,
               extend_z, 1, ElemsPerItem,
               reduciton_helper::Identity<InitValueT>(), out_func, op);
      cgh.parallel_for<reduciton_helper::SubGroupColReductionKernel<
          InitValueT, OutputT, InitValueT,
          reduciton_helper::LocalAcc<InitValueT>,
          reduciton_helper::Identity<InitValueT>, OutputFunctor, Op,
          MaxGroupSize, SubGroupSize>>(sycl::nd_range<3>(global, local), task);
    });

  } else {
    sycl::range<3> local(1, num_sub_group, SubGroupSize);
    sycl::range<3> global(extend_x, local[1], num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<InitValueT> scratch(
          num_sub_group * SubGroupSize, cgh);
      reduciton_helper::SubGroupColReductionKernel<
          InputT, OutputT, InitValueT, reduciton_helper::LocalAcc<InitValueT>,
          InputFunctor, OutputFunctor, Op, MaxGroupSize, SubGroupSize>
          task(in_data, out_data, init, scratch, extend_x, extend_y, extend_z,
               1, elems_per_item, in_func, out_func, op);
      cgh.parallel_for<reduciton_helper::SubGroupColReductionKernel<
          InputT, OutputT, InitValueT, reduciton_helper::LocalAcc<InitValueT>,
          InputFunctor, OutputFunctor, Op, MaxGroupSize, SubGroupSize>>(
          sycl::nd_range<3>(global, local), task);
    });
  }
}

template <typename InputT, typename OutputT, typename InitValueT, typename Op,
          typename InputFunctor = reduciton_helper::Identity<InputT>,
          typename OutputFunctor = reduciton_helper::Identity<InitValueT>>
void LaunchColReduction(
    OpKernelContext* ctx, InputT* in_data, OutputT* out_data,
    InitValueT init_val, int extend_x, int extend_y, int extend_z, Op op,
    InputFunctor in_func = reduciton_helper::Identity<InputT>(),
    OutputFunctor out_func = reduciton_helper::Identity<InitValueT>()) {
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();

  int elems_per_item = extend_y / (extend_x * extend_z);
  if (elems_per_item < 4) {
    const int out_size = extend_x * extend_z;
    int GroupSize = std::min(512, max_group_size);
    int num_wg = (out_size + GroupSize - 1) / GroupSize;
    sycl::nd_range<1> range(num_wg * GroupSize, GroupSize);

    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::SimpleColReduction<InputT, OutputT, InitValueT,
                                           InputFunctor, OutputFunctor, Op>
          task(in_data, out_data, init_val, extend_x, extend_y, extend_z,
               in_func, out_func, op);
      cgh.parallel_for<reduciton_helper::SimpleColReduction<
          InputT, OutputT, InitValueT, InputFunctor, OutputFunctor, Op>>(range,
                                                                         task);
    });

  } else {
    if (max_group_size >= 1024) {
      SubGroupColReduction<InputT, OutputT, InitValueT, 1024, 32, Op,
                           InputFunctor, OutputFunctor>(
          ctx, in_data, out_data, extend_x, extend_y, extend_z, init_val, op,
          in_func, out_func);
    } else if (max_group_size >= 256) {
      SubGroupColReduction<InputT, OutputT, InitValueT, 256, 16, Op,
                           InputFunctor, OutputFunctor>(
          ctx, in_data, out_data, extend_x, extend_y, extend_z, init_val, op,
          in_func, out_func);
    } else {
      std::stringstream ss;
      ss << "Unsupported col reduce algorithm for group size == "
         << max_group_size << " which is lower than 256";
      ITEX_LOG(FATAL) << ss.str();
    }
  }
}

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_REDUCTION_DPCPP_KERNELS_H_
