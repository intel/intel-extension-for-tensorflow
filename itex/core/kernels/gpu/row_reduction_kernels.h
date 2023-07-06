#ifndef ITEX_CORE_KERNELS_GPU_ROW_REDUCTION_KERNELS_H_
#define ITEX_CORE_KERNELS_GPU_ROW_REDUCTION_KERNELS_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

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
      : in_data(in_data),
        out_data(out_data),
        init(init),
        extend_x(extend_x),
        extend_y(extend_y),
        in_func(in_func),
        out_func(out_func),
        op(op) {}

  static constexpr int GROUP_SIZE = 256;
  static constexpr int ITEMS_PER_THREAD = 8;
  static constexpr int TILES_SIZE = ITEMS_PER_THREAD * GROUP_SIZE;
  static constexpr int VEC_LENGTH = 4 * sizeof(float) / sizeof(InputT);
  static constexpr int WORDS = ITEMS_PER_THREAD / VEC_LENGTH;

  // valid_items < TILES_SIZE
  inline void ConsumRange(sycl::nd_item<1> item, int offset, int valid_items,
                          Int2Type<false> /*is_full_tile*/,
                          InitValueT* aggregate) const {
    auto lid = item.get_local_linear_id();
    auto g = item.get_group();

    InitValueT thread_aggregate = init;
#pragma unroll
    for (int index = offset + lid; index < offset + valid_items;
         index += GROUP_SIZE) {
      InitValueT data = InitValueT(in_func(in_data[index]));
      thread_aggregate = op(thread_aggregate, data);
    }
    InitValueT updated_thread_aggregate =
        sycl::reduce_over_group(g, thread_aggregate, op);
    *aggregate = op(*aggregate, updated_thread_aggregate);
  }

  inline void ConsumRange(sycl::nd_item<1> item, int offset,
                          int /*valid_items*/, Int2Type<true> /*is_full_tile*/,
                          InitValueT* aggregate) const {
    auto lid = item.get_local_linear_id();
    auto g = item.get_group();

    typedef sycl::vec<InputT, VEC_LENGTH> VecT;
    InputT input_items[ITEMS_PER_THREAD];
    VecT* vec_items = reinterpret_cast<VecT*>(input_items);

    VecT* vec_in =
        reinterpret_cast<VecT*>(in_data + offset + (lid * VEC_LENGTH));

#pragma unroll
    for (int i = 0; i < WORDS; ++i) {
      vec_items[i] = vec_in[i * GROUP_SIZE];
    }

    InitValueT thread_aggregate = init;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      thread_aggregate =
          op(thread_aggregate, InitValueT(in_func(input_items[i])));
    }

    InitValueT updated_thread_aggregate =
        sycl::reduce_over_group(g, thread_aggregate, op);
    *aggregate = op(*aggregate, updated_thread_aggregate);
  }

  void operator()(sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    auto lid = item.get_local_linear_id();
    InitValueT aggregate(init);
    int start_offset = group_id * extend_y;
    if (extend_y < TILES_SIZE) {
      ConsumRange(item, start_offset, extend_y, Int2Type<false>(), &aggregate);
    } else {
      int loops = extend_y / TILES_SIZE;
      int extra = extend_y - loops * TILES_SIZE;
#pragma unroll
      for (int i = 0; i < loops; ++i) {
        ConsumRange(item, start_offset + i * TILES_SIZE, TILES_SIZE,
                    Int2Type<true>(), &aggregate);
      }
      if (extra)
        ConsumRange(item, start_offset + loops * TILES_SIZE, extra,
                    Int2Type<false>(), &aggregate);
    }

    if (lid == 0) out_data[group_id] = OutputT(out_func(aggregate));
  }

  InputT* in_data;
  OutputT* out_data;
  InitValueT init;
  int extend_x;
  int extend_y;
  InputFunctor in_func;
  OutputFunctor out_func;
  BinaryOp op;
};

// map one row to one workgroup
template <typename OutputT, typename InitValueT, typename InputFunctor,
          typename OutputFunctor, typename BinaryOp>
struct GroupRowReduction<Eigen::bfloat16, OutputT, InitValueT, InputFunctor,
                         OutputFunctor, BinaryOp> {
  typedef Eigen::bfloat16 InputT;
  GroupRowReduction(InputT* in_data, OutputT* out_data, InitValueT init,
                    int extend_x, int extend_y, InputFunctor in_func,
                    OutputFunctor out_func, BinaryOp op)
      : in_data(in_data),
        out_data(out_data),
        init(init),
        extend_x(extend_x),
        extend_y(extend_y),
        in_func(in_func),
        out_func(out_func),
        op(op) {}

  static constexpr int GROUP_SIZE = 256;
  static constexpr int ITEMS_PER_THREAD = 8;
  static constexpr int TILES_SIZE = ITEMS_PER_THREAD * GROUP_SIZE;
  static constexpr int VEC_LENGTH = 4 * sizeof(float) / sizeof(InputT);
  static constexpr int WORDS = ITEMS_PER_THREAD / VEC_LENGTH;

  inline void ConsumRange(sycl::nd_item<1> item, int offset,
                          int /*valid_items*/, Int2Type<true> /*is_full_tile*/,
                          InitValueT* aggregate) const {
    auto lid = item.get_local_linear_id();
    auto g = item.get_group();

    typedef sycl::vec<uint16_t, VEC_LENGTH> VecT;
    uint16_t input_items[ITEMS_PER_THREAD];
    VecT* vec_items = reinterpret_cast<VecT*>(input_items);

    VecT* vec_in =
        reinterpret_cast<VecT*>(in_data + offset + (lid * VEC_LENGTH));

#pragma unroll
    for (int i = 0; i < WORDS; ++i) {
      vec_items[i] = vec_in[i * GROUP_SIZE];
    }

    InitValueT thread_aggregate = init;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      InputT data =
          Eigen::bfloat16_impl::raw_uint16_to_bfloat16(input_items[i]);
      thread_aggregate = op(thread_aggregate, InitValueT(in_func(data)));
    }

    InitValueT updated_thread_aggregate =
        sycl::reduce_over_group(g, thread_aggregate, op);
    *aggregate = op(*aggregate, updated_thread_aggregate);
  }

  // valid_items < TILES_SIZE
  inline void ConsumRange(sycl::nd_item<1> item, int offset, int valid_items,
                          Int2Type<false> /*is_full_tile*/,
                          InitValueT* aggregate) const {
    auto lid = item.get_local_id(0);
    auto g = item.get_group();

    InitValueT thread_aggregate = init;
#pragma unroll
    for (int index = offset + lid; index < offset + valid_items;
         index += GROUP_SIZE) {
      InitValueT data = InitValueT(in_func(in_data[index]));
      thread_aggregate = op(thread_aggregate, data);
    }
    InitValueT updated_thread_aggregate =
        sycl::reduce_over_group(g, thread_aggregate, op);
    *aggregate = op(*aggregate, updated_thread_aggregate);
  }

  void operator()(sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    auto lid = item.get_local_linear_id();
    InitValueT aggregate(init);
    int start_offset = group_id * extend_y;
    if (extend_y < TILES_SIZE) {
      ConsumRange(item, start_offset, extend_y, Int2Type<false>(), &aggregate);
    } else {
      int loops = extend_y / TILES_SIZE;
      int extra = extend_y - loops * TILES_SIZE;
#pragma unroll
      for (int i = 0; i < loops; ++i) {
        ConsumRange(item, start_offset + i * TILES_SIZE, TILES_SIZE,
                    Int2Type<true>(), &aggregate);
      }
      if (extra)
        ConsumRange(item, start_offset + loops * TILES_SIZE, extra,
                    Int2Type<false>(), &aggregate);
    }

    if (lid == 0) out_data[group_id] = OutputT(out_func(aggregate));
  }

  InputT* in_data;
  OutputT* out_data;
  InitValueT init;
  int extend_x;
  int extend_y;
  InputFunctor in_func;
  OutputFunctor out_func;
  BinaryOp op;
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
    SubGroupRowReduction<InputT, OutputT, InitValueT, InputFunctor,
                         OutputFunctor, Op, SubGroupSize>
        sg_reduction(in_data, out_data, init_val, extend_x, extend_y, in_func,
                     out_func, op);

    cgh.parallel_for<
        SubGroupRowReduction<InputT, OutputT, InitValueT, InputFunctor,
                             OutputFunctor, Op, SubGroupSize>>(range,
                                                               sg_reduction);
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

  typedef typename std::remove_cv<
      typename std::remove_reference<InputT>::type>::type BaseInputT;
  BaseInputT* in_unqualified = const_cast<BaseInputT*>(in_data);

  if (extend_y <= 32) {
    int group_size = std::min(512, max_group_size);
    int num_wg = DivUp(extend_x, group_size);
    sycl::nd_range<1> range(num_wg * group_size, group_size);
    auto event = stream->submit([&](sycl::handler& cgh) {
      SimpleRowReduction<BaseInputT, OutputT, InitValueT, InputFunctor,
                         OutputFunctor, Op>
          simple_row_reduction(in_unqualified, out_data, init_val, extend_x,
                               extend_y, in_func, out_func, op);
      cgh.parallel_for<SimpleRowReduction<BaseInputT, OutputT, InitValueT,
                                          InputFunctor, OutputFunctor, Op>>(
          range, simple_row_reduction);
    });

  } else if (extend_y <= 1024) {
    int group_size = std::min(512, max_group_size);
    int max_sub_group_size =
        (stream->get_device())
            .template get_info<sycl::info::device::sub_group_sizes>()
            .back();
    if (max_sub_group_size == 32) {
      launchSugGroupRowReduction<BaseInputT, OutputT, InitValueT, InputFunctor,
                                 OutputFunctor, Op, 32>(
          ctx, in_unqualified, out_data, init_val, extend_x, extend_y,
          group_size, in_func, out_func, op);
    } else if (max_sub_group_size == 16) {
      launchSugGroupRowReduction<BaseInputT, OutputT, InitValueT, InputFunctor,
                                 OutputFunctor, Op, 16>(
          ctx, in_unqualified, out_data, init_val, extend_x, extend_y,
          group_size, in_func, out_func, op);
    } else {
      std::stringstream ss;
      ss << "Unsupported row reduce algorithm for max sub group size == "
         << max_group_size << " which is lower than 16";
      ITEX_LOG(FATAL) << ss.str();
    }
  } else {
    int group_size =
        std::min(GroupRowReduction<BaseInputT, OutputT, InitValueT,
                                   InputFunctor, OutputFunctor, Op>::GROUP_SIZE,
                 max_group_size);
    sycl::range<1> local(group_size);
    sycl::range<1> global(extend_x * local[0]);
    sycl::nd_range<1> range(global, local);
    stream->submit([&](sycl::handler& cgh) {
      GroupRowReduction<BaseInputT, OutputT, InitValueT, InputFunctor,
                        OutputFunctor, Op>
          g_reduction(in_unqualified, out_data, init_val, extend_x, extend_y,
                      in_func, out_func, op);
      cgh.parallel_for<GroupRowReduction<BaseInputT, OutputT, InitValueT,
                                         InputFunctor, OutputFunctor, Op>>(
          range, g_reduction);
    });
  }
}
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_ROW_REDUCTION_KERNELS_H_
