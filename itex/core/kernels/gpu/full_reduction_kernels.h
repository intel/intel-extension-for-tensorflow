#ifndef ITEX_CORE_KERNELS_GPU_FULL_REDUCTION_KERNELS_H_
#define ITEX_CORE_KERNELS_GPU_FULL_REDUCTION_KERNELS_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

struct FullReducePolicy {
  static constexpr int ITEMS_PER_THREAD = 8;
};

template <typename InputT, typename OutputT, typename InputFunctor,
          typename OutputFunctor, typename InitValueT, typename Op>
struct GroupReduceKernel {
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
    constexpr int VEC_LENGTH = sizeof(float) * 4 / sizeof(InputT);
    constexpr int WORDS = FullReducePolicy::ITEMS_PER_THREAD / VEC_LENGTH;

    auto lid = item.get_local_id(0);
    auto g = item.get_group();
    auto group_id = item.get_group(0);
    auto group_size = item.get_local_range(0);

    typedef sycl::vec<InputT, VEC_LENGTH> VecT;
    InputT input_items[FullReducePolicy::ITEMS_PER_THREAD];
    VecT* vec_items = reinterpret_cast<VecT*>(input_items);

    int group_start = group_id * elems_per_group_;

    InitValueT aggregate = init_val_;
    int loops =
        elems_per_group_ / (group_size * FullReducePolicy::ITEMS_PER_THREAD);
    for (int l = 0; l < loops; ++l) {
      int start_offset =
          group_start + l * group_size * FullReducePolicy::ITEMS_PER_THREAD;
      VecT* vec_in =
          reinterpret_cast<VecT*>(in_data_ + start_offset + (lid * VEC_LENGTH));

#pragma unroll
      for (int i = 0; i < WORDS; ++i) {
        vec_items[i] = vec_in[i * group_size];
      }

#pragma unroll
      for (int i = 0; i < FullReducePolicy::ITEMS_PER_THREAD; ++i) {
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

// TODO(itex): it's a workaround for the case when op is mulitplies and data
// type is int64, currently compiler has accuracy issue for that, remove this
// workaround when compiler fix is ready
template <typename OutputT, typename InputFunctor, typename OutputFunctor,
          typename InitValueT, typename Op>
struct GroupReduceKernel<itex::int64, OutputT, InputFunctor, OutputFunctor,
                         InitValueT, Op> {
  typedef itex::int64 InputT;
  static constexpr int SubGroupSize = 16;
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
  [[sycl::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<1> item) const {
    auto group_id = item.get_group(0);
    if ((group_id + 1) * elems_per_group_ > in_size_)
      ConsumRange(item, Int2Type<false>());
    else
      ConsumRange(item, Int2Type<true>());
  }

  inline InitValueT SGReduce(sycl::sub_group sg, const InitValueT value) const {
    InitValueT result = value;
#pragma unroll
    for (int i = SubGroupSize / 2; i > 0; i >>= 1) {
      InitValueT new_value = sg.shuffle_down(result, i);
      result = op_(result, new_value);
    }
    return result;
  }

  inline InitValueT GroupReduce(sycl::nd_item<1> item,
                                const InitValueT data) const {
    auto sg = item.get_sub_group();
    auto sg_id = sg.get_group_linear_id();
    auto lane_id = sg.get_local_linear_id();
    const int sg_num = sg.get_group_linear_range();

    // subgroup number is expected to <= 32
    sycl::multi_ptr<InitValueT[32], sycl::access::address_space::local_space>
        scratch = sycl::ext::oneapi::group_local_memory<InitValueT[32]>(
            item.get_group());
    auto* ref_scratch = scratch.get();

    int cur_sg_num = sg_num;
    InitValueT result = SGReduce(sg, data);
    while (cur_sg_num > 1) {
      if (lane_id == 0 && sg_id < cur_sg_num) ref_scratch[0][sg_id] = result;
      item.barrier(sycl::access::fence_space::local_space);
      if (sg_id < std::max(cur_sg_num / SubGroupSize, 1)) {
        int offset = sg_id * SubGroupSize + lane_id;
        if (offset < cur_sg_num)
          result = ref_scratch[0][offset];
        else
          result = init_val_;
        result = SGReduce(sg, result);
      }
      cur_sg_num /= SubGroupSize;
    }
    return result;
  }

  void ConsumRange(sycl::nd_item<1> item, Int2Type<false> /*can_vec*/) const {
    auto lid = item.get_local_id(0);
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
    InitValueT res = GroupReduce(item, sum);
    if (lid == 0) out_data_[group_id] = OutputT(out_func_(res));
  }

  inline void ConsumRange(sycl::nd_item<1> item,
                          Int2Type<true> /*can_vec*/) const {
    auto lid = item.get_local_id(0);
    auto group_id = item.get_group(0);
    auto group_size = item.get_local_range(0);

    typedef sycl::vec<InputT, FullReducePolicy::VEC_LENGTH> VecT;
    InputT input_items[FullReducePolicy::ITEMS_PER_THREAD];
    VecT* vec_items = reinterpret_cast<VecT*>(input_items);

    int group_start = group_id * elems_per_group_;

    InitValueT aggregate = init_val_;
    int loops =
        elems_per_group_ / (group_size * FullReducePolicy::ITEMS_PER_THREAD);
    for (int l = 0; l < loops; ++l) {
      int start_offset =
          group_start + l * group_size * FullReducePolicy::ITEMS_PER_THREAD;
      VecT* vec_in = reinterpret_cast<VecT*>(
          in_data_ + start_offset + (lid * FullReducePolicy::VEC_LENGTH));

#pragma unroll
      for (int i = 0; i < FullReducePolicy::WORDS; ++i) {
        vec_items[i] = vec_in[i * group_size];
      }

#pragma unroll
      for (int i = 0; i < FullReducePolicy::ITEMS_PER_THREAD; ++i) {
        aggregate = op_(aggregate, InitValueT(in_func_(input_items[i])));
      }
    }
    InitValueT res = GroupReduce(item, aggregate);
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
          typename InitValueT, typename Op>
struct GroupReduceKernel<Eigen::bfloat16, OutputT, InputFunctor, OutputFunctor,
                         InitValueT, Op> {
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
    auto group_id = item.get_group(0);
    if ((group_id + 1) * elems_per_group_ > in_size_)
      ConsumRange(item, Int2Type<false>());
    else
      ConsumRange(item, Int2Type<true>());
  }

  inline void ConsumRange(sycl::nd_item<1> item,
                          Int2Type<true> /*can_vec*/) const {
    constexpr int VEC_LENGTH = sizeof(float) * 4 / sizeof(InputT);
    constexpr int WORDS = FullReducePolicy::ITEMS_PER_THREAD / VEC_LENGTH;

    auto lid = item.get_local_id(0);
    auto g = item.get_group();
    auto group_id = item.get_group(0);
    auto group_size = item.get_local_range(0);

    typedef sycl::vec<uint16_t, VEC_LENGTH> VecT;
    uint16_t input_items[FullReducePolicy::ITEMS_PER_THREAD];
    VecT* vec_items = reinterpret_cast<VecT*>(input_items);

    int group_start = group_id * elems_per_group_;

    InitValueT aggregate = init_val_;
    int loops =
        elems_per_group_ / (group_size * FullReducePolicy::ITEMS_PER_THREAD);
    for (int l = 0; l < loops; ++l) {
      int start_offset =
          group_start + l * group_size * FullReducePolicy::ITEMS_PER_THREAD;
      VecT* vec_in =
          reinterpret_cast<VecT*>(in_data_ + start_offset + (lid * VEC_LENGTH));

#pragma unroll
      for (int i = 0; i < WORDS; ++i) {
        vec_items[i] = vec_in[i * group_size];
      }

#pragma unroll
      for (int i = 0; i < FullReducePolicy::ITEMS_PER_THREAD; ++i) {
        InputT data =
            Eigen::bfloat16_impl::raw_uint16_to_bfloat16(input_items[i]);
        aggregate = op_(aggregate, InitValueT(in_func_(data)));
      }
    }
    InitValueT res = sycl::reduce_over_group(g, aggregate, op_);
    if (lid == 0) out_data_[group_id] = OutputT(out_func_(res));
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

template <typename InputT, typename OutputT, typename InitValueT, typename Op,
          typename InputFunctor = reduciton_helper::Identity<InputT>,
          typename OutputFunctor = reduciton_helper::Identity<InitValueT>>
void LaunchFullReduction(
    OpKernelContext* ctx, InputT* in, OutputT* out, InitValueT init_val,
    int in_size, Op op,
    InputFunctor in_func = reduciton_helper::Identity<InputT>(),
    OutputFunctor out_func = reduciton_helper::Identity<InitValueT>()) {
  typedef typename std::remove_cv<
      typename std::remove_reference<InputT>::type>::type BaseInputT;

  BaseInputT* in_unqualified = const_cast<BaseInputT*>(in);
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  int max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_group_size);
  const int Max_Elems_Per_Group =
      group_size * FullReducePolicy::ITEMS_PER_THREAD * 6;

  if (in_size <= Max_Elems_Per_Group) {
    int elems_per_group =
        RoundUp(in_size, group_size * FullReducePolicy::ITEMS_PER_THREAD);
    sycl::range<1> local(group_size);
    sycl::range<1> global(group_size);
    stream->submit([&](sycl::handler& cgh) {
      GroupReduceKernel<BaseInputT, OutputT, InputFunctor, OutputFunctor,
                        InitValueT, Op>
          task(in_unqualified, out, in_func, out_func, init_val, in_size,
               elems_per_group, op);
      cgh.parallel_for<GroupReduceKernel<BaseInputT, OutputT, InputFunctor,
                                         OutputFunctor, InitValueT, Op>>(
          sycl::nd_range<1>(global, local), task);
    });
  } else {
    int num_wg = std::min(
        group_size,
        DivUp(in_size, group_size * FullReducePolicy::ITEMS_PER_THREAD));

    int elems_per_group =
        RoundUp(DivUp(in_size, num_wg),
                group_size * FullReducePolicy::ITEMS_PER_THREAD);
    num_wg = DivUp(in_size, elems_per_group);

    Tensor scratch_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<InitValueT>::value,
                                      TensorShape({num_wg}), &scratch_tensor));
    InitValueT* scratch = scratch_tensor.flat<InitValueT>().data();

    sycl::range<1> local(group_size);
    sycl::range<1> global(group_size * num_wg);
    stream->submit([&](sycl::handler& cgh) {
      GroupReduceKernel<BaseInputT, InitValueT, InputFunctor,
                        reduciton_helper::Identity<InitValueT>, InitValueT, Op>
          task(in_unqualified, scratch, in_func,
               reduciton_helper::Identity<InitValueT>(), init_val, in_size,
               elems_per_group, op);
      cgh.parallel_for<GroupReduceKernel<BaseInputT, InitValueT, InputFunctor,
                                         reduciton_helper::Identity<InitValueT>,
                                         InitValueT, Op>>(
          sycl::nd_range<1>(global, local), task);
    });

    local = sycl::range<1>(group_size);
    elems_per_group =
        RoundUp(num_wg, group_size * FullReducePolicy::ITEMS_PER_THREAD);
    stream->submit([&](sycl::handler& cgh) {
      GroupReduceKernel<InitValueT, OutputT,
                        reduciton_helper::Identity<InitValueT>, OutputFunctor,
                        InitValueT, Op>
          task(scratch, out, reduciton_helper::Identity<InitValueT>(), out_func,
               init_val, num_wg, elems_per_group, op);
      cgh.parallel_for<GroupReduceKernel<InitValueT, OutputT,
                                         reduciton_helper::Identity<InitValueT>,
                                         OutputFunctor, InitValueT, Op>>(
          sycl::nd_range<1>(local, local), task);
    });
  }
}

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_FULL_REDUCTION_KERNELS_H_
