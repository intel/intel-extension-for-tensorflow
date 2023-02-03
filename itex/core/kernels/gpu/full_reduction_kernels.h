#ifndef ITEX_CORE_KERNELS_GPU_FULL_REDUCTION_KERNELS_H_
#define ITEX_CORE_KERNELS_GPU_FULL_REDUCTION_KERNELS_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

struct FullReducePolicy {
  static constexpr int ITEMS_PER_THREAD = 8;
  static constexpr int VEC_LENGTH = 4;
  static constexpr int WORDS = ITEMS_PER_THREAD / VEC_LENGTH;
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
    auto lid = item.get_local_id(0);
    auto g = item.get_group();
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
