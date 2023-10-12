/*******************************************************************************
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in wriscalar_tg, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "itex/core/kernels/gpu/linalg/fused_einsum_impl.h"

#include "itex/core/kernels/gpu/linalg/einsum_helper.h"
#include "itex/core/utils/op_requires.h"

namespace itex {
namespace functor {

#define DIMS_SWITCH(dims, DimsT, ...) \
  switch (dims) {                     \
    case 4: {                         \
      using DimsT = DIMS4;            \
      { __VA_ARGS__ }                 \
      break;                          \
    }                                 \
    case 3: {                         \
      using DimsT = DIMS3;            \
      { __VA_ARGS__ }                 \
      break;                          \
    }                                 \
    default: {                        \
      using DimsT = DIMS2;            \
      { __VA_ARGS__ }                 \
      break;                          \
    }                                 \
  }

template <typename T>
using DispatchCallback = einsum_dispatcher::DispatchCallback<T>;

template <typename T>
using Args = einsum_dispatcher::Args<T>;

template <typename T, typename Config, typename... Configs>
DispatchCallback<T> einsum_dispatcher::FindOrDie(Args<T>& args) {  // NOLINT
  int batch_num = 1;
  std::for_each(args.out_shape_.begin(), args.out_shape_.end() - 2,
                [&](int val) { batch_num *= val; });

  int dims = args.out_shape_.size();
  int group_range_n =
      (args.out_shape_[dims - 1] + Config::wg_n - 1) / Config::wg_n;
  int group_range_m =
      (args.out_shape_[dims - 2] + Config::wg_m - 1) / Config::wg_m;
  int eu_num = group_range_n * group_range_m * batch_num;

  bool a_row_major = args.lhs_stride_.back() == 1;
  bool b_row_major = args.rhs_stride_.back() == 1;
  bool c_row_major = args.out_stride_.back() == 1;

  DispatchCallback<T> callback(nullptr, -1);
  if (eu_num >= MIN_EU_NUM && a_row_major == Config::a_row_major &&
      b_row_major == Config::b_row_major &&
      c_row_major == Config::c_row_major) {
    auto executor = [=](Args<T>& args) {
      const auto& d = (args.ctx_)->eigen_gpu_device();
      auto& stream = d.stream();

      Tensor* result;
      OP_REQUIRES_OK(args.ctx_, args.ctx_->allocate_output(
                                    0, args.result_shape_, &result));
      auto out = result->flat<T>().data();

      constexpr uint32_t wg_m = Config::wg_m;
      constexpr uint32_t wg_n = Config::wg_n;
      constexpr uint32_t sg_m = Config::sg_m;
      constexpr uint32_t sg_n = Config::sg_n;

      uint32_t subgroup_range_m = (wg_m + sg_m - 1) / sg_m;
      uint32_t subgroup_range_n = (wg_n + sg_n - 1) / sg_n;

      cl::sycl::range<3> group_range{static_cast<size_t>(batch_num),
                                     static_cast<size_t>(group_range_m),
                                     static_cast<size_t>(group_range_n)};
      cl::sycl::range<3> local_range{1, static_cast<size_t>(subgroup_range_m),
                                     static_cast<size_t>(subgroup_range_n)};
      cl::sycl::nd_range<3> thread_range(group_range * local_range,
                                         local_range);

      using InT =
          std::conditional_t<std::is_same_v<T, float>, float,
                             std::conditional_t<std::is_same_v<T, Eigen::half>,
                                                sycl::half, gpu::xetla::bf16>>;

#define CAST(ptr, src_t, dst_t) \
  reinterpret_cast<dst_t*>(const_cast<src_t*>(ptr))

      DIMS_SWITCH(dims, DimsT, stream->submit([&](sycl::handler& cgh) {
        gpu::xetla::FusedEinsumKernel<InT, DimsT::dims, wg_m, wg_n, sg_m, sg_n,
                                      Config::a_row_major, Config::b_row_major,
                                      Config::c_row_major>
            task(CAST(args.lhs_, T, InT), args.lhs_shape_, args.lhs_stride_,
                 CAST(args.rhs_, T, InT), args.rhs_shape_, args.rhs_stride_,
                 CAST(out, T, InT), args.out_shape_, args.out_stride_,
                 args.min_batch_);
        cgh.parallel_for<class gpu::xetla::FusedEinsumKernel<
            InT, DimsT::dims, wg_m, wg_n, sg_m, sg_n, Config::a_row_major,
            Config::b_row_major, Config::c_row_major>>(thread_range, task);
      }););
    };
    callback.first = executor;
    callback.second = eu_num;
  }

  if constexpr (sizeof...(Configs) == 0) {
    return callback;
  } else {
    auto result = einsum_dispatcher::FindOrDie<T, Configs...>(args);
    if (callback.first != nullptr &&
        (result.first == nullptr ||
         (callback.second <= MAX_EU_NUM &&
          (callback.second > result.second || result.second > MAX_EU_NUM)) ||
         (callback.second > MAX_EU_NUM && callback.second < result.second))) {
      return callback;
    }
    return result;
  }
#undef CAST
}

#undef DIMS_SWITCH

#define DISPATCHER_DECLARATION(T)                                     \
  template DispatchCallback<T>                                        \
      einsum_dispatcher::FindOrDie<T, FusedEinsumConfigSet>(Args<T> & \
                                                            args);  // NOLINT

DISPATCHER_DECLARATION(float);
DISPATCHER_DECLARATION(Eigen::half);
DISPATCHER_DECLARATION(Eigen::bfloat16);

}  // namespace functor
}  // namespace itex
