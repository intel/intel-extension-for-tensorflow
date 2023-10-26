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

#ifndef ITEX_CORE_KERNELS_GPU_LINALG_FUSED_EINSUM_HELPER_H_
#define ITEX_CORE_KERNELS_GPU_LINALG_FUSED_EINSUM_HELPER_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"

namespace itex {

namespace functor {

template <int wg_m_, int wg_n_, int sg_m_, int sg_n_, bool a_row_major_,
          bool b_row_major_, bool c_row_major_>
struct FusedEinsumConfig {
  static constexpr int wg_m = wg_m_;
  static constexpr int wg_n = wg_n_;
  static constexpr int sg_m = sg_m_;
  static constexpr int sg_n = sg_n_;
  static constexpr bool a_row_major = a_row_major_;
  static constexpr bool b_row_major = b_row_major_;
  static constexpr bool c_row_major = c_row_major_;
};

#define DIMS_STRUCT(dim)             \
  struct DIMS##dim {                 \
    static constexpr int dims = dim; \
  };
DIMS_STRUCT(4);
DIMS_STRUCT(3);
DIMS_STRUCT(2);
#undef DIMS_STRUCT

template <typename T>
struct FusedEinsum {
  struct Arguments {
    OpKernelContext* ctx_;
    const T* lhs_;
    std::vector<int>& lhs_shape_;
    std::vector<int>& lhs_stride_;
    const T* rhs_;
    std::vector<int>& rhs_shape_;
    std::vector<int>& rhs_stride_;
    std::vector<int>& out_shape_;
    std::vector<int>& out_stride_;
    int min_batch_;
    TensorShape& result_shape_;
    bool finish_;

    Arguments(OpKernelContext* ctx, const T* lhs,
              std::vector<int>& lhs_shape,                 // NOLINT
              std::vector<int>& lhs_stride, const T* rhs,  // NOLINT
              std::vector<int>& rhs_shape,                 // NOLINT
              std::vector<int>& rhs_stride,                // NOLINT
              std::vector<int>& out_shape,                 // NOLINT
              std::vector<int>& out_stride,                // NOLINT
              int min_batch, TensorShape& result_shape)    // NOLINT
        : ctx_(ctx),
          lhs_(lhs),
          lhs_shape_(lhs_shape),
          lhs_stride_(lhs_stride),
          rhs_(rhs),
          rhs_shape_(rhs_shape),
          rhs_stride_(rhs_stride),
          out_shape_(out_shape),
          out_stride_(out_stride),
          min_batch_(min_batch),
          result_shape_(result_shape),
          finish_(false) {}
  };
  void operator()(Arguments& args);  // NOLINT
};

// clang-format off
#define FusedEinsumConfigSet                               \
  FusedEinsumConfig<256, 128, 32, 32, true, false, false>, \
  FusedEinsumConfig<256, 128, 32, 32, true, false, true>,  \
  FusedEinsumConfig<256, 128, 32, 32, true, true, true>
  // FusedEinsumConfig<256, 128, 32, 32, false, true, true>,
  // FusedEinsumConfig<256, 128, 32, 32, false, false, false>
// clang-format on

namespace einsum_dispatcher {

static constexpr int MIN_EU_NUM = 448;
static constexpr int MAX_EU_NUM = 512;

template <typename T>
using Args = typename FusedEinsum<T>::Arguments;

template <typename T>
using DispatchCallback =
    std::pair<std::function<void(Args<T>& args)>, int>;  // NOLINT

template <typename T, typename Config, typename... Configs>
DispatchCallback<T> FindOrDie(Args<T>& args);  // NOLINT

template <typename T>
void Dispatch(Args<T>& args) {  // NOLINT
  auto result = FindOrDie<T, FusedEinsumConfigSet>(args);
  if (result.first != nullptr) {
    result.first(args);
    args.finish_ = true;
  }
}
}  // namespace einsum_dispatcher

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_LINALG_FUSED_EINSUM_HELPER_H_
