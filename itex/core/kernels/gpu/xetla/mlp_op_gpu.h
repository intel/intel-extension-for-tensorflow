/* Copyright (c) 2023 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_GPU_XETLA_MLP_OP_GPU_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_MLP_OP_GPU_H_

#include <stdint.h>

#include <xetla.hpp>

namespace gpu::xetla {

struct KernelAttr {
  static constexpr uint32_t wg_tile_m = 256;
  static constexpr uint32_t wg_tile_n = 256;
  static constexpr uint32_t sg_tile_m = 32;
  static constexpr uint32_t sg_tile_n = 64;
  static constexpr uint32_t sg_tile_k = 32;
  using data_type_acc = float;
  static constexpr post_kind post_op = post_kind::gelu_bwd_w;
};

template <typename dtype_a, typename dtype_b, typename dtype_c,
          typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
          uint32_t sg_n, uint32_t sg_k, mem_layout layout_a,
          mem_layout layout_b, post_kind post_op_kind>
struct fused_dense_func {
  using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
  static constexpr uint32_t periodic_sync_interval = 8;
  static constexpr uint32_t prefetch_distance = 3;
  using brgemm_t = typename group::brgemm_selector_t<
      dtype_a, dtype_b, layout_a, layout_b, mem_space::global,
      mem_space::global, 8, 8, dtype_acc, tile_shape, sg_k, mma_engine::xmx,
      gpu_arch::Xe, prefetch_distance, periodic_sync_interval>::brgemm;

  using bias_op_t = typename subgroup::bias_add_op_t<dtype_c, gpu_arch::Xe>;
  using gelu_op_t = typename std::conditional<
      post_op_kind == post_kind::gelu, subgroup::gelu_fwd_op_t,
      subgroup::gelu_fwd_w_op_t<dtype_c, gpu_arch::Xe>>::type;
  using post_op_t = subgroup::chained_tile_op_t<bias_op_t, gelu_op_t>;

  using epilogue_t = group::epilogue_t<
      group::epilogue_policy_tile_op<post_op_t, result_overwrite, gpu_arch::Xe>,
      tile_shape,
      mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>>;
  using gemm_op_t =
      kernel::gemm_t<kernel::dispatch_policy_default<gpu_arch::Xe>, brgemm_t,
                     epilogue_t>;

  static constexpr uint32_t barrier_count = gemm_op_t::brgemm_t::barrier_count;
  static constexpr uint32_t slm_size = gemm_op_t::brgemm_t::slm_size;

  static inline void run(xetla_exec_item<3>* ei, dtype_a* A, dtype_b* B,
                         dtype_c* C, dtype_c* bias, uint32_t mat_m,
                         uint32_t mat_n, uint32_t mat_k, uint32_t lda,
                         uint32_t ldb, uint32_t ldc, dtype_c* gelu_w) {
    gemm_op_t gemm_op;
    typename bias_op_t::arguments_t bias_op_arg({bias}, {mat_n, 1, mat_n});
    if constexpr (post_op_kind == post_kind::gelu) {
      typename gemm_op_t::arguments_t args(mat_m, mat_k, mat_n, A, lda, B, ldb,
                                           C, ldc, {{bias_op_arg, {}}});
      gemm_op(*ei, args);
    } else {
      typename gelu_op_t::arguments_t gelu_op_arg({gelu_w},
                                                  {mat_n, mat_m, ldc});
      typename gemm_op_t::arguments_t args(mat_m, mat_k, mat_n, A, lda, B, ldb,
                                           C, ldc,
                                           {{bias_op_arg, gelu_op_arg}});
      gemm_op(*ei, args);
    }
  }
};

template <typename T>
class FusedDenseBiasAddGeluKernel {
 public:
  explicit FusedDenseBiasAddGeluKernel(T* feature_ptr, T* weights_ptr,
                                       T* bias_ptr, T* output_ptr,
                                       T* workspace_ptr,
                                       const uint32_t matrix_m,
                                       const uint32_t matrix_n,
                                       const uint32_t matrix_k)
      : feature_ptr(feature_ptr),
        weights_ptr(weights_ptr),
        bias_ptr(bias_ptr),
        output_ptr(output_ptr),
        workspace_ptr(workspace_ptr),
        matrix_m(matrix_m),
        matrix_n(matrix_n),
        matrix_k(matrix_k) {}

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    xetla_exec_item<3> ei(item);
    using fused_dense_functor =
        fused_dense_func<T, T, T, KernelAttr::data_type_acc,
                         KernelAttr::wg_tile_m, KernelAttr::wg_tile_n,
                         KernelAttr::sg_tile_m, KernelAttr::sg_tile_n,
                         KernelAttr::sg_tile_k, mem_layout::row_major,
                         mem_layout::row_major, KernelAttr::post_op>;
    constexpr uint32_t barrier_count = fused_dense_functor::barrier_count;
    constexpr uint32_t slm_size = fused_dense_functor::slm_size;
    if constexpr (barrier_count != 0) {
      xetla_nbarrier_init<barrier_count>();
    }
    if constexpr (slm_size != 0) {
      xetla_local_init<slm_size>();
    }

    fused_dense_functor::run(&ei, feature_ptr, weights_ptr, output_ptr,
                             bias_ptr, matrix_m, matrix_n, matrix_k, matrix_k,
                             matrix_n, matrix_n, workspace_ptr);
  }

 private:
  T* feature_ptr;
  T* weights_ptr;
  T* bias_ptr;
  T* output_ptr;
  T* workspace_ptr;
  const uint32_t matrix_m;
  const uint32_t matrix_n;
  const uint32_t matrix_k;
};

}  // namespace gpu::xetla

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_MLP_OP_GPU_H_
