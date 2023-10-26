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

#ifndef ITEX_CORE_KERNELS_GPU_XETLA_FUSED_EINSUM_IMPL_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_FUSED_EINSUM_IMPL_H_

#include <xetla.hpp>

namespace gpu::xetla {

#define ASSIGN_META(x)                \
  for (int i = 0; i < DIMS; ++i) {    \
    x##_shape_[i] = x##_shape[i];     \
    x##_strides_[i] = x##_strides[i]; \
  }

template <typename dtype, int DIMS, int wg_m, int wg_n, int sg_m, int sg_n,
          bool a_row_major, bool b_row_major, bool c_row_major>
class FusedEinsumKernel {
 public:
  explicit FusedEinsumKernel(
      dtype* lhs_ptr, std::vector<int>& lhs_shape,    // NOLINT
      std::vector<int>& lhs_strides, dtype* rhs_ptr,  // NOLINT
      std::vector<int>& rhs_shape,                    // NOLINT
      std::vector<int>& rhs_strides, dtype* out_ptr,  // NOLINT
      std::vector<int>& out_shape,                    // NOLINT
      std::vector<int>& out_strides, int min_batch)   // NOLINT
      : lhs_ptr_(lhs_ptr),
        rhs_ptr_(rhs_ptr),
        out_ptr_(out_ptr),
        min_batch_(min_batch) {
    // Assign meta data: shape and stride.
    ASSIGN_META(lhs);
    ASSIGN_META(rhs);
    ASSIGN_META(out);
  }

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    using namespace gpu::xetla;            // NOLINT
    using namespace gpu::xetla::group;     // NOLINT
    using namespace gpu::xetla::kernel;    // NOLINT
    using namespace gpu::xetla::subgroup;  // NOLINT

    xetla_exec_item<3> ei(item);

    static constexpr uint32_t periodic_sync_interval = 8;
    static constexpr uint32_t prefetch_distance = 3;
    static constexpr uint32_t k_iter_num = 32;
    using perf_tuning_knob = perf_tuning_knob_t<k_iter_num, prefetch_distance,
                                                periodic_sync_interval>;

    using compute_attr = compute_attr_t<dtype, dtype, float>;
    using compute_policy =
        compute_policy_default_xmx<compute_attr, perf_tuning_knob,
                                   gpu_arch::Xe>;

    using mem_desc_input_a =
        mem_desc_t<dtype,
                   a_row_major ? mem_layout::row_major : mem_layout::col_major,
                   mem_space::global>;
    using mem_desc_input_b =
        mem_desc_t<dtype,
                   b_row_major ? mem_layout::row_major : mem_layout::col_major,
                   mem_space::global>;
    using mem_desc_output_c =
        mem_desc_t<dtype, mem_layout::row_major, mem_space::global>;

    using tile_shape = tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
    using brgemm_t = brgemm_t<compute_policy, tile_shape, mem_desc_input_a,
                              mem_desc_input_b>;
    brgemm_t brgemm;

    static constexpr uint32_t barrier_count = brgemm_t::barrier_count;
    static constexpr uint32_t slm_size = brgemm_t::slm_size;
    xetla_nbarrier_init<barrier_count>();
    xetla_local_init<slm_size>();

    uint32_t matrix_m = lhs_shape_[DIMS - 2];
    uint32_t matrix_n = rhs_shape_[DIMS - 1];
    uint32_t matrix_k = rhs_shape_[DIMS - 2];

    uint32_t lda = lhs_strides_[DIMS - 2];
    if constexpr (!a_row_major) {
      lda = lhs_strides_[DIMS - 1];
    }

    uint32_t ldb = rhs_strides_[DIMS - 2];
    if constexpr (!b_row_major) {
      ldb = rhs_strides_[DIMS - 1];
    }

    uint32_t ldc = out_strides_[DIMS - 2];
    if constexpr (!c_row_major) {
      ldc = out_strides_[DIMS - 1];
    }

    int start_n = ei.get_group(2) * wg_n;
    int start_m = ei.get_group(1) * wg_m;

    int start_k = 0;
    uint32_t wg_tile_k = matrix_k;
    uint32_t inner_loop_count = (wg_tile_k + k_iter_num - 1) / k_iter_num;

    auto lhs_cur = const_cast<dtype*>(lhs_ptr_);
    auto rhs_cur = const_cast<dtype*>(rhs_ptr_);
    auto out_cur = const_cast<dtype*>(out_ptr_);
    int lhs_offset = 0, rhs_offset = 0, out_offset = 0;
    int batch = ei.get_group(0);
#pragma unroll
    for (int i = DIMS - 3; i >= 0; --i) {
      int pred = batch / lhs_shape_[i];
      int mod = batch - pred * lhs_shape_[i];
      if (i > min_batch_) {
        lhs_offset += mod * lhs_strides_[i];
        rhs_offset += mod * rhs_strides_[i];
        out_offset += mod * out_strides_[i];
      } else {
        lhs_cur += mod * lhs_strides_[i];
        rhs_cur += mod * rhs_strides_[i];
        out_cur += mod * out_strides_[i];
      }
      batch = pred;
    }

    mem_desc_input_a md_a;
    if constexpr (a_row_major) {
      md_a.init(lhs_cur, {lhs_offset + matrix_k, matrix_m, lda},
                {lhs_offset + start_k, start_m});
    } else {
      md_a.init(lhs_cur, {matrix_k, lhs_offset + matrix_m, lda},
                {start_k, lhs_offset + start_m});
    }

    mem_desc_input_b md_b;
    if constexpr (b_row_major) {
      md_b.init(rhs_cur, {rhs_offset + matrix_n, matrix_k, ldb},
                {rhs_offset + start_n, start_k});
    } else {
      md_b.init(rhs_cur, {matrix_n, rhs_offset + matrix_k, ldb},
                {start_n, rhs_offset + start_k});
    }

    typename brgemm_t::matAcc_t matAcc;
    matAcc.init(0);

    typename brgemm_t::arguments_t brgemm_args(md_a, md_b, inner_loop_count);

    typename brgemm_t::work_group_t g(ei.get_local_linear_id());
    brgemm(g, matAcc, brgemm_args);

    using matAcc_t = typename brgemm_t::matAcc_t;
    using matAcc_tile_desc = typename matAcc_t::tile_desc;
    using matC_tile_desc_t = subgroup::tile_desc_t<
        c_row_major ? matAcc_t::tile_size_x : matAcc_t::tile_size_y,
        c_row_major ? matAcc_t::tile_size_y : matAcc_t::tile_size_x,
        c_row_major ? matAcc_t::block_size_x : matAcc_t::block_size_y,
        c_row_major ? matAcc_t::block_size_y : matAcc_t::block_size_x,
        reg_layout::tiled>;
    static constexpr msg_type msg_type_c = msg_type::block_2d;
    using matC_t = subgroup::tile_t<dtype, matC_tile_desc_t>;
    using matC_payload_t =
        subgroup::mem_payload_t<dtype, matC_tile_desc_t, msg_type_c,
                                mem_layout::row_major, mem_space::global,
                                gpu_arch::Xe>;
    mem_desc_output_c md_c;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t block_elems = matAcc_tile_desc::block_elems;
    static constexpr uint32_t num_block_y = matAcc_tile_desc::num_block_y;
    static constexpr uint32_t num_block_x = matAcc_tile_desc::num_block_x;
    static constexpr uint32_t block_size_x = matAcc_tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_tile_desc::block_size_y;
    int32_t sg_idx = g.get_id() % wg_size_x;
    int32_t sg_idy = g.get_id() / wg_size_x;
    int32_t tile_offset_n;
    int32_t tile_offset_m;
    matC_t matC;
    if constexpr (!c_row_major) {
      md_c.init(out_cur, {out_offset + matrix_m, matrix_n, ldc},
                {out_offset + start_m, start_n});
      tile_offset_n = sg_idy * sg_m;
      tile_offset_m = sg_idx * sg_n;
#pragma unroll
      for (int i = 0; i < num_block_y; ++i) {
#pragma unroll
        for (int j = 0; j < num_block_x; ++j) {
          auto src_block = matAcc.reg.xetla_select<block_elems, 1>(
              (i * num_block_x + j) * block_elems);
          auto dst_block = matC.reg.xetla_select<block_elems, 1>(
              (j * num_block_y + i) * block_elems);
#pragma unroll
          for (int k = 0; k < block_size_y; ++k) {
            dst_block.xetla_select<block_size_x, block_size_y>(k) =
                src_block.xetla_select<block_size_x, 1>(k * block_size_x);
          }
        }
      }
    } else {
      tile_offset_n = sg_idx * sg_n;
      tile_offset_m = sg_idy * sg_m;
      md_c.init(out_cur, {out_offset + matrix_n, matrix_m, ldc},
                {out_offset + start_n, start_m});
      subgroup::elemwise_cvt(matC, matAcc);
    }
    md_c.update_coord(tile_offset_n, tile_offset_m);
    matC_payload_t matC_payload(md_c);
    subgroup::tile_store(matC, matC_payload);
  }

 private:
  const dtype* lhs_ptr_;
  const dtype* rhs_ptr_;
  dtype* out_ptr_;
  int min_batch_;
  int lhs_shape_[DIMS];
  int lhs_strides_[DIMS];
  int rhs_shape_[DIMS];
  int rhs_strides_[DIMS];
  int out_shape_[DIMS];
  int out_strides_[DIMS];
};

#undef ASSIGN_META

}  // namespace gpu::xetla

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_FUSED_EINSUM_IMPL_H_
