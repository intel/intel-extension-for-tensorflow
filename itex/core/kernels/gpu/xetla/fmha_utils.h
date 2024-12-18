/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef ITEX_CORE_KERNELS_GPU_XETLA_FMHA_UTILS_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_FMHA_UTILS_H_

#include <algorithm>
#include <cmath>
#include <xetla.hpp>

namespace gpu::xetla {

namespace fmha {

template <typename mat_t>
struct tile_mask_t {
  using accum_t = typename mat_t::dtype;
  static constexpr accum_t kNegInfinity = INFINITY * -1;
  static constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  static constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  static constexpr uint32_t block_size_x = mat_t::block_size_x;
  static constexpr uint32_t block_size_y = mat_t::block_size_y;
  static constexpr int32_t num_block_x = mat_t::num_block_x;
  static constexpr uint32_t block_elems = mat_t::block_elems;

  // --------------------- // causal_mask // ---------------------- //

  inline static void causal_mask(mat_t* src, uint32_t start_x,
                                 uint32_t start_y) {
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
      uint32_t blk_start_y = start_y + i * block_size_y;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        uint32_t blk_start_x = start_x + j * block_size_x;
        if (blk_start_x + block_size_x > blk_start_y) {
          xetla_vector<uint32_t, block_size_x> blk_seq_x =
              xetla_vector_gen<uint32_t, block_size_x>(blk_start_x, 1);
          auto src_sub =
              src->reg
                  .xetla_select<block_elems, 1>((i * num_block_x + j) *
                                                block_elems)
                  .xetla_format<accum_t, block_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < block_size_y; k++) {
            xetla_mask<block_size_x> mask = blk_seq_x > blk_start_y + k;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;

      uint32_t blk_start_y = start_y + tail_start_y;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        uint32_t blk_start_x = start_x + j * block_size_x;
        if (blk_start_x + block_size_x > blk_start_y) {
          xetla_vector<uint32_t, block_size_x> blk_seq_x =
              xetla_vector_gen<uint32_t, block_size_x>(blk_start_x, 1);
          auto src_sub =
              src->reg
                  .xetla_select<tail_block_elems, 1>(
                      tail_start_y * tile_size_x + j * tail_block_elems)
                  .xetla_format<accum_t, tail_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < tail_size_y; k++) {
            xetla_mask<block_size_x> mask = blk_seq_x > blk_start_y + k;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }
  }

  // -------------------- // padding_mask // ---------------------- //

  inline static void padding_mask(mat_t* src, uint32_t num_keep) {
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        int start_x = j * block_size_x;
        int num_keep_blk = std::max(static_cast<int>(num_keep) - start_x, 0);

        if (num_keep_blk < block_size_x) {
          xetla_mask<block_size_x> mask =
              xetla_vector_gen<uint32_t, block_size_x>(1, 1) > num_keep_blk;
          auto src_sub =
              src->reg
                  .xetla_select<block_elems, 1>((i * num_block_x + j) *
                                                block_elems)
                  .xetla_format<accum_t, block_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < block_size_y; k++) {
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        int start_x = j * block_size_x;
        int num_keep_blk = std::max(static_cast<int>(num_keep) - start_x, 0);

        if (num_keep_blk < block_size_x) {
          xetla_mask<block_size_x> mask =
              xetla_vector_gen<uint32_t, block_size_x>(1, 1) > num_keep_blk;
          auto src_sub =
              src->reg
                  .xetla_select<tail_block_elems, 1>(
                      tail_start_y * tile_size_x + j * tail_block_elems)
                  .xetla_format<accum_t, tail_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < tail_size_y; k++) {
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }
  }
};

// ==================== // group_row_reduce_t // ================== //

template <typename mat_t, uint32_t kNumSg, reduce_op reduce_kind>
struct group_row_reduce_t {
  using T = typename mat_t::dtype;
  static constexpr uint32_t kNum = mat_t::tile_desc::tile_size_y;
  static constexpr uint32_t kTotal = kNum * kNumSg;

  // store results of subgroup to slm
  using store_tile_desc =
      subgroup::tile_desc_t<kNum, 1, kNum, 1, reg_layout::tiled>;
  using store_tile_t = subgroup::tile_t<T, store_tile_desc>;
  using mem_desc_store_t =
      mem_desc_t<T, mem_layout::row_major, mem_space::local>;
  using mem_desc_load_t =
      mem_desc_t<T, mem_layout::row_major, mem_space::local>;
  using store_payload_t =
      subgroup::mem_payload_t<mem_desc_store_t, store_tile_desc,
                              msg_type::block_1d, gpu_arch::Xe>;
  // load all subgroup results together
  using load_tile_desc =
      subgroup::tile_desc_t<kTotal, 1, kTotal, 1, reg_layout::tiled>;
  using load_tile_t = subgroup::tile_t<T, load_tile_desc>;
  using load_payload_t = subgroup::mem_payload_t<
      mem_desc_load_t, load_tile_desc,
      subgroup::msg_type_v<load_tile_desc, mem_space::local>, gpu_arch::Xe>;

  xetla_nbarrier_t<kNumSg, kNumSg> nbarrier;
  uint32_t slm_base;
  uint32_t sg_id;
  inline group_row_reduce_t() = default;
  inline group_row_reduce_t(uint32_t sg_id_, uint32_t nbarrier_id,
                            uint32_t slm_base_) {
    nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
    sg_id = sg_id_;
    slm_base = slm_base_;
  }

  inline KERNEL_FUNC xetla_vector<T, kNum> operator()(mat_t* src) {
    xetla_vector<T, kNum> ret =
        subgroup::tile_reduce<reduce_kind, T, T, 1>(*src);
    if constexpr (kNumSg == 1) return ret;

    store_tile_t sg_store;
    store_payload_t sg_store_payload(slm_base, kTotal, 1, kTotal, sg_id * kNum,
                                     0);
    sg_store.reg = ret;
    subgroup::tile_store(sg_store, sg_store_payload);

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    load_tile_t sg_load;
    load_payload_t sg_load_payload(slm_base, kTotal, 1, kTotal, 0, 0);
    subgroup::tile_load(sg_load, sg_load_payload);

    auto data_2d = sg_load.reg.xetla_format<T, kNumSg, kNum>();
    ret = data_2d.row(0);
#pragma unroll
    for (int i = 1; i < kNumSg; i++) {
      ret = reduce_helper<reduce_kind, T, kNum>(data_2d.row(i), ret);
    }
    return ret;
  }
};

// ==================== // dropout_t // ================== //

template <typename mat_t, typename mem_desc_mask_t>
struct dropout_t {
  static constexpr uint32_t num_flag = 4;
  static constexpr uint32_t unroll_size = num_flag * 16;

  static constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  static constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  static constexpr uint32_t block_size_x = mat_t::block_size_x;
  static constexpr uint32_t block_size_y = mat_t::block_size_y;
  static constexpr int32_t num_block_x = mat_t::num_block_x;
  static constexpr int32_t num_block_y = mat_t::num_block_y;
  static constexpr uint32_t tile_elems = mat_t::tile_elems;
  static constexpr uint32_t block_elems = mat_t::block_elems;

  using dtype_mask = typename mem_desc_mask_t::dtype;
  using mask_in_tile_desc_t =
      subgroup::tile_desc_t<tile_size_x, tile_size_y, block_size_x,
                            block_size_y, reg_layout::tiled>;
  using mask_in_tile_t = subgroup::tile_t<dtype_mask, mask_in_tile_desc_t>;
  using mask_in_payload_t = subgroup::mem_payload_t<
      mem_desc_mask_t, mask_in_tile_desc_t,
      subgroup::msg_type_v<mask_in_tile_desc_t, mem_desc_mask_t::space>,
      gpu_arch::Xe>;

  inline KERNEL_FUNC void operator()(mat_t* src, mem_desc_mask_t mem_desc_mask,
                                     float prob) {
    if (prob == 0.f) {
      return;
    }
    float scale = 1.f / (1.f - prob);

    mask_in_tile_t mask_in;
    mask_in_payload_t mask_in_payload(mem_desc_mask);
    tile_load<cache_hint::cached, cache_hint::cached>(mask_in, mask_in_payload);
    src->reg = src->reg * scale;

#pragma unroll
    for (int i = 0; i < tile_elems / unroll_size; i++) {
      xetla_mask<unroll_size> mask_flag =
          mask_in.reg.xetla_select<unroll_size, 1>(i * unroll_size) <= 0;
      (src->reg)
          .xetla_select<unroll_size, 1>(i * unroll_size)
          .xetla_merge(0, mask_flag);
    }
    if constexpr (tile_elems % unroll_size != 0) {
      constexpr uint32_t remain_len = tile_elems % unroll_size;
      constexpr uint32_t remain_start = tile_elems / unroll_size * unroll_size;
      xetla_mask<remain_len> mask_flag =
          mask_in.reg.xetla_select<remain_len, 1>(remain_start) <= 0;
      (src->reg)
          .xetla_select<remain_len, 1>(remain_start)
          .xetla_merge(0, mask_flag);
    }
  }
};

template <typename scalar_t, typename tile_desc_t, typename mem_desc_t>
void store_tile(subgroup::tile_t<scalar_t, tile_desc_t>* src, mem_desc_t dst) {
  using store_t = subgroup::mem_payload_t<
      mem_desc_t, tile_desc_t,
      subgroup::msg_type_v<tile_desc_t, mem_desc_t::space>, gpu_arch::Xe>;
  store_t store(dst);
  subgroup::tile_store(*src, store);
}

template <typename scalar_t, typename tile_desc_t, typename mem_desc_t>
void store_tile(subgroup::tile_t<scalar_t, tile_desc_t>* src, mem_desc_t dst,
                int32_t tile_offset_x, int32_t tile_offset_y) {
  using store_t = subgroup::mem_payload_t<
      mem_desc_t, tile_desc_t,
      subgroup::msg_type_v<tile_desc_t, mem_desc_t::space>, gpu_arch::Xe>;
  dst.update_coord(tile_offset_x, tile_offset_y);
  store_t store(dst);
  subgroup::tile_store(*src, store);
}

template <typename scalar_t, typename tile_desc_t, typename mem_desc_t>
void load_tile(subgroup::tile_t<scalar_t, tile_desc_t>* dst, mem_desc_t src) {
  using load_t = subgroup::mem_payload_t<
      mem_desc_t, tile_desc_t,
      subgroup::msg_type_v<tile_desc_t, mem_desc_t::space>, gpu_arch::Xe>;
  load_t load(src);
  subgroup::tile_load(*dst, load);
}

template <typename scalar_t, typename tile_desc_t, typename mem_desc_t>
void load_tile(subgroup::tile_t<scalar_t, tile_desc_t>* dst, mem_desc_t src,
               int32_t tile_offset_x, int32_t tile_offset_y) {
  using load_t = subgroup::mem_payload_t<
      mem_desc_t, tile_desc_t,
      subgroup::msg_type_v<tile_desc_t, mem_desc_t::space>, gpu_arch::Xe>;
  src.update_coord(tile_offset_x, tile_offset_y);
  load_t load(src);
  subgroup::tile_load(*dst, load);
}

}  // namespace fmha

}  // namespace gpu::xetla

namespace gpu::xetla::group {

template <typename epilogue_policy, typename tile_shape_,
          typename mem_desc_c_t_>
class epilogue_transp_t {};

template <typename tile_op_t_, typename tile_shape_, typename mem_desc_c_t_>
class epilogue_transp_t<epilogue_policy_tile_op<tile_op_t_, gpu_arch::Xe>,
                        tile_shape_, mem_desc_c_t_> {
 public:
  using tile_shape = tile_shape_;
  using mem_desc_c_t = mem_desc_c_t_;
  static constexpr gpu_arch arch_tag = gpu_arch::Xe;
  static constexpr uint32_t barrier_count = 0;
  static constexpr uint32_t slm_size = 0;

  struct arguments_t {};

 private:
  using work_group_t = typename tile_shape::work_group_t;
  static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
  static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
  static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
  static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
  using dtype_c = typename mem_desc_c_t::dtype;
  static constexpr mem_layout mem_layout_c = mem_desc_c_t::layout;
  static constexpr mem_space mem_space_c = mem_desc_c_t::space;
  static constexpr msg_type msg_type_c =
      (mem_space_c == mem_space::global ? msg_type::block_2d
                                        : msg_type::scatter);
  /// @brief Updates tile base descriptor based on the tid.
  __XETLA_API static void update_sg_tile_tdesc(
      work_group_t& g,             // NOLINT(runtime/references)
      mem_desc_c_t& mem_desc_c) {  // NOLINT(runtime/references)
    int32_t sg_idy = g.get_id() % wg_size_x;
    int32_t sg_idx = g.get_id() / wg_size_x;
    int32_t tile_offset_n = sg_idx * sg_tile_m;
    int32_t tile_offset_m = sg_idy * sg_tile_n;
    mem_desc_c.update_coord(tile_offset_n, tile_offset_m);
  }

 public:
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      work_group_t& g, matAcc_t& matAcc,  // NOLINT(runtime/references)
      mem_desc_c_t mem_desc_c, arguments_t args = {}, uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    static_assert(mem_layout_c == mem_layout::row_major &&
                      mem_space_c == mem_space::local,
                  "layout should be row_major and space should be local");
    using matC_tile_desc_t =
        subgroup::tile_desc_t<matAcc_t::tile_size_x, matAcc_t::tile_size_y,
                              matAcc_t::block_size_x, matAcc_t::block_size_y,
                              reg_layout::vnni_tiled_col_major>;

    using matC_t = subgroup::tile_t<dtype_c, matC_tile_desc_t>;
    using matC_payload_t =
        subgroup::mem_payload_t<mem_desc_c_t, matC_tile_desc_t, msg_type_c,
                                arch_tag>;

    update_sg_tile_tdesc(g, mem_desc_c);
    matC_payload_t matC_payload(mem_desc_c);
    matC_t matC;
    subgroup::vnni_transform(matC, matAcc);
    subgroup::tile_store(matC, matC_payload);
  }
};

}  // namespace gpu::xetla::group

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_FMHA_UTILS_H_
