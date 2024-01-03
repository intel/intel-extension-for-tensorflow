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

#ifndef ITEX_CORE_KERNELS_GPU_XETLA_FMHA_BACKWARD_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_FMHA_BACKWARD_H_

#include <algorithm>
#include <xetla.hpp>

#include "itex/core/kernels/gpu/xetla/fmha_policy.h"
#include "itex/core/kernels/gpu/xetla/fmha_utils.h"

namespace gpu::xetla {

namespace fmha {

// For tile desc, align with the xetla brgemm MatAcc_t.
const uint32_t kDefaultBlockSize = 16;

template <typename scalar_t, typename accum_t>
struct arguments_t {
  // Input tensors
  scalar_t* Q_ptr;            // [B, N, F, H] - query
  scalar_t* K_ptr;            // [B, N, T, H] - key
  scalar_t* V_ptr;            // [B, N, T, H] - value
  scalar_t* O_ptr;            // [B, F, N, H] - out
  scalar_t* B_ptr = nullptr;  // [B, 1, F, T] - bias
  scalar_t* dO_ptr;           // [B, F, N, H] - grad_out
  uint8_t* dp_mask_ptr;       // [B, N, F, T] - dropout
  accum_t dp_prob;
  accum_t dp_scale;
  accum_t* dp_sum;  // [B, N, F]
  accum_t* L_ptr;   // [B, N, F]
  // Output tensors
  scalar_t* dQ_ptr;      // [B, N, F, H] - grad_query
  accum_t* dQaccum_ptr;  // [B, N, F, H] - grad_query accumulates
  scalar_t* dK_ptr;      // [B, N, T, H] - grad_key
  scalar_t* dV_ptr;      // [B, N, T, H] - grad_value
  // Dimension size
  uint32_t uB;
  uint32_t uN;
  uint32_t uH;
  uint32_t uF;
  uint32_t uT;
  // Softmax scale is the reciprocal square root of head size by default
  accum_t sm_scale;

  inline arguments_t() = default;
  inline arguments_t(scalar_t* query, scalar_t* key, scalar_t* value,
                     scalar_t* out, scalar_t* bias, scalar_t* grad_out,
                     uint8_t* dropout, accum_t dropout_prob, accum_t* dp_sum,
                     accum_t* L_ptr, scalar_t* grad_query,
                     accum_t* grad_query_accum, scalar_t* grad_key,
                     scalar_t* grad_value, uint32_t num_batches,
                     uint32_t num_heads, uint32_t head_size,
                     uint32_t num_queries, uint32_t num_keys)
      : Q_ptr(query),
        K_ptr(key),
        V_ptr(value),
        O_ptr(out),
        B_ptr(bias),
        dO_ptr(grad_out),
        dp_mask_ptr(dropout),
        dp_prob(dropout_prob),
        dp_scale(1.f / (1.f - dropout_prob)),
        dp_sum(dp_sum),
        L_ptr(L_ptr),
        dQ_ptr(grad_query),
        dQaccum_ptr(grad_query_accum),
        dK_ptr(grad_key),
        dV_ptr(grad_value),
        uB(num_batches),
        uN(num_heads),
        uH(head_size),
        uF(num_queries),
        uT(num_keys),
        sm_scale(xetla_rsqrt<accum_t>(accum_t(head_size))) {}
};

template <typename mha_policy, typename scalar_t>
class fmha_backward_dot_do_o_t {
 public:
  using accum_t = float;
  using args_t = arguments_t<scalar_t, float>;

 private:
  static constexpr uint32_t kBr = mha_policy::kBr;
  static constexpr uint32_t kHm = mha_policy::kHm;
  static constexpr uint32_t kSgBr = mha_policy::kSgBr;
  static constexpr uint32_t kSgHm = mha_policy::kSgHm;

  using tile_shape_BrHm = group::tile_shape_t<kHm, kBr, kSgHm, kSgBr>;

  static constexpr uint32_t wg_size_x = tile_shape_BrHm::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_BrHm::wg_size_y;
  using work_group_t = typename tile_shape_BrHm::work_group_t;
  static constexpr uint32_t wg_size = work_group_t::size;

  static constexpr uint32_t block_size_x = kDefaultBlockSize;
  static constexpr uint32_t block_size_y =
      (kDefaultBlockSize > kSgBr) ? kSgBr : kDefaultBlockSize;

  static_assert(wg_size <= 32, "The number of threads should be less than 32!");

  using dO_tile_desc_t = subgroup::tile_desc_t<kSgHm, kSgBr, block_size_x,
                                               block_size_y, reg_layout::tiled>;
  using dO_tile_t = subgroup::tile_t<scalar_t, dO_tile_desc_t>;
  using dO_tile_acc_t = subgroup::tile_t<accum_t, dO_tile_desc_t>;
  using dq_tile_t = subgroup::tile_t<accum_t, dO_tile_desc_t>;

  using mem_desc_dO_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_O_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dPsum_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dQaccum_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;

  static constexpr uint32_t slm_size =
      (wg_size_x > 1) ? wg_size * kSgBr * sizeof(accum_t) : 0;
  static constexpr uint32_t nbarrier_cnt = (wg_size_x > 1) ? wg_size_y : 0;

  struct context_t {
    // thread id
    work_group_t g;
    uint32_t sg_idx;
    uint32_t sg_idy;
    // nbarrier
    xetla_nbarrier_t<wg_size_x, wg_size_x> nbarrier;
    // mem desc variables
    mem_desc_dO_t mem_desc_dO;
    mem_desc_O_t mem_desc_O;
    mem_desc_dPsum_t mem_desc_dPsum;
    mem_desc_dQaccum_t mem_desc_dQaccum;

    inline context_t() = default;

    /// @brief Initialize variables used in the fmha dot_do_o
    inline void init_context(const sycl::nd_item<3>& ei, const args_t& args) {
      uint32_t sg_id = ei.get_local_linear_id();
      uint32_t gid = ei.get_group(0);

      // thread id and nbarrier
      g.init(sg_id);
      sg_idx = sg_id % wg_size_x;
      sg_idy = sg_id / wg_size_x;
      nbarrier.init_nbarrier(sg_idy, nbarrier_role::producer_consumer);

      int32_t start_y_dq = gid * args.uF + ei.get_group(1) * kBr;
      uint32_t end_y_dq = start_y_dq + kBr;
      uint32_t boundary_y = (gid + 1) * args.uF;
      end_y_dq = end_y_dq > boundary_y ? boundary_y : end_y_dq;

      mem_desc_dQaccum.init(args.dQaccum_ptr, {args.uH, end_y_dq, args.uH},
                            {0, start_y_dq});

      uint32_t bid = ei.get_group(0) / args.uN;
      uint32_t nid = ei.get_group(0) % args.uN;
      int32_t start_x_tr = nid * args.uH;
      uint32_t end_x_tr = (nid + 1) * args.uH;
      int32_t start_y_tr = bid * args.uF + ei.get_group(1) * kBr;
      uint32_t end_y_tr = start_y_tr + kBr;
      uint32_t boundry_y_tr = (bid + 1) * args.uF;
      end_y_tr = end_y_tr > boundry_y_tr ? boundry_y_tr : end_y_tr;
      mem_desc_O.init(args.O_ptr, {end_x_tr, end_y_tr, args.uH * args.uN},
                      {start_x_tr, start_y_tr});
      mem_desc_dO.init(args.dO_ptr, {end_x_tr, end_y_tr, args.uH * args.uN},
                       {start_x_tr, start_y_tr});

      int32_t start_x_dPsum = ei.get_group(1) * kBr + sg_idy * kSgBr;
      int32_t start_y_dPsum = gid;
      mem_desc_dPsum.init(args.dp_sum, {args.uF, args.uB * args.uN, args.uF},
                          {start_x_dPsum, start_y_dPsum});
    }
  };

  context_t ctx;

  /// @brief sum_dot_dO_O is used to compute Di.
  inline void sum_dot_dO_O(const args_t& args) {
    // load dO and O from global memory
    int32_t tile_offset_x = ctx.sg_idx * kSgHm;
    int32_t tile_offset_y = ctx.sg_idy * kSgBr;
    dO_tile_t rdO, rO;
    load_tile(&rdO, ctx.mem_desc_dO, tile_offset_x, tile_offset_y);
    load_tile(&rO, ctx.mem_desc_O, tile_offset_x, tile_offset_y);

    // Initialize the dQaccum tensor to 0.
    dq_tile_t rdq(0);
    store_tile(&rdq, ctx.mem_desc_dQaccum, tile_offset_x, tile_offset_y);

    dO_tile_acc_t rdO_acc, rO_acc, dot_acc;
    subgroup::elemwise_cvt(rdO_acc, rdO);
    subgroup::elemwise_cvt(rO_acc, rO);

    // compute dO * O
    dot_acc.reg = rdO_acc.reg * rO_acc.reg;
    uint32_t reducer_slm = ctx.sg_idy * wg_size_x * kSgBr * sizeof(accum_t);
    using wg_row_sum_t =
        group_row_reduce_t<dO_tile_acc_t, wg_size_x, reduce_op::sum>;

    wg_row_sum_t wg_row_sum(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    xetla_vector<accum_t, kSgBr> row_sum = wg_row_sum(&dot_acc);

    // store result
    using store_tile_desc =
        subgroup::tile_desc_t<kSgBr, 1, kSgBr, 1, reg_layout::tiled>;
    using store_tile_t = subgroup::tile_t<accum_t, store_tile_desc>;
    using store_payload_t =
        subgroup::mem_payload_t<mem_desc_dPsum_t, store_tile_desc,
                                msg_type::block_2d, gpu_arch::Xe>;
    store_tile_t dPsum_store;
    store_payload_t dPsum_store_payload(ctx.mem_desc_dPsum);

    dPsum_store.reg = row_sum;
    if (ctx.sg_idx == 0) {
      subgroup::tile_store(dPsum_store, dPsum_store_payload);
    }
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t count = nbarrier_cnt;
    static_assert(count <= 32,
                  "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size;
    static_assert(size <= (128 * 1024),
                  "The local memory size should be less than 128KB!");
    return size;
  }

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(uint32_t total_batches,
                                        uint32_t num_queries) {
    // local range
    sycl::range<3> local_range = sycl::range<3>{1, wg_size_y, wg_size_x};
    // group range
    uint32_t group_range_m = (num_queries + kBr - 1) / kBr;
    sycl::range<3> group_range =
        sycl::range<3>{total_batches, group_range_m, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  }

  inline KERNEL_FUNC void operator()(const sycl::nd_item<3>& ei,
                                     const args_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // initialize context
    ctx.init_context(ei, args);
    sum_dot_dO_O(args);
  }
};

template <typename mha_policy, typename scalar_t, bool kUseBias,
          bool kUseDropout>
class fmha_backward_t {
 public:
  using accum_t = float;
  using args_t = arguments_t<scalar_t, float>;

 private:
  // -------------------- // Compute policy // -------------------- //
  static constexpr uint32_t accum_step = mha_policy::accum_step;
  static constexpr uint32_t stages = mha_policy::stages;
  static constexpr uint32_t sync_freq = mha_policy::sync_freq;

  using compute_attr = group::compute_attr_t<scalar_t, scalar_t, accum_t>;
  using perf_tuning_knob =
      group::perf_tuning_knob_t<accum_step, stages, sync_freq>;
  using compute_policy =
      group::compute_policy_default_xmx<compute_attr, perf_tuning_knob,
                                        gpu_arch::Xe>;

  // ---------------- // Tile shape and Threads // ---------------- //
  static constexpr uint32_t kBr = mha_policy::kBr;
  static constexpr uint32_t kBc = mha_policy::kBc;
  static constexpr uint32_t kHm = mha_policy::kHm;
  static constexpr uint32_t kSgBr = mha_policy::kSgBr;
  static constexpr uint32_t kSgBc = mha_policy::kSgBc;
  static constexpr uint32_t kSgHm = mha_policy::kSgHm;
  static constexpr uint32_t kSgBc_M = mha_policy::kSgBc_M;

  using tile_shape_BrBc = group::tile_shape_t<kBc, kBr, kSgBc, kSgBr>;
  using tile_shape_BrHm = group::tile_shape_t<kHm, kBr, kSgHm, kSgBr>;
  using tile_shape_BcHm = group::tile_shape_t<kHm, kBc, kSgHm, kSgBc_M>;

  using work_group_BrBc_t = typename tile_shape_BrBc::work_group_t;
  using work_group_BrHm_t = typename tile_shape_BrHm::work_group_t;
  using work_group_BcHm_t = typename tile_shape_BcHm::work_group_t;

  static constexpr uint32_t wg_size_x = tile_shape_BrBc::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_BrBc::wg_size_y;
  using work_group_t = typename tile_shape_BrBc::work_group_t;
  static constexpr uint32_t wg_size = work_group_t::size;

  static_assert(kHm / kSgHm == kBc / kSgBc,
                "wg_size_x must be the same between Hm and Bc");
  static_assert(kBr / kSgBr == kBc / kSgBc_M,
                "wg_size_y must be the same between Br and Bc_M");
  static_assert(wg_size <= 32, "The number of threads should be less than 32!");

  // --------------------- // Memory desc // ---------------------- //
  // suffix: L -> local; T -> transpose
  using mem_desc_Q_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_K_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_K_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_V_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_V_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_Bij_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_P_LT_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_dO_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Dp_mask_t =
      mem_desc_t<uint8_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dS_L_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_dS_LT_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_dQ_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dK_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dV_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dQaccum_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Li_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dPsum_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;

  // ------------------- // Slm and nbarrier // ------------------- //
  static constexpr uint32_t P_slm_size = kBr * kBc * sizeof(scalar_t);
  static constexpr uint32_t dS_slm_size = kBr * kBc * sizeof(scalar_t);

  static constexpr uint32_t P_slm = 0;
  static constexpr uint32_t dS_slm = P_slm + P_slm_size;
  static constexpr uint32_t total_slm = P_slm_size + dS_slm_size;

  static constexpr uint32_t nbarrier_cnt = 1;

  using brgemm_Sij_t = group::gemm_t<compute_policy, tile_shape_BrBc,
                                     mem_desc_Q_t, mem_desc_K_T_t>;
  using matAccSij_t = typename brgemm_Sij_t::matAcc_t;
  using tile_P_t = matAccSij_t;

  // ======================== // Context // ======================= //

  /// @brief Variables used in the mha backward
  struct context_t {
    // thread id
    work_group_t g;
    uint32_t sg_idx;
    uint32_t sg_idy;
    uint32_t startT;
    uint32_t startF;
    work_group_BrBc_t g_brbc;
    work_group_BrHm_t g_brhm;
    work_group_BcHm_t g_bchm;
    // nbarrier
    xetla_nbarrier_t<wg_size, wg_size> nbarrier;
    // mem desc variables
    mem_desc_Q_t mem_desc_Q;
    mem_desc_K_t mem_desc_K;
    mem_desc_K_T_t mem_desc_K_T;
    mem_desc_V_t mem_desc_V;
    mem_desc_V_T_t mem_desc_V_T;
    mem_desc_Bij_t mem_desc_Bij;
    mem_desc_P_LT_t mem_desc_P_LT;
    mem_desc_dO_t mem_desc_dO;
    mem_desc_Dp_mask_t mem_desc_Dp_mask;
    mem_desc_dS_L_t mem_desc_dS_L;
    mem_desc_dS_LT_t mem_desc_dS_LT;
    mem_desc_dQ_t mem_desc_dQ;
    mem_desc_dK_t mem_desc_dK;
    mem_desc_dV_t mem_desc_dV;
    mem_desc_dQaccum_t mem_desc_dQaccum;
    mem_desc_Li_t mem_desc_Li;
    mem_desc_dPsum_t mem_desc_dPsum;

    inline context_t() = default;

    /// @brief Initialize variables used in the mha backward
    inline void init_context(const sycl::nd_item<3>& ei, const args_t& args) {
      uint32_t sg_id = ei.get_local_linear_id();
      uint32_t gid = ei.get_group(0);
      startT = ei.get_group(1) * kBc;

      // thread id and nbarrier
      g.init(sg_id);
      g_brbc.init(sg_id);
      g_brhm.init(sg_id);
      g_bchm.init(sg_id);
      nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);

      // for shape [B,N,T,H]
      int32_t start_x = gid * args.uT + ei.get_group(1) * kBc;
      uint32_t end_x = start_x + kBc;
      uint32_t boundary_x = (gid + 1) * args.uT;
      end_x = end_x > boundary_x ? boundary_x : end_x;

      mem_desc_K.init(args.K_ptr, {args.uH, end_x, args.uH}, {0, start_x});
      mem_desc_K_T.init(args.K_ptr, {end_x, args.uH, args.uH}, {start_x, 0});
      mem_desc_V_T.init(args.V_ptr, {end_x, args.uH, args.uH}, {start_x, 0});
      mem_desc_V.init(args.V_ptr, {args.uH, end_x, args.uH}, {0, start_x});
      mem_desc_dK.init(args.dK_ptr, {args.uH, end_x, args.uH}, {0, start_x});
      mem_desc_dV.init(args.dV_ptr, {args.uH, end_x, args.uH}, {0, start_x});

      // local memory
      mem_desc_P_LT.init(P_slm, {kBr, kBc, kBr}, {0, 0});
      mem_desc_dS_L.init(dS_slm, {kBc, kBr, kBc}, {0, 0});
      mem_desc_dS_LT.init(dS_slm, {kBr, kBc, kBr}, {0, 0});
    }

    /// @brief Update variables for each flash mha loop
    inline void update_context(const sycl::nd_item<3>& ei, const args_t& args,
                               uint32_t fstart) {
      uint32_t sg_id = ei.get_local_linear_id();
      uint32_t gid = ei.get_group(0);
      startF = fstart;

      sg_idx = sg_id % wg_size_x;
      sg_idy = sg_id / wg_size_x;
      // mem desc variables
      // for shape [B,N,F,T] and [B,N,F,H]
      int32_t start_y = gid * args.uF + startF;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = (gid + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;

      mem_desc_Q.init(args.Q_ptr, {args.uH, end_y, args.uH}, {0, start_y});
      mem_desc_dQ.init(args.dQ_ptr, {args.uH, end_y, args.uH}, {0, start_y});
      mem_desc_dQaccum.init(args.dQaccum_ptr, {args.uH, end_y, args.uH},
                            {0, start_y});

      if constexpr (kUseDropout) {
        int32_t offset_x = ei.get_group(1) * kBc + sg_idx * kSgBc;
        int32_t offset_y = start_y + sg_idy * kSgBr;
        mem_desc_Dp_mask.init(args.dp_mask_ptr, {args.uT, end_y, args.uT},
                              {offset_x, offset_y});
      }

      int32_t start_x_ml = startF + sg_idy * kSgBr;
      int32_t start_y_ml = gid;
      mem_desc_Li.init(args.L_ptr, {args.uF, args.uB * args.uN, args.uF},
                       {start_x_ml, start_y_ml});
      mem_desc_dPsum.init(args.dp_sum, {args.uF, args.uB * args.uN, args.uF},
                          {start_x_ml, start_y_ml});

      if constexpr (kUseBias) {
        int32_t start_x = ei.get_group(1) * kBc;
        uint32_t end_x = start_x + kBc;
        uint32_t boundary_x = args.uT;
        end_x = end_x > boundary_x ? boundary_x : end_x;

        uint32_t batch_id = gid / args.uN;
        int32_t start_y = batch_id * args.uF + startF;
        uint32_t end_y = start_y + kBr;
        uint32_t boundary_y = (batch_id + 1) * args.uF;
        end_y = end_y > boundary_y ? boundary_y : end_y;

        mem_desc_Bij.init(args.B_ptr, {end_x, end_y, args.uT},
                          {start_x, start_y});
      }

      // for dO shape [B,F,N,H]
      uint32_t head_id = gid % args.uN;
      uint32_t batch_id = gid / args.uN;
      int32_t start_x = head_id * args.uH;
      uint32_t end_x = (head_id + 1) * args.uH;
      start_y = batch_id * args.uF + startF;
      end_y = start_y + kBr;
      boundary_y = (batch_id + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;

      mem_desc_dO.init(args.dO_ptr, {end_x, end_y, args.uH * args.uN},
                       {start_x, start_y});
    }
  };

  context_t ctx;

  // ======================= // gemm_Sij // ======================= //
  // Define kernel to compute Sij = Q x K.T
  using dp_mask_tile_t =
      subgroup::tile_t<uint8_t, typename tile_P_t::tile_desc>;

  /// @brief compute_Pij is used to compute Sij = Q x K.T and store Pij to slm
  /// # [Br,H] x [H,Bc] = [Br,Bc]
  inline void compute_Pij(tile_P_t* rP, dp_mask_tile_t* mask_in,
                          const args_t& args) {
    using load_ml_desc =
        subgroup::tile_desc_t<kSgBr, 1, kSgBr, 1, reg_layout::tiled>;
    using ml_tile_t = subgroup::tile_t<accum_t, load_ml_desc>;
    ml_tile_t l_load;
    load_tile(&l_load, ctx.mem_desc_Li);

    using brgemm_args_t = typename brgemm_Sij_t::arguments_t;

    // Gemm to comput Sij
    brgemm_Sij_t brgemm;
    uint32_t loop_count = (args.uH + accum_step - 1) / accum_step;
    brgemm_args_t brgemm_args(ctx.mem_desc_Q, ctx.mem_desc_K_T, loop_count);
    brgemm(ctx.g_brbc, *rP, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);

    // Multiply by softmax scaling factor
    rP->reg *= args.sm_scale;

    int32_t tile_offset_x = ctx.sg_idx * kSgBc;
    int32_t tile_offset_y = ctx.sg_idy * kSgBr;
    // Add bias if needed
    if constexpr (kUseBias) {
      using bias_op_t = subgroup::elemwise_reduce_op_t<reduce_op::sum, scalar_t,
                                                       gpu_arch::Xe>;
      using bias_args_t = typename bias_op_t::arguments_t;
      ctx.mem_desc_Bij.update_coord(tile_offset_x, tile_offset_y);
      bias_op_t bias_op;
      bias_args_t bias_args(ctx.mem_desc_Bij.base, ctx.mem_desc_Bij.shape);
      bias_op(*rP, ctx.mem_desc_Bij.coord, bias_args);
    }

    subgroup::tile_broadcast_op<subgroup::tile_minus, tile_P_t>(*rP,
                                                                l_load.reg);
    rP->reg = xetla_exp<accum_t>(rP->reg);
    // apply dropout mask
    tile_P_t rp_drop;
    rp_drop.reg = rP->reg;
    if constexpr (kUseDropout) {
      load_tile(mask_in, ctx.mem_desc_Dp_mask);
      rp_drop.reg = rp_drop.reg * mask_in->reg * args.dp_scale;
    }
    // store Pij to local memory, transpose it while saving
    using epilogue_p_t = group::epilogue_transp_t<
        group::epilogue_policy_tile_op<subgroup::chained_tile_op_t<>,
                                       gpu_arch::Xe>,
        tile_shape_BrBc, mem_desc_P_LT_t>;
    epilogue_p_t epilogue;
    epilogue(ctx.g_brbc, rp_drop, ctx.mem_desc_P_LT);
  }

  // ======================= // gemm_dP // ======================= //
  // Define kernel to compute dP = dO x V.T
  using brgemm_dP_t = group::gemm_t<compute_policy, tile_shape_BrBc,
                                    mem_desc_dO_t, mem_desc_V_T_t>;
  using tile_dP_t = typename brgemm_dP_t::matAcc_t;
  using tile_dS_t = tile_dP_t;

  /// @brief compute_dSij is used to compute dP = dO x V.T and store dSij to slm
  /// # [Br,H] x [H,Bc] = [Br,Bc]
  inline void compute_dSij(tile_P_t* rP, tile_dS_t* rdS,
                           dp_mask_tile_t* mask_in, const args_t& args) {
    using load_sum_desc =
        subgroup::tile_desc_t<kSgBr, 1, kSgBr, 1, reg_layout::tiled>;
    using load_tile_t = subgroup::tile_t<accum_t, load_sum_desc>;
    load_tile_t sum_load;
    load_tile(&sum_load, ctx.mem_desc_dPsum);

    using brgemm_args_t = typename brgemm_dP_t::arguments_t;
    tile_dP_t rdP(0);
    // Gemm to comput dP
    brgemm_dP_t brgemm;
    uint32_t loop_count = (args.uH + accum_step - 1) / accum_step;
    brgemm_args_t brgemm_args(ctx.mem_desc_dO, ctx.mem_desc_V_T, loop_count);
    brgemm(ctx.g_brbc, rdP, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);

    if constexpr (kUseDropout) {
      rdP.reg = rdP.reg * mask_in->reg * args.dp_scale;
    }
    subgroup::tile_broadcast_op<subgroup::tile_minus, tile_dP_t>(rdP,
                                                                 sum_load.reg);
    rdS->reg = rP->reg * rdP.reg;
    // Store dS to slm
    using epilogue_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrBc, mem_desc_dS_L_t>;
    epilogue_t epilogue;
    epilogue(ctx.g_brbc, *rdS, ctx.mem_desc_dS_L);

    xetla_fence<memory_kind::shared_local>();
    ctx.nbarrier.arrive_wait();
  }

  // ======================== // gemm_dQ // ======================= //
  // define brgemm kernel
  using brgemm_dQ_t = group::gemm_t<compute_policy, tile_shape_BrHm,
                                    mem_desc_dS_L_t, mem_desc_K_t>;
  using tile_dQ_t = typename brgemm_dQ_t::matAcc_t;

  /// @brief gemm_dQ is used to compute dQ = dS x K
  /// # [F,T] x [T,H] = [F,H]
  inline void compute_dQ(const args_t& args) {
    using brgemm_args_t = typename brgemm_dQ_t::arguments_t;
    uint32_t remainT = args.uT - ctx.startT;
    uint32_t boundary_k = remainT > kBc ? kBc : remainT;
    uint32_t loop_count = (boundary_k + accum_step - 1) / accum_step;

    // Gemm to comput dQ
    brgemm_dQ_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_dS_L, ctx.mem_desc_K, loop_count);
    tile_dQ_t rdQ(0);
    brgemm(ctx.g_brhm, rdQ, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);

    xetla_fence<memory_kind::shared_local>();
    ctx.nbarrier.arrive_wait();
    // Store and atomic add dQ to dQaccum
    using store_t = subgroup::mem_payload_t<mem_desc_dQaccum_t,
                                            typename tile_dQ_t::tile_desc,
                                            msg_type::atomic_add, gpu_arch::Xe>;
    ctx.mem_desc_dQaccum.update_coord(ctx.sg_idx * kSgHm, ctx.sg_idy * kSgBr);
    store_t dQ_store(ctx.mem_desc_dQaccum);
    subgroup::tile_store(rdQ, dQ_store);
  }

  // ======================= // gemm_dV // ======================= //
  // Define kernel to compute dV = P.T x dO
  using brgemm_dV_t = group::gemm_t<compute_policy, tile_shape_BcHm,
                                    mem_desc_P_LT_t, mem_desc_dO_t>;
  using tile_dV_t = typename brgemm_dV_t::matAcc_t;

  /// @brief gemm_dV is used to compute dV = P.T x dO
  /// # [T,F] x [F,H] = [T,H]
  inline void compute_dV(tile_dV_t* rdV, const args_t& args) {
    using brgemm_args_t = typename brgemm_dV_t::arguments_t;

    uint32_t remainF = args.uF - ctx.startF;
    uint32_t boundary_k = remainF > kBr ? kBr : remainF;
    uint32_t loop_count = (boundary_k + accum_step - 1) / accum_step;

    // Gemm to comput dV
    brgemm_dV_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_P_LT, ctx.mem_desc_dO, loop_count);
    brgemm(ctx.g_bchm, *rdV, brgemm_args);
  }

  // ======================= // gemm_dK // ======================= //
  // Define kernel to compute dK = dS.T x Q
  using brgemm_dK_t = group::gemm_t<compute_policy, tile_shape_BcHm,
                                    mem_desc_dS_LT_t, mem_desc_Q_t>;
  using tile_dK_t = typename brgemm_dK_t::matAcc_t;

  /// @brief gemm_dK is used to compute dK = dS.T x Q
  /// # [T,F] x [F,H] = [T,H]
  inline void compute_dK(tile_dK_t* rdK, tile_dS_t* rdS, const args_t& args) {
    // store dSij transpose to local
    using epilogue_s_t = group::epilogue_transp_t<
        group::epilogue_policy_tile_op<subgroup::chained_tile_op_t<>,
                                       gpu_arch::Xe>,
        tile_shape_BrBc, mem_desc_dS_LT_t>;
    epilogue_s_t epilogue;
    epilogue(ctx.g_brbc, *rdS, ctx.mem_desc_dS_LT);
    xetla_fence<memory_kind::shared_local>();
    ctx.nbarrier.arrive_wait();

    using brgemm_args_t = typename brgemm_dK_t::arguments_t;

    uint32_t remainF = args.uF - ctx.startF;
    uint32_t boundary_k = remainF > kBr ? kBr : remainF;
    uint32_t loop_count = (boundary_k + accum_step - 1) / accum_step;

    // Gemm to comput dK
    brgemm_dK_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_dS_LT, ctx.mem_desc_Q, loop_count);
    brgemm(ctx.g_bchm, *rdK, brgemm_args);
  }

  /// @brief store_dKdV is used to store dK and dV to global memory.
  inline void store_dKdV(tile_dK_t* rdK, tile_dK_t* rdV, const args_t& args) {
    rdK->reg *= args.sm_scale;
    subgroup::tile_t<scalar_t, typename tile_dK_t::tile_desc> rdK_s, rdV_s;
    subgroup::elemwise_cvt(rdK_s, *rdK);
    subgroup::elemwise_cvt(rdV_s, *rdV);
    using epilogue_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BcHm, mem_desc_dK_t>;
    epilogue_t epilogue_dK, epilogue_dV;
    epilogue_dK(ctx.g_bchm, rdK_s, ctx.mem_desc_dK);
    epilogue_dV(ctx.g_bchm, rdV_s, ctx.mem_desc_dV);
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count_dP = brgemm_dP_t::barrier_count;
    constexpr uint32_t barrier_count_dQ = brgemm_dQ_t::barrier_count;
    constexpr uint32_t barrier_count_dK = brgemm_dK_t::barrier_count;
    constexpr uint32_t count =
        std::max(std::max(barrier_count_dP, barrier_count_dQ),
                 barrier_count_dK) +
        nbarrier_cnt;
    static_assert(count <= 32,
                  "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = total_slm;
    static_assert(size <= (128 * 1024),
                  "The local memory size should be less than 128KB!");
    return size;
  }

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(uint32_t total_batches,
                                        uint32_t num_keys) {
    // local range
    sycl::range<3> local_range = sycl::range<3>{1, wg_size_y, wg_size_x};
    // group range
    uint32_t group_range_n = (num_keys + kBc - 1) / kBc;
    sycl::range<3> group_range =
        sycl::range<3>{total_batches, group_range_n, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  }

  // ================= // Entry of the functor // ================= //

  inline KERNEL_FUNC void operator()(const sycl::nd_item<3>& ei,
                                     const args_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // initialize context
    ctx.init_context(ei, args);

    tile_dV_t rdV(0);
    tile_dK_t rdK(0);

    for (uint32_t startF = 0; startF < args.uF; startF += kBr) {
      ctx.update_context(ei, args, startF);
      tile_P_t rP(0);
      dp_mask_tile_t mask_in;
      compute_Pij(&rP, &mask_in, args);
      tile_dS_t rdS(0);
      compute_dSij(&rP, &rdS, &mask_in, args);
      compute_dV(&rdV, args);
      compute_dQ(args);
      compute_dK(&rdK, &rdS, args);
    }
    store_dKdV(&rdK, &rdV, args);
  }
};

template <typename mha_policy, typename scalar_t>
class fmha_backward_convert_dq_t {
 public:
  using accum_t = float;
  using args_t = arguments_t<scalar_t, float>;

 private:
  static constexpr uint32_t kBr = mha_policy::kBr;
  static constexpr uint32_t kHm = mha_policy::kHm;
  static constexpr uint32_t kSgBr = mha_policy::kSgBr;
  static constexpr uint32_t kSgHm = mha_policy::kSgHm;

  using tile_shape_BrHm = group::tile_shape_t<kHm, kBr, kSgHm, kSgBr>;
  static constexpr uint32_t wg_size_x = tile_shape_BrHm::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_BrHm::wg_size_y;
  using work_group_t = typename tile_shape_BrHm::work_group_t;
  static constexpr uint32_t wg_size = work_group_t::size;

  static constexpr uint32_t block_size_x = kDefaultBlockSize;
  static constexpr uint32_t block_size_y =
      (kDefaultBlockSize > kSgBr) ? kSgBr : kDefaultBlockSize;

  static_assert(wg_size <= 32, "The number of threads should be less than 32!");

  using mem_desc_dQaccum_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dQ_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;

  struct context_t {
    // thread id
    work_group_t g;
    uint32_t sg_idx;
    uint32_t sg_idy;
    // mem desc variables
    mem_desc_dQaccum_t mem_desc_dQaccum;
    mem_desc_dQ_t mem_desc_dQ;

    inline context_t() = default;

    /// @brief Initialize variables used in the fmha convert_dq
    inline void init_context(const sycl::nd_item<3>& ei, const args_t& args) {
      uint32_t sg_id = ei.get_local_linear_id();
      uint32_t gid = ei.get_group(0);
      g.init(sg_id);

      sg_idx = sg_id % wg_size_x;
      sg_idy = sg_id / wg_size_x;
      int32_t start_y = gid * args.uF + ei.get_group(1) * kBr;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = (gid + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;

      mem_desc_dQaccum.init(args.dQaccum_ptr, {args.uH, end_y, args.uH},
                            {0, start_y});
      mem_desc_dQ.init(args.dQ_ptr, {args.uH, end_y, args.uH}, {0, start_y});
    }
  };

  context_t ctx;

 public:
  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(uint32_t total_batches,
                                        uint32_t num_queries) {
    // local range
    sycl::range<3> local_range = sycl::range<3>{1, wg_size_y, wg_size_x};
    // group range
    uint32_t group_range_m = (num_queries + kBr - 1) / kBr;
    sycl::range<3> group_range =
        sycl::range<3>{total_batches, group_range_m, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  }

  inline KERNEL_FUNC void operator()(const sycl::nd_item<3>& ei,
                                     const args_t& args) {
    // initialize context
    ctx.init_context(ei, args);

    using dQ_tile_desc_t =
        subgroup::tile_desc_t<kSgHm, kSgBr, block_size_x, block_size_y,
                              reg_layout::tiled>;
    using dQ_tile_t = subgroup::tile_t<scalar_t, dQ_tile_desc_t>;
    using dQaccum_tile_t = subgroup::tile_t<accum_t, dQ_tile_desc_t>;

    dQaccum_tile_t rdQaccum;
    dQ_tile_t rdQ;
    load_tile(&rdQaccum, ctx.mem_desc_dQaccum, ctx.sg_idx * kSgHm,
              ctx.sg_idy * kSgBr);
    rdQaccum.reg *= args.sm_scale;
    // Convert dQ from fp32 to fp16/bf16
    subgroup::elemwise_cvt(rdQ, rdQaccum);

    using epilogue_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrHm, mem_desc_dQ_t>;
    epilogue_t epilogue_dQ;
    epilogue_dQ(ctx.g, rdQ, ctx.mem_desc_dQ);
  }
};

template <typename mha_policy, typename T, bool kUseBias, bool kUseDropout>
class FmhaBackwardDotDOO;
template <typename mha_policy, typename T, bool kUseBias, bool kUseDropout>
class FmhaBackwardKernel;
template <typename mha_policy, typename T, bool kUseBias, bool kUseDropout>
class FmhaBackwardConvertDQ;

// The launcher of fmha backward kernel
template <typename mha_policy, typename T, bool kUseBias, bool kUseDropout>
void fmha_backward_impl(sycl::queue* q, T* query, T* key, T* value, T* out,
                        T* bias, T* grad_out, uint8_t* dropout,
                        float dropout_prob, float* dp_sum, float* L_ptr,
                        T* grad_query, float* grad_query_accum, T* grad_key,
                        T* grad_value, uint32_t num_batches, uint32_t num_heads,
                        uint32_t head_size, uint32_t num_queries,
                        uint32_t num_keys) {
  arguments_t<T, float> args(
      query, key, value, out, bias, grad_out, dropout, dropout_prob, dp_sum,
      L_ptr, grad_query, grad_query_accum, grad_key, grad_value, num_batches,
      num_heads, head_size, num_queries, num_keys);

  using fmha_bwd_dot_do_op_t = fmha_backward_dot_do_o_t<mha_policy, T>;
  sycl::nd_range<3> NdRange0 =
      fmha_bwd_dot_do_op_t::get_nd_range(num_batches * num_heads, num_queries);

  q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<
        class FmhaBackwardDotDOO<mha_policy, T, kUseBias, kUseDropout>>(
        NdRange0, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      sycl::nd_item<3> ei(item);
      fmha_bwd_dot_do_op_t fmha_bwd_dot_do_o_op;
      fmha_bwd_dot_do_o_op(ei, args);
        });
  });

  using fmha_backward_op_t =
      fmha_backward_t<mha_policy, T, kUseBias, kUseDropout>;
  sycl::nd_range<3> NdRange1 =
      fmha_backward_op_t::get_nd_range(num_batches * num_heads, num_keys);

  q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<
        class FmhaBackwardKernel<mha_policy, T, kUseBias, kUseDropout>>(
        NdRange1, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      sycl::nd_item<3> ei(item);
      fmha_backward_op_t fmha_bwd_op;
      fmha_bwd_op(ei, args);
        });
  });

  using fmha_bwd_convert_dq_op_t = fmha_backward_convert_dq_t<mha_policy, T>;
  sycl::nd_range<3> NdRange2 = fmha_bwd_convert_dq_op_t::get_nd_range(
      num_batches * num_heads, num_queries);

  q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<
        class FmhaBackwardConvertDQ<mha_policy, T, kUseBias, kUseDropout>>(
        NdRange2, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      sycl::nd_item<3> ei(item);
      fmha_bwd_convert_dq_op_t fmha_bwd_convert_dq_op;
      fmha_bwd_convert_dq_op(ei, args);
        });
  });
}

}  // namespace fmha
}  // namespace gpu::xetla

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_FMHA_BACKWARD_H_
