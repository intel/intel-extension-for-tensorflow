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

#ifndef ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_BACKWARD_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_BACKWARD_H_

#include <algorithm>
#include <xetla.hpp>

#include "itex/core/kernels/gpu/xetla/non_flash_sdp/mha_policy.h"
#include "itex/core/kernels/gpu/xetla/non_flash_sdp/mha_util.h"

namespace gpu::xetla {

namespace mha {

// backward is divided into two parts due to the need of group sync

template <typename mha_policy, typename scalar_t>
class mha_backward0_t {
 public:
  using accum_t = float;

  struct arguments_t {
    // Input tensors
    scalar_t* K_ptr;       // [B, N, T, H] - key
    scalar_t* V_ptr;       // [B, N, T, H] - value
    scalar_t* P_ptr;       // [B, N, F, T] - attention
    scalar_t* dO_ptr;      // [B, F, N, H] - grad_out
    uint8_t* dp_mask_ptr;  // [B, N, F, T] - dropout
    accum_t dp_prob;
    // Output tensors
    scalar_t* dS_ptr;  // [B, N, F, T]
    scalar_t* dQ_ptr;  // [B, N, F, H]
    // Dimension size
    uint32_t uB;
    uint32_t uN;
    uint32_t uH;
    uint32_t uF;
    uint32_t uT;
    // Softmax scale is the reciprocal square root of head size by default
    accum_t sm_scale;

    inline arguments_t() = default;
    inline arguments_t(scalar_t* key, scalar_t* value, scalar_t* attention,
                       scalar_t* grad_out, uint8_t* dropout,
                       accum_t dropout_prob, scalar_t* grad_score,
                       scalar_t* grad_query, uint32_t num_batches,
                       uint32_t num_heads, uint32_t head_size,
                       uint32_t num_queries, uint32_t num_keys)
        : K_ptr(key),
          V_ptr(value),
          P_ptr(attention),
          dO_ptr(grad_out),
          dp_mask_ptr(dropout),
          dp_prob(dropout_prob),
          dS_ptr(grad_score),
          dQ_ptr(grad_query),
          uB(num_batches),
          uN(num_heads),
          uH(head_size),
          uF(num_queries),
          uT(num_keys),
          sm_scale(xetla_rsqrt<accum_t>(accum_t(head_size))) {}
  };

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
  static constexpr uint32_t kTm = mha_policy::kTm;
  static constexpr uint32_t kHm = mha_policy::kHm;
  static constexpr uint32_t kSgBr = mha_policy::kSgBr;
  static constexpr uint32_t kSgTm = mha_policy::kSgTm;
  static constexpr uint32_t kSgHm = mha_policy::kSgHm;

  using tile_shape_BrTm = group::tile_shape_t<kTm, kBr, kSgTm, kSgBr>;
  using tile_shape_BrHm = group::tile_shape_t<kHm, kBr, kSgHm, kSgBr>;

  static constexpr uint32_t wg_size_x = tile_shape_BrTm::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_BrTm::wg_size_y;
  using work_group_t = typename tile_shape_BrTm::work_group_t;
  static constexpr uint32_t wg_size = work_group_t::size;

  static_assert(kHm / kSgHm == kTm / kSgTm,
                "wg_size_x must be the same between Hm and Tm");
  static_assert(wg_size <= 32, "The number of threads should be less than 32!");

  // --------------------- // Memory desc // ---------------------- //
  // suffix: L -> local; T -> transpose
  using mem_desc_K_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_V_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_P_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dO_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Dp_mask_t =
      mem_desc_t<uint8_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dS_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dS_L_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_dQ_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;

  // ------------------- // Slm and nbarrier // ------------------- //
  static constexpr uint32_t slm_size_dS = 0;
  static constexpr uint32_t slm_size_softmax =
      (wg_size_x > 1) ? wg_size * kSgBr * sizeof(accum_t) : 0;

  static constexpr uint32_t dS_slm = 0;
  static constexpr uint32_t softmax_slm = dS_slm + slm_size_dS;

  static constexpr uint32_t nbarrier_cnt = (wg_size_x > 1) ? wg_size_y : 0;

  // ======================== // Context // ======================= //

  /// @brief Variables used in the mha backward0
  struct context_t {
    // thread id
    work_group_t g;
    uint32_t sg_idx;
    uint32_t sg_idy;
    // nbarrier
    xetla_nbarrier_t<wg_size_x, wg_size_x> nbarrier;
    // mem desc variables
    mem_desc_K_t mem_desc_K;
    mem_desc_V_T_t mem_desc_V_T;
    mem_desc_P_t mem_desc_P;
    mem_desc_dO_t mem_desc_dO;
    mem_desc_Dp_mask_t mem_desc_Dp_mask;
    mem_desc_dS_t mem_desc_dS;
    mem_desc_dS_L_t mem_desc_dS_L;
    mem_desc_dQ_t mem_desc_dQ;

    inline context_t() = default;

    /// @brief Initialize variables used in the mha backward0
    inline void init_context(const sycl::nd_item<3>& ei,
                             const arguments_t& args) {
      uint32_t sg_id = ei.get_local_linear_id();
      uint32_t gid = ei.get_group(0);

      // thread id and nbarrier
      g.init(sg_id);
      sg_idx = sg_id % wg_size_x;
      sg_idy = sg_id / wg_size_x;
      nbarrier.init_nbarrier(sg_idy, nbarrier_role::producer_consumer);

      // mem desc variables
      // for shape [B,N,F,T] and [B,N,F,H]
      int32_t start_y = gid * args.uF + ei.get_group(1) * kBr;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = (gid + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;

      mem_desc_dS.init(args.dS_ptr, {args.uT, end_y, args.uT}, {0, start_y});
      mem_desc_dQ.init(args.dQ_ptr, {args.uH, end_y, args.uH}, {0, start_y});

      // add sg offset to Dropout
      int32_t offset_x = sg_idx * kSgTm;
      int32_t offset_y = start_y + sg_idy * kSgBr;
      mem_desc_Dp_mask.init(args.dp_mask_ptr, {args.uT, end_y, args.uT},
                            {offset_x, offset_y});
      mem_desc_P.init(args.P_ptr, {args.uT, end_y, args.uT},
                      {offset_x, offset_y});

      // for shape [B,N,T,H]
      int32_t start_x = gid * args.uT;
      uint32_t end_x = start_x + args.uT;

      mem_desc_V_T.init(args.V_ptr, {end_x, args.uH, args.uH}, {start_x, 0});
      mem_desc_K.init(args.K_ptr, {args.uH, end_x, args.uH}, {0, start_x});

      // for dO shape [B,F,N,H]
      uint32_t head_id = gid % args.uN;
      uint32_t batch_id = gid / args.uN;
      start_x = head_id * args.uH;
      end_x = (head_id + 1) * args.uH;
      start_y = batch_id * args.uF + ei.get_group(1) * kBr;
      end_y = start_y + kBr;
      boundary_y = (batch_id + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;

      mem_desc_dO.init(args.dO_ptr, {end_x, end_y, args.uH * args.uN},
                       {start_x, start_y});

      // local dS
      mem_desc_dS_L.init(dS_slm, {kTm, kBr, kTm}, {0, 0});
    }
  };

  context_t ctx;

  // ======================== // gemm_dP // ======================= //
  // define brgemm kernel
  using brgemm_dP_t = group::gemm_t<compute_policy, tile_shape_BrTm,
                                    mem_desc_dO_t, mem_desc_V_T_t>;
  using matAcc_dP_t = typename brgemm_dP_t::matAcc_t;

  /// @brief gemm_dP is used to compute dP = dO x V.T
  /// # [F,H] x [H,T] = [F,T]
  inline void gemm_dP(matAcc_dP_t* matAcc_dP, const arguments_t& args) {
    using brgemm_args_t = typename brgemm_dP_t::arguments_t;

    uint32_t loop_count = (args.uH + accum_step - 1) / accum_step;

    // Gemm to comput S
    brgemm_dP_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_dO, ctx.mem_desc_V_T, loop_count);
    brgemm(ctx.g, *matAcc_dP, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);
  }

  // ====================== // softmax_bwd // ===================== //
  using dropout_op_t = dropout_t<matAcc_dP_t, mem_desc_Dp_mask_t>;
  using mat_P_tile_desc_t = typename brgemm_dP_t::matAcc_tile_desc_t;
  using mat_P_t = subgroup::tile_t<scalar_t, mat_P_tile_desc_t>;
  using mat_P_payload_t = subgroup::mem_payload_t<
      mem_desc_P_t, mat_P_tile_desc_t,
      subgroup::msg_type_v<mat_P_tile_desc_t, mem_desc_P_t::space>,
      gpu_arch::Xe>;
  using wg_row_sum_t =
      group_row_reduce_t<matAcc_dP_t, wg_size_x, reduce_op::sum>;

  /// @brief softmax_bwd is used to do softmax.
  inline void softmax_bwd(matAcc_dP_t* matAcc_dP, const arguments_t& args) {
    // apply dropout mask
    dropout_op_t dropout_op;
    dropout_op(matAcc_dP, &(ctx.mem_desc_Dp_mask), args.dp_prob);

    // load P from global memory
    mat_P_t mat_P;
    mat_P_payload_t mat_P_payload(ctx.mem_desc_P);
    subgroup::tile_load<cache_hint::cached, cache_hint::cached>(mat_P,
                                                                mat_P_payload);
    matAcc_dP_t matAcc_P, matAcc_DP_P;
    subgroup::elemwise_cvt(matAcc_P, mat_P);

    // compute dP * P
    matAcc_DP_P.reg = matAcc_dP->reg * matAcc_P.reg;

    uint32_t reducer_slm =
        softmax_slm + ctx.sg_idy * wg_size_x * kSgBr * sizeof(accum_t);

    // compute dS
    wg_row_sum_t wg_row_sum(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    xetla_vector<accum_t, kSgBr> row_sum = wg_row_sum(&matAcc_DP_P);

    subgroup::tile_broadcast_op<subgroup::tile_minus, matAcc_dP_t>(*matAcc_dP,
                                                                   row_sum);
    matAcc_dP->reg = matAcc_dP->reg * matAcc_P.reg * args.sm_scale;

    // store dS to global memory
    using epilogue_dS_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrTm, mem_desc_dS_t>;
    epilogue_dS_t epilogue_dS;
    epilogue_dS(ctx.g, *matAcc_dP, ctx.mem_desc_dS);
    xetla_fence<memory_kind::untyped_global>();
    if constexpr (wg_size_x > 1) ctx.nbarrier.arrive_wait();
  }

  // ======================== // gemm_dQ // ======================= //
  // define brgemm kernel
  using brgemm_dQ_t = group::gemm_t<compute_policy, tile_shape_BrHm,
                                    mem_desc_dS_t, mem_desc_K_t>;
  using matAcc_dQ_t = typename brgemm_dQ_t::matAcc_t;

  /// @brief gemm_dQ is used to compute dQ = dS x K
  /// # [F,T] x [T,H] = [F,H]
  inline void gemm_dQ(const arguments_t& args) {
    using brgemm_args_t = typename brgemm_dQ_t::arguments_t;

    uint32_t loop_count = (args.uT + accum_step - 1) / accum_step;

    // Gemm to comput dQ
    brgemm_dQ_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_dS, ctx.mem_desc_K, loop_count);
    matAcc_dQ_t matAcc_dQ(0);
    brgemm(ctx.g, matAcc_dQ, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);

    // store dQ to global memory
    using epilogue_dQ_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrHm, mem_desc_dQ_t>;
    epilogue_dQ_t epilogue;
    epilogue(ctx.g, matAcc_dQ, ctx.mem_desc_dQ);
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count_dP = brgemm_dP_t::barrier_count;
    constexpr uint32_t barrier_count_dQ = brgemm_dQ_t::barrier_count;
    constexpr uint32_t count =
        std::max(barrier_count_dP, barrier_count_dQ) + nbarrier_cnt;
    static_assert(count <= 32,
                  "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_dS + slm_size_softmax;
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

  // ================= // Entry of the functor // ================= //

  inline KERNEL_FUNC void operator()(const sycl::nd_item<3>& ei,
                                     const arguments_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // initialize context
    ctx.init_context(ei, args);

    // compute dP
    matAcc_dP_t matAcc_dP(0);
    gemm_dP(&matAcc_dP, args);

    // do Softmax
    softmax_bwd(&matAcc_dP, args);

    // compute dQ
    gemm_dQ(args);
  }
};

template <typename mha_policy, typename scalar_t>
class mha_backward1_t {
 public:
  using accum_t = float;

  struct arguments_t {
    // Input tensors
    scalar_t* Q_ptr;     // [B, N, F, H] - query
    scalar_t* P_dp_ptr;  // [B, N, F, T] - attention_dp
    scalar_t* dO_ptr;    // [B, F, N, H] - grad_out
    scalar_t* dS_ptr;    // [B, N, F, T] - grad_score
    // Output tensors
    scalar_t* dK_ptr;  // [B, N, T, H] - grad_key
    scalar_t* dV_ptr;  // [B, N, T, H] - grad_value
    // Dimension size
    uint32_t uB;
    uint32_t uN;
    uint32_t uH;
    uint32_t uF;
    uint32_t uT;

    inline arguments_t() = default;
    inline arguments_t(scalar_t* query, scalar_t* attention_dp,
                       scalar_t* grad_out, scalar_t* grad_score,
                       scalar_t* grad_key, scalar_t* grad_value,
                       uint32_t num_batches, uint32_t num_heads,
                       uint32_t head_size, uint32_t num_queries,
                       uint32_t num_keys)
        : Q_ptr(query),
          P_dp_ptr(attention_dp),
          dO_ptr(grad_out),
          dS_ptr(grad_score),
          dK_ptr(grad_key),
          dV_ptr(grad_value),
          uB(num_batches),
          uN(num_heads),
          uH(head_size),
          uF(num_queries),
          uT(num_keys) {}
  };

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
  static constexpr uint32_t kHm = mha_policy::kHm;
  static constexpr uint32_t kSgBr = mha_policy::kSgBr;
  static constexpr uint32_t kSgHm = mha_policy::kSgHm;
  using tile_shape_BrHm = group::tile_shape_t<kHm, kBr, kSgHm, kSgBr>;

  static constexpr uint32_t wg_size_x = tile_shape_BrHm::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_BrHm::wg_size_y;
  using work_group_t = typename tile_shape_BrHm::work_group_t;
  static constexpr uint32_t wg_size = work_group_t::size;

  static_assert(wg_size <= 32, "The number of threads should be less than 32!");

  // --------------------- // Memory desc // ---------------------- //
  // suffix: L -> local; T -> transpose
  using mem_desc_Q_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Pdp_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_dO_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dS_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_dK_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dV_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;

  // ======================== // Context // ======================= //

  /// @brief Used to store variables in the flash mha loops
  struct context_t {
    // thread id
    work_group_t g;
    // mem desc variables
    mem_desc_Q_t mem_desc_Q;
    mem_desc_Pdp_T_t mem_desc_Pdp_T;
    mem_desc_dO_t mem_desc_dO;
    mem_desc_dS_T_t mem_desc_dS_T;
    mem_desc_dK_t mem_desc_dK;
    mem_desc_dV_t mem_desc_dV;

    inline context_t() = default;

    /// @brief Initialize invariant variables in the flash mha loop
    inline void init_context(const sycl::nd_item<3>& ei,
                             const arguments_t& args) {
      uint32_t sg_id = ei.get_local_linear_id();
      uint32_t gid = ei.get_group(0);

      // thread id
      g.init(sg_id);

      // mem desc variables
      // for shape [B,N,F,T]
      int32_t start_x = gid * args.uF;
      uint32_t end_x = start_x + args.uF;
      int32_t start_y = ei.get_group(1) * kBr;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = args.uT;
      end_y = end_y > boundary_y ? boundary_y : end_y;

      mem_desc_Pdp_T.init(args.P_dp_ptr, {end_x, end_y, args.uT},
                          {start_x, start_y});
      mem_desc_dS_T.init(args.dS_ptr, {end_x, end_y, args.uT},
                         {start_x, start_y});

      // for shape [B,N,F,H]
      start_y = gid * args.uF;
      end_y = start_y + args.uF;
      mem_desc_Q.init(args.Q_ptr, {args.uH, end_y, args.uH}, {0, start_y});

      // for shape [B,N,T,H]
      start_y = gid * args.uT + ei.get_group(1) * kBr;
      end_y = start_y + kBr;
      boundary_y = (gid + 1) * args.uT;
      end_y = end_y > boundary_y ? boundary_y : end_y;
      mem_desc_dV.init(args.dV_ptr, {args.uH, end_y, args.uH}, {0, start_y});
      mem_desc_dK.init(args.dK_ptr, {args.uH, end_y, args.uH}, {0, start_y});

      // for dO shape [B,F,N,H]
      uint32_t head_id = gid % args.uN;
      uint32_t batch_id = gid / args.uN;
      start_x = head_id * args.uH;
      end_x = (head_id + 1) * args.uH;
      start_y = batch_id * args.uF;
      end_y = (batch_id + 1) * args.uF;

      mem_desc_dO.init(args.dO_ptr, {end_x, end_y, args.uH * args.uN},
                       {start_x, start_y});
    }
  };

  context_t ctx;

  // ======================== // gemm_dV // ======================= //
  // define brgemm kernel
  using brgemm_dV_t = group::gemm_t<compute_policy, tile_shape_BrHm,
                                    mem_desc_Pdp_T_t, mem_desc_dO_t>;

  /// @brief gemm_dV is used to compute dV = P_dp x dO
  /// # [T,F] x [F,H] = [T,H]
  inline void gemm_dV(const arguments_t& args) {
    using brgemm_args_t = typename brgemm_dV_t::arguments_t;
    using matAcc_dV_t = typename brgemm_dV_t::matAcc_t;

    uint32_t loop_count = (args.uF + accum_step - 1) / accum_step;

    // Gemm to comput dV
    brgemm_dV_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_Pdp_T, ctx.mem_desc_dO, loop_count);
    matAcc_dV_t matAcc_dV(0);
    brgemm(ctx.g, matAcc_dV, brgemm_args);

    // store dV to global memory
    using epilogue_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrHm, mem_desc_dV_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, matAcc_dV, ctx.mem_desc_dV);
  }

  // ======================== // gemm_dK // ======================= //
  // define brgemm kernel
  using brgemm_dK_t = group::gemm_t<compute_policy, tile_shape_BrHm,
                                    mem_desc_dS_T_t, mem_desc_Q_t>;

  /// @brief gemm_dK is used to compute dK = dS.T x Q
  /// # [T,F] x [F,H] = [T,H]
  inline void gemm_dK(const arguments_t& args) {
    using brgemm_args_t = typename brgemm_dK_t::arguments_t;
    using matAcc_dK_t = typename brgemm_dK_t::matAcc_t;

    uint32_t loop_count = (args.uF + accum_step - 1) / accum_step;

    // Gemm to comput dK
    brgemm_dK_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_dS_T, ctx.mem_desc_Q, loop_count);
    matAcc_dK_t matAcc_dK(0);
    brgemm(ctx.g, matAcc_dK, brgemm_args);

    // store dK to global memory
    using epilogue_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrHm, mem_desc_dK_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, matAcc_dK, ctx.mem_desc_dK);
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count_dV = brgemm_dV_t::barrier_count;
    constexpr uint32_t barrier_count_dK = brgemm_dK_t::barrier_count;
    constexpr uint32_t count = std::max(barrier_count_dV, barrier_count_dK);
    static_assert(count <= 32,
                  "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() { return 0u; }

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(uint32_t total_batches,
                                        uint32_t num_keys) {
    // local range
    sycl::range<3> local_range = sycl::range<3>{1, wg_size_y, wg_size_x};
    // group range
    uint32_t group_range_m = (num_keys + kBr - 1) / kBr;
    sycl::range<3> group_range =
        sycl::range<3>{total_batches, group_range_m, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  }

  // ================= // Entry of the functor // ================= //

  inline KERNEL_FUNC void operator()(const sycl::nd_item<3>& ei,
                                     const arguments_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // initialize context
    ctx.init_context(ei, args);

    // compute dV
    gemm_dV(args);

    // compute dK
    gemm_dK(args);
  }
};

template <typename mha_policy, typename T>
class MhaBackwardKernel0;
template <typename mha_policy, typename T>
class MhaBackwardKernel1;

// The launcher of mha backward kernel
template <typename mha_policy, typename T>
void mha_backward_impl(sycl::queue* q, T* query, T* key, T* value, T* attention,
                       T* attention_dp, T* grad_out, uint8_t* dropout,
                       float dropout_prob, T* grad_score, T* grad_query,
                       T* grad_key, T* grad_value, uint32_t num_batches,
                       uint32_t num_heads, uint32_t head_size,
                       uint32_t num_queries, uint32_t num_keys) {
  // The first mha backward kernel
  using mha_backward0_op_t = mha_backward0_t<mha_policy, T>;
  sycl::nd_range<3> NdRange0 =
      mha_backward0_op_t::get_nd_range(num_batches * num_heads, num_queries);

  q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class MhaBackwardKernel0<mha_policy, T>>(
        NdRange0, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      // exec item
      sycl::nd_item<3> ei(item);

      // init mha backward0 op and arguments
      mha_backward0_op_t mha_bwd0_op;
      typename mha_backward0_op_t::arguments_t args(
          key, value, attention, grad_out, dropout, dropout_prob, grad_score,
          grad_query, num_batches, num_heads, head_size, num_queries, num_keys);

      // call the functor
      mha_bwd0_op(ei, args);
        });
  });
  // The second mha backward kernel
  using mha_backward1_op_t = mha_backward1_t<mha_policy, T>;
  sycl::nd_range<3> NdRange1 =
      mha_backward1_op_t::get_nd_range(num_batches * num_heads, num_keys);

  q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class MhaBackwardKernel1<mha_policy, T>>(
        NdRange1, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      // exec item
      sycl::nd_item<3> ei(item);

      // init mha backward1 op and arguments
      mha_backward1_op_t mha_bwd1_op;
      typename mha_backward1_op_t::arguments_t args(
          query, attention_dp, grad_out, grad_score, grad_key, grad_value,
          num_batches, num_heads, head_size, num_queries, num_keys);

      // call the functor
      mha_bwd1_op(ei, args);
        });
  });
}

}  // namespace mha
}  // namespace gpu::xetla

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_BACKWARD_H_
