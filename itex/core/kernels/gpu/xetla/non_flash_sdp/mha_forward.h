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

#ifndef ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_FORWARD_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_FORWARD_H_

#include <algorithm>
#include <xetla.hpp>

#include "itex/core/kernels/gpu/xetla/non_flash_sdp/mha_policy.h"
#include "itex/core/kernels/gpu/xetla/non_flash_sdp/mha_util.h"

#define _RAW_OUTPUT 0

namespace gpu::xetla {

namespace mha {

template <typename mha_policy, typename scalar_t, bool kUseBias,
          bool kIsTraining>
class mha_forward_t {
 public:
  using accum_t = float;

  struct arguments_t {
    // Input tensors
    scalar_t* Q_ptr;            // [B, N, F, H] - query
    scalar_t* K_ptr;            // [B, N, T, H] - key
    scalar_t* V_ptr;            // [B, N, T, H] - value
    scalar_t* B_ptr = nullptr;  // [B, 1, F, T] - bias
    uint8_t* Dp_ptr = nullptr;  // [B, N, F, T] - dropout
    accum_t dp_prob;
    // Output tensor
    scalar_t* O_ptr;     // raw: [B, N, F, H]; permute: [B, F, N, H] - out
    scalar_t* P_ptr;     // [B, N, F, T] - attention
    scalar_t* P_dp_ptr;  // [B, N, F, T] - attention_dp
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
                       scalar_t* bias, uint8_t* dropout, accum_t dropout_prob,
                       scalar_t* out, scalar_t* attention,
                       scalar_t* attention_dp, uint32_t num_batches,
                       uint32_t num_heads, uint32_t head_size,
                       uint32_t num_queries, uint32_t num_keys)
        : Q_ptr(query),
          K_ptr(key),
          V_ptr(value),
          B_ptr(bias),
          Dp_ptr(dropout),
          dp_prob(dropout_prob),
          O_ptr(out),
          P_ptr(attention),
          P_dp_ptr(attention_dp),
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
  using mem_desc_Q_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_K_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_V_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_B_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Dp_t =
      mem_desc_t<uint8_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_O_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_P_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Pdp_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_P_L_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;

  // ------------------- // Slm and nbarrier // ------------------- //
  static constexpr uint32_t slm_size_P =
      kIsTraining ? 0 : kBr * kTm * sizeof(scalar_t);
  static constexpr uint32_t slm_size_softmax =
      (wg_size_x > 1) ? wg_size * kSgBr * sizeof(accum_t) : 0;

  static constexpr uint32_t P_slm = 0;
  static constexpr uint32_t softmax_slm = P_slm + slm_size_P;

  static constexpr uint32_t nbarrier_cnt = (wg_size_x > 1) ? wg_size_y : 0;

  // ======================== // Context // ======================= //

  /// @brief Variables used in the mha forward
  struct context_t {
    // thread id
    work_group_t g;
    uint32_t sg_idx;
    uint32_t sg_idy;
    // nbarrier
    xetla_nbarrier_t<wg_size_x, wg_size_x> nbarrier;
    // mem desc variables
    mem_desc_Q_t mem_desc_Q;
    mem_desc_K_T_t mem_desc_K_T;
    mem_desc_V_t mem_desc_V;
    mem_desc_B_t mem_desc_B;
    mem_desc_Dp_t mem_desc_Dp;
    mem_desc_O_t mem_desc_O;
    mem_desc_P_t mem_desc_P;
    mem_desc_Pdp_t mem_desc_Pdp;
    mem_desc_P_L_t mem_desc_P_L;

    inline context_t() = default;

    /// @brief Initialize variables used in the mha forward
    inline void init_context(const xetla_exec_item<3>& ei,
                             const arguments_t& args) {
      uint32_t sg_id = ei.get_local_linear_id();
      uint32_t gid = ei.get_group(0);

      // thread id and nbarrier
      g.init(sg_id);
      sg_idx = sg_id % wg_size_x;
      sg_idy = sg_id / wg_size_x;
      nbarrier.init_nbarrier(sg_idy, nbarrier_role::producer_consumer);

      // mem desc variables
      // local P
      mem_desc_P_L.init(P_slm, {kTm, kBr, kTm}, {0, 0});

      // for shape [B,N,T,H]
      int32_t start_x = gid * args.uT;
      uint32_t end_x = start_x + args.uT;
      mem_desc_K_T.init(args.K_ptr, {end_x, args.uH, args.uH}, {start_x, 0});
      mem_desc_V.init(args.V_ptr, {args.uH, end_x, args.uH}, {0, start_x});

      // for shape [B,N,F,T] and [B,N,F,H]
      int32_t start_y = gid * args.uF + ei.get_group(1) * kBr;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = (gid + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;

      if constexpr (kIsTraining) {
        // shape [B,N,F,T]
        mem_desc_P.init(args.P_ptr, {args.uT, end_y, args.uT}, {0, start_y});
        mem_desc_Pdp.init(args.P_dp_ptr, {args.uT, end_y, args.uT},
                          {0, start_y});

        int32_t offset_x = sg_idx * kSgTm;
        int32_t offset_y = start_y + sg_idy * kSgBr;
        mem_desc_Dp.init(args.Dp_ptr, {args.uT, end_y, args.uT},
                         {offset_x, offset_y});
      }

      // shape [B,N,F,H]
      mem_desc_Q.init(args.Q_ptr, {args.uH, end_y, args.uH}, {0, start_y});
#if _RAW_OUTPUT
      mem_desc_O.init(args.O_ptr, {args.uH, end_y, args.uH}, {0, start_y});
#else
      // for O shape [B,F,N,H]
      uint32_t head_id = gid % args.uN;
      uint32_t batch_id = gid / args.uN;
      start_x = head_id * args.uH;
      end_x = (head_id + 1) * args.uH;
      start_y = batch_id * args.uF + ei.get_group(1) * kBr;
      end_y = start_y + kBr;
      boundary_y = (batch_id + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;
      mem_desc_O.init(args.O_ptr, {end_x, end_y, args.uH * args.uN},
                      {start_x, start_y});
#endif

      if constexpr (kUseBias) {
        uint32_t batch_id = gid / args.uN;
        start_y = batch_id * args.uF + ei.get_group(1) * kBr;
        end_y = start_y + kBr;
        boundary_y = (batch_id + 1) * args.uF;
        end_y = end_y > boundary_y ? boundary_y : end_y;

        int32_t offset_x = sg_idx * kSgTm;
        int32_t offset_y = start_y + sg_idy * kSgBr;
        mem_desc_B.init(args.B_ptr, {args.uT, end_y, args.uT},
                        {offset_x, offset_y});
      }
    }
  };

  context_t ctx;

  // ======================== // gemm_S // ======================== //
  // define brgemm kernel
  using brgemm_S_t = group::brgemm_t<compute_policy, tile_shape_BrTm,
                                     mem_desc_Q_t, mem_desc_K_T_t>;
  using matAcc_S_t = typename brgemm_S_t::matAcc_t;

  /// @brief gemm_S is used to compute S = Q x K.T
  /// # [F,H] x [H,T] = [F,T]
  inline void gemm_S(matAcc_S_t* matAcc_S, const arguments_t& args) {
    using brgemm_args_t = typename brgemm_S_t::arguments_t;

    uint32_t loop_count = (args.uH + accum_step - 1) / accum_step;

    // Gemm to comput S
    brgemm_S_t brgemm;
    brgemm_args_t brgemm_args(ctx.mem_desc_Q, ctx.mem_desc_K_T, loop_count);
    brgemm(ctx.g, *matAcc_S, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);
  }

  // ====================== // apply_mask // ====================== //

  /// @brief apply mask to matAcc_S.
  inline void apply_mask(matAcc_S_t* matAcc_S, const arguments_t& args) {
    using tile_mask = tile_mask_t<matAcc_S_t>;

    uint32_t sg_startT = ctx.sg_idx * kSgTm;
    uint32_t remainT = (args.uT < sg_startT) ? 0 : (args.uT - sg_startT);
    if (remainT < kSgTm) {
      tile_mask::padding_mask(matAcc_S, remainT);
    }
  }

  // ====================== // softmax_fwd // ===================== //
  using wg_row_max_t =
      group_row_reduce_t<matAcc_S_t, wg_size_x, reduce_op::max>;
  using wg_row_sum_t =
      group_row_reduce_t<matAcc_S_t, wg_size_x, reduce_op::sum>;

  /// @brief softmax_fwd is used to do forward softmax.
  inline void softmax_fwd(matAcc_S_t* matAcc_S, const arguments_t& args) {
    // multiply by softmax scaling factor
    (matAcc_S->reg) *= args.sm_scale;

    if constexpr (kUseBias) {
      // add bias if needed
      using bias_op_t = subgroup::elemwise_reduce_op_t<reduce_op::sum, scalar_t,
                                                       gpu_arch::Xe>;
      using bias_args_t = typename bias_op_t::arguments_t;

      bias_op_t bias_op;
      bias_args_t bias_args(ctx.mem_desc_B.base, ctx.mem_desc_B.shape);
      bias_op(*matAcc_S, ctx.mem_desc_B.coord, bias_args);
    }

    // apply padding mask to matAcc_S
    apply_mask(matAcc_S, args);

    // compute row max
    uint32_t reducer_slm =
        softmax_slm + ctx.sg_idy * wg_size_x * kSgBr * sizeof(accum_t);
    wg_row_max_t wg_row_max(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    xetla_vector<accum_t, kSgBr> row_max = wg_row_max(matAcc_S);

    if constexpr (wg_size_x > 1) ctx.nbarrier.arrive();

    // compute exp
    subgroup::tile_broadcast_op<subgroup::tile_minus, matAcc_S_t>(*matAcc_S,
                                                                  row_max);
    (matAcc_S->reg) = xetla_exp<accum_t>(matAcc_S->reg);

    if constexpr (wg_size_x > 1) ctx.nbarrier.wait();

    // compute P
    wg_row_sum_t wg_row_sum(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    xetla_vector<accum_t, kSgBr> exp_sum = wg_row_sum(matAcc_S);
    subgroup::tile_broadcast_op<subgroup::tile_div, matAcc_S_t>(*matAcc_S,
                                                                exp_sum);

    if constexpr (kIsTraining) {
      // store P to global memory
      using epilogue_P_t =
          group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                            tile_shape_BrTm, mem_desc_P_t>;
      epilogue_P_t epilogue_P;
      epilogue_P(ctx.g, *matAcc_S, ctx.mem_desc_P);

      // apply dropout mask
      using dropout_op_t = dropout_t<matAcc_S_t, mem_desc_Dp_t>;
      dropout_op_t dropout_op;
      dropout_op(matAcc_S, &(ctx.mem_desc_Dp), args.dp_prob);

      // store P_dp to global memory
      using epilogue_Pdp_t =
          group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                            tile_shape_BrTm, mem_desc_Pdp_t>;
      epilogue_Pdp_t epilogue_Pdp;
      epilogue_Pdp(ctx.g, *matAcc_S, ctx.mem_desc_Pdp);
      xetla_fence<memory_kind::untyped_global>();
    } else {
      // store Pij to local memory
      using epilogue_t =
          group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                            tile_shape_BrTm, mem_desc_P_L_t>;
      epilogue_t epilogue;
      epilogue(ctx.g, *matAcc_S, ctx.mem_desc_P_L);
      xetla_fence<memory_kind::shared_local>();
    }
    if constexpr (wg_size_x > 1) ctx.nbarrier.arrive_wait();
  }

  // ======================== // gemm_O // ======================= //
  // Define kernel to compute O = P x V
  using P_t =
      std::conditional<kIsTraining, mem_desc_Pdp_t, mem_desc_P_L_t>::type;
  using brgemm_O_t =
      group::brgemm_t<compute_policy, tile_shape_BrHm, P_t, mem_desc_V_t>;
  using matAccO_t = typename brgemm_O_t::matAcc_t;

  /// @brief gemm_O is used to compute O = P x V
  /// # [Br,T] x [T,H] = [Br,Hm]
  inline void gemm_O(const arguments_t& args) {
    using brgemm_args_t = typename brgemm_O_t::arguments_t;

    uint32_t loop_count = (args.uT + accum_step - 1) / accum_step;

    // Gemm to comput O
    brgemm_O_t brgemm;
    matAccO_t matAccO(0);
    brgemm_args_t brgemm_args;
    if constexpr (kIsTraining) {
      brgemm_args.init(ctx.mem_desc_Pdp, ctx.mem_desc_V, loop_count);
    } else {
      brgemm_args.init(ctx.mem_desc_P_L, ctx.mem_desc_V, loop_count);
    }
    brgemm(ctx.g, matAccO, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);

    // store O to global memory
    using epilogue_t =
        group::epilogue_t<group::epilogue_policy_default<gpu_arch::Xe>,
                          tile_shape_BrHm, mem_desc_O_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, matAccO, ctx.mem_desc_O);
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count_S = brgemm_S_t::barrier_count;
    constexpr uint32_t barrier_count_O = brgemm_O_t::barrier_count;
    constexpr uint32_t count =
        std::max(barrier_count_S, barrier_count_O) + nbarrier_cnt;
    static_assert(count <= 32,
                  "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_P + slm_size_softmax;
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

  inline KERNEL_FUNC void operator()(const xetla_exec_item<3>& ei,
                                     const arguments_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // initialize context
    ctx.init_context(ei, args);

    // compute S
    matAcc_S_t matAcc_S(0);
    gemm_S(&matAcc_S, args);

    // do softmax
    softmax_fwd(&matAcc_S, args);

    // compute O
    gemm_O(args);
  }
};

template <typename mha_policy, typename T, bool kUseBias, bool kIsTraining>
class MhaForwardKernel;

template <typename mha_policy, typename T, bool kUseBias, bool kIsTraining>
void mha_forward_impl(sycl::queue* q, T* query, T* key, T* value, T* bias,
                      uint8_t* dropout, float dropout_prob, T* out,
                      T* attention, T* attention_dp, uint32_t num_batches,
                      uint32_t num_heads, uint32_t head_size,
                      uint32_t num_queries, uint32_t num_keys) {
  // mha forward kernel
  using mha_forward_op_t = mha_forward_t<mha_policy, T, kUseBias, kIsTraining>;

  sycl::nd_range<3> NdRange =
      mha_forward_op_t::get_nd_range(num_batches * num_heads, num_queries);

  q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<
        class MhaForwardKernel<mha_policy, T, kUseBias, kIsTraining>>(
        NdRange, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      // exec item
      xetla_exec_item<3> ei(item);

      // init mha forward op and arguments
      mha_forward_op_t mha_fwd_op;
      typename mha_forward_op_t::arguments_t args(query, key, value, bias,
                                                  dropout, dropout_prob, out,
                                                  attention, attention_dp,
                                                  num_batches, num_heads,
                                                  head_size, num_queries,
                                                  num_keys);

      // call the functor
      mha_fwd_op(ei, args);
        });
  });
}

}  // namespace mha

}  // namespace gpu::xetla

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_FORWARD_H_
