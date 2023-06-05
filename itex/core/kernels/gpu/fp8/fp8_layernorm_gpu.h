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

#ifndef ITEX_CORE_KERNELS_GPU_FP8_FP8_LAYERNORM_GPU_H_
#define ITEX_CORE_KERNELS_GPU_FP8_FP8_LAYERNORM_GPU_H_

#include "itex/core/kernels/gpu/fp8/utils.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace layernorm {

template <typename Params>
struct LaunchParams {
  OpKernelContext* context_;
  Params params_;
};

struct ParamsBase {
  ParamsBase()
      : wg_num_(0),
        rows_(0),
        cols_(0),
        x_(nullptr),
        mu_(nullptr),
        rs_(nullptr),
        gamma_(nullptr) {}

  int wg_num_;

  // Input is interpreted as matrix. We normalize across columns.
  int rows_;
  int cols_;

  const void* x_;
  float* mu_;
  float* rs_;
  const void* gamma_;
};

struct FwdParams : public ParamsBase {
  FwdParams()
      : ParamsBase(),
        z_(nullptr),
        beta_(nullptr),
        epsilon_(0.f),
        z_scale_(nullptr),
        z_amax_(nullptr) {}

  // Output of LN FWD.
  void* z_;
  const void* beta_;
  float epsilon_;

  const void* z_scale_;
  void* z_amax_;
};

struct BwdParams : public ParamsBase {
  BwdParams()
      : ParamsBase(),
        dz_(nullptr),
        dbeta_part_(nullptr),
        dgamma_part_(nullptr),
        dx_(nullptr),
        dbeta_(nullptr),
        dgamma_(nullptr),
        dz_scale_inv_(nullptr) {}

  // Input: gradient wrt. LN FWD output.
  const void* dz_;

  // Workspace for Wgrad pre-reduction.
  void* dbeta_part_;
  void* dgamma_part_;

  // Output: Dgrad.
  void* dx_;
  // Output: Wgrad.
  void* dbeta_;
  void* dgamma_;

  const void* dz_scale_inv_;
};

template <typename input_t, typename weight_t, typename output_t,
          int SUBGROUPS_M, int SUBGROUPS_N, int BYTES_PER_LOAD, int HIDDEN_SIZE>
struct Fp8LayerNormFwdKernel {
  static constexpr int NUM_ELTS = BYTES_PER_LOAD / sizeof(weight_t);
  static constexpr int THREADS_PER_ROW = SUBGROUPS_N * THREADS_PER_SUBGROUP;
  static constexpr int LOADS = HIDDEN_SIZE / (THREADS_PER_ROW * NUM_ELTS);
  static constexpr int THREADS_PER_GROUP = SUBGROUPS_M * THREADS_PER_ROW;
  static constexpr int COLS_PER_GROUP = NUM_ELTS * THREADS_PER_ROW;
  static constexpr int SMEM_SIZE = SUBGROUPS_M * SUBGROUPS_N;

  using index_t = int;
  using compute_t = float;
  using Ivec = Vec<input_t, NUM_ELTS>;
  using Ovec = Vec<output_t, NUM_ELTS>;
  using Wvec = Vec<weight_t, NUM_ELTS>;
  using Cvec = Vec<compute_t, NUM_ELTS>;
  using Smem = sycl::local_accessor<compute_t, 1>;
  using Reducer = Reducer<compute_t, Smem, SUBGROUPS_M, SUBGROUPS_N>;

  Fp8LayerNormFwdKernel(FwdParams params, Smem scratch)
      : params_(params), scratch_(scratch) {}

  [[intel::reqd_sub_group_size(THREADS_PER_SUBGROUP)]] void operator()(
      sycl::nd_item<1> item) const {
    auto g = item.get_group();
    auto sg = item.get_sub_group();
    const index_t tid = item.get_local_linear_id();
    const index_t gid = item.get_group(0);
    const index_t lane = sg.get_local_id();
    const index_t sg_id = tid / THREADS_PER_SUBGROUP;
    const index_t sg_m = sg_id / SUBGROUPS_N;
    const index_t sg_n = sg_id % SUBGROUPS_N;

    const index_t group_row = gid * SUBGROUPS_M;
    const index_t vec_col = sg_n * THREADS_PER_SUBGROUP + lane;

    Reducer reducer(g, sg, sg_m, sg_n, lane, scratch_);

    compute_t* mu_ptr = static_cast<compute_t*>(params_.mu_);
    compute_t* rs_ptr = static_cast<compute_t*>(params_.rs_);
    compute_t rn = compute_t(1.f) / params_.cols_;

    compute_t z_amax = 0.f, z_scale = 1.f;
    if constexpr (is_fp8<output_t>::value) {
      z_scale = *reinterpret_cast<const float*>(params_.z_scale_);
    }

    Cvec gamma[LOADS];
    Cvec beta[LOADS];
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && col < params_.cols_; ++it, col += COLS_PER_GROUP) {
      Wvec gamma_in, beta_in;
      gamma_in.load_from_elts(params_.gamma_, col, params_.cols_ - col);
      beta_in.load_from_elts(params_.beta_, col, params_.cols_ - col);
      gamma_in.to(gamma[it]);
      beta_in.to(beta[it]);
    }

    const int row = group_row + sg_m;
    // Load input
    Cvec xs[LOADS];
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows_ && col < params_.cols_;
         it++, col += COLS_PER_GROUP) {
      Ivec x_in;
      x_in.load_from_elts(params_.x_, row * params_.cols_ + col,
                          params_.cols_ - col);
      x_in.to(xs[it]);
    }

    // Compute mean
    compute_t mu = 0.f;
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows_ && col < params_.cols_;
         it++, col += COLS_PER_GROUP) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        mu += xs[it].data[jt];
      }
    }
    mu = reducer.allreduce(mu, sycl::plus<compute_t>()) * rn;

    // Compute variance
    compute_t rs = 0.f;
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows_ && col < params_.cols_;
         it++, col += COLS_PER_GROUP) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; ++jt) {
        if (col + jt < params_.cols_) {
          compute_t diff = xs[it].data[jt] - mu;
          rs += diff * diff;
        }
      }
    }

    rs = reducer.allreduce(rs, sycl::plus<compute_t>()) * rn;
    rs = sycl::rsqrt(rs + compute_t(params_.epsilon_));

    // Write statistics
    if (sg_n == 0 && lane == 0 && row < params_.rows_) {
      mu_ptr[row] = mu;
      rs_ptr[row] = rs;
    }

    // Compute output
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows_ && col < params_.cols_;
         it++, col += COLS_PER_GROUP) {
      // Compute output values
      Cvec z;
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t y = rs * (xs[it].data[jt] - mu);
        compute_t g_ij = gamma[it].data[jt];
        compute_t b_ij = beta[it].data[jt];
        z.data[jt] = g_ij * y + b_ij;
      }

      // Apply fp8 factors
      if constexpr (is_fp8<output_t>::value) {
#pragma unroll
        for (int jt = 0; jt < NUM_ELTS; jt++) {
          if (col + jt < params_.cols_) {
            compute_t z_ij = z.data[jt];
            z_amax = sycl::max(z_amax, sycl::abs(z_ij));  // NOLINT
            z.data[jt] = z_ij * z_scale;
          }
        }
      }

      // Store output
      Ovec z_out;
      z.to(z_out);
      z_out.store_to_elts(params_.z_, row * params_.cols_ + col,
                          params_.cols_ - col);
    }

    // Finalize fp8 factors
    if constexpr (is_fp8<output_t>::value) {
      z_amax = sycl::reduce_over_group(g, z_amax, sycl::maximum<float>());
      if (tid == 0) {
        static_assert(std::is_same<compute_t, float>::value);
        ItexAtomicMax(reinterpret_cast<compute_t*>(params_.z_amax_), z_amax);
      }
    }
  }

 private:
  FwdParams params_;
  Smem scratch_;
};

template <typename input_t, typename grad_t, typename weight_t,
          typename output_t, int SUBGROUPS_M, int SUBGROUPS_N,
          int BYTES_PER_LOAD, int HIDDEN_SIZE>
struct Fp8LayerNormBwdKernel {
  static constexpr int NUM_ELTS = BYTES_PER_LOAD / sizeof(weight_t);
  static constexpr int THREADS_PER_ROW = SUBGROUPS_N * THREADS_PER_SUBGROUP;
  static constexpr int LOADS = HIDDEN_SIZE / (THREADS_PER_ROW * NUM_ELTS);
  static constexpr int THREADS_PER_GROUP = SUBGROUPS_M * THREADS_PER_ROW;
  static constexpr int COLS_PER_GROUP = NUM_ELTS * THREADS_PER_ROW;
  static constexpr int SMEM_SIZE_FOR_STATS = SUBGROUPS_M * SUBGROUPS_N;

  using index_t = int;
  using compute_t = float;
  using Ivec = Vec<input_t, NUM_ELTS>;
  using Gvec = Vec<grad_t, NUM_ELTS>;
  using Ovec = Vec<output_t, NUM_ELTS>;
  using Wvec = Vec<weight_t, NUM_ELTS>;
  using Cvec = Vec<compute_t, NUM_ELTS>;
  using reduce_t = typename TypeToVec2<compute_t>::Type;
  using Smem0 = sycl::local_accessor<reduce_t, 1>;
  using Smem1 = sycl::local_accessor<Cvec, 3>;
  using Reducer = Reducer<reduce_t, Smem0, SUBGROUPS_M, SUBGROUPS_N>;

  Fp8LayerNormBwdKernel(BwdParams params, Smem0 stats_scratch,
                        Smem1 weight_scratch)
      : params_(params),
        stats_scratch_(stats_scratch),
        weight_scratch_(weight_scratch) {}

  [[intel::reqd_sub_group_size(THREADS_PER_SUBGROUP)]] void operator()(
      sycl::nd_item<1> item) const {
    auto g = item.get_group();
    auto sg = item.get_sub_group();
    const index_t tid = item.get_local_linear_id();
    const index_t gid = item.get_group(0);
    const index_t lane = sg.get_local_id();
    const index_t sg_id = tid / THREADS_PER_SUBGROUP;
    const index_t sg_m = sg_id / SUBGROUPS_N;
    const index_t sg_n = sg_id % SUBGROUPS_N;

    const index_t group_row = gid * SUBGROUPS_M;
    const index_t vec_col = sg_n * THREADS_PER_SUBGROUP + lane;

    Cvec dzy_sum[LOADS];
    Cvec dz_sum[LOADS];
#pragma unroll
    for (int it = 0; it < LOADS; ++it) {
      dzy_sum[it].clear();
      dz_sum[it].clear();
    }

    Reducer reducer(g, sg, sg_m, sg_n, lane, stats_scratch_);
    const compute_t rn = 1.f / static_cast<compute_t>(params_.cols_);

    compute_t dz_scale_inv = 1.f;
    if constexpr (is_fp8<grad_t>::value) {
      dz_scale_inv = *reinterpret_cast<const float*>(params_.dz_scale_inv_);
    }

    Cvec gamma[LOADS];
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && col < params_.cols_; it++, col += COLS_PER_GROUP) {
      Wvec gamma_in;
      gamma_in.load_from_elts(params_.gamma_, col, params_.cols_ - col);
      gamma_in.to(gamma[it]);
    }

    const int row = group_row + sg_m;
    compute_t mu = 0.f;
    compute_t rs = 0.f;
    if (row < params_.rows_) {
      mu = static_cast<const compute_t*>(params_.mu_)[row];
      rs = static_cast<const compute_t*>(params_.rs_)[row];
    }

    Cvec dy[LOADS];
    Cvec y[LOADS];
    compute_t mdy = 0.f;
    compute_t mdyy = 0.f;

#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows_ && col < params_.cols_;
         it++, col += COLS_PER_GROUP) {
      Ivec x;
      Gvec dz;
      Cvec xf, dzf;
      x.load_from_elts(params_.x_, row * params_.cols_ + col,
                       params_.cols_ - col);
      x.to(xf);
      dz.load_from_elts(params_.dz_, row * params_.cols_ + col,
                        params_.cols_ - col);
      dz.to(dzf);
      if constexpr (is_fp8<grad_t>::value) {
        dzf.scale(dz_scale_inv);
      }

#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        const compute_t x_ij = xf.data[jt];
        const compute_t y_ij = rs * (x_ij - mu);
        const compute_t g_ij = gamma[it].data[jt];
        const compute_t dz_ij = dzf.data[jt];
        const compute_t dy_ij = g_ij * dz_ij;

        y[it].data[jt] = y_ij;
        dy[it].data[jt] = dy_ij;

        mdy += dy_ij;
        mdyy += dy_ij * y_ij;

        dz_sum[it].data[jt] += dz_ij;
        dzy_sum[it].data[jt] += dz_ij * y_ij;
      }
    }

    // Reduce over row
    reduce_t result = reducer.allreduce({mdy, mdyy}, sycl::plus<reduce_t>());
    mdy = result[0] * rn;
    mdyy = result[1] * rn;

    // Compute dx
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows_ && col < params_.cols_;
         it++, col += COLS_PER_GROUP) {
      Ovec dx;
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t dy_ij = dy[it].data[jt];
        compute_t y_ij = y[it].data[jt];
        compute_t dx_ij = rs * (dy_ij - (mdyy * y_ij + mdy));
        dx.data[jt] = typename Ovec::Scalar_type(dx_ij);
      }
      dx.store_to_elts(params_.dx_, row * params_.cols_ + col,
                       params_.cols_ - col);
    }

    if constexpr (SUBGROUPS_M == 1) {
      // Write out local weight grad contributions
#pragma unroll
      for (int it = 0, col = vec_col * NUM_ELTS;
           it < LOADS && col < params_.cols_; it++, col += COLS_PER_GROUP) {
        dz_sum[it].store_to_elts(params_.dbeta_part_, gid * params_.cols_ + col,
                                 params_.cols_ - col);
        dzy_sum[it].store_to_elts(params_.dgamma_part_,
                                  gid * params_.cols_ + col,
                                  params_.cols_ - col);
      }
    } else {
      // Reduce ln_dz
#pragma unroll
      for (int it = 0, col = vec_col * NUM_ELTS;
           it < LOADS && col < params_.cols_; it++, col += COLS_PER_GROUP) {
        if (it != sg_m) {
          weight_scratch_[it][sg_m * SUBGROUPS_N + sg_n][lane] = dz_sum[it];
        }
      }
      sycl::group_barrier(g);
#pragma unroll
      for (int it = sg_m, col = (vec_col + it * THREADS_PER_ROW) * NUM_ELTS;
           it < LOADS && col < params_.cols_;
           it += SUBGROUPS_M, col += THREADS_PER_GROUP * NUM_ELTS) {
#pragma unroll
        for (int kt = 0; kt < SUBGROUPS_M; kt++) {
          if (kt != sg_m) {
#pragma unroll
            for (int jt = 0; jt < NUM_ELTS; jt++) {
              dz_sum[it].data[jt] +=
                  weight_scratch_[it][kt * SUBGROUPS_N + sg_n][lane].data[jt];
            }
          }
        }
        dz_sum[it].store_to_elts(params_.dbeta_part_, gid * params_.cols_ + col,
                                 params_.cols_ - col);
      }

      // Reduce ln_dzy
      sycl::group_barrier(g);
#pragma unroll
      for (int it = 0, col = vec_col * NUM_ELTS;
           it < LOADS && col < params_.cols_; it++, col += COLS_PER_GROUP) {
        if (it != sg_m) {
          weight_scratch_[it][sg_m * SUBGROUPS_N + sg_n][lane] = dzy_sum[it];
        }
      }
      sycl::group_barrier(g);
#pragma unroll
      for (int it = sg_m, col = (vec_col + it * THREADS_PER_ROW) * NUM_ELTS;
           it < LOADS && col < params_.cols_;
           it += SUBGROUPS_M, col += THREADS_PER_GROUP * NUM_ELTS) {
#pragma unroll
        for (int kt = 0; kt < SUBGROUPS_M; kt++) {
          if (kt != sg_m) {
#pragma unroll
            for (int jt = 0; jt < NUM_ELTS; jt++) {
              dzy_sum[it].data[jt] +=
                  weight_scratch_[it][kt * SUBGROUPS_N + sg_n][lane].data[jt];
            }
          }
        }
        dzy_sum[it].store_to_elts(params_.dgamma_part_,
                                  gid * params_.cols_ + col,
                                  params_.cols_ - col);
      }
    }
  }

 private:
  BwdParams params_;
  Smem0 stats_scratch_;
  Smem1 weight_scratch_;
};

template <typename weight_t, int HIDDEN_SIZE>
struct Fp8LayerNormBwdFinalizeKernel {
  using compute_t = float;
  using index_t = int;

  static constexpr int BYTES_PER_LOAD = 4;
  static constexpr int NUM_ELTS = BYTES_PER_LOAD / sizeof(compute_t);

  using Wvec = Vec<weight_t, NUM_ELTS>;
  using Cvec = Vec<compute_t, NUM_ELTS>;
  using reduce_t = compute_t;

  static constexpr int SUBGROUPS_M = 32;
  static constexpr int SUBGROUPS_N = 1;
  static_assert(SUBGROUPS_N == 1);
  static constexpr int THREADS_PER_COL_PER_GROUP =
      SUBGROUPS_N * THREADS_PER_SUBGROUP;
  static constexpr int THREADS_PER_GROUP =
      SUBGROUPS_M * SUBGROUPS_N * THREADS_PER_SUBGROUP;
  static constexpr int SMEM_SIZE = THREADS_PER_GROUP * 2;
  using Smem = sycl::local_accessor<Cvec, 1>;
  using Reducer = Reducer<compute_t, Smem, SUBGROUPS_M, 1>;

  Fp8LayerNormBwdFinalizeKernel(BwdParams params, Smem scratch)
      : params_(params), scratch_(scratch) {}

  [[intel::reqd_sub_group_size(THREADS_PER_SUBGROUP)]] void operator()(
      sycl::nd_item<1> item) const {
    auto g = item.get_group();
    auto sg = item.get_sub_group();
    const index_t tid = item.get_local_linear_id();
    const index_t gid = item.get_group(0);
    const index_t lane = sg.get_local_id();
    const index_t sg_id = tid / THREADS_PER_SUBGROUP;
    const index_t sg_m = sg_id / SUBGROUPS_N;
    const index_t sg_n = sg_id % SUBGROUPS_N;
    const index_t col =
        (gid * THREADS_PER_COL_PER_GROUP + sg_n * THREADS_PER_SUBGROUP + lane) *
        NUM_ELTS;

    Cvec dgamma, dbeta;
    dgamma.clear();
    dbeta.clear();
    for (int row = sg_m; row < params_.wg_num_ && col < params_.cols_;
         row += SUBGROUPS_M) {
      Cvec dgamma_part, dbeta_part;
      dgamma_part.load_from_elts(
          params_.dgamma_part_, row * params_.cols_ + col, params_.cols_ - col);
      dbeta_part.load_from_elts(params_.dbeta_part_, row * params_.cols_ + col,
                                params_.cols_ - col);
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        dgamma.data[jt] += dgamma_part.data[jt];
        dbeta.data[jt] += dbeta_part.data[jt];
      }
    }
    // Reduce dgamma and dbeta.
    Reducer reducer(g, sg, sg_m, sg_n, lane, scratch_);
    const index_t write_row = sg_m;
    // Avoid bank conflict.
    const index_t write_col = lane ^ write_row;
    const index_t write_idx = write_row * THREADS_PER_COL_PER_GROUP + write_col;
    scratch_[write_idx] = dgamma;
    scratch_[THREADS_PER_GROUP + write_idx] = dbeta;
    sycl::group_barrier(g);

    // More than one iter iff SUBGROUPS < 32.
#pragma unroll
    for (int m = sg_m; m < THREADS_PER_SUBGROUP; m += SUBGROUPS_M) {
      const int read_row = lane;
      const int read_col = m ^ read_row;
      const int read_idx = read_row * THREADS_PER_COL_PER_GROUP + read_col;

      Cvec dbeta_local, dgamma_local;
      dbeta_local.clear();
      dgamma_local.clear();

      // Load beta and gamma transposed
      if (read_row < SUBGROUPS_M) {
        dgamma_local = scratch_[read_idx];
        dbeta_local = scratch_[THREADS_PER_GROUP + read_idx];
      }

      // Call reducer.
#pragma unroll
      for (int it = 0; it < NUM_ELTS; it++) {
        compute_t b_i = dbeta_local.data[it];
        compute_t g_i = dgamma_local.data[it];
        b_i = reducer.reduce(b_i, sycl::plus<reduce_t>());
        g_i = reducer.reduce(g_i, sycl::plus<reduce_t>());

        dgamma_local.data[it] = g_i;
        dbeta_local.data[it] = b_i;
      }

      // Leader stores the result at the current column.
      const index_t reduce_col =
          (m + gid * THREADS_PER_COL_PER_GROUP) * NUM_ELTS;
      if (lane == 0 && reduce_col < params_.cols_) {
        Wvec dgamma_final, dbeta_final;
        dgamma_local.to(dgamma_final);
        dbeta_local.to(dbeta_final);
        dgamma_final.store_to_elts(params_.dgamma_, reduce_col,
                                   params_.cols_ - reduce_col);
        dbeta_final.store_to_elts(params_.dbeta_, reduce_col,
                                  params_.cols_ - reduce_col);
      }
    }
  }

 private:
  BwdParams params_;
  Smem scratch_;
};

template <typename Tin, typename Tweight, typename Tout, int SUBGROUPS_M,
          int SUBGROUPS_N, int HIDDEN_SIZE, int BYTES_PER_LOAD>
void launch_fp8_layernorm_fwd(
    LaunchParams<FwdParams>& launch_params) {  // NOLINT
  using kernel_type =
      Fp8LayerNormFwdKernel<Tin, Tweight, Tout, SUBGROUPS_M, SUBGROUPS_N,
                            BYTES_PER_LOAD, HIDDEN_SIZE>;
  // Set kernel launch parameter.
  constexpr int THREADS_PER_GROUP = kernel_type::THREADS_PER_GROUP;

  // Reserve random state for bwd kernel.
  FwdParams& params = launch_params.params_;
  params.wg_num_ = DivUp(params.rows_, SUBGROUPS_M);

  auto stream = launch_params.context_->GetDeviceStream();
  stream->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> scratch(kernel_type::SMEM_SIZE, cgh);
    kernel_type task(params, scratch);
    cgh.parallel_for<kernel_type>(
        sycl::nd_range<1>(params.wg_num_ * THREADS_PER_GROUP,
                          THREADS_PER_GROUP),
        task);
  });
}

template <typename Tin, typename Tgrad, typename Tweight, typename Tout,
          int SUBGROUPS_M, int SUBGROUPS_N, int HIDDEN_SIZE, int BYTES_PER_LOAD>
void launch_fp8_layernorm_bwd(
    LaunchParams<BwdParams>& launch_params) {  // NOLINT
  auto context = launch_params.context_;
  using kernel_type =
      Fp8LayerNormBwdKernel<Tin, Tgrad, Tweight, Tout, SUBGROUPS_M, SUBGROUPS_N,
                            BYTES_PER_LOAD, HIDDEN_SIZE>;

  BwdParams& params = launch_params.params_;
  params.wg_num_ = DivUp(params.rows_, SUBGROUPS_M);

  Tensor dgamma_part, dbeta_part;
  OP_REQUIRES_OK(
      context, context->allocate_temp(
                   DataTypeToEnum<typename kernel_type::compute_t>::v(),
                   TensorShape({params.wg_num_, params.cols_}), &dgamma_part));
  OP_REQUIRES_OK(context,
                 context->allocate_temp(
                     DataTypeToEnum<typename kernel_type::compute_t>::v(),
                     TensorShape({params.wg_num_, params.cols_}), &dbeta_part));
  params.dgamma_part_ =
      dgamma_part.flat<typename kernel_type::compute_t>().data();
  params.dbeta_part_ =
      dbeta_part.flat<typename kernel_type::compute_t>().data();
  auto stream = context->GetDeviceStream();
  stream->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<typename kernel_type::reduce_t, 1> stats_scratch(
        kernel_type::SMEM_SIZE_FOR_STATS, cgh);
    // We set the last dims as THREADS_PER_SUBGROUP + 1 to avoid
    // bank conflict.
    sycl::local_accessor<typename kernel_type::Cvec, 3> weight_scratch(
        {kernel_type::LOADS, SUBGROUPS_M * SUBGROUPS_N,
         THREADS_PER_SUBGROUP + 1},
        cgh);
    kernel_type task(launch_params.params_, stats_scratch, weight_scratch);
    cgh.parallel_for<kernel_type>(
        sycl::nd_range<1>(params.wg_num_ * kernel_type::THREADS_PER_GROUP,
                          kernel_type::THREADS_PER_GROUP),
        task);
  });

  using finalize_kernel_type =
      Fp8LayerNormBwdFinalizeKernel<Tweight, HIDDEN_SIZE>;
  stream->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<typename finalize_kernel_type::Cvec, 1> scratch(
        finalize_kernel_type::SMEM_SIZE, cgh);
    finalize_kernel_type task(params, scratch);
    cgh.parallel_for<finalize_kernel_type>(
        sycl::nd_range<1>(
            DivUp(params.cols_,
                  THREADS_PER_SUBGROUP * finalize_kernel_type::NUM_ELTS) *
                finalize_kernel_type::THREADS_PER_GROUP,
            finalize_kernel_type::THREADS_PER_GROUP),
        task);
  });
}

};  // namespace layernorm

namespace functor {

template <typename Device, typename Tin, typename Tweight, typename Tout>
struct Fp8LayerNormFwd {
  void operator()(OpKernelContext* context, const void* x, const Tweight* gamma,
                  const Tweight* beta, float* mu, float* rsigma, void* z,
                  float* z_amax, const float* z_scale, const float epsilon,
                  int rows, int cols) {
    using namespace itex::layernorm;  // NOLINT

    auto launcher =
        &launch_fp8_layernorm_fwd<Tin, Tweight, Tout, 1, 4, 1024, 16>;

    if (cols <= 128) {
      launcher = &launch_fp8_layernorm_fwd<Tin, Tweight, Tout, 4, 1, 128, 8>;
    } else if (cols <= 512) {
      launcher = &launch_fp8_layernorm_fwd<Tin, Tweight, Tout, 4, 1, 512, 16>;
    } else if (cols <= 1024) {
      launcher = &launch_fp8_layernorm_fwd<Tin, Tweight, Tout, 1, 4, 1024, 16>;
    } else if (cols <= 2048) {
      launcher = &launch_fp8_layernorm_fwd<Tin, Tweight, Tout, 1, 8, 2048, 16>;
    } else if (cols <= 8192) {
      launcher = &launch_fp8_layernorm_fwd<Tin, Tweight, Tout, 1, 16, 8192, 16>;
    } else {
      /* TODO(itex): support welford updating for large cols. */
      context->SetStatus(errors::InvalidArgument("Unsupported shape"));
      return;
    }

    LaunchParams<FwdParams> launch_params;
    launch_params.context_ = context;

    FwdParams& params = launch_params.params_;
    params.rows_ = rows;
    params.cols_ = cols;
    params.x_ = x;
    params.mu_ = mu;
    params.rs_ = rsigma;
    params.gamma_ = gamma;
    params.beta_ = beta;
    params.epsilon_ = epsilon;
    params.z_ = z;
    params.z_amax_ = z_amax;
    params.z_scale_ = z_scale;

    launcher(launch_params);
  }
};

template <typename Device, typename Tin, typename Tgrad, typename Tweight,
          typename Tout>
struct Fp8LayerNormBwd {
  void operator()(OpKernelContext* context, const void* dz, const void* x,
                  const float* mu, const float* rsigma, const Tweight* gamma,
                  void* dx, Tweight* dgamma, Tweight* dbeta,
                  const float* dz_scale_inv, int rows, int cols) {
    using namespace itex::layernorm;  // NOLINT

    auto launcher =
        &launch_fp8_layernorm_bwd<Tin, Tgrad, Tweight, Tout, 4, 4, 1024, 16>;

    if (cols <= 128) {
      launcher =
          &launch_fp8_layernorm_bwd<Tin, Tgrad, Tweight, Tout, 4, 1, 128, 8>;
    } else if (cols <= 512) {
      launcher =
          &launch_fp8_layernorm_bwd<Tin, Tgrad, Tweight, Tout, 4, 1, 512, 16>;
    } else if (cols <= 1024) {
      launcher =
          &launch_fp8_layernorm_bwd<Tin, Tgrad, Tweight, Tout, 4, 4, 1024, 16>;
    } else if (cols <= 2048) {
      launcher =
          &launch_fp8_layernorm_bwd<Tin, Tgrad, Tweight, Tout, 2, 8, 2048, 16>;
    } else if (cols <= 4096) {
      launcher =
          &launch_fp8_layernorm_bwd<Tin, Tgrad, Tweight, Tout, 2, 8, 4096, 16>;
    } else if (cols <= 8192) {
      launcher =
          &launch_fp8_layernorm_bwd<Tin, Tgrad, Tweight, Tout, 1, 16, 8192, 16>;
    } else {
      /* TODO(itex): support welford updating for large cols. */
      context->SetStatus(errors::InvalidArgument("Unsupported shape"));
      return;
    }

    LaunchParams<BwdParams> launch_params;
    launch_params.context_ = context;

    BwdParams& params = launch_params.params_;
    params.rows_ = rows;
    params.cols_ = cols;
    params.dz_ = dz;
    params.x_ = x;
    params.mu_ = const_cast<float*>(mu);
    params.rs_ = const_cast<float*>(rsigma);
    params.gamma_ = gamma;
    params.dx_ = dx;
    params.dgamma_ = dgamma;
    params.dbeta_ = dbeta;
    params.dz_scale_inv_ = dz_scale_inv;

    launcher(launch_params);
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FP8_FP8_LAYERNORM_GPU_H_
