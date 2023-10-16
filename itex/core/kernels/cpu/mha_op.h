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

#ifndef ITEX_CORE_KERNELS_CPU_MHA_OP_H_
#define ITEX_CORE_KERNELS_CPU_MHA_OP_H_

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/kernels/cpu/cpu_blas.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/parallel_openmp.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

// is_reduced_floating_point represents bf16/fp16 type.
template <typename T>
struct is_reduced_floating_point
    : std::integral_constant<bool,
                             std::is_same<T, Eigen::half>::value ||
                                 std::is_same<T, Eigen::bfloat16>::value> {};

template <typename T>
constexpr bool is_reduced_floating_point_v =
    is_reduced_floating_point<T>::value;

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  ITEX_CHECK(ptr2 == nullptr);
  return ptr;
}

template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

template <typename T, int64_t qSplitSize, int64_t kvSplitSize>
class FmhaFunctor {
 public:
  using Tvec = typename TTypes<T>::Flat;
  using Fvec = typename TTypes<float>::Flat;

  void operator()(const Tensor& query, const Tensor& key, const Tensor& value,
                  int64_t batch_size, int64_t q_seq_len, int64_t num_heads,
                  int64_t head_size, int64_t k_seq_len, bool use_mask,
                  bool use_causal, bool use_dropout, const Tensor& atten_mask,
                  const Tensor& dropout_mask, float dropout_prob,
                  Tensor* output) {
    int64_t q_stride_b = num_heads * q_seq_len * head_size;
    int64_t q_stride_h = q_seq_len * head_size;
    int64_t q_stride_m = head_size;
    int64_t k_stride_b = num_heads * k_seq_len * head_size;
    int64_t k_stride_h = k_seq_len * head_size;
    int64_t k_stride_n = head_size;
    int64_t v_stride_b = num_heads * k_seq_len * head_size;
    int64_t v_stride_h = k_seq_len * head_size;
    int64_t v_stride_n = head_size;
    int64_t o_stride_b = num_heads * q_seq_len * head_size;
    int64_t o_stride_h = head_size;
    int64_t o_stride_m = num_heads * head_size;

    float scaling_factor = 1.0 / std::sqrt(static_cast<double>(head_size));

    int64_t q_split_size = qSplitSize > q_seq_len ? q_seq_len : qSplitSize;
    int64_t kv_split_size = kvSplitSize > k_seq_len ? k_seq_len : kvSplitSize;
    int64_t q_slice = (q_seq_len - 1) / q_split_size + 1;
    int64_t num_thread = omp_get_max_threads();

    // Allocate per thread temp buf (float type).
    // Use float for intermediate computation to avoid overflow issues.
    int64_t size_per_thread =
        /* qk     */ q_split_size * kv_split_size +
        /* qk_max */ q_split_size +
        /* qk_sum */ q_split_size +
        /* dst    */ q_split_size * head_size;

    constexpr bool is_reduced_type = is_reduced_floating_point_v<T>;
    Tensor buf(DT_FLOAT, {num_thread, size_per_thread});
    // We should convert the intermediate type back to compute (P_ij @ V). This
    // buf stores the temporary P_ij that has been converted to the T type.
    Tensor buf_reduced(
        DataTypeToEnum<T>::v(),
        {num_thread, q_split_size, is_reduced_type ? kv_split_size : 0});

    T* q_data = const_cast<Tensor&>(query).flat<T>().data();
    T* k_data = const_cast<Tensor&>(key).flat<T>().data();
    T* v_data = const_cast<Tensor&>(value).flat<T>().data();
    T* out_data = const_cast<Tensor&>(*output).flat<T>().data();

    float* buf_data = buf.flat<float>().data();
    T* buf_reduced_data =
        is_reduced_type ? buf_reduced.flat<T>().data() : nullptr;

    parallel_for(
        0, batch_size * num_heads * q_slice, 1,
        [&](int64_t begin, int64_t end) {
          int64_t i = 0, j = 0, k = 0;
          DataIndexInit(begin, &i, batch_size, &j, num_heads, &k, q_slice);

          int omp_idx = omp_get_thread_num();
          float* buf_ptr = buf_data + omp_idx * size_per_thread;
          float* qk_data = buf_ptr;
          float* qk_max_data = qk_data + q_split_size * kv_split_size;
          float* qk_sum_data = qk_max_data + q_split_size;
          float* dst_data = qk_sum_data + q_split_size;
          T* qk_reduced_data =
              is_reduced_type
                  ? buf_reduced_data + omp_idx * q_split_size * kv_split_size
                  : nullptr;

          for (int x = begin; x < end; ++x) {
            int64_t m = k * q_split_size;
            int64_t q_block_size = std::min(q_split_size, q_seq_len - m);
            // Initialize max and sum
            Fvec qk_max_vec(qk_max_data, q_block_size);
            Fvec qk_sum_vec(qk_sum_data, q_block_size);
            qk_max_vec.setConstant(-std::numeric_limits<float>::infinity());
            qk_sum_vec.setConstant(0.f);

            int64_t causal_q_start = k_seq_len - q_seq_len + m;
            int64_t num_keys =
                use_causal ? std::min(causal_q_start + q_block_size, k_seq_len)
                           : k_seq_len;

            for (int64_t n = 0; n < num_keys; n += kv_split_size) {
              int64_t kv_block_size = std::min(kv_split_size, num_keys - n);
              // Calculate scale * q @ k.T
              // And the output is float type.
              cpublas::gemm(
                  'N', 'T', q_block_size, kv_block_size, head_size,
                  scaling_factor,
                  q_data + i * q_stride_b + j * q_stride_h + m * q_stride_m,
                  q_stride_m,
                  k_data + i * k_stride_b + j * k_stride_h + n * k_stride_n,
                  k_stride_n, 0.f, qk_data, kv_block_size);

              // Apply causal mask, fill unused with -inf.
              // Suppose there is a lower_triangle_mask whose lower triangle is
              // all 1, cut a portion of it like lower_triangle_mask[:, :,
              // key_length - query_length : key_length, :key_length], this
              // portion is then applied to the qk result as causal mask.
              if (use_causal && n + kv_block_size > causal_q_start + 1) {
                int64_t row_end = std::min(
                    n + kv_block_size - causal_q_start - 1, q_block_size);
                for (int row = 0; row < row_end; ++row) {
                  int64_t last_col = causal_q_start + 1 + row;
                  float* row_ptr = qk_data + row * kv_block_size;
                  Fvec causal_mask_vec(row_ptr + last_col,
                                       n + kv_block_size - last_col);
                  causal_mask_vec.setConstant(
                      -std::numeric_limits<float>::infinity());
                }
              }

              if (use_mask) {
                T* atten_mask_data =
                    const_cast<Tensor&>(atten_mask).flat<T>().data();
                int64_t mask_batch_size = atten_mask.dim_size(0);
                int64_t mask_num_heads = atten_mask.dim_size(1);
                int64_t mask_q_size = atten_mask.dim_size(2);
                for (int row = 0; row < q_block_size; ++row) {
                  int b_index = mask_batch_size > 1 ? i : 0;
                  int h_index = mask_num_heads > 1 ? j : 0;
                  int q_index = mask_q_size > 1 ? (m + row) : 0;
                  T* mask_start_data =
                      atten_mask_data +
                      b_index * mask_num_heads * mask_q_size * k_seq_len +
                      h_index * mask_q_size * k_seq_len + q_index * k_seq_len +
                      n;
                  Fvec qk_vec(qk_data + row * kv_block_size, kv_block_size);
                  Tvec mask_vec(mask_start_data, kv_block_size);
                  qk_vec += mask_vec.template cast<float>();
                }
              }

              // Below is applying the softmax.
              float tmp_max = 0.f, tmp_sum = 0.f, sum_old = 0.f, exp_tmp = 0.f;

              for (int row = 0; row < q_block_size; ++row) {
                sum_old = qk_sum_data[row];

                Fvec qk_max_vec(qk_data + row * kv_block_size, kv_block_size);
                Eigen::Tensor<float, 0, Eigen::RowMajor> max_t =
                    qk_max_vec.maximum();
                tmp_max = max_t(0);

                tmp_max =
                    qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;

                Fvec pij_vec(qk_data + row * kv_block_size, kv_block_size);
                // P_ij = exp(S_ij - m_inew)
                pij_vec = (pij_vec - tmp_max).exp();
                Eigen::Tensor<float, 0, Eigen::RowMajor> sum_t = pij_vec.sum();
                tmp_sum = sum_t(0);

                // exp_tmp = exp(m_i - m_inew)
                exp_tmp = std::exp(qk_max_data[row] - tmp_max);
                // l_inew = exp_tmp * l_i + l_icur
                qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
                qk_max_data[row] = tmp_max;

                pij_vec = pij_vec / qk_sum_data[row];
                if (is_reduced_type) {
                  Tvec reduced_vec(qk_reduced_data + row * kv_block_size,
                                   kv_block_size);
                  reduced_vec = pij_vec.cast<T>();
                }
                // dst <- dst * sum_old / sum_new * exp_tmp
                if (n > 0) {
                  Fvec origin_out_vec(dst_data + row * head_size, head_size);
                  Fvec update_out_vec(dst_data + row * head_size, head_size);
                  float sum_cor = sum_old / qk_sum_data[row];
                  update_out_vec = origin_out_vec * sum_cor * exp_tmp;
                }
              }
              // Calculate Softmax(q @ k.T) @ v
              cpublas::gemm(
                  'N', 'N', q_block_size, head_size, kv_block_size, 1.0f,
                  conditional_data_ptr(qk_data, qk_reduced_data), kv_block_size,
                  v_data + i * v_stride_b + j * v_stride_h + n * v_stride_n,
                  v_stride_n, n == 0 ? 0.0f : 1.0f, dst_data, head_size);
            }
            T* out_cur_data =
                out_data + i * o_stride_b + j * o_stride_h + m * o_stride_m;
            for (int64_t i = 0; i < q_block_size; ++i) {
              Tvec out_vec(out_cur_data + i * o_stride_m, head_size);
              Fvec dst_vec(dst_data + i * head_size, head_size);
              out_vec = dst_vec.cast<T>();
            }
            // Move to the next query
            DataIndexStep(&i, batch_size, &j, num_heads, &k, q_slice);
          }
        });
  }
};
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_CPU_MHA_OP_H_
