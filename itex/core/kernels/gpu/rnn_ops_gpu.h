/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_RNN_OPS_GPU_H_
#define ITEX_CORE_KERNELS_GPU_RNN_OPS_GPU_H_

#include "itex/core/utils/gpu_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace internal {

// ------------------------------------------------------------------
// UTILS

template <typename T>
using LocalAcc = sycl::accessor<T, 2, sycl::access::mode::read_write,
                                sycl::access::target::local>;

template <typename T>
struct TanhFunc {
  T operator()(T x) const { return Eigen::numext::tanh(x); }
};

template <typename T>
struct ReluFunc {
  T operator()(T x) const { return Eigen::numext::tanh(x); }
};

template <typename T>
T sigmoid(T in) {
  T one = static_cast<T>(1.0);
  return one / (one + Eigen::numext::exp(-in));
}

template <typename T>
inline T square(const T& a) {
  return a * a;
}

// ------------------------------------------------------------------
// ApplyMask kernel
// ------------------------------------------------------------------

template <typename T>
struct ApplyMaskKernel {
  enum { DIM = 4 };
  ApplyMaskKernel(const T* input_, const T* mask_, T* out_,
                  const int num_gates_, const int max_seq_length_,
                  const int batch_size_, const int output_size_)
      : input(input_),
        mask(mask_),
        out(out_),
        num_gates(num_gates_),
        max_seq_length(max_seq_length_),
        batch_size(batch_size_),
        output_size(output_size_) {
    std::array<int, DIM> dim_size{max_seq_length, num_gates, batch_size,
                                  output_size};
    strides[DIM - 1] = 1;
    for (int i = DIM - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dim_size[i + 1];
    }
  }
  inline void operator()(sycl::nd_item<1> item) const {
    int num_elems = max_seq_length * num_gates * batch_size * output_size;
    int id = item.get_global_linear_id();
    if (id >= num_elems) return;
    int indexs[DIM];
    int offset = id;
    for (int i = 0; i < DIM; ++i) {
      int index = offset / strides[i];
      indexs[i] = index;
      offset -= index * strides[i];
    }

    int input_offset = indexs[0] * batch_size * output_size +
                       indexs[2] * output_size + indexs[3];
    int mask_offset = indexs[1] * batch_size * output_size +
                      indexs[2] * output_size + indexs[3];
    out[id] = input[input_offset] * (mask[mask_offset]);
  }

  const T* input;
  const T* mask;
  T* out;
  const int num_gates;
  const int max_seq_length;
  const int batch_size;
  const int output_size;
  std::array<int, DIM> strides;
};

template <typename T>
void ApplyMask(const GPUDevice& d, const T* input, const T* mask, T* out,
               const int num_gates, const int max_seq_length,
               const int batch_size, const int output_size) {
  // input: (max_seq_length, batch_size, output_size)
  // mask: (num_gates, batch_size, output_size)
  // out: (max_seq_length, num_gates, batch_size, output_size)
  auto stream = d.stream();
  auto group_size = (*stream)
                        .get_device()
                        .get_info<sycl::info::device::max_work_group_size>();

  int num_elems = max_seq_length * num_gates * batch_size * output_size;
  int num_group = (num_elems + group_size - 1) / group_size;

  sycl::nd_range<1> range(num_group * group_size, group_size);

  stream->submit([&](sycl::handler& cgh) {
    ApplyMaskKernel<T> task(input, mask, out, num_gates, max_seq_length,
                            batch_size, output_size);
    cgh.parallel_for<ApplyMaskKernel<T>>(range, task);
  });
}

// ------------------------------------------------------------------
template <typename T, bool UseMask>
struct ApplyMaskThenReduceKernel {
  ApplyMaskThenReduceKernel(const T* input_, const T* mask_, T* output_,
                            const int extend_x_, const int extend_y_,
                            const int extend_z_)
      : input(input_),
        mask(mask_),
        output(output_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_) {}
  inline void operator()(sycl::nd_item<1> item) const {
    int num_elems = extend_x * extend_z;
    int id = item.get_global_linear_id();
    if (id >= num_elems) return;

    int x = id / extend_z;
    int z = id - x * extend_z;

    T res = T(0);
    for (int y = 0; y < extend_y; ++y) {
      int offset1 = x * extend_y * extend_z + y * extend_z + z;
      int offset2 = y * extend_z + z;
      if (UseMask)
        res += input[offset1] * mask[offset2];
      else
        res += input[offset1];
    }
    output[id] = res;
  }
  const T* input;
  const T* mask;
  T* output;
  const int extend_x;
  const int extend_y;
  const int extend_z;
};

template <typename T, bool UseMask>
void ApplyMaskThenReduce(const GPUDevice& d, const T* input, const T* mask,
                         T* output, const int extend_x, const int extend_y,
                         const int extend_z) {
  auto stream = d.stream();
  auto group_size = (*stream)
                        .get_device()
                        .get_info<sycl::info::device::max_work_group_size>();

  int num_elems = extend_x * extend_z;
  int num_group = (num_elems + group_size - 1) / group_size;

  sycl::range<1> global(num_group * group_size);
  sycl::range<1> local(group_size);

  stream->submit([&](sycl::handler& cgh) {
    ApplyMaskThenReduceKernel<T, UseMask> task(input, mask, output, extend_x,
                                               extend_y, extend_z);
    cgh.parallel_for<ApplyMaskThenReduceKernel<T, UseMask>>(
        sycl::nd_range<1>(global, local), task);
  });
}

// ------------------------------------------------------------------

template <typename T, int cP, int cM, int BSX, int BSY, int TILE_K, int TAB,
          bool TransA, bool TransB, bool BcastA>
struct InputGemmKernel {
  InputGemmKernel(const T* A_, const T* B_, T* C_, const int timesteps_,
                  const int bs_, const int M_, const int N_, const int P_,
                  LocalAcc<T> Asub_, LocalAcc<T> Bsub_)
      : A(A_),
        B(B_),
        C(C_),
        timesteps(timesteps_),
        bs(bs_),
        M(M_),
        N(N_),
        P(P_),
        Asub(Asub_),
        Bsub(Bsub_) {}

  inline void operator()(sycl::nd_item<3> block) const {
    int i = cM * block.get_global_id(1);  // col
    int j = cP * block.get_global_id(2);  // row

    int item_i = block.get_local_id(1);  // col
    int item_j = block.get_local_id(2);  // row

    int group_i = block.get_group(1);  // col
    int group_j = block.get_group(2);  // row
    int group_z = block.get_group(0);
    int group_z_t = group_z / bs;
    int group_z_b = group_z - group_z_t * bs;

    // register tile (A, B, C) for better performancee
    T a_reg[TAB][cM];
    T b_reg[TAB][cP];
    T sum[cM][cP] = {{T(0)}};

    int stride_a;
    if (BcastA) {
      stride_a = group_z_t * M * N;
    } else {
      stride_a = group_z_t * bs * M * N + group_z_b * M * N;
    }

    int stride_b = group_z_b * N * P;

    const T* A_per_batch = A + stride_a;
    const T* B_per_batch = B + stride_b;
    T* C_per_batch = C + group_z * M * P;

    // reduction in K direction
    for (int k = 0; k < N; k += TILE_K) {
      // Load matrix A into local memory
#pragma unroll
      for (int k1 = 0; k1 < DivUp(TILE_K, BSY); k1++) {
#pragma unroll
        for (int m1 = 0; m1 < cM; m1++) {
          int tile_k_x = k1 * BSY + item_j;
          int tile_k_y = m1 * BSX + item_i;
          int global_y = tile_k_y + group_i * cM * BSX;
          int global_x = k + tile_k_x;
          if (global_y < M && global_x < N) {
            if (!TransA)
              Asub[tile_k_y][tile_k_x] = A_per_batch[global_y * N + global_x];
            else
              Asub[tile_k_y][tile_k_x] = A_per_batch[global_x * M + global_y];
          } else {
            Asub[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

      // Load matrix B into local memory
#pragma unroll
      for (int k1 = 0; k1 < DivUp(TILE_K, BSX); k1++) {
#pragma unroll
        for (int p1 = 0; p1 < cP; p1++) {
          int tile_k_x = p1 * BSY + item_j;
          int tile_k_y = k1 * BSX + item_i;
          int global_y = k + tile_k_y;
          int global_x = group_j * cP * BSY + tile_k_x;
          if (global_y < N && global_x < P) {
            if (!TransB)
              Bsub[tile_k_y][tile_k_x] = B_per_batch[global_y * P + global_x];
            else
              Bsub[tile_k_y][tile_k_x] = B_per_batch[global_x * N + global_y];
          } else {
            Bsub[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

      // wait all memory has been stored to Asub & Bsub
      block.barrier(sycl::access::fence_space::local_space);

      // Per thread consideration in core computation part
      for (int k1 = 0; k1 < TILE_K; k1 += TAB) {
        // load SLM to registers
#pragma unroll
        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int m = 0; m < cM; m++) {
            // a_reg is a transposed sahpe for better memory acess
            a_reg[tab][m] = Asub[item_i * cM + m][k1 + tab];
          }
        }

        // load SLM to registers
#pragma unroll
        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int p = 0; p < cP; p++) {
            b_reg[tab][p] = Bsub[k1 + tab][item_j * cP + p];
          }
        }

        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int m = 0; m < cM; m++) {
#pragma unroll
            for (int p = 0; p < cP; p++) {
              // does broadcast need?
              sum[m][p] += a_reg[tab][m] * b_reg[tab][p];
            }
          }
        }  // end of tab
      }

      // wait all memory has been consumed
      block.barrier(sycl::access::fence_space::local_space);
    }

    // write results back
#pragma unroll
    for (int m = 0; m < cM; m++) {
#pragma unroll
      for (int p = 0; p < cP; p++) {
        int global_i = i + m;
        int global_j = j + p;
        if (global_i < M && global_j < P) {
          C_per_batch[global_i * P + global_j] = sum[m][p];
        }
      }
    }
  }
  const T* A;
  const T* B;
  T* C;
  const int timesteps;
  const int bs;
  const int M;
  const int N;
  const int P;
  LocalAcc<T> Asub;
  LocalAcc<T> Bsub;
};

template <typename T, int cP, int cM, int BSX, int BSY, int TILE_K, int TAB,
          bool TransA, bool TransB, bool BcastA>
void LaunchInputGemm(const GPUDevice& d, const T* A, const T* B, T* C,
                     const int timesteps, const int bs, const int M,
                     const int N, const int P) {
  auto stream = d.stream();
  const int MN = DivUp(M, cM);
  const int PN = DivUp(P, cP);
  sycl::range<3> local{1, BSX /* col */, BSY /* row */};
  sycl::range<3> global{static_cast<size_t>(timesteps * bs),
                        static_cast<size_t>(RoundUp(MN, BSX)),
                        static_cast<size_t>(RoundUp(PN, BSY))};

  stream->submit([&](sycl::handler& h) {
    /* Memory Management */
    LocalAcc<T> Asub(sycl::range<2>{cM * BSX, TILE_K}, h);
    LocalAcc<T> Bsub(sycl::range<2>{TILE_K, cP * BSY}, h);
    InputGemmKernel<T, cP, cM, BSX, BSY, TILE_K, TAB, TransA, TransB, BcastA>
        task(A, B, C, timesteps, bs, M, N, P, Asub, Bsub);
    h.parallel_for<InputGemmKernel<T, cP, cM, BSX, BSY, TILE_K, TAB, TransA,
                                   TransB, BcastA>>(
        sycl::nd_range<3>(global, local), task);
  });
}

// A: timesteps * bs * m * n  if no broadcast, else timesteps * m * n
// B: bs * n * p
template <typename T, bool TransA, bool TransB, bool BcastA>
void InputGemm(const GPUDevice& d, const T* A, const T* B, T* C,
               const int timesteps, const int bs, const int M, const int N,
               const int P) {
  enum { cM = 1, cP = 1, BSX = 16, BSY = 16, TILE_K = 16, TAB = 2 };

  return LaunchInputGemm<T, cM, cP, BSX, BSY, TILE_K, TAB, TransA, TransB,
                         BcastA>(d, A, B, C, timesteps, bs, M, N, P);
}

// ------------------------------------------------------------------
// Gemm kernel

template <typename T, int cP, int cM, int BSX, int BSY, int TILE_K, int TAB,
          bool TransA, bool TransB, bool BcastA, bool BcastB>
struct GemmKernel {
  GemmKernel(const T* A_, const T* B_, T* C_, const int bs_, const int M_,
             const int N_, const int P_, LocalAcc<T> Asub_, LocalAcc<T> Bsub_)
      : A(A_),
        B(B_),
        C(C_),
        bs(bs_),
        M(M_),
        N(N_),
        P(P_),
        Asub(Asub_),
        Bsub(Bsub_) {}
  inline void operator()(sycl::nd_item<3> item) const {
    int i = cM * item.get_global_id(1);  // col
    int j = cP * item.get_global_id(2);  // row

    int item_i = item.get_local_id(1);  // col
    int item_j = item.get_local_id(2);  // row

    int group_i = item.get_group(1);  // col
    int group_j = item.get_group(2);  // row
    int group_z = item.get_group(0);

    // register tile (A, B, C) for better performancee

    T a_reg[TAB][cM];
    T b_reg[TAB][cP];
    T sum[cM][cP] = {{T(0)}};

    int stride_a = !BcastA * M * N;
    int stride_b = !BcastB * N * P;

    const T* A_per_batch = A + group_z * stride_a;
    const T* B_per_batch = B + group_z * stride_b;
    T* C_per_batch = C + group_z * M * P;

    // reduction in K direction
    for (int k = 0; k < N; k += TILE_K) {
      // Load matrix A into local memory
#pragma unroll
      for (int k1 = 0; k1 < DivUp(TILE_K, BSY); k1++) {
#pragma unroll
        for (int m1 = 0; m1 < cM; m1++) {
          int tile_k_x = k1 * BSY + item_j;
          int tile_k_y = m1 * BSX + item_i;
          int global_y = tile_k_y + group_i * cM * BSX;
          int global_x = k + tile_k_x;
          if (global_y < M && global_x < N) {
            if (!TransA)
              Asub[tile_k_y][tile_k_x] = A_per_batch[global_y * N + global_x];
            else
              Asub[tile_k_y][tile_k_x] = A_per_batch[global_x * M + global_y];
          } else {
            Asub[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

      // Load matrix B into local memory
#pragma unroll
      for (int k1 = 0; k1 < DivUp(TILE_K, BSX); k1++) {
#pragma unroll
        for (int p1 = 0; p1 < cP; p1++) {
          int tile_k_x = p1 * BSY + item_j;
          int tile_k_y = k1 * BSX + item_i;
          int global_y = k + tile_k_y;
          int global_x = group_j * cP * BSY + tile_k_x;
          if (global_y < N && global_x < P) {
            if (!TransB)
              Bsub[tile_k_y][tile_k_x] = B_per_batch[global_y * P + global_x];
            else
              Bsub[tile_k_y][tile_k_x] = B_per_batch[global_x * N + global_y];
          } else {
            Bsub[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

      // wait all memory has been stored to Asub & Bsub
      item.barrier(sycl::access::fence_space::local_space);

      // Per thread consideration in core computation part
      for (int k1 = 0; k1 < TILE_K; k1 += TAB) {
        // load SLM to registers
#pragma unroll
        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int m = 0; m < cM; m++) {
            // a_reg is a transposed sahpe for better memory acess
            a_reg[tab][m] = Asub[item_i * cM + m][k1 + tab];
          }
        }

        // load SLM to registers
#pragma unroll
        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int p = 0; p < cP; p++) {
            b_reg[tab][p] = Bsub[k1 + tab][item_j * cP + p];
          }
        }

        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int m = 0; m < cM; m++) {
#pragma unroll
            for (int p = 0; p < cP; p++) {
              // does broadcast need?
              sum[m][p] += a_reg[tab][m] * b_reg[tab][p];
            }
          }
        }  // end of tab
      }

      // wait all memory has been consumed
      item.barrier(sycl::access::fence_space::local_space);
    }

    // write results back
#pragma unroll
    for (int m = 0; m < cM; m++) {
#pragma unroll
      for (int p = 0; p < cP; p++) {
        int global_i = i + m;
        int global_j = j + p;
        if (global_i < M && global_j < P) {
          C_per_batch[global_i * P + global_j] = sum[m][p];
        }
      }
    }
  }
  const T* A;
  const T* B;
  T* C;
  const int bs;
  const int M;
  const int N;
  const int P;
  LocalAcc<T> Asub;
  LocalAcc<T> Bsub;
};

template <typename T, int cP, int cM, int BSX, int BSY, int TILE_K, int TAB,
          bool TransA, bool TransB, bool BcastA, bool BcastB>
void LaunchGemmKernel(const GPUDevice& d, const T* A, const T* B, T* C,
                      const int bs, const int M, const int N, const int P) {
  auto stream = d.stream();

  const int MN = DivUp(M, cM);
  const int PN = DivUp(P, cP);
  sycl::range<3> local{1, BSX /* col */, BSY /* row */};
  sycl::range<3> global{static_cast<size_t>(bs),
                        static_cast<size_t>(RoundUp(MN, BSX)),
                        static_cast<size_t>(RoundUp(PN, BSY))};

  stream->submit([&](sycl::handler& cgh) {
    // Memory Management
    LocalAcc<T> Asub(sycl::range<2>{cM * BSX, TILE_K}, cgh);
    LocalAcc<T> Bsub(sycl::range<2>{TILE_K, cP * BSY}, cgh);
    GemmKernel<T, cP, cM, BSX, BSY, TILE_K, TAB, TransA, TransB, BcastA, BcastB>
        task(A, B, C, bs, M, N, P, Asub, Bsub);
    cgh.parallel_for<GemmKernel<T, cP, cM, BSX, BSY, TILE_K, TAB, TransA,
                                TransB, BcastA, BcastB>>(
        sycl::nd_range<3>(global, local), task);  // end of parallel_for
  });
}

// Genernal GEMM operation for RNN
// A: (bs * M, N)
// B: (bs * N, P)
template <typename T, bool TransA, bool TransB, bool BcastA, bool BcastB>
void LstmGemm(const GPUDevice& d, const T* A, const T* B, T* C, const int bs,
              const int M, const int N, const int P) {
  enum { cM = 1, cP = 1, BSX = 16, BSY = 16, TILE_K = 16, TAB = 2 };
  LaunchGemmKernel<T, cM, cP, BSX, BSY, TILE_K, TAB, TransA, TransB, BcastA,
                   BcastB>(d, A, B, C, bs, M, N, P);
}

// ------------------------------------------------------------------
// LstmGradEltwise kernel
template <typename T, int cP, int cM, int BSX, int BSY, int TILE_K, int TAB,
          bool TransA, bool TransB, int BcastB>
struct ParamsGemmKernel {
  ParamsGemmKernel(const T* A_, const T* B_, T* C_, const int timesteps_,
                   const int bs_, const int M_, const int N_, const int P_,
                   LocalAcc<T> Asub_, LocalAcc<T> Bsub_)
      : A(A_),
        B(B_),
        C(C_),
        timesteps(timesteps_),
        bs(bs_),
        M(M_),
        N(N_),
        P(P_),
        Asub(Asub_),
        Bsub(Bsub_) {}
  inline void operator()(sycl::nd_item<3> block) const {
    int i = cM * block.get_global_id(1);  // col
    int j = cP * block.get_global_id(2);  // row

    int item_i = block.get_local_id(1);  // col
    int item_j = block.get_local_id(2);  // row

    int group_i = block.get_group(1);  // col
    int group_j = block.get_group(2);  // row
    int group_z = block.get_group(0);
    int group_z_t = group_z / bs;
    int group_z_b = group_z - group_z_t * bs;

    // register tile (A, B, C) for better performancee

    T a_reg[TAB][cM];
    T b_reg[TAB][cP];
    T sum[cM][cP] = {{T(0)}};

    int stride_a = group_z_t * bs * M * N + group_z_b * M * N;
    int stride_b;
    if (BcastB) {
      stride_b = group_z_t * N * P;
    } else {
      stride_b = group_z_t * bs * N * P + group_z_b * N * P;
    }

    const T* A_per_batch = A + stride_a;
    const T* B_per_batch = B + stride_b;
    T* C_per_batch = C + group_z * M * P;

    // reduction in K direction
    for (int k = 0; k < N; k += TILE_K) {
      // Load matrix A into local memory
#pragma unroll
      for (int k1 = 0; k1 < DivUp(TILE_K, BSY); k1++) {
#pragma unroll
        for (int m1 = 0; m1 < cM; m1++) {
          int tile_k_x = k1 * BSY + item_j;
          int tile_k_y = m1 * BSX + item_i;
          int global_y = tile_k_y + group_i * cM * BSX;
          int global_x = k + tile_k_x;
          if (global_y < M && global_x < N) {
            if (!TransA)
              Asub[tile_k_y][tile_k_x] = A_per_batch[global_y * N + global_x];
            else
              Asub[tile_k_y][tile_k_x] = A_per_batch[global_x * M + global_y];
          } else {
            Asub[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

      // Load matrix B into local memory
#pragma unroll
      for (int k1 = 0; k1 < DivUp(TILE_K, BSX); k1++) {
#pragma unroll
        for (int p1 = 0; p1 < cP; p1++) {
          int tile_k_x = p1 * BSY + item_j;
          int tile_k_y = k1 * BSX + item_i;
          int global_y = k + tile_k_y;
          int global_x = group_j * cP * BSY + tile_k_x;
          if (global_y < N && global_x < P) {
            if (!TransB)
              Bsub[tile_k_y][tile_k_x] = B_per_batch[global_y * P + global_x];
            else
              Bsub[tile_k_y][tile_k_x] = B_per_batch[global_x * N + global_y];
          } else {
            Bsub[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

      // wait all memory has been stored to Asub & Bsub
      block.barrier(sycl::access::fence_space::local_space);

      // Per thread consideration in core computation part
      for (int k1 = 0; k1 < TILE_K; k1 += TAB) {
        // load SLM to registers
#pragma unroll
        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int m = 0; m < cM; m++) {
            // a_reg is a transposed sahpe for better memory acess
            a_reg[tab][m] = Asub[item_i * cM + m][k1 + tab];
          }
        }

        // load SLM to registers
#pragma unroll
        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int p = 0; p < cP; p++) {
            b_reg[tab][p] = Bsub[k1 + tab][item_j * cP + p];
          }
        }

        for (int tab = 0; tab < TAB; tab++) {
#pragma unroll
          for (int m = 0; m < cM; m++) {
#pragma unroll
            for (int p = 0; p < cP; p++) {
              // does broadcast need?
              sum[m][p] += a_reg[tab][m] * b_reg[tab][p];
            }
          }
        }  // end of tab
      }

      // wait all memory has been consumed
      block.barrier(sycl::access::fence_space::local_space);
    }

    // write results back
#pragma unroll
    for (int m = 0; m < cM; m++) {
#pragma unroll
      for (int p = 0; p < cP; p++) {
        int global_i = i + m;
        int global_j = j + p;
        if (global_i < M && global_j < P) {
          C_per_batch[global_i * P + global_j] = sum[m][p];
        }
      }
    }
  }
  const T* A;
  const T* B;
  T* C;
  const int timesteps;
  const int bs;
  const int M;
  const int N;
  const int P;
  LocalAcc<T> Asub;
  LocalAcc<T> Bsub;
};

template <typename T, int cP, int cM, int BSX, int BSY, int TILE_K, int TAB,
          bool TransA, bool TransB, int BcastB>
void LaunchParamsGemm(const GPUDevice& d, const T* A, const T* B, T* C,
                      const int timesteps, const int bs, const int M,
                      const int N, const int P) {
  const int MN = DivUp(M, cM);
  const int PN = DivUp(P, cP);
  sycl::range<3> local{1, BSX /* col */, BSY /* row */};
  sycl::range<3> global{static_cast<size_t>(timesteps * bs),
                        static_cast<size_t>(RoundUp(MN, BSX)),
                        static_cast<size_t>(RoundUp(PN, BSY))};
  auto stream = d.stream();

  stream->submit([&](sycl::handler& h) {
    /* Memory Management */
    LocalAcc<T> Asub(sycl::range<2>{cM * BSX, TILE_K}, h);
    LocalAcc<T> Bsub(sycl::range<2>{TILE_K, cP * BSY}, h);
    ParamsGemmKernel<T, cP, cM, BSX, BSY, TILE_K, TAB, TransA, TransB, BcastB>
        task(A, B, C, timesteps, bs, M, N, P, Asub, Bsub);
    h.parallel_for<ParamsGemmKernel<T, cP, cM, BSX, BSY, TILE_K, TAB, TransA,
                                    TransB, BcastB>>(
        sycl::nd_range<3>(global, local), task);  // end of parallel_for
  });
}

// A: timesteps * bs * m * n  if no broadcast, else timesteps * m * n
// B: bs * n * p
template <typename T, bool TransA, bool TransB, const bool BcastB>
void ParamsGemm(const GPUDevice& d, const T* A, const T* B, T* C,
                const int timesteps, const int bs, const int M, const int N,
                const int P) {
  enum { cM = 1, cP = 1, BSX = 16, BSY = 16, TILE_K = 16, TAB = 2 };

  return LaunchParamsGemm<T, cM, cP, BSX, BSY, TILE_K, TAB, TransA, TransB,
                          BcastB>(d, A, B, C, timesteps, bs, M, N, P);
}

// ------------------------------------------------------------------
// LstmEltwise kernel
template <typename T, bool IsTraining>
struct LstmEltwiseKernel {
  LstmEltwiseKernel(const T* igates_, const T* hgates_, const T* bias_,
                    const T* c_prev_, T* h_next_, T* c_next_, T* gates_,
                    const int batch_size_, const int output_size_)
      : igates(igates_),
        hgates(hgates_),
        bias(bias_),
        c_prev(c_prev_),
        h_next(h_next_),
        c_next(c_next_),
        gates(gates_),
        batch_size(batch_size_),
        output_size(output_size_) {}
  inline void operator()(sycl::nd_item<1> item) const {
    int num_elems = batch_size * output_size;
    int id = item.get_global_linear_id();
    if (id >= num_elems) return;

    int bs = id / output_size;
    int col = id - bs * output_size;

    const T ii_gates = igates[id];
    const T hi_gates = hgates[id];
    const T i_bias = bias[col];
    const T it = sigmoid(ii_gates + hi_gates + i_bias);

    const T if_gates = igates[id + num_elems];
    const T hf_gates = hgates[id + num_elems];
    const T f_bias = bias[col + output_size];
    const T ft = sigmoid(if_gates + hf_gates + f_bias);

    const T ic_gates = igates[id + num_elems * 2];
    const T hc_gates = hgates[id + num_elems * 2];
    const T c_bias = bias[col + output_size * 2];
    const T ct = Eigen::numext::tanh(ic_gates + hc_gates + c_bias);

    const T io_gates = igates[id + num_elems * 3];
    const T ho_gates = hgates[id + num_elems * 3];
    const T o_bias = bias[col + output_size * 3];
    const T ot = sigmoid(io_gates + ho_gates + o_bias);

    const T c_nt = ft * c_prev[id] + it * ct;
    const T h_nt = ot * Eigen::numext::tanh(c_nt);

    c_next[id] = c_nt;
    h_next[id] = h_nt;

    if (IsTraining) {
      gates[id] = it;
      gates[id + num_elems] = ft;
      gates[id + num_elems * 2] = ct;
      gates[id + num_elems * 3] = ot;
    }
  }

  const T* igates;
  const T* hgates;
  const T* bias;
  const T* c_prev;
  T* h_next;
  T* c_next;
  T* gates;
  const int batch_size;
  const int output_size;
};

template <typename T, bool IsTraining>
void LstmEltwise(const GPUDevice& d, const T* igates, const T* hgates,
                 const T* bias, const T* c_prev, T* h_next, T* c_next, T* gates,
                 const int batch_size, const int output_size) {
  auto stream = d.stream();
  int group_size = (*stream)
                       .get_device()
                       .get_info<sycl::info::device::max_work_group_size>();

  int num_elems = batch_size * output_size;
  int num_group = DivUp(num_elems, group_size);

  sycl::range<1> global(num_group * group_size);
  sycl::range<1> local(group_size);

  stream->submit([&](sycl::handler& cgh) {
    LstmEltwiseKernel<T, IsTraining> task(igates, hgates, bias, c_prev, h_next,
                                          c_next, gates, batch_size,
                                          output_size);
    cgh.parallel_for<LstmEltwiseKernel<T, IsTraining>>(
        sycl::nd_range<1>(global, local), task);
  });
}

// ------------------------------------------------------------------
// LstmGradEltwise kernel

template <typename T>
struct LstmGradEltwiseKernel {
  LstmGradEltwiseKernel(const T* c_prev_, const T* c_next_,
                        const T* output_grad_, const T* h_prev_grad_,
                        const T* c_next_grad_, const T* gates_, T* c_prev_grad_,
                        T* gates_grad_, const int batch_size_,
                        const int output_size_)
      : c_prev(c_prev_),
        c_next(c_next_),
        output_grad(output_grad_),
        h_prev_grad(h_prev_grad_),
        c_next_grad(c_next_grad_),
        gates(gates_),
        c_prev_grad(c_prev_grad_),
        gates_grad(gates_grad_),
        batch_size(batch_size_),
        output_size(output_size_) {}
  inline void operator()(sycl::nd_item<1> item) const {
    int num_elems = batch_size * output_size;
    int id = item.get_global_linear_id();
    if (id >= num_elems) return;

    const T dh_next = h_prev_grad[id] + output_grad[id];

    const T it = gates[id];
    const T ft = gates[id + num_elems];
    const T gt = gates[id + num_elems * 2];
    const T ot = gates[id + num_elems * 3];

    const T tanh_c_next = Eigen::numext::tanh(c_next[id]);
    const T dc_next =
        c_next_grad[id] + dh_next * ot * (T(1) - square(tanh_c_next));
    // dc_prev = dct * ft
    const T dc_prev = dc_next * ft;

    // dot = dst_diff_h * tanh(c_next) * ot * (1 - ot)
    const T dot = dh_next * tanh_c_next * ot * (T(1) - ot);
    // dft = dct * c_prev * ft * (1 - ft)
    const T dft = dc_next * c_prev[id] * ft * (T(1) - ft);
    //  dit = dct * gt * it * (1 - it)
    const T dit = dc_next * gt * it * (T(1) - it);
    // dgt = dct * it * (1 - np.square(gt))
    const T dgt = dc_next * it * (T(1) - square(gt));

    c_prev_grad[id] = dc_prev;
    gates_grad[id] = dit;
    gates_grad[id + num_elems] = dft;
    gates_grad[id + num_elems * 2] = dgt;
    gates_grad[id + num_elems * 3] = dot;
  }

  const T* c_prev;
  const T* c_next;
  const T* output_grad;
  const T* h_prev_grad;
  const T* c_next_grad;
  const T* gates;
  T* c_prev_grad;
  T* gates_grad;
  const int batch_size;
  const int output_size;
};

template <typename T>
void LstmGradEltwise(const GPUDevice& d, const T* c_prev, const T* c_next,
                     const T* output_grad, const T* h_prev_grad,
                     const T* c_next_grad, const T* gates, T* c_prev_grad,
                     T* gates_grad, const int batch_size,
                     const int output_size) {
  auto stream = d.stream();
  int group_size = (*stream)
                       .get_device()
                       .get_info<sycl::info::device::max_work_group_size>();
  int num_elems = batch_size * output_size;
  int num_group = DivUp(num_elems, group_size);

  stream->submit([&](sycl::handler& cgh) {
    LstmGradEltwiseKernel<T> task(c_prev, c_next, output_grad, h_prev_grad,
                                  c_next_grad, gates, c_prev_grad, gates_grad,
                                  batch_size, output_size);
    cgh.parallel_for<LstmGradEltwiseKernel<T>>(
        sycl::nd_range<1>(num_group * group_size, group_size), task);
  });
}

}  // namespace internal
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_RNN_OPS_GPU_H_
