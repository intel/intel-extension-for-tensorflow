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

#include "itex/core/kernels/cpu/cpu_blas.h"

#include "itex/core/utils/onednn/onednn_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace cpublas {

extern "C" {
dnnl_status_t dnnl_gemm_bf16bf16f32(char transa, char transb, dnnl_dim_t M,
                                    dnnl_dim_t N, dnnl_dim_t K, float alpha,
                                    const dnnl_port::bfloat16_t* A,
                                    dnnl_dim_t lda,
                                    const dnnl_port::bfloat16_t* B,
                                    dnnl_dim_t ldb, float beta, float* C,
                                    dnnl_dim_t ldc);
}

void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
          float alpha, float* a, int64_t lda, float* b, int64_t ldb, float beta,
          float* c, int64_t ldc) {
  dnnl_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
          float alpha, Eigen::bfloat16* a, int64_t lda, Eigen::bfloat16* b,
          int64_t ldb, float beta, float* c, int64_t ldc) {
  dnnl_port::bfloat16_t* dnnl_a = reinterpret_cast<dnnl_port::bfloat16_t*>(a);
  dnnl_port::bfloat16_t* dnnl_b = reinterpret_cast<dnnl_port::bfloat16_t*>(b);
  dnnl_gemm_bf16bf16f32(transa, transb, m, n, k, alpha, dnnl_a, lda, dnnl_b,
                        ldb, beta, c, ldc);
}
}  // namespace cpublas
}  // namespace itex
