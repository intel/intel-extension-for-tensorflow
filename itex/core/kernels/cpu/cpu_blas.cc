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

#include "dnnl.hpp"  // NOLINT(build/include_subdir)
#include "itex/core/utils/plugin_tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace cpublas {

void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
          float alpha, float* a, int64_t lda, float* b, int64_t ldb, float beta,
          float* c, int64_t ldc) {
  dnnl_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
          float alpha, Eigen::bfloat16* a, int64_t lda, Eigen::bfloat16* b,
          int64_t ldb, float beta, float* c, int64_t ldc) {
  int64_t dim_0 = m;
  int64_t dim_1 = k;
  int64_t dim_2 = k;
  int64_t dim_3 = n;
  if (transa == 'T') {
    dim_0 = k;
    dim_1 = m;
  }
  if (transb == 'T') {
    dim_2 = n;
    dim_3 = k;
  }
  Tensor a_float(DT_FLOAT, {dim_0, dim_1});
  Tensor b_float(DT_FLOAT, {dim_2, dim_3});
  float* a_float_data = a_float.flat<float>().data();
  float* b_float_data = b_float.flat<float>().data();
  for (int i = 0; i < dim_0; ++i) {
    typename TTypes<float>::Flat a_float_flat(a_float_data + i * dim_1, dim_1);
    typename TTypes<Eigen::bfloat16>::Flat a_flat(a + i * lda, dim_1);
    a_float_flat = a_flat.cast<float>();
  }
  for (int j = 0; j < dim_2; ++j) {
    typename TTypes<float>::Flat b_float_flat(b_float_data + j * dim_3, dim_3);
    typename TTypes<Eigen::bfloat16>::Flat b_flat(b + j * ldb, dim_3);
    b_float_flat = b_flat.cast<float>();
  }
  dnnl_sgemm(transa, transb, m, n, k, alpha, a_float_data, lda, b_float_data,
             ldb, beta, c, ldc);
}
}  // namespace cpublas
}  // namespace itex
