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

#ifndef ITEX_CORE_KERNELS_CPU_CPU_BLAS_H_
#define ITEX_CORE_KERNELS_CPU_CPU_BLAS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace cpublas {

void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
          float alpha, float* a, int64_t lda, float* b, int64_t ldb, float beta,
          float* c, int64_t ldc);

void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
          float alpha, Eigen::bfloat16* a, int64_t lda, Eigen::bfloat16* b,
          int64_t ldb, float beta, float* c, int64_t ldc);
}  // namespace cpublas
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_CPU_CPU_BLAS_H_
