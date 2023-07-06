/* Copyright (c) 2021-2022 Intel Corporation

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

#if ITEX_USE_MKL
#include "itex/core/kernels/gpu/linalg/matrix_triangular_solve_op_impl.h"
#include "itex/core/utils/register_types.h"

namespace itex {
#define REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU(TYPE)             \
  REGISTER_KERNEL_BUILDER(Name("MatrixTriangularSolve")              \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<TYPE>("T"),            \
                          MatrixTriangularSolveOp<GPUDevice, TYPE>); \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixTriangularSolve")         \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<TYPE>("T"),            \
                          MatrixTriangularSolveOp<GPUDevice, TYPE>);

TF_CALL_float(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU);
TF_CALL_complex128(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_complex64(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU);

#undef REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU
}  // namespace itex
#endif  // ITEX_USE_MKL
