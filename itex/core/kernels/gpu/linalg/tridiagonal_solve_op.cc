/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "mkl.h"  // NOLINT(build/include_subdir)
#include "oneapi/mkl/lapack.hpp"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct SparseToDense {
  SparseToDense(int64_t m, int64_t n, T* sparsed_matrix, T* densed_matrix)
      : m(m),
        n(n),
        sparsed_matrix(sparsed_matrix),
        densed_matrix(densed_matrix) {}
  void operator()(sycl::item<2> item) const {
    auto batch_id = item.get_id()[0];
    auto diag_id = item.get_id()[1];
    const int starting_col = 1 - diag_id;

    for (int i = 0; i < n; ++i) {
      int row_densed = i;
      int col_densed = starting_col + i;
      if (row_densed >= 0 && row_densed < n && col_densed >= 0 &&
          col_densed < n) {
        int offset_dense = batch_id * (n * n) + row_densed * n + col_densed;
        int offset_sparse = batch_id * (m * n) + diag_id * n + i;
        densed_matrix[offset_dense] = sparsed_matrix[offset_sparse];
      }
    }
  }

 private:
  int64_t m;
  int64_t n;
  T* sparsed_matrix;
  T* densed_matrix;
};

template <typename Scalar>
void LaunchSparseToDenseKernel(sycl::queue* stream, int64_t batch_size,
                               int64_t m, int64_t n, Scalar* sparsed_matrix,
                               Scalar* densed_matrix) {
  stream->submit([&](sycl::handler& cgh) {
    SparseToDense<Scalar> task(m, n, sparsed_matrix, densed_matrix);
    cgh.parallel_for<SparseToDense<Scalar>>(
        sycl::range<2>{static_cast<size_t>(batch_size), static_cast<size_t>(m)},
        task);
  });
}

template <class Scalar>
class TridiagonalSolveOpGpu : public OpKernel {
 public:
  explicit TridiagonalSolveOpGpu(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("partial_pivoting", &pivoting_));
  }

  void Compute(OpKernelContext* context) final {
    auto* stream = context->GetDeviceStream();
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    const Tensor& lhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const int ndims = lhs.dims();
    const int64_t num_rhs = rhs.dim_size(rhs.dims() - 1);
    const int64_t matrix_size = lhs.dim_size(ndims - 1);
    const int num_diags = lhs.dim_size(ndims - 2);

    OP_REQUIRES(
        context, num_diags == 3,
        errors::InvalidArgument("Expected diagonals to be provided as a "
                                "matrix with 3 rows, got ",
                                num_diags, " rows."));

    int64_t batch_size = 1;
    for (int i = 0; i < ndims - 2; i++) {
      batch_size *= lhs.dim_size(i);
    }

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 0, rhs.shape(), &output));
    DeviceMemcpy<GPUDevice>(output->flat<Scalar>().data(),
                            rhs.flat<Scalar>().data(),
                            rhs.NumElements() * sizeof(Scalar), stream);

    if (lhs.NumElements() == 0) {
      return;
    }

    Tensor densed_input;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                        {batch_size, matrix_size, matrix_size},
                                        &densed_input));
    DeviceFill<GPUDevice, Scalar>(densed_input.flat<Scalar>().data(), Scalar(0),
                                  densed_input.NumElements(), stream);
    LaunchSparseToDenseKernel<Scalar>(
        stream, batch_size, num_diags, matrix_size,
        const_cast<Scalar*>(lhs.flat<Scalar>().data()),
        const_cast<Scalar*>(densed_input.flat<Scalar>().data()));

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<int64_t>::value,
                                TensorShape{batch_size, matrix_size}, &pivots));
    auto pivots_mat = pivots.template matrix<int64_t>();

    int64_t lda = matrix_size, ldb = matrix_size, stride_ipiv = matrix_size;
    int64_t stride_a = lda * matrix_size, stride_b = ldb * num_rhs;

    int64_t info = 0;
    // 1. Compute the partially pivoted LU factorization(s) of the
    // matrix/matrices.
    try {
      int64_t getrf_scratchpad_size =
          oneapi::mkl::lapack::getrf_batch_scratchpad_size<Scalar>(
              *stream, matrix_size, matrix_size, lda, stride_a, stride_ipiv,
              batch_size);
      Tensor getrf_scratchpad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::v(),
                                  {getrf_scratchpad_size}, &getrf_scratchpad));
      oneapi::mkl::lapack::getrf_batch(
          *stream, matrix_size, matrix_size,
          static_cast<Scalar*>(densed_input.data()), lda, stride_a,
          const_cast<int64_t*>(pivots_mat.data()), stride_ipiv, batch_size,
          static_cast<Scalar*>(getrf_scratchpad.data()), getrf_scratchpad_size);
    } catch (oneapi::mkl::lapack::batch_error const& be) {
      int i = 0;
      auto& ids = be.ids();
      for (auto const& e : be.exceptions()) {
        try {
          std::rethrow_exception(e);
        } catch (oneapi::mkl::lapack::exception& e) {
          ITEX_LOG(ERROR) << "Exception " << ids[i++]
                          << " in a batch says: " << e.what()
                          << " (info code: " << e.info() << ")";
          info = e.info();
        }
      }
    }
    OP_REQUIRES(context, info == 0,
                errors::Internal("Unexpected exception caught during "
                                 "call to LAPACK API."));

    // 2. Make a transposed copy of the right-hand sides. This is necessary
    // because OneMKL assumes column-major storage while TensorFlow TF uses
    // row-major.
    TensorShape transposed_rhs_shape(rhs.shape());
    transposed_rhs_shape.RemoveLastDims(2);
    transposed_rhs_shape.AddDim(num_rhs);
    transposed_rhs_shape.AddDim(matrix_size);
    Tensor transposed_rhs;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                        transposed_rhs_shape, &transposed_rhs));
    if (num_rhs > 1) {
      OP_REQUIRES_OK(context,
                     DoConjugateMatrixTranspose(device, rhs, &transposed_rhs));
    } else {
      DeviceMemcpy<GPUDevice>(transposed_rhs.flat<Scalar>().data(),
                              rhs.flat<Scalar>().data(),
                              rhs.NumElements() * sizeof(Scalar), stream);
    }

    // 3. Solve op(A) X = B (in column major form).
    auto transposed_rhs_reshaped =
        transposed_rhs.template flat_inner_dims<Scalar, 3>();
    auto trans = oneapi::mkl::transpose::conjtrans;
    try {
      int64_t getrs_scratchpad_size =
          oneapi::mkl::lapack::getrs_batch_scratchpad_size<Scalar>(
              *stream, trans, matrix_size, num_rhs, lda, stride_a, stride_ipiv,
              ldb, stride_b, batch_size);
      Tensor getrs_scratchpad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::v(),
                                  {getrs_scratchpad_size}, &getrs_scratchpad));

      oneapi::mkl::lapack::getrs_batch(
          *stream, trans, matrix_size, num_rhs,
          static_cast<Scalar*>(densed_input.data()), lda, stride_a,
          const_cast<int64_t*>(pivots_mat.data()), stride_ipiv,
          transposed_rhs_reshaped.data(), ldb, stride_b, batch_size,
          static_cast<Scalar*>(getrs_scratchpad.data()), getrs_scratchpad_size);
    } catch (oneapi::mkl::lapack::batch_error const& be) {
      int i = 0;
      auto& ids = be.ids();
      for (auto const& e : be.exceptions()) {
        try {
          std::rethrow_exception(e);
        } catch (oneapi::mkl::lapack::exception& e) {
          ITEX_LOG(ERROR) << "Exception " << ids[i++]
                          << " in a batch says: " << e.what()
                          << " (info code: " << e.info() << ")";
          info = e.info();
        }
      }
    }
    OP_REQUIRES(context, info == 0,
                errors::Internal("Unexpected exception caught during "
                                 "call to LAPACK API."));

    // 4. Transpose X to get the final result in row-major form.
    if (num_rhs > 1) {
      OP_REQUIRES_OK(
          context, DoConjugateMatrixTranspose(device, transposed_rhs, output));
    } else {
      DeviceMemcpy<GPUDevice>(
          output->flat<Scalar>().data(), transposed_rhs.flat<Scalar>().data(),
          transposed_rhs.NumElements() * sizeof(Scalar), stream);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TridiagonalSolveOpGpu);

  bool pivoting_;
};

REGISTER_KERNEL_BUILDER(
    Name("TridiagonalSolve").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    TridiagonalSolveOpGpu<float>);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("TridiagonalSolve").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    TridiagonalSolveOpGpu<double>);
REGISTER_KERNEL_BUILDER(
    Name("TridiagonalSolve").Device(DEVICE_GPU).TypeConstraint<complex128>("T"),
    TridiagonalSolveOpGpu<complex128>);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("TridiagonalSolve").Device(DEVICE_GPU).TypeConstraint<complex64>("T"),
    TridiagonalSolveOpGpu<complex64>);
}  // namespace itex

#endif  // ITEX_USE_MKL
