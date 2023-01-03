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

template <class Scalar>
class MatrixSolveOpGpu : public OpKernel {
 public:
  explicit MatrixSolveOpGpu(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  void Compute(OpKernelContext* context) final {
    auto* stream = context->GetDeviceStream();
    const Tensor& input = context->input(0);
    const Tensor& rhs = context->input(1);
    const int ndims = input.dims();
    const int64_t n = input.dim_size(ndims - 1);
    const int64_t nrhs = rhs.dim_size(ndims - 1);

    // Validate inputs.
    OP_REQUIRES(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims));
    OP_REQUIRES(context, rhs.dims() == ndims,
                errors::InvalidArgument(
                    "Input and right-hand side must have same rank, got ",
                    ndims, " != ", rhs.dims()));
    OP_REQUIRES(context, input.dim_size(ndims - 2) == n,
                errors::InvalidArgument("Input matrices must be squares, got",
                                        input.dim_size(ndims - 2), " != ", n));
    OP_REQUIRES(context, rhs.dim_size(ndims - 2) == n,
                errors::InvalidArgument(
                    "Input matrix and right-hand side must have the "
                    "same number of rows, got",
                    n, " != ", rhs.dim_size(ndims - 2)));
    for (int dim = 0; dim < ndims - 2; dim++) {
      OP_REQUIRES(
          context, input.dim_size(dim) == rhs.dim_size(dim),
          errors::InvalidArgument(
              "All input tensors must have the same outer dimensions."));
    }

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 0, rhs.shape(), &output));

    // To be consistent with the MatrixInverse op, we define the solution for
    // an empty set of equations as the empty matrix.
    if (input.NumElements() == 0 || rhs.NumElements() == 0) {
      return;
    }

    // Make a copy of the input for the factorization step, or, if adjoint_ is
    // false, try to reuse the input buffer if this op owns it exclusively.
    Tensor input_copy;
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                          input.shape(), &input_copy));
    if (adjoint_) {
      // For the adjoint case, it is simpler to always make a transposed copy up
      // front.
      OP_REQUIRES_OK(context, DoMatrixTranspose(device, input, &input_copy));
    } else {
      DeviceMemcpy<GPUDevice>(input_copy.flat<Scalar>().data(),
                              input.flat<Scalar>().data(),
                              input.NumElements() * sizeof(Scalar), stream);
    }

    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64_t batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<int64_t>::value,
                                          TensorShape{batch_size, n}, &pivots));
    auto pivots_mat = pivots.template matrix<int64_t>();

    int64_t lda = n, ldb = n, stride_ipiv = n;
    int64_t stride_a = lda * n, stride_b = ldb * nrhs;
    // 1. Compute the partially pivoted LU factorization(s) of the
    // matrix/matrices.
    int64_t info = 0;
    try {
      int64_t getrf_scratchpad_size =
          oneapi::mkl::lapack::getrf_batch_scratchpad_size<Scalar>(
              *stream, n, n, lda, stride_a, stride_ipiv, batch_size);
      Tensor getrf_scratchpad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::v(),
                                  {getrf_scratchpad_size}, &getrf_scratchpad));
      oneapi::mkl::lapack::getrf_batch(
          *stream, n, n, const_cast<Scalar*>(input_copy_reshaped.data()), lda,
          stride_a, const_cast<int64_t*>(pivots_mat.data()), stride_ipiv,
          batch_size, static_cast<Scalar*>(getrf_scratchpad.data()),
          getrf_scratchpad_size);
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
    transposed_rhs_shape.AddDim(nrhs);
    transposed_rhs_shape.AddDim(n);
    Tensor transposed_rhs;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                        transposed_rhs_shape, &transposed_rhs));
    if (nrhs > 1) {
      OP_REQUIRES_OK(context, DoMatrixTranspose(device, rhs, &transposed_rhs));
    } else {
      DeviceMemcpy<GPUDevice>(transposed_rhs.flat<Scalar>().data(),
                              rhs.flat<Scalar>().data(),
                              rhs.NumElements() * sizeof(Scalar), stream);
    }

    // 3. Solve op(A) X = B (in column major form).
    auto transposed_rhs_reshaped =
        transposed_rhs.template flat_inner_dims<Scalar, 3>();

    auto trans = adjoint_ ? oneapi::mkl::transpose::conjtrans
                          : oneapi::mkl::transpose::trans;
    try {
      int64_t getrs_scratchpad_size =
          oneapi::mkl::lapack::getrs_batch_scratchpad_size<Scalar>(
              *stream, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb,
              stride_b, batch_size);
      Tensor getrs_scratchpad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::v(),
                                  {getrs_scratchpad_size}, &getrs_scratchpad));
      oneapi::mkl::lapack::getrs_batch(
          *stream, trans, n, nrhs,
          const_cast<Scalar*>(input_copy_reshaped.data()), lda, stride_a,
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
    if (nrhs > 1) {
      OP_REQUIRES_OK(context,
                     DoMatrixTranspose(device, transposed_rhs, output));
    } else {
      DeviceMemcpy<GPUDevice>(
          output->flat<Scalar>().data(), transposed_rhs.flat<Scalar>().data(),
          transposed_rhs.NumElements() * sizeof(Scalar), stream);
    }
  }

 private:
  bool adjoint_;
};

REGISTER_KERNEL_BUILDER(
    Name("MatrixSolve").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    MatrixSolveOpGpu<float>);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("MatrixSolve").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    MatrixSolveOpGpu<double>);
REGISTER_KERNEL_BUILDER(
    Name("MatrixSolve").Device(DEVICE_GPU).TypeConstraint<complex128>("T"),
    MatrixSolveOpGpu<complex128>);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("MatrixSolve").Device(DEVICE_GPU).TypeConstraint<complex64>("T"),
    MatrixSolveOpGpu<complex64>);
}  // namespace itex
#endif  // ITEX_USE_MKL
