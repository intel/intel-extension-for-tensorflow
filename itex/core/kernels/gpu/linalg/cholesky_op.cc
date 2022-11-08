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

#include "itex/core/kernels/gpu/matrix_band_part_op.h"
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
class CholeskyOpGpu : public OpKernel {
 public:
  explicit CholeskyOpGpu(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) final {
    auto* stream = context->GetDeviceStream();
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64 n = input.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims));
    OP_REQUIRES(context, input.dim_size(ndims - 2) == n,
                errors::InvalidArgument("Input matrices must be squares, got",
                                        input.dim_size(ndims - 2), " != ", n));

    if (input.NumElements() == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      context->set_output(0, input);
      return;
    }

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));

    // Copy the lower triangular part of the input matrices to the output and
    // set the strictly upper triangular part to zero. We use a pre-existing
    // kernel MatrixBandPart to do this for all matrices in the batch at once,
    // before we launch each of the Cholesky factorization kernels.
    auto input_reshaped = input.template flat_inner_dims<Scalar, 3>();
    auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();
    functor::MatrixBandPartFunctor<GPUDevice, Scalar> band_part;
    band_part(context, context->eigen_device<GPUDevice>(),
              n /* num_lower_diags */, 0 /* num_upper_diags */, input_reshaped,
              output_reshaped);

    // Launch a Cholesky kernel for each matrix in the batch.
    const int64 batch_size = input_reshaped.dimension(0);
    auto lda = n;
    auto stride_a = n * n;

    int64_t info = 0;
    try {
      int64_t scratchpad_size =
          oneapi::mkl::lapack::potrf_batch_scratchpad_size<Scalar>(
              *stream, oneapi::mkl::uplo::upper, n, lda, stride_a, batch_size);
      Tensor scratchpad;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Scalar>::v(),
                                            {scratchpad_size}, &scratchpad));
      oneapi::mkl::lapack::potrf_batch(
          *stream, oneapi::mkl::uplo::upper, n,
          const_cast<Scalar*>(output_reshaped.data()), lda, stride_a,
          batch_size, static_cast<Scalar*>(scratchpad.data()), scratchpad_size);
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
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Cholesky").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    CholeskyOpGpu<float>);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("Cholesky").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    CholeskyOpGpu<double>);
REGISTER_KERNEL_BUILDER(
    Name("Cholesky").Device(DEVICE_GPU).TypeConstraint<complex128>("T"),
    CholeskyOpGpu<complex128>);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("Cholesky").Device(DEVICE_GPU).TypeConstraint<complex64>("T"),
    CholeskyOpGpu<complex64>);
}  // namespace itex
#endif  // ITEX_USE_MKL
