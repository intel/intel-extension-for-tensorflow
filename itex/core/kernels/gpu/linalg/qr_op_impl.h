/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_LINALG_QR_OP_IMPL_H_
#define ITEX_CORE_KERNELS_GPU_LINALG_QR_OP_IMPL_H_

#if ITEX_USE_MKL
#include <algorithm>

#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/kernels/gpu/matrix_band_part_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "mkl.h"  // NOLINT(build/include_subdir)
#include "oneapi/mkl/lapack.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class QrOpGpu : public OpKernel {
 public:
  explicit QrOpGpu(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t m = input.dim_size(ndims - 2);
    const int64_t n = input.dim_size(ndims - 1);
    const int64_t min_size = std::min(m, n);
    const int64_t batch_size =
        input.template flat_inner_dims<Scalar, 3>().dimension(0);

    // Validate inputs.
    OP_REQUIRES(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims));

    // Allocate output.
    // If full_matrices_ is true then Q is m x m and R is m x n.
    // Otherwise, Q is m x min(m, n), and R is min(m, n) x n.
    Tensor* q;
    TensorShape q_shape = input.shape();
    q_shape.set_dim(ndims - 1, full_matrices_ ? m : min_size);
    OP_REQUIRES_OK(context, context->allocate_output(0, q_shape, &q));
    Tensor* r;
    TensorShape r_shape = input.shape();
    r_shape.set_dim(ndims - 2, full_matrices_ ? m : min_size);
    OP_REQUIRES_OK(context, context->allocate_output(1, r_shape, &r));

    if (input.NumElements() == 0) {
      return;
    }

    // Allocate temporaries.
    Tensor input_transposed;
    TensorShape transposed_shape = input.shape();
    transposed_shape.set_dim(ndims - 2, input.dim_size(ndims - 1));
    transposed_shape.set_dim(ndims - 1, input.dim_size(ndims - 2));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                          transposed_shape, &input_transposed));
    Tensor tau;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<Scalar>::value,
                                TensorShape({batch_size, min_size}), &tau));

    // Transpose input, since oneMKL LAPACK uses column-major, while TensorFlow
    // uses row-major storage.
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    OP_REQUIRES_OK(context,
                   DoMatrixTranspose(device, input, &input_transposed));

    auto* stream = device.stream();
    // Compute QR decomposition in-place in input_transposed.
    auto input_transposed_reshaped =
        input_transposed.flat_inner_dims<Scalar, 3>();
    auto tau_matrix = tau.matrix<Scalar>();
    auto r_reshaped = r->flat_inner_dims<Scalar, 3>();
    int64_t lda = m, stride_a = lda * n;
    int64_t stride_tau = stride_a;

    int64_t info = 0;
    try {
      int64_t geqrf_scratchpad_size =
          oneapi::mkl::lapack::geqrf_batch_scratchpad_size<Scalar>(
              *stream, m, n, lda, stride_a, stride_tau, batch_size);
      Tensor geqrf_scratchpad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::v(),
                                  {geqrf_scratchpad_size}, &geqrf_scratchpad));
      oneapi::mkl::lapack::geqrf_batch(
          *stream, m, n, input_transposed_reshaped.data(), lda, stride_a,
          tau_matrix.data(), stride_tau, batch_size,
          static_cast<Scalar*>(geqrf_scratchpad.data()), geqrf_scratchpad_size);
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

    // Generate R. R is equal to the upper triangle of the decomposition
    // stored in input_transposed. Crop, transpose (to get back to row-major)
    // and copy it to the output buffer.
    // TODO(itex): if full_matrices_ == false?
    OP_REQUIRES_OK(context, DoMatrixTranspose(device, input_transposed, r));
    // Extract the upper triangle of r (i.e. zero out the strictly lower
    // triangle).
    functor::MatrixBandPartFunctor<GPUDevice, Scalar> band_part;
    auto r_reshaped_const =
        const_cast<const Tensor*>(r)->flat_inner_dims<Scalar, 3>();
    band_part(context, device, 0 /* num_lower_diags */,
              -1 /* num_upper_diags */, r_reshaped_const, r_reshaped);

    // Generate Q from the decomposition in input_transposed.
    int64_t k = std::min(m, n);
    try {
      int64_t orgqr_scratchpad_size =
          oneapi::mkl::lapack::orgqr_batch_scratchpad_size<Scalar>(
              *stream, m, n, k, lda, stride_a, stride_tau, batch_size);
      Tensor orgqr_scratchpad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::v(),
                                  {orgqr_scratchpad_size}, &orgqr_scratchpad));

      oneapi::mkl::lapack::orgqr_batch(
          *stream, m, n, k, input_transposed_reshaped.data(), lda, stride_a,
          tau_matrix.data(), stride_tau, batch_size,
          static_cast<Scalar*>(orgqr_scratchpad.data()), orgqr_scratchpad_size);
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

    OP_REQUIRES_OK(context, DoMatrixTranspose(device, input_transposed, q));
  }

 private:
  bool full_matrices_;

  TF_DISALLOW_COPY_AND_ASSIGN(QrOpGpu);
};
}  // namespace itex
#endif  // ITEX_USE_MKL
#endif  // ITEX_CORE_KERNELS_GPU_LINALG_QR_OP_IMPL_H_
