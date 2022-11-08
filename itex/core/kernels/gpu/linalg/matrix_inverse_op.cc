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
#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "mkl.h"  // NOLINT(build/include_subdir)
#include "oneapi/mkl/lapack.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class MatrixInverseOpGpu : public OpKernel {
 public:
  explicit MatrixInverseOpGpu(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t n = input.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims));
    OP_REQUIRES(context, input.dim_size(ndims - 2) == n,
                errors::InvalidArgument("Input matrices must be squares, got",
                                        input.dim_size(ndims - 2), " != ", n));

    // By definition, an empty matrix's inverse is an empty matrix.
    if (input.NumElements() == 0) {
      context->set_output(0, input);
      return;
    }

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));

    // Make a copy of the (possible adjointed) input that we will use for the
    // factorization step.
    Tensor input_copy;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                          input.shape(), &input_copy));
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();

    const GPUDevice& device = context->eigen_device<GPUDevice>();
    auto* stream = device.stream();
    if (!adjoint_) {
      DeviceMemcpy<GPUDevice>(input_copy.flat<Scalar>().data(),
                              input.flat<Scalar>().data(),
                              input.NumElements() * sizeof(Scalar), stream);
    } else {
      OP_REQUIRES_OK(context,
                     DoConjugateMatrixTranspose(device, input, &input_copy));
    }

    const int64_t batch_size = input_copy_reshaped.dimension(0);
    Tensor pivots;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<int64_t>::value,
                                          TensorShape{batch_size, n}, &pivots));
    auto pivots_mat = pivots.template matrix<int64_t>();
    auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();
    int64_t lda = n, stride_ipiv = n;
    int64_t stride_a = lda * n;

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

      int64_t getri_scratchpad_size =
          oneapi::mkl::lapack::getri_batch_scratchpad_size<Scalar>(
              *stream, (int64_t)n, lda, stride_a, stride_ipiv,
              (int64_t)batch_size);
      Tensor getri_scratchpad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::v(),
                                  {getri_scratchpad_size}, &getri_scratchpad));
      oneapi::mkl::lapack::getri_batch(
          *stream, n, const_cast<Scalar*>(input_copy_reshaped.data()), lda,
          stride_a, const_cast<int64_t*>(pivots_mat.data()), stride_ipiv,
          batch_size, static_cast<Scalar*>(getri_scratchpad.data()),
          getri_scratchpad_size);
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

    DeviceMemcpy<GPUDevice>(output_reshaped.data(), input_copy_reshaped.data(),
                            input.NumElements() * sizeof(Scalar), stream);
  }

 private:
  bool adjoint_;
};

REGISTER_KERNEL_BUILDER(
    Name("MatrixInverse").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    MatrixInverseOpGpu<float>);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("MatrixInverse").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    MatrixInverseOpGpu<double>);
REGISTER_KERNEL_BUILDER(
    Name("MatrixInverse").Device(DEVICE_GPU).TypeConstraint<complex128>("T"),
    MatrixInverseOpGpu<complex128>);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("MatrixInverse").Device(DEVICE_GPU).TypeConstraint<complex64>("T"),
    MatrixInverseOpGpu<complex64>);
}  // namespace itex
#endif  // ITEX_USE_MKL
