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
class SelfAdjointEigV2OpGpu : public OpKernel {
 public:
  explicit SelfAdjointEigV2OpGpu(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compute_v", &compute_v_));
  }

  void Compute(OpKernelContext* context) final {
    auto* stream = context->GetDeviceStream();
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    OP_REQUIRES(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims));
    const int64_t n = input.dim_size(ndims - 1);
    OP_REQUIRES(context, input.dim_size(ndims - 2) == n,
                errors::InvalidArgument("Input matrices must be squares, got",
                                        input.dim_size(ndims - 2), " != ", n));
    const int64_t batch_size =
        input.template flat_inner_dims<Scalar, 3>().dimension(0);

    // Allocate outputs.
    Tensor* eigenvalues;
    TensorShape eigenvalues_shape = input.shape();
    eigenvalues_shape.RemoveLastDims(1);
    OP_REQUIRES_OK(
        context, context->allocate_output(0, eigenvalues_shape, &eigenvalues));
    Tensor* eigenvectors;
    TensorShape eigenvectors_shape =
        compute_v_ ? input.shape() : TensorShape({});
    OP_REQUIRES_OK(context, context->allocate_output(1, eigenvectors_shape,
                                                     &eigenvectors));

    if (input.NumElements() == 0) {
      return;
    }

    Tensor eigenvalues_real;
    eigenvalues_real = *eigenvalues;

    Tensor input_copy;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                          input.shape(), &input_copy));
    // For real symmetric matrices, row-major and column-major are the same. For
    // complex Hermitian, row-major and column-major differ by a conjugation,
    // which is still cheaper than a transpose.
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    DeviceMemcpy<GPUDevice>(input_copy.flat<Scalar>().data(),
                            input.flat<Scalar>().data(),
                            input.NumElements() * sizeof(Scalar), stream);

    // Compute eigen decomposition in-place in input_copy.
    auto input_copy_reshaped = input_copy.flat_inner_dims<Scalar, 3>();
    auto eigenvalues_real_reshaped =
        eigenvalues_real.flat_inner_dims<Scalar, 2>();

    int64_t info = 0;
    std::string err_msg;
    try {
      oneapi::mkl::job jobz =
          compute_v_ ? oneapi::mkl::job::vec : oneapi::mkl::job::novec;
      int64_t scratchpad_size =
          oneapi::mkl::lapack::syevd_scratchpad_size<Scalar>(
              *stream, jobz, oneapi::mkl::uplo::upper, (int64_t)n, (int64_t)n);
      Tensor scratchpad;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Scalar>::v(),
                                            {scratchpad_size}, &scratchpad));

      for (int batch = 0; batch < batch_size; ++batch) {
        oneapi::mkl::lapack::syevd(
            *stream, jobz, oneapi::mkl::uplo::upper, n,
            const_cast<Scalar*>(&input_copy_reshaped(batch, 0, 0)), n,
            const_cast<Scalar*>(&eigenvalues_real_reshaped(batch, 0)),
            static_cast<Scalar*>(scratchpad.data()), scratchpad_size);
      }
    } catch (oneapi::mkl::lapack::exception const& e) {
      info = e.info();
      err_msg = std::string(e.what());
    }
    OP_REQUIRES(context, info == 0,
                errors::Internal("Unexpected exception caught during call to "
                                 "LAPACK API, error message: ",
                                 err_msg));

    if (compute_v_) {
      // Transpose eigenvectors now stored in input_copy in column-major form to
      // output in row-major form.
      OP_REQUIRES_OK(context,
                     DoMatrixTranspose(device, input_copy, eigenvectors));
    }
  }

 private:
  bool compute_v_;

  TF_DISALLOW_COPY_AND_ASSIGN(SelfAdjointEigV2OpGpu);
};

REGISTER_KERNEL_BUILDER(
    Name("SelfAdjointEigV2").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    SelfAdjointEigV2OpGpu<float>);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("SelfAdjointEigV2").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    SelfAdjointEigV2OpGpu<double>);
#endif  // ITEX_ENABLE_DOUBLE
// complex64 and complex128 are not supported by oneMKL syevd routine
}  // namespace itex

#endif  // ITEX_USE_MKL
