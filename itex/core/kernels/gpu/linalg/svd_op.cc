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

#if ITEX_USE_MKL
#include <algorithm>
#include <string>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/kernels/gpu/linalg/eye_functor.h"
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
class SvdOpGpu : public OpKernel {
 public:
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  explicit SvdOpGpu(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compute_uv", &compute_uv_));
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  void RunSVD(OpKernelContext* context, int64_t m, int64_t n, int64_t p,
              const Tensor& M_copy, Tensor* S, Tensor* U, Tensor* V) {
    // Compute U S V* = M.
    auto* stream = context->GetDeviceStream();
    RealScalar* outputS_ptr;
    auto input_reshaped = M_copy.template flat_inner_dims<Scalar, 3>();
    const Scalar* input_ptr = input_reshaped.data();
    const int64_t batch_size =
        M_copy.dims() > 2 ? input_reshaped.dimension(0) : 1;

    // Copies of U and V if required so can take transposes after SVD.
    Tensor u_copy, v_copy;
    Scalar* outputU_ptr = nullptr;
    Scalar* outputV_ptr = nullptr;
    if (compute_uv_) {
      TensorShape u_shape, v_shape;
      if (full_matrices_) {
        u_shape = U->shape();
        v_shape = V->shape();
      } else {
        TensorShape shapeRaw = M_copy.shape();
        shapeRaw.RemoveLastDims(2);
        u_shape = shapeRaw;
        u_shape.AddDim(p);
        u_shape.AddDim(m);
        v_shape = shapeRaw;
        v_shape.AddDim(p);
        v_shape.AddDim(n);
      }

      OP_REQUIRES_OK(context,
                     context->allocate_temp(U->dtype(), u_shape, &u_copy));
      outputU_ptr = u_copy.template flat_inner_dims<Scalar, 3>().data();
      outputV_ptr = V->template flat_inner_dims<Scalar, 3>().data();
    }

    outputS_ptr = S->template flat_inner_dims<RealScalar, 2>().data();

    // Save the input matrix
    // Needed for the n=1 fix, see below, since SVD destroys the input
    Tensor input_copy;
    if (compute_uv_ && n == 1) {
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::v(),
                                  TensorShape({batch_size, m}), &input_copy));
      DeviceMemcpy<GPUDevice>(input_copy.flat<Scalar>().data(), input_ptr,
                              batch_size * m * sizeof(Scalar), stream);
    }

    Scalar* outputU = nullptr;
    Scalar* outputVT = nullptr;
    oneapi::mkl::jobsvd jobu = oneapi::mkl::jobsvd::N;
    oneapi::mkl::jobsvd jobvt = oneapi::mkl::jobsvd::N;

    int64_t info = 0;
    std::string err_msg;
    try {
      for (int64_t batch = 0; batch < batch_size; ++batch) {
        const Scalar* input = input_ptr + batch * m * n;
        RealScalar* outputS = outputS_ptr + batch * p;
        if (compute_uv_) {
          if (full_matrices_) {
            outputU = outputU_ptr + batch * m * m;
            outputVT = outputV_ptr + batch * n * n;
            jobu = oneapi::mkl::jobsvd::A;
            jobvt = oneapi::mkl::jobsvd::A;
          } else {
            outputU = outputU_ptr + batch * m * p;
            outputVT = outputV_ptr + batch * n * p;
            jobu = oneapi::mkl::jobsvd::S;
            jobvt = oneapi::mkl::jobsvd::S;
          }
        }

        std::int64_t lda = m;
        std::int64_t ldu = n;
        std::int64_t ldvt = std::min(m, n);

        std::int64_t scratchpad_size =
            oneapi::mkl::lapack::gesvd_scratchpad_size<Scalar>(
                *stream, jobu, jobvt, m, n, lda, ldu, ldvt);
        Tensor scratchpad;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<Scalar>::v(),
                                              {scratchpad_size}, &scratchpad));

        oneapi::mkl::lapack::gesvd(
            *stream, jobu, jobvt, m, n, const_cast<Scalar*>(input), lda,
            outputS, outputU, ldu, outputVT, ldvt,
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

    if (compute_uv_) {
      auto device = context->eigen_device<GPUDevice>();
      OP_REQUIRES_OK(context, DoMatrixTranspose(device, u_copy, U));
    }
  }

  // The SVD if m >= n
  void PerformSVD_MgeqN(OpKernelContext* context, int64_t m, int64_t n,
                        int64_t p, const Tensor& M, Tensor* S, Tensor* U,
                        Tensor* V) {
    auto device = context->eigen_device<GPUDevice>();
    // Transpose M, because OneMKL expects it to be column-major
    TensorShape shapeRaw = M.shape();
    shapeRaw.RemoveLastDims(2);
    TensorShape input_shape = shapeRaw;
    input_shape.AddDim(n);
    input_shape.AddDim(m);
    Tensor input_copy;

    OP_REQUIRES_OK(context,
                   context->allocate_temp(M.dtype(), input_shape, &input_copy));
    OP_REQUIRES_OK(context, DoMatrixTranspose(device, M, &input_copy));

    // Call the SVD: compute U S V* = M.
    RunSVD(context, m, n, p, input_copy, S, U, V);
  }

  // The SVD if m < n
  void PerformSVD_MlessN(OpKernelContext* context, int64_t m, int64_t n,
                         int64_t p, const Tensor& M, Tensor* S, Tensor* U,
                         Tensor* V) {
    auto* stream = context->GetDeviceStream();
    // Perform the SVD on M'. OneMKL works column major so don't need to
    // transpose M.

    // Reuse the input buffer or make a copy for the SVD depending on whether
    // this op owns the input buffer exclusively. This is needed because the
    // SVD modifies the input
    Tensor input_copy;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                          M.shape(), &input_copy));
    DeviceMemcpy<GPUDevice>(input_copy.flat<Scalar>().data(),
                            M.flat<Scalar>().data(),
                            M.NumElements() * sizeof(Scalar), stream);

    // Call the SVD: compute V S U* = M*.
    // Note (m, n) and (U, V) are swapped accordingly.
    RunSVD(context, n, m, p, input_copy, S, V, U);
  }

  void Compute(OpKernelContext* context) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();

    // Validate inputs.
    OP_REQUIRES(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims));

    const int64_t m = input.dim_size(ndims - 2);
    const int64_t n = input.dim_size(ndims - 1);
    const int64_t p = std::min(m, n);

    // output tensors.
    Tensor* outputU = nullptr;
    Tensor* outputS = nullptr;
    Tensor* outputV = nullptr;

    // compute  shapes
    TensorShape shapeRaw = input.shape();
    shapeRaw.RemoveLastDims(2);
    TensorShape shapeS = shapeRaw;
    TensorShape shapeU = shapeRaw;
    TensorShape shapeV = shapeRaw;
    shapeS.AddDim(p);
    if (compute_uv_) {
      if (full_matrices_) {
        shapeU.AddDim(m);
        shapeU.AddDim(m);
        shapeV.AddDim(n);
        shapeV.AddDim(n);
      } else {
        shapeU.AddDim(m);
        shapeU.AddDim(p);
        shapeV.AddDim(n);
        shapeV.AddDim(p);
      }
    } else {
      shapeU = TensorShape({0});
      shapeV = TensorShape({0});
    }

    // allocate output
    OP_REQUIRES_OK(context, context->allocate_output(0, shapeS, &outputS));
    OP_REQUIRES_OK(context, context->allocate_output(1, shapeU, &outputU));
    OP_REQUIRES_OK(context, context->allocate_output(2, shapeV, &outputV));

    if (n == 0 || m == 0) {
      if (n == m || !compute_uv_ || !full_matrices_) {
        // S, U, and V are all empty. Nothing to do.
        return;
      }
      auto* stream = context->GetDeviceStream();
      functor::EyeFunctor<GPUDevice, Scalar> eye;
      if (m > 0) {
        // Return a full canonical basis for the column space.
        auto outputU_reshaped = outputU->flat_inner_dims<Scalar, 3>();
        eye(stream, outputU_reshaped);
      } else if (n > 0) {
        // Return a full canonical basis for the row space.
        auto outputV_reshaped = outputV->flat_inner_dims<Scalar, 3>();
        eye(stream, outputV_reshaped);
      }
      return;
    }

    // call implementations
    if (m >= n) {
      PerformSVD_MgeqN(context, m, n, p, input, outputS, outputU, outputV);
    } else {
      PerformSVD_MlessN(context, m, n, p, input, outputS, outputU, outputV);
    }
  }

 private:
  bool compute_uv_;
  bool full_matrices_;
};

#define REGISTER_GPU(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Svd").Device(DEVICE_GPU).TypeConstraint<T>("T"), SvdOpGpu<T>); \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("BatchSvd").Device(DEVICE_GPU).TypeConstraint<T>("T"),          \
      SvdOpGpu<T>);

REGISTER_GPU(float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_GPU(double);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU
}  // namespace itex
#endif  // ITEX_USE_MKL
