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
#include "itex/core/kernels/common/fill_functor.h"
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

namespace {
inline int PermutationOrder(int n, const int64_t* pivots) {
  // Compute the order of the permutation from the number of transpositions
  // encoded in the pivot array, see:
  // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=340
  int order = 0;
  for (int i = 0; i < n - 1; ++i) {
    order += pivots[i] != (i + 1);
  }
  return order;
}

template <typename Scalar, bool compute_log_abs_det = true>
void DeterminantFromPivotedLUKernel(sycl::item<1> item, int n,
                                    const Scalar* lu_factor,
                                    const int64_t* all_pivots, Scalar* sign,
                                    Scalar* log_abs_det) {
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  const int64_t o_idx = item.get_id();
  const int matrix_size = n * n;
  const int stride = n + 1;
  // We only parallelize over batches here. Performance is not critical,
  // since this cheap O(n) kernel always follows an O(n^3) LU factorization.
  // The main purpose is to avoid having to copy the LU decomposition to
  // host memory.

  // Initialize sign to (-1)^order.
  const int order = PermutationOrder(n, all_pivots + o_idx * n);
  Scalar prod_sign = order % 2 ? Scalar(-1) : Scalar(1);
  RealScalar sum_log_abs_det = RealScalar(0);
  int i_idx = matrix_size * o_idx;
  for (int i = 0; i < n; ++i, i_idx += stride) {
    const RealScalar abs_i = Eigen::numext::abs(lu_factor[i_idx]);
    sum_log_abs_det += Eigen::numext::log(abs_i);
    prod_sign = prod_sign * (lu_factor[i_idx] / abs_i);
  }

  if (!Eigen::numext::isfinite(sum_log_abs_det)) {
    prod_sign = Scalar(0);
    sum_log_abs_det = sum_log_abs_det > 0 ? -Eigen::numext::log(RealScalar(0))
                                          : Eigen::numext::log(RealScalar(0));
  }
  if (compute_log_abs_det) {
    sign[o_idx] = prod_sign;
    log_abs_det[o_idx] = Scalar(sum_log_abs_det);
  } else {
    log_abs_det[o_idx] = prod_sign * Eigen::numext::exp(sum_log_abs_det);
  }
}

template <typename T, bool compute_log_abs_det>
struct DeterminantFromPivotedLU;

template <typename T>
struct DeterminantFromPivotedLU<T, true> {
  DeterminantFromPivotedLU(int64_t n, const T* lu_factor_ptr,
                           const int64_t* pivots, T* sign, T* output_ptr)
      : n(n),
        lu_factor_ptr(lu_factor_ptr),
        pivots(pivots),
        sign(sign),
        output_ptr(output_ptr) {}
  void operator()(sycl::item<1> item) const {
    DeterminantFromPivotedLUKernel<T, true>(item, n, lu_factor_ptr, pivots,
                                            sign, output_ptr);
  }

 private:
  int64_t n;
  const T* lu_factor_ptr;
  const int64_t* pivots;
  T* sign;
  T* output_ptr;
};

template <typename T>
struct DeterminantFromPivotedLU<T, false> {
  DeterminantFromPivotedLU(int64_t n, const T* lu_factor_ptr,
                           const int64_t* pivots, T* output_ptr)
      : n(n),
        lu_factor_ptr(lu_factor_ptr),
        pivots(pivots),
        output_ptr(output_ptr) {}
  void operator()(sycl::item<1> item) const {
    DeterminantFromPivotedLUKernel<T, false>(item, n, lu_factor_ptr, pivots,
                                             nullptr, output_ptr);
  }

 private:
  int64_t n;
  const T* lu_factor_ptr;
  const int64_t* pivots;
  T* output_ptr;
};

template <typename Scalar>
void LaunchDeterminantFromPivotedLU(
    sycl::queue* stream, typename TTypes<Scalar, 3>::ConstTensor lu_factor,
    const int64_t* pivots, typename TTypes<Scalar, 1>::Tensor output) {
  const int64 num_matrices = output.size();
  const int64 n = lu_factor.dimension(2);

  stream->submit([&](sycl::handler& cgh) {
    DeterminantFromPivotedLU<Scalar, false> task(n, lu_factor.data(), pivots,
                                                 output.data());
    cgh.parallel_for<
        DeterminantFromPivotedLU<Scalar, /*compute_log_abs_det=*/false>>(
        sycl::range<1>(num_matrices), task);
  });
}

template <typename Scalar>
void LaunchLogDeterminantFromPivotedLU(
    sycl::queue* stream, typename TTypes<Scalar, 3>::ConstTensor lu_factor,
    const int64_t* pivots, typename TTypes<Scalar, 1>::Tensor sign,
    typename TTypes<Scalar, 1>::Tensor log_abs_det) {
  const int64 num_matrices = sign.size();
  const int64 n = lu_factor.dimension(2);
  stream->submit([&](sycl::handler& cgh) {
    DeterminantFromPivotedLU<Scalar, true> task(
        n, lu_factor.data(), pivots, sign.data(), log_abs_det.data());
    cgh.parallel_for<DeterminantFromPivotedLU<Scalar, true>>(
        sycl::range<1>(num_matrices), [=](sycl::item<1> item) {
          DeterminantFromPivotedLUKernel<Scalar, /*compute_log_abs_det=*/true>(
              item, n, lu_factor.data(), pivots, sign.data(),
              log_abs_det.data());
        });
  });
}

}  // namespace

template <class Scalar>
class DeterminantOpGpu : public OpKernel {
 public:
  explicit DeterminantOpGpu(OpKernelConstruction* context)
      : OpKernel(context) {}

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
                errors::InvalidArgument("Input matrices must be square, got",
                                        input.dim_size(ndims - 2), " != ", n));

    // Allocate output.
    TensorShape out_shape;
    for (int dim = 0; dim < ndims - 2; ++dim) {
      out_shape.AddDim(input.dim_size(dim));
    }
    out_shape.AppendShape(TensorShape({}));
    Tensor* out;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));

    // By definition, the determinant of an empty matrix is equal to one.
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    if (input.NumElements() == 0) {
      functor::SetOneFunctor<GPUDevice, Scalar> f;
      f(d, out->template flat<Scalar>());
      return;
    }

    // Make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor input_copy;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                          input.shape(), &input_copy));
    DeviceMemcpy<GPUDevice>(input_copy.flat<Scalar>().data(),
                            input.flat<Scalar>().data(),
                            input.NumElements() * sizeof(Scalar), stream);

    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64 batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<int64_t>::value,
                                          TensorShape{batch_size, n}, &pivots));
    auto pivots_mat = pivots.template matrix<int64_t>();
    auto output_reshaped = out->template flat_inner_dims<Scalar, 1>();

    // Compute the partially pivoted LU factorization(s) of the matrix/matrices.
    int64_t lda = n;
    int64_t stride_a = lda * n;
    int64_t stride_ipiv = n;
    auto* a = const_cast<Scalar*>(input_copy_reshaped.data());
    auto* ipiv = const_cast<int64_t*>(pivots_mat.data());

    int64_t info = 0;
    try {
      int64_t scratchpad_size =
          oneapi::mkl::lapack::getrf_batch_scratchpad_size<Scalar>(
              *stream, n, n, lda, stride_a, stride_ipiv, batch_size);
      Tensor scratchpad;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Scalar>::v(),
                                            {scratchpad_size}, &scratchpad));

      oneapi::mkl::lapack::getrf_batch(
          *stream, n, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size,
          reinterpret_cast<Scalar*>(scratchpad.data()), scratchpad_size);
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
                errors::Internal("Unexpected exception caught during call to "
                                 "LAPACK API."));

    // Compute the determinant for each batch as (-1)^s * prod(diag(U)),
    // where s is the order of the permutation encoded in pivots and U is the
    // upper triangular factor of the LU factorization, which is written to
    // input_copy.
    LaunchDeterminantFromPivotedLU<Scalar>(
        stream,
        const_cast<const Tensor*>(&input_copy)
            ->template flat_inner_dims<Scalar, 3>(),
        pivots_mat.data(), output_reshaped);
  }
};

template <class Scalar>
class LogDeterminantOpGpu : public OpKernel {
 public:
  explicit LogDeterminantOpGpu(OpKernelConstruction* context)
      : OpKernel(context) {}

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
                errors::InvalidArgument("Input matrices must be square, got",
                                        input.dim_size(ndims - 2), " != ", n));

    // Allocate output.
    TensorShape out_shape;
    for (int dim = 0; dim < ndims - 2; ++dim) {
      out_shape.AddDim(input.dim_size(dim));
    }
    out_shape.AppendShape(TensorShape({}));
    Tensor* sign;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &sign));
    Tensor* log_abs_det;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, out_shape, &log_abs_det));

    // By definition, the determinant of an empty matrix is equal to one.
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    if (input.NumElements() == 0) {
      functor::SetOneFunctor<GPUDevice, Scalar> one_func;
      one_func(d, sign->template flat<Scalar>());
      functor::SetZeroFunctor<GPUDevice, Scalar> zero_func;
      zero_func(d, log_abs_det->template flat<Scalar>());
      return;
    }

    // Reuse the input buffer or make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor input_copy;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                          input.shape(), &input_copy));

    DeviceMemcpy<GPUDevice>(input_copy.flat<Scalar>().data(),
                            input.flat<Scalar>().data(),
                            input.NumElements() * sizeof(Scalar), stream);

    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64 batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<int64_t>::value,
                                          TensorShape{batch_size, n}, &pivots));
    auto pivots_mat = pivots.template matrix<int64_t>();

    // Compute the partially pivoted LU factorization(s) of the matrix /
    // matrices
    int64_t lda = n;
    int64_t stride_a = lda * n;
    int64_t stride_ipiv = n;
    auto* a = reinterpret_cast<Scalar*>(input_copy_reshaped.data());
    auto* ipiv = reinterpret_cast<int64_t*>(pivots_mat.data());

    int64_t info = 0;
    try {
      Tensor scratchpad;
      int64_t scratchpad_size =
          oneapi::mkl::lapack::getrf_batch_scratchpad_size<Scalar>(
              *stream, n, n, lda, stride_a, stride_ipiv, batch_size);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Scalar>::v(),
                                            {scratchpad_size}, &scratchpad));
      oneapi::mkl::lapack::getrf_batch(
          *stream, n, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size,
          reinterpret_cast<Scalar*>(scratchpad.data()), scratchpad_size);
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
                errors::Internal("Unexpected exception caught during call to "
                                 "LAPACK API."));

    auto input_copy_reshaped_const =
        const_cast<const Tensor*>(&input_copy)
            ->template flat_inner_dims<Scalar, 3>();
    auto sign_reshaped = sign->flat<Scalar>();
    auto log_abs_det_reshaped = log_abs_det->flat<Scalar>();
    // Compute the determinant for each batch as (-1)^s * prod(diag(U)),
    // where s is the order of the permutation encoded in pivots and U is the
    // upper triangular factor of the LU factorization, which is written to
    // input_copy.
    LaunchLogDeterminantFromPivotedLU<Scalar>(stream, input_copy_reshaped_const,
                                              pivots_mat.data(), sign_reshaped,
                                              log_abs_det_reshaped);
  }
};

#define REGISTER_GPU(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("MatrixDeterminant").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
      DeterminantOpGpu<T>);                                                   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("LogMatrixDeterminant").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LogDeterminantOpGpu<T>);

REGISTER_GPU(float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_GPU(double);
REGISTER_GPU(complex128);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_GPU(complex64);
#undef REGISTER_GPU

}  // namespace itex
#endif  // ITEX_USE_MKL
