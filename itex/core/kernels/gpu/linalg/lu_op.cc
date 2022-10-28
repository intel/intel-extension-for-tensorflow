/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

namespace {
template <typename Scalar>
void ComputePermutationFromTranspositionsKernel(
    sycl::item<1> item, const int64_t num_rows, const int64_t* all_pivots,
    Scalar* all_permutation_indices) {
  const int64_t index = item.get_id();
  const int64_t* pivots = all_pivots + index * num_rows;
  Scalar* permutation_indices = all_permutation_indices + index * num_rows;
  // Fill in the output array with the identity permutation.
  for (int i = 0; i < num_rows; ++i) {
    permutation_indices[i] = static_cast<Scalar>(i);
  }

  // Compute the permutation from a sequence of transpositions encoded
  // in the pivot array by applying the transpositions in order on the
  // identity permutation.
  for (int i = 0; i < num_rows; ++i) {
    Scalar t = permutation_indices[i];
    int64_t curr_pivot = pivots[i];
    // TODO(itex): it seems a bug exited in pivot calculation
    // double check, and file a JIRA when confirmed.
    if (curr_pivot < 1) {
      curr_pivot = 1;
    }
    permutation_indices[i] = permutation_indices[curr_pivot - 1];
    permutation_indices[curr_pivot - 1] = t;
  }
}

template <typename Tidx>
struct PermFromTrans {
  PermFromTrans(int64_t num_rows, const int64_t* all_pivots,
                Tidx* all_permutation_indices)
      : num_rows(num_rows),
        all_pivots(all_pivots),
        all_permutation_indices(all_permutation_indices) {}
  void operator()(sycl::item<1> item) const {
    ComputePermutationFromTranspositionsKernel(item, num_rows, all_pivots,
                                               all_permutation_indices);
  }

 private:
  int64_t num_rows;
  const int64_t* all_pivots;
  Tidx* all_permutation_indices;
};

template <typename Tidx>
void LaunchComputePermutationFromTranspositions(sycl::queue* stream,
                                                const int64_t batch_size,
                                                const int64_t num_rows,
                                                const int64_t* all_pivots,
                                                Tidx* all_permutation_indices) {
  stream->submit([&](sycl::handler& cgh) {
    PermFromTrans<Tidx> task(num_rows, all_pivots, all_permutation_indices);
    cgh.parallel_for<PermFromTrans<Tidx>>(sycl::range<1>(batch_size), task);
  });
}
}  // namespace

template <class Scalar, class Tidx>
class LuOpGpu : public OpKernel {
 public:
  explicit LuOpGpu(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) final {
    auto* stream = context->GetDeviceStream();
    const Tensor& input = context->input(0);

    // Analyze shape and validate inputs.
    const int input_rank = input.dims();

    OP_REQUIRES(
        context, input_rank >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", input_rank));

    const int64 num_rows = input.dim_size(input_rank - 2);
    const int64 num_cols = input.dim_size(input_rank - 1);

    OP_REQUIRES(context, num_rows == num_cols,
                errors::InvalidArgument("Input matrices must be squares, got",
                                        num_rows, " != ", num_cols));

    TensorShape batch_shape;
    for (int dim = 0; dim < input_rank - 2; ++dim) {
      batch_shape.AddDim(input.dim_size(dim));
    }
    TensorShape permutation_indices_shape = batch_shape;
    permutation_indices_shape.AddDim(num_rows);

    const GPUDevice& device = context->eigen_device<GPUDevice>();

    // We output the packed triangular factors in a dense form.
    // The lower triangular factor L corresponds to the strictly lower
    // triangular part of packed_triangular_factors with an implicit unit
    // diagonal. The upper triangular factor U is the upper triangular part of
    // packed_triangular_factors. The triangular factors satisfy the equation
    //     P * input_matrix = L * U
    // where P is the permutation matrix corresponding to the indices in
    // permutation_indices.
    //
    // Reuse the input buffer or make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor* packed_triangular_factors;
    OP_REQUIRES_OK(context,
                   context->forward_input_or_allocate_output(
                       {0}, 0, input.shape(), &packed_triangular_factors));

    DeviceMemcpy<GPUDevice>(packed_triangular_factors->flat<Scalar>().data(),
                            input.flat<Scalar>().data(),
                            input.NumElements() * sizeof(Scalar), stream);

    // Allocate output permutation.
    Tensor* permutation_indices = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, permutation_indices_shape,
                                            &permutation_indices));

    if (input.NumElements() == 0) {
      return;
    }

    // Allocate a temporary Tensor to store the transposed packed triangular
    // factors.
    Tensor packed_triangular_factors_transpose;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<Scalar>::value, input.shape(),
                                &packed_triangular_factors_transpose));
    auto packed_triangular_factors_transpose_reshaped =
        packed_triangular_factors_transpose
            .template flat_inner_dims<Scalar, 3>();
    const int64 batch_size =
        packed_triangular_factors_transpose_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<int64>::value,
                                TensorShape{batch_size, num_rows}, &pivots));
    auto pivots_mat = pivots.template matrix<int64>();

    // Transpose the input. This is necessary because OneMKL assumes
    // column-major storage while TensorFlow uses row-major.
    OP_REQUIRES_OK(context,
                   DoMatrixTranspose(device, *packed_triangular_factors,
                                     &packed_triangular_factors_transpose));

    int64_t lda = num_rows;
    int64_t stride_a = lda * num_cols;
    int64_t stride_ipiv = (num_rows < num_cols) ? num_rows : num_cols;
    auto* a = const_cast<Scalar*>(
        packed_triangular_factors_transpose_reshaped.data());
    auto* ipiv = const_cast<int64_t*>(pivots_mat.data());
    int64_t info = 0;
    try {
      Tensor scratchpad;
      int64_t scratchpad_size =
          oneapi::mkl::lapack::getrf_batch_scratchpad_size<Scalar>(
              *stream, num_rows, num_cols, lda, stride_a, stride_ipiv,
              batch_size);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Scalar>::v(),
                                            {scratchpad_size}, &scratchpad));
      oneapi::mkl::lapack::getrf_batch(
          *stream, num_rows, num_cols, a, lda, stride_a, ipiv, stride_ipiv,
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

    // Transpose the result since we had transposed the input.
    OP_REQUIRES_OK(
        context, DoMatrixTranspose(device, packed_triangular_factors_transpose,
                                   packed_triangular_factors));

    // Pivots encode the permutation of the rows as a sequences of row swaps.
    // For each index i, row i is swapped with row pivots[i].
    int64* pivots_ptr = pivots.flat<int64>().data();
    Tidx* permutation_indices_ptr =
        permutation_indices->template flat<Tidx>().data();

    LaunchComputePermutationFromTranspositions<Tidx>(
        device.stream(), batch_size, num_rows, const_cast<int64_t*>(pivots_ptr),
        permutation_indices_ptr);
  }
};

#define REGISTER_LU_GPU(type, idx_type)                                     \
  REGISTER_KERNEL_BUILDER(Name("Lu")                                        \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<type>("T")                    \
                              .TypeConstraint<idx_type>("output_idx_type"), \
                          LuOpGpu<type, idx_type>);

REGISTER_LU_GPU(float, int32);
REGISTER_LU_GPU(float, int64);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_LU_GPU(double, int32);
REGISTER_LU_GPU(double, int64);
REGISTER_LU_GPU(complex128, int32);
REGISTER_LU_GPU(complex128, int64);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_LU_GPU(complex64, int32);
REGISTER_LU_GPU(complex64, int64);
#undef REGISTER_LU_GPU

}  // namespace itex
#endif  // ITEX_USE_MKL
