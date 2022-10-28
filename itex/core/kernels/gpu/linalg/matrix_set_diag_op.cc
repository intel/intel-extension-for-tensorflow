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

#include "itex/core/kernels/gpu/linalg/matrix_set_diag_op.h"

#include <algorithm>

#include "itex/core/kernels/gpu/linalg/matrix_diag_op.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

int ComputeContentOffset(const int diag_index, const int max_diag_len,
                         const int num_rows, const int num_cols,
                         const bool left_align_superdiagonal,
                         const bool left_align_subdiagonal) {
  const bool left_align = (diag_index >= 0 && left_align_superdiagonal) ||
                          (diag_index <= 0 && left_align_subdiagonal);
  if (left_align) return 0;
  const int y_offset = sycl::min(0, diag_index);
  const int x_offset = sycl::max(0, diag_index);
  const int diag_len = sycl::min(num_rows + y_offset, num_cols - x_offset);
  return max_diag_len - diag_len;
}

template <typename Device, typename T>
class MatrixSetDiagOp : public OpKernel {
 public:
  explicit MatrixSetDiagOp(OpKernelConstruction* context) : OpKernel(context) {
    // MatrixSetDiagV3-specific.
    if (context->HasAttr("align")) {
      functor::ReadAlignment(context, &left_align_superdiagonal_,
                             &left_align_subdiagonal_);
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& diag = context->input(1);

    // MatrixSetDiag and MatrixSetDiagV2 both use this OpKernel. MatrixSetDiag
    // only has two inputs, so we have to check the number of inputs before
    // reading additional parameters in MatrixSetDiagV2.
    int32 lower_diag_index = 0;
    int32 upper_diag_index = 0;

    // MatrixSetDiagV2-specific.
    if (context->num_inputs() > kNumV1Inputs) {
      auto& diag_index = context->input(2);
      OP_REQUIRES(context,
                  TensorShapeUtils::IsScalar(diag_index.shape()) ||
                      TensorShapeUtils::IsVector(diag_index.shape()),
                  errors::InvalidArgument(
                      "diag_index must be a scalar or vector, received shape: ",
                      diag_index.shape().DebugString()));
      lower_diag_index = diag_index.flat<int32>()(0);
      upper_diag_index = lower_diag_index;
      if (TensorShapeUtils::IsVector(diag_index.shape())) {
        auto diag_index_size = diag_index.dim_size(0);
        OP_REQUIRES(
            context, 0 < diag_index_size && diag_index_size <= 2,
            errors::InvalidArgument(
                "diag_index must have only one or two elements, received ",
                diag_index_size, " elements."));
        if (diag_index_size > 1) {
          upper_diag_index = diag_index.flat<int32>()(1);
        }
      }
    }

    const TensorShape& input_shape = input.shape();
    const TensorShape& diag_shape = diag.shape();
    const int input_rank = input_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(diag_shape),
                errors::InvalidArgument(
                    "diagonal must be at least 1-dim, received shape: ",
                    diag_shape.DebugString()));

    // Make sure lower_diag_index and upper_diag_index is valid.
    const Eigen::Index num_rows = input_shape.dim_size(input_rank - 2);
    const Eigen::Index num_cols = input_shape.dim_size(input_rank - 1);
    OP_REQUIRES(  // Checks lower_diag_index == 0 for when matrix shape = 0.
        context,
        (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
            lower_diag_index == 0,
        errors::InvalidArgument(
            "lower_diag_index is out of bound: ", lower_diag_index,
            " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(context,
                (-num_rows < upper_diag_index && upper_diag_index < num_cols) ||
                    upper_diag_index == 0,
                errors::InvalidArgument(
                    "upper_diag_index is out of bound: ", upper_diag_index,
                    " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(
        context, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));

    // Check if diag size is consistent with input.
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    OP_REQUIRES(
        context,
        lower_diag_index == upper_diag_index ||
            (diag_shape.dim_size(input_rank - 2) == num_diags),
        errors::InvalidArgument("The number of diagonals provided in `diag` "
                                "is not consistent with `lower_diag_index` and "
                                "`upper_diag_index`"));

    TensorShape expected_diag_shape = input_shape;
    expected_diag_shape.RemoveLastDims(2);
    if (num_diags > 1) expected_diag_shape.AddDim(num_diags);
    const int32 max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0),
                 num_cols - std::max(lower_diag_index, 0));
    expected_diag_shape.AddDim(max_diag_len);
    OP_REQUIRES(
        context, expected_diag_shape == diag_shape,
        errors::InvalidArgument(
            "Either first dimensions of diagonal don't match input.shape[:-2], "
            "or diagonal.shape[:-1] is not equal to the longests diagonal in "
            "range [lower_diag_index:upper_diag_index].\nInput shape: ",
            input_shape.DebugString(),
            "\nDiagonal shape: ", diag_shape.DebugString(),
            "\nExpected diagonal shape: ", expected_diag_shape.DebugString()));

    if (input.NumElements() == 0) {
      // This is a no-op.
      context->set_output(0, input);
      return;
    }

    auto input_reshaped = input.flat_inner_dims<T, 3>();
    auto diag_reshaped = diag.flat<T>();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input_shape, &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();
    functor::MatrixSetDiag<Device, T>::Compute(
        context, context->eigen_device<Device>(), input_reshaped, diag_reshaped,
        output_reshaped, lower_diag_index, upper_diag_index, max_diag_len,
        left_align_superdiagonal_, left_align_subdiagonal_);
  }

 private:
  bool left_align_superdiagonal_ = true;
  bool left_align_subdiagonal_ = true;
  static constexpr int kNumV1Inputs = 2;
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixSetDiagOp);
};

namespace functor {

template <typename Scalar, bool shared_input>
class MatrixSetDiagKernel;

template <typename Scalar>
struct MatrixSetDiag<GPUDevice, Scalar> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      const typename TTypes<Scalar, 3>::ConstTensor& input,
                      const typename TTypes<Scalar>::ConstTensor& diag,
                      const typename TTypes<Scalar, 3>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
    const int batch_size = input.dimension(0);
    const int m = input.dimension(1);
    const int n = input.dimension(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;

    if (batch_size == 0 || max_diag_len == 0 || m == 0 || n == 0) return;
    if (input.data() == output.data()) {
      auto stream = context->eigen_gpu_device().stream();
      auto work_group_size =
          (*stream)
              .get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      auto num_work_items = batch_size * num_diags * max_diag_len;
      auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
      stream->submit([&](sycl::handler& cgh) {
        auto diag_ptr = diag.data();
        auto output_ptr = output.data();
        cgh.parallel_for<MatrixSetDiagKernel<Scalar, true>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                              sycl::range<1>(work_group_size)),
            [=](sycl::nd_item<1> item) {
              auto id = item.get_global_linear_id();
              if (id >= num_work_items) {
                return;
              }

              const int batch_and_diag_index = id / max_diag_len;
              int index_in_the_diagonal =
                  id - batch_and_diag_index * max_diag_len;
              const int batch = batch_and_diag_index / num_diags;
              const int diag_index_in_input =
                  batch_and_diag_index - batch * num_diags;
              const int diag_index = upper_diag_index - diag_index_in_input;
              index_in_the_diagonal -= ComputeContentOffset(
                  diag_index, max_diag_len, m, n, left_align_superdiagonal,
                  left_align_subdiagonal);

              const int y_index =
                  index_in_the_diagonal - sycl::min(0, diag_index);
              const int x_index =
                  index_in_the_diagonal + sycl::max(0, diag_index);

              // Upper-bound checks for diagonals shorter than max_diag_len.
              if (index_in_the_diagonal >= 0 && y_index < m && x_index < n) {
                const int out_index = batch * m * n + y_index * n + x_index;
                output_ptr[out_index] = diag_ptr[id];
              }
            });
      });
    } else {
      auto stream = context->eigen_gpu_device().stream();
      auto work_group_size =
          (*stream)
              .get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      auto num_work_items = batch_size * m * n;
      auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
      stream->submit([&](sycl::handler& cgh) {
        auto diag_ptr = diag.data();
        auto input_ptr = input.data();
        auto output_ptr = output.data();
        cgh.parallel_for<MatrixSetDiagKernel<Scalar, false>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                              sycl::range<1>(work_group_size)),
            [=](sycl::nd_item<1> item) {
              auto id = item.get_global_linear_id();
              if (id >= num_work_items) {
                return;
              }

              const int batch_and_row_index = id / n;
              const int col = id - batch_and_row_index * n;
              const int batch = batch_and_row_index / m;
              const int row = batch_and_row_index - batch * m;
              const int diag_index = col - row;
              const int diag_index_in_input = upper_diag_index - diag_index;
              const int index_in_the_diagonal =
                  col - sycl::max(0, diag_index) +
                  ComputeContentOffset(diag_index, max_diag_len, m, n,
                                       left_align_superdiagonal,
                                       left_align_subdiagonal);

              if (lower_diag_index <= diag_index &&
                  diag_index <= upper_diag_index) {
                output_ptr[id] = diag_ptr[batch * num_diags * max_diag_len +
                                          diag_index_in_input * max_diag_len +
                                          index_in_the_diagonal];
              } else {
                output_ptr[id] = input_ptr[id];
              }
            });
      });
    }
  }
};

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_MATRIX_SET_DIAG_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MatrixSetDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      MatrixSetDiagOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(Name("MatrixSetDiagV2")                         \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<type>("T")                  \
                              .HostMemory("k"),                           \
                          MatrixSetDiagOp<GPUDevice, type>);              \
  REGISTER_KERNEL_BUILDER(Name("MatrixSetDiagV3")                         \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<type>("T")                  \
                              .HostMemory("k"),                           \
                          MatrixSetDiagOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_MATRIX_SET_DIAG_GPU);
TF_CALL_complex64(REGISTER_MATRIX_SET_DIAG_GPU);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_MATRIX_SET_DIAG_GPU);
TF_CALL_complex128(REGISTER_MATRIX_SET_DIAG_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_MATRIX_SET_DIAG_GPU

}  // namespace itex
