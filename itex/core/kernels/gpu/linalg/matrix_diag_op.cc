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

#include "itex/core/kernels/gpu/linalg/matrix_diag_op.h"

#include <algorithm>
#include <string>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class MatrixDiagPartOp : public OpKernel {
 public:
  explicit MatrixDiagPartOp(OpKernelConstruction* context) : OpKernel(context) {
    // MatrixDiagPartV3-specific.
    if (context->HasAttr("align")) {
      functor::ReadAlignment(context, &left_align_superdiagonal_,
                             &left_align_subdiagonal_);
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    // MatrixDiagPart and MatrixDiagPartV2 both use this OpKernel.
    // MatrixDiagPart only has one input, so we have to check the number of
    // inputs before reading additional parameters in MatrixDiagV2.
    int32 lower_diag_index = 0;
    int32 upper_diag_index = 0;
    T padding_value(0);

    // MatrixDiagPartV2-specific.
    if (context->num_inputs() > kNumV1Inputs) {
      auto& diag_index = context->input(1);
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
      padding_value = context->input(2).flat<T>()(0);
    }
    const TensorShape& input_shape = input.shape();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));

    // Make sure lower_diag_index and upper_diag_index is valid.
    const int rank = input_shape.dims();
    const Eigen::Index num_rows = input_shape.dim_size(rank - 2);
    const Eigen::Index num_cols = input_shape.dim_size(rank - 1);
    OP_REQUIRES(  // Checks lower_diag_index == 0 for when matrix shape = 0.
        context,
        (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
            lower_diag_index == 0,
        errors::InvalidArgument(
            "lower_diag_index is out of bound: ", lower_diag_index,
            ". It must be between ", -num_rows, " and ", num_cols));
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

    TensorShape output_shape;
    for (int i = 0; i < rank - 2; ++i) {
      output_shape.AddDim(input_shape.dim_size(i));
    }
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    if (num_diags > 1) output_shape.AddDim(num_diags);
    const int32 max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0),
                 num_cols - std::max(lower_diag_index, 0));
    output_shape.AddDim(max_diag_len);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_reshaped = output->flat<T>();
    auto input_reshaped = input.flat_inner_dims<T, 3>();
    functor::MatrixDiagPart<Device, T>::Compute(
        context, context->eigen_device<Device>(), input_reshaped,
        output_reshaped, lower_diag_index, upper_diag_index, max_diag_len,
        padding_value, left_align_superdiagonal_, left_align_subdiagonal_);
  }

 private:
  bool left_align_superdiagonal_ = true;
  bool left_align_subdiagonal_ = true;
  static constexpr int kNumV1Inputs = 1;
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixDiagPartOp);
};

template <typename Device, typename T>
class MatrixDiagOp : public OpKernel {
 public:
  explicit MatrixDiagOp(OpKernelConstruction* context) : OpKernel(context) {
    // MatrixDiagV3-specific.
    if (context->HasAttr("align")) {
      functor::ReadAlignment(context, &left_align_superdiagonal_,
                             &left_align_subdiagonal_);
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& diagonal = context->input(0);

    // MatrixDiag and MatrixDiagV2 both use this OpKernel. MatrixDiag only has
    // one input, so we have to check the number of inputs before reading
    // additional parameters in MatrixDiagV2.
    int32 lower_diag_index = 0;
    int32 upper_diag_index = 0;
    int32 num_rows = -1;
    int32 num_cols = -1;
    T padding_value(0);

    // MatrixDiagOpV2-specific.
    if (context->num_inputs() > kNumV1Inputs) {
      auto& diag_index = context->input(1);
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
      num_rows = context->input(2).flat<int32>()(0);
      num_cols = context->input(3).flat<int32>()(0);
      padding_value = context->input(4).flat<T>()(0);
    }

    // Size validations.
    const TensorShape& diagonal_shape = diagonal.shape();
    const int diag_rank = diagonal_shape.dims();
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(diagonal_shape),
                errors::InvalidArgument(
                    "diagonal must be at least 1-dim, received shape: ",
                    diagonal.shape().DebugString()));
    OP_REQUIRES(
        context, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));
    OP_REQUIRES(context,
                lower_diag_index == upper_diag_index ||
                    diagonal_shape.dim_size(diag_rank - 2) == num_diags,
                errors::InvalidArgument(
                    "The number of diagonals provided in the input does not "
                    "match the lower_diag_index and upper_diag_index range."));

    const Eigen::Index max_diag_len = diagonal_shape.dim_size(diag_rank - 1);
    const int32 min_num_rows = max_diag_len - std::min(upper_diag_index, 0);
    const int32 min_num_cols = max_diag_len + std::max(lower_diag_index, 0);
    OP_REQUIRES(context, num_rows == -1 || num_rows >= min_num_rows,
                errors::InvalidArgument("The number of rows is too small."));
    OP_REQUIRES(context, num_cols == -1 || num_cols >= min_num_cols,
                errors::InvalidArgument("The number of columns is too small."));

    // If both num_rows and num_cols are unknown, assume that output is square.
    // Otherwise, use smallest possible values.
    if (num_rows == -1 && num_cols == -1) {
      num_rows = std::max(min_num_rows, min_num_cols);
      num_cols = num_rows;
    } else if (num_rows == -1) {
      num_rows = min_num_rows;
    } else if (num_cols == -1) {
      num_cols = min_num_cols;
    }
    OP_REQUIRES(context, num_rows == min_num_rows || num_cols == min_num_cols,
                errors::InvalidArgument(
                    "The number of rows or columns is not consistent with "
                    "the specified d_lower, d_upper, and diagonal."));

    TensorShape output_shape = diagonal_shape;
    if (num_diags == 1) {  // Output has rank `rank+1`.
      output_shape.set_dim(diag_rank - 1, num_rows);
      output_shape.AddDim(num_cols);
    } else {  // Output has rank `rank`.
      output_shape.set_dim(diag_rank - 2, num_rows);
      output_shape.set_dim(diag_rank - 1, num_cols);
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();
    auto diag_reshaped = diagonal.flat<T>();
    functor::MatrixDiag<Device, T>::Compute(
        context, context->eigen_device<Device>(), diag_reshaped,
        output_reshaped, lower_diag_index, upper_diag_index, max_diag_len,
        padding_value, left_align_superdiagonal_, left_align_subdiagonal_);
  }

 private:
  bool left_align_superdiagonal_ = true;
  bool left_align_subdiagonal_ = true;
  static constexpr int kNumV1Inputs = 1;
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixDiagOp);
};

namespace functor {

void ReadAlignment(OpKernelConstruction* context,
                   bool* left_align_superdiagonal,
                   bool* left_align_subdiagonal) {
  string align;
  OP_REQUIRES_OK(context, context->GetAttr("align", &align));

  *left_align_superdiagonal = align == "LEFT_LEFT" || align == "LEFT_RIGHT";
  *left_align_subdiagonal = align == "LEFT_LEFT" || align == "RIGHT_LEFT";
}

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

template <typename T>
struct MatrixDiagKernel {
  MatrixDiagKernel(size_t num_work_items, int num_cols, int num_rows,
                   Eigen::Index lower_diag_index, Eigen::Index upper_diag_index,
                   bool left_align_superdiagonal, bool left_align_subdiagonal,
                   Eigen::Index max_diag_len, T padding_value, int num_diags,
                   const T* diag_ptr, T* output_ptr)
      : num_work_items(num_work_items),
        num_cols(num_cols),
        num_rows(num_rows),
        lower_diag_index(lower_diag_index),
        upper_diag_index(upper_diag_index),
        left_align_superdiagonal(left_align_superdiagonal),
        left_align_subdiagonal(left_align_subdiagonal),
        max_diag_len(max_diag_len),
        padding_value(padding_value),
        num_diags(num_diags),
        diag_ptr(diag_ptr),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }

    const int batch_and_row_index = id / num_cols;
    const int col = id - batch_and_row_index * num_cols;
    const int batch = batch_and_row_index / num_rows;
    const int row = batch_and_row_index - batch * num_rows;
    const int diag_index = col - row;
    const int diag_index_in_input = upper_diag_index - diag_index;

    const int content_offset =
        ComputeContentOffset(diag_index, max_diag_len, num_rows, num_cols,
                             left_align_superdiagonal, left_align_subdiagonal);

    const int index_in_the_diagonal =
        col - sycl::max(diag_index, 0) + content_offset;
    if (lower_diag_index <= diag_index && diag_index <= upper_diag_index) {
      output_ptr[id] =
          diag_ptr[batch * num_diags * max_diag_len +
                   diag_index_in_input * max_diag_len + index_in_the_diagonal];
    } else {
      output_ptr[id] = padding_value;
    }
  }

 private:
  size_t num_work_items;
  int num_cols;
  int num_rows;
  Eigen::Index lower_diag_index;
  Eigen::Index upper_diag_index;
  bool left_align_superdiagonal;
  bool left_align_subdiagonal;
  Eigen::Index max_diag_len;
  T padding_value;
  int num_diags;
  const T* diag_ptr;
  T* output_ptr;
};

template <typename T>
struct MatrixDiag<GPUDevice, T> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      const typename TTypes<T>::ConstTensor& diag,
                      const typename TTypes<T, 3>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len, const T padding_value,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
    const int batch_size = output.dimension(0);
    const int num_rows = output.dimension(1);
    const int num_cols = output.dimension(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;
    if (batch_size == 0 || max_diag_len == 0 || num_rows == 0 ||
        num_cols == 0) {
      return;
    }

    auto stream = context->eigen_gpu_device().stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = batch_size * num_rows * num_cols;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      auto diag_ptr = diag.data();
      auto output_ptr = output.data();
      MatrixDiagKernel<T> task(
          num_work_items, num_cols, num_rows, lower_diag_index,
          upper_diag_index, left_align_superdiagonal, left_align_subdiagonal,
          max_diag_len, padding_value, num_diags, diag_ptr, output_ptr);
      cgh.parallel_for<MatrixDiagKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          task);
    });
  }
};

template <typename T>
struct MatrixDiagPartKernel {
  MatrixDiagPartKernel(size_t num_work_items, int num_cols, int num_rows,
                       Eigen::Index lower_diag_index,
                       Eigen::Index upper_diag_index,
                       bool left_align_superdiagonal,
                       bool left_align_subdiagonal, Eigen::Index max_diag_len,
                       T padding_value, int num_diags, const T* input_ptr,
                       T* output_ptr)
      : num_work_items(num_work_items),
        num_cols(num_cols),
        num_rows(num_rows),
        lower_diag_index(lower_diag_index),
        upper_diag_index(upper_diag_index),
        left_align_superdiagonal(left_align_superdiagonal),
        left_align_subdiagonal(left_align_subdiagonal),
        max_diag_len(max_diag_len),
        padding_value(padding_value),
        num_diags(num_diags),
        input_ptr(input_ptr),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }

    const int batch_and_mapped_diag_index = id / max_diag_len;
    const int index_in_the_diagonal =
        id - batch_and_mapped_diag_index * max_diag_len;
    const int batch = batch_and_mapped_diag_index / num_diags;
    const int mapped_diag_index =
        batch_and_mapped_diag_index - batch * num_diags;
    const int diag_index = upper_diag_index - mapped_diag_index;

    const int content_offset =
        ComputeContentOffset(diag_index, max_diag_len, num_rows, num_cols,
                             left_align_superdiagonal, left_align_subdiagonal);

    const int y_offset = std::max(0, -diag_index);
    const int x_offset = std::max(0, diag_index);
    const int y_index = index_in_the_diagonal + y_offset - content_offset;
    const int x_index = index_in_the_diagonal + x_offset - content_offset;
    if (0 <= y_index && y_index < num_rows && 0 <= x_index &&
        x_index < num_cols) {
      output_ptr[id] =
          input_ptr[batch * num_rows * num_cols + y_index * num_cols + x_index];
    } else {
      output_ptr[id] = padding_value;
    }
  }

 private:
  size_t num_work_items;
  int num_cols;
  int num_rows;
  Eigen::Index lower_diag_index;
  Eigen::Index upper_diag_index;
  bool left_align_superdiagonal;
  bool left_align_subdiagonal;
  Eigen::Index max_diag_len;
  T padding_value;
  int num_diags;
  const T* input_ptr;
  T* output_ptr;
};

template <typename T>
struct MatrixDiagPart<GPUDevice, T> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      const typename TTypes<T, 3>::ConstTensor& input,
                      const typename TTypes<T>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len, const T padding_value,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
    const int batch_size = input.dimension(0);
    const int num_rows = input.dimension(1);
    const int num_cols = input.dimension(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;
    if (batch_size == 0 || max_diag_len == 0 || num_rows == 0 ||
        num_cols == 0) {
      return;
    }

    auto stream = context->eigen_gpu_device().stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = batch_size * num_diags * max_diag_len;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      auto input_ptr = input.data();
      auto output_ptr = output.data();
      MatrixDiagPartKernel<T> task(
          num_work_items, num_cols, num_rows, lower_diag_index,
          upper_diag_index, left_align_superdiagonal, left_align_subdiagonal,
          max_diag_len, padding_value, num_diags, input_ptr, output_ptr);

      cgh.parallel_for<MatrixDiagPartKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          task);
    });
  }
};
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_MATRIX_DIAG_GPU(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      MatrixDiagOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagV2")                             \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .HostMemory("k")                             \
                              .HostMemory("num_rows")                      \
                              .HostMemory("num_cols")                      \
                              .HostMemory("padding_value"),                \
                          MatrixDiagOp<GPUDevice, type>);                  \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagV3")                             \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .HostMemory("k")                             \
                              .HostMemory("num_rows")                      \
                              .HostMemory("num_cols")                      \
                              .HostMemory("padding_value"),                \
                          MatrixDiagOp<GPUDevice, type>);                  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixDiagPart").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      MatrixDiagPartOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagPartV2")                         \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .HostMemory("k")                             \
                              .HostMemory("padding_value"),                \
                          MatrixDiagPartOp<GPUDevice, type>);              \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagPartV3")                         \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .HostMemory("k")                             \
                              .HostMemory("padding_value"),                \
                          MatrixDiagPartOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_MATRIX_DIAG_GPU);
TF_CALL_complex64(REGISTER_MATRIX_DIAG_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_MATRIX_DIAG_GPU);
TF_CALL_complex128(REGISTER_MATRIX_DIAG_GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_MATRIX_DIAG_GPU

}  // namespace itex
