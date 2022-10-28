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

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/ctc/ctc_loss_calculator.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename T>
class CTCLossOp : public OpKernel {
  typedef Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      InputMap;
  typedef Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      OutputMap;

 public:
  explicit CTCLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated",
                                     &preprocess_collapse_repeated_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ctc_merge_repeated", &ctc_merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ignore_longer_outputs_than_inputs",
                                     &ignore_longer_outputs_than_inputs_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs;
    const Tensor* labels_indices;
    const Tensor* labels_values;
    const Tensor* seq_len;
    OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
    OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
    OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
    OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &seq_len));

    OP_REQUIRES(ctx, inputs->shape().dims() == 3,
                errors::InvalidArgument("inputs is not a 3-Tensor"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(seq_len->shape()),
                errors::InvalidArgument("sequence_length is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(labels_indices->shape()),
                errors::InvalidArgument("labels_indices is not a matrix"));
    OP_REQUIRES(ctx, labels_indices->dim_size(1) > 1,
                errors::InvalidArgument(
                    "labels_indices second dimension must be >= 1. Received ",
                    labels_indices->dim_size(1)));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_values->shape()),
                errors::InvalidArgument("labels_values is not a vector"));

    const TensorShape& inputs_shape = inputs->shape();
    const int64 max_time = inputs_shape.dim_size(0);
    OP_REQUIRES(ctx, max_time != 0,
                errors::InvalidArgument(
                    "Max time or first dimension of input cannot be 0."));
    const int64 batch_size = inputs_shape.dim_size(1);
    const int64 num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    OP_REQUIRES(
        ctx, batch_size == seq_len->dim_size(0),
        errors::InvalidArgument("len(sequence_length) != batch_size.  ",
                                "len(sequence_length):  ", seq_len->dim_size(0),
                                " batch_size: ", batch_size));
    auto seq_len_t = seq_len->vec<int32>();

    OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                errors::InvalidArgument(
                    "labels_indices and labels_values must contain the "
                    "same number of rows, but saw shapes: ",
                    labels_indices->shape().DebugString(), " vs. ",
                    labels_values->shape().DebugString()));

    OP_REQUIRES(ctx, batch_size != 0,
                errors::InvalidArgument("batch_size must not be 0"));

    // Figure out the maximum label length to use as sparse tensor dimension.
    auto labels_indices_t = labels_indices->matrix<int64>();
    int64 max_label_len = 0;
    for (int i = 0; i < labels_indices->dim_size(0); i++) {
      max_label_len = std::max(max_label_len, labels_indices_t(i, 1) + 1);
    }

    // TODO(itex): for now, we only hanle case when batch_size and
    // max_label_len can be represented by int32, this limit will be removed
    // after adding SparseTensor support.
    Status labels_sp_valid =
        IndicesValid(labels_indices, batch_size, max_label_len);
    OP_REQUIRES(ctx, labels_sp_valid.ok(),
                errors::InvalidArgument("label SparseTensor is not valid: ",
                                        labels_sp_valid.error_message()));

    typename ctc::CTCLossCalculator<T>::LabelSequences labels_t(batch_size);
    auto labels_values_t = labels_values->flat<int32>();
    for (int i = 0; i < labels_indices->dim_size(0); ++i) {
      const int batch_indices = labels_indices_t(i, 0);
      OP_REQUIRES(ctx, FastBoundsCheck(batch_indices, batch_size),
                  errors::InvalidArgument("labels batch index must be between ",
                                          0, " and ", batch_size,
                                          " but saw: ", batch_indices));
      labels_t[batch_indices].emplace_back(labels_values_t(i));
    }

    OP_REQUIRES(ctx, static_cast<size_t>(batch_size) == labels_t.size(),
                errors::InvalidArgument("len(labels) != batch_size.  ",
                                        "len(labels):  ", labels_t.size(),
                                        " batch_size: ", batch_size));

    for (int64 b = 0; b < batch_size; ++b) {
      OP_REQUIRES(
          ctx, seq_len_t(b) <= max_time,
          errors::InvalidArgument("sequence_length(", b, ") <= ", max_time));
    }

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, seq_len->shape(), &loss));
    auto loss_t = loss->vec<T>();

    Tensor* gradient;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, inputs_shape, &gradient));
    auto gradient_t = gradient->tensor<T, 3>();
    auto inputs_t = inputs->tensor<T, 3>();
    std::vector<OutputMap> gradient_list_t;
    std::vector<InputMap> input_list_t;

    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
      gradient_list_t.emplace_back(
          gradient_t.data() + t * batch_size * num_classes, batch_size,
          num_classes);
    }

    gradient_t.setZero();

    // Assumption: the blank index is num_classes - 1
    ctc::CTCLossCalculator<T> ctc_loss_calculator(num_classes - 1, 0);
    OP_REQUIRES_OK(ctx, ctc_loss_calculator.CalculateLoss(
                            seq_len_t, labels_t, input_list_t,
                            preprocess_collapse_repeated_, ctc_merge_repeated_,
                            ignore_longer_outputs_than_inputs_, &loss_t,
                            &gradient_list_t));
  }

 private:
  bool preprocess_collapse_repeated_;
  bool ctc_merge_repeated_;
  bool ignore_longer_outputs_than_inputs_;

  Status IndicesValid(const Tensor* ix, const int64 rows, const int64 cols) {
    const auto ix_t = ix->matrix<int64>();
    ITEX_DCHECK_LE(rows, std::numeric_limits<int32>::max());
    ITEX_DCHECK_LE(cols, std::numeric_limits<int32>::max());

    const int32 max_rows = static_cast<int32>(rows);
    const int32 max_cols = static_cast<int32>(cols);

    // We maintain separate bools for each validation predicate to enable
    // vectorization across loop iterations.
    bool row_zeros_valid = true;
    bool row_in_range_valid = true;
    bool col_zeros_valid = true;
    bool col_in_range_valid = true;
    bool order_valid = true;

    int64 prev_index = -1;

    // Points to the beginning of the current row of the indices matrix.
    // Each row has two int64 elements, but we use an int32 pointer to access
    // the low and high 32 bits of each element separately. This means that our
    // stride per row is 4 elements.
    const int32* const index_base_ptr =
        reinterpret_cast<const int32*>(ix_t.data());
    const size_t kInt32ElementsPerRow = 4;

    for (std::size_t n = 0; n < ix_t.dimension(0); ++n) {
      const int32* const index_ptr = index_base_ptr + n * kInt32ElementsPerRow;

      // Unpack the values on the current row of the indices matrix.
      // Note: the byte order of intel machine is always Little Endian
      const int32 row_32 = index_ptr[0];
      const int32 row_zeros = index_ptr[1];
      const int32 col_32 = index_ptr[2];
      const int32 col_zeros = index_ptr[3];

      // Validate that the high 32 bits of the row and column indices are zero.
      row_zeros_valid = row_zeros_valid & (row_zeros == 0);
      col_zeros_valid = col_zeros_valid & (col_zeros == 0);

      // Validate that the low 32 bits of the row and column indices are within
      // range of the shape.
      row_in_range_valid =
          row_in_range_valid & (row_32 >= 0) & (row_32 < max_rows);
      col_in_range_valid =
          col_in_range_valid & (col_32 >= 0) & (col_32 < max_cols);

      // Interpret the row and column as a concatenated 64-bit integer, and
      // validate that the concatenated indices are in strictly increasing
      // order.
      const int64 concatenated_index =
          (static_cast<int64>(row_32) << 32) + col_32;
      order_valid = order_valid & (concatenated_index > prev_index);
      prev_index = concatenated_index;
    }

    if (!(row_zeros_valid & row_in_range_valid & col_zeros_valid &
          col_in_range_valid)) {
      return errors::InvalidArgument("labels_indices is out of bounds.\n");
    }
    if (!order_valid) {
      return errors::InvalidArgument(
          " labels_indices is out of order. Many sparse ops require sorted "
          "indices.\n"
          "    Use `tf.sparse.reorder` to create a correctly ordered copy."
          "\n\n");
    }
    return Status::OK();
  }

  TF_DISALLOW_COPY_AND_ASSIGN(CTCLossOp<T>);
};

#define REGISTER_GPU(T)                                      \
  REGISTER_KERNEL_BUILDER(Name("CTCLoss")                    \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<T>("T")        \
                              .HostMemory("inputs")          \
                              .HostMemory("labels_indices")  \
                              .HostMemory("labels_values")   \
                              .HostMemory("sequence_length") \
                              .HostMemory("loss")            \
                              .HostMemory("gradient"),       \
                          CTCLossOp<T>);

REGISTER_GPU(float);
#undef REGISTER_GPU
}  // namespace itex
