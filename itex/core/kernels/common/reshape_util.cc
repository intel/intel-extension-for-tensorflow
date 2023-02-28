/* Copyright (c) 2023 Intel Corporation

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/common/reshape_util.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "itex/core/utils/op_requires.h"

namespace itex {

using GPUDevice = Eigen::GpuDevice;

template <typename Device>
void ReshapeSparseTensor(OpKernelContext* context,
                         const Tensor& input_indices_in,
                         const Tensor& input_shape_in,
                         const Tensor& target_shape_in, int output_indices_idx,
                         int output_shape_idx) {
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices_in.shape()),
              errors::InvalidArgument(
                  "Input indices should be a matrix but received shape ",
                  input_indices_in.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
              errors::InvalidArgument(
                  "Input shape should be a vector but received shape ",
                  input_shape_in.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(target_shape_in.shape()),
              errors::InvalidArgument(
                  "Target shape should be a vector but received shape ",
                  target_shape_in.shape().DebugString()));

  const int64_t input_rank = input_shape_in.NumElements();
  const int64_t output_rank = target_shape_in.NumElements();
  OP_REQUIRES(context, input_rank <= SPARSE_RESHAPE_MAX_SHAPE_DIMS,
              errors::Unimplemented(
                  "Target shape dims should less than 8 but received shape ",
                  input_shape_in.shape().DebugString()));
  OP_REQUIRES(context, output_rank <= SPARSE_RESHAPE_MAX_SHAPE_DIMS,
              errors::Unimplemented(
                  "Target shape dims should less than 8 but received shape ",
                  target_shape_in.shape().DebugString()));

  TensorShape input_shape;
  OP_REQUIRES_OK(context, TensorShape::BuildTensorShape(
                              input_shape_in.vec<int64_t>(), &input_shape));
  const int64_t dense_size = input_shape.num_elements();
  const int64_t nnz = input_indices_in.shape().dim_size(0);

  // Compute the output shape. Determine product of specified dimensions, and
  // find the index of the unspecified one.
  TensorShape output_shape;
  int64_t product = 1;
  int unknown_index = -1;
  auto target_shape = target_shape_in.vec<int64_t>();
  for (int d = 0; d < output_rank; ++d) {
    const int64_t size = target_shape(d);
    if (size == -1) {
      OP_REQUIRES(
          context, unknown_index == -1,
          errors::InvalidArgument("only one output dimension may be -1, "
                                  "not both ",
                                  unknown_index, " and ", d));
      unknown_index = d;
      output_shape.AddDim(1);
    } else {
      OP_REQUIRES(context, size >= 0,
                  errors::InvalidArgument("size ", d,
                                          " must be non-negative, not ", size));
      product *= size;
      output_shape.AddDim(size);
    }
  }
  if (unknown_index != -1) {
    OP_REQUIRES(
        context, product > 0,
        errors::InvalidArgument("reshape cannot infer the missing "
                                "input size for an empty tensor unless all "
                                "specified input sizes are non-zero"));
    const int64_t missing = dense_size / product;
    OP_REQUIRES(
        context, product * missing == dense_size,
        errors::InvalidArgument(
            "Input to reshape is a SparseTensor with ", dense_size,
            " dense values, but the requested shape requires a multiple of ",
            product, ". input_shape=", input_shape.DebugString(),
            " output_shape=", output_shape.DebugString()));
    output_shape.set_dim(unknown_index, missing);
  }

  OP_REQUIRES(
      context, output_shape.num_elements() == dense_size,
      errors::InvalidArgument("Input to reshape is a tensor with ", dense_size,
                              " dense values, but the requested shape has ",
                              output_shape.num_elements(),
                              ". input_shape=", input_shape.DebugString(),
                              " output_shape=", output_shape.DebugString()));

  // Optimize for reshaping to the same shape.
  if (input_shape == output_shape) {
    context->set_output(output_indices_idx, input_indices_in);
    context->set_output(output_shape_idx, input_shape_in);
    return;
  }

  Tensor* result_shape = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(output_shape_idx,
                                                   TensorShape({output_rank}),
                                                   &result_shape));
  auto output_shape_vec = result_shape->vec<int64_t>();
  for (int j = 0; j < output_shape.dims(); ++j) {
    output_shape_vec(j) = output_shape.dim_size(j);
  }

  Tensor* result_indices = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(output_indices_idx,
                                          TensorShape({nnz, output_rank}),
                                          &result_indices));
  if (nnz > 0) {
    OP_REQUIRES(
        context, dense_size > 0 && product > 0,
        errors::InvalidArgument(
            "Input tensor has ", nnz, " non zero elements but input shape (",
            input_shape.DebugString(), ") or output shape (",
            output_shape.DebugString(), ") is empty"));
    OP_REQUIRES_OK(context, functor::ReshapeSparseTensorFunctor<Device>()(
                                context, input_shape, output_shape,
                                input_indices_in.matrix<int64_t>(),
                                result_indices->matrix<int64_t>()));
  }
}

#define EXPLICITLY_INSTANTIATE_FUNCTION(Device)                    \
  template void ReshapeSparseTensor<Device>(                       \
      OpKernelContext * context, const Tensor& input_indices_in,   \
      const Tensor& input_shape_in, const Tensor& target_shape_in, \
      int output_indices_idx, int output_shape_idx)

EXPLICITLY_INSTANTIATE_FUNCTION(GPUDevice);
#undef EXPLICITLY_INSTANTIATE_FUNCTION

}  // namespace itex
