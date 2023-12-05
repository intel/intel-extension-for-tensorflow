/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_COMMON_SHAPE_OPS_H_
#define ITEX_CORE_KERNELS_COMMON_SHAPE_OPS_H_

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/overflow.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/types.h"

namespace itex {

void AllocateOutputAndReshapePjrtBuffer(OpKernelContext* context,
                                        const Tensor& input,
                                        TensorShape& shape);  // NOLINT

// Note that this op is subclassed for QuantizedReshapeOp.
class ReshapeOp : public OpKernel {
 public:
  explicit ReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& sizes = context->input(1);
    // Preliminary validation of sizes.
    OP_REQUIRES(
        context,
        (TensorShapeUtils::IsVector(sizes.shape()) ||
         // TODO(rmlarsen): Disallow legacy use of scalars to represent shape.
         TensorShapeUtils::IsScalar(sizes.shape())),
        errors::InvalidArgument("sizes input must be 1-D, not ",
                                sizes.shape().DebugString()));

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64_t product = 1;
    int unknown_index = -1;
    bool sizes_has_zero_dim;
    switch (sizes.dtype()) {
      case DT_INT32:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int32>(sizes, &product, &unknown_index,
                                            &shape, &sizes_has_zero_dim));
        break;
      case DT_INT64:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int64_t>(sizes, &product, &unknown_index,
                                              &shape, &sizes_has_zero_dim));
        break;
      default:
        context->CtxFailure(errors::InvalidArgument(
            "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
            DataTypeString(sizes.dtype())));
        return;
    }
    if (unknown_index != -1) {
      int64_t input_num_elements = 1;
      bool input_has_zero_dim = false;
      for (int dim = 0; dim < input.dims(); dim++) {
        // For zero dimension, we don't count it into `input_num_elements`
        // unless `sizes` has no zero dimension, so we are still able to
        // infer shapes for other dimensions.
        if (input.dim_size(dim) > 0 || !sizes_has_zero_dim) {
          input_num_elements *= input.dim_size(dim);
        } else {
          input_has_zero_dim = true;
        }
      }

      const int64_t missing = input_num_elements / product;
      if (!input_has_zero_dim) {
        OP_REQUIRES(
            context, product * missing == input_num_elements,
            errors::InvalidArgument(
                "Input to reshape is a tensor with ", input_num_elements,
                " values, but the requested shape requires a multiple of ",
                product));
      }
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(context, shape.num_elements() == input.NumElements(),
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        input.NumElements(),
                                        " values, but the requested shape has ",
                                        shape.num_elements()));

    // Actually produce the reshaped output.
#ifdef USING_NEXTPLUGGABLE_DEVICE
    AllocateOutputAndReshapePjrtBuffer(context, input, shape);
#else
    Tensor output(input.dtype());
    ITEX_CHECK(output.CopyFrom(input, shape));
    context->set_output(0, output);
#endif  // USING_NEXTPLUGGABLE_DEVICE
  }

 private:
  template <typename Tshape>
  Status ValidateSizes(const Tensor& sizes, int64_t* product,
                       int* unknown_index, TensorShape* shape,
                       bool* has_zero_dim) {
    *product = 1;
    *unknown_index = -1;
    *has_zero_dim = false;
    const int64_t num_dims = sizes.NumElements();
    auto Svec = sizes.flat<Tshape>();
    for (int d = 0; d < num_dims; ++d) {
      const Tshape size = Svec(d);
      if (size == -1) {
        if (*unknown_index != -1) {
          return errors::InvalidArgument(
              "Only one input size may be -1, not both ", *unknown_index,
              " and ", d);
        }
        *unknown_index = d;
        shape->AddDim(1);
      } else if (size < 0) {
        return errors::InvalidArgument("Size ", d,
                                       " must be non-negative, not ", size);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape->AddDim(size);
        *has_zero_dim = true;
      } else {
        if (MultiplyWithoutOverflow(shape->num_elements(), size) < 0) {
          string msg;
          for (int ii = 0; ii < num_dims; ++ii) {
            if (ii != 0) {
              strings::StrAppend(&msg, ", ");
            }
            strings::StrAppend(&msg, Svec(ii));
          }
          return errors::InvalidArgument("Shape [", msg,
                                         "] has too many elements");
        }
        shape->AddDim(size);
        (*product) *= size;
      }
    }
    return Status::OK();
  }
};

template <typename Tdim>
class ExpandDimsOp : public OpKernel {
 public:
  explicit ExpandDimsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_t = ctx->input(0);
    OP_REQUIRES(ctx, input_t.dtype() != DataType::DT_VARIANT,
                errors::InvalidArgument("ExpandDims on Variant not supported"));

    const Tensor& dim_t = ctx->input(1);
    OP_REQUIRES(
        ctx, (dim_t.NumElements() == 1),
        errors::InvalidArgument("'dim' must be a tensor with a single value"));
    ITEX_CHECK_EQ(dim_t.dtype(), DataTypeToEnum<Tdim>::v());
    Tdim dim = *(dim_t.flat<Tdim>().data());
    const TensorShape& input_shape = input_t.shape();
    int input_dims = input_shape.dims();
    OP_REQUIRES(ctx, dim >= -1 - input_dims && dim <= input_dims,
                errors::InvalidArgument("Tried to expand dim index ", dim,
                                        " for tensor with ", input_dims,
                                        " dimensions."));

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    if (dim < 0) {
      // Clamp to the end if needed.
      dim = std::min<Tdim>(dim + input_dims + 1, input_dims);
    }

    // Compute new shape with an additional dimension.
    absl::InlinedVector<int64_t, 8> output_shape_vec(input_dims + 1);
    for (int64_t i = 0; i < dim; ++i) {
      output_shape_vec[i] = input_shape.dim_size(i);
    }
    output_shape_vec[dim] = 1;
    for (int64_t i = dim + 1; i < input_dims + 1; ++i) {
      output_shape_vec[i] = input_shape.dim_size(i - 1);
    }
    TensorShape output_shape(output_shape_vec);
    AllocateOutputAndReshapePjrtBuffer(ctx, input_t, output_shape);
  }

  bool IsExpensive() { return false; }
};

class SqueezeOp : public OpKernel {
 public:
  explicit SqueezeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<int32> squeeze_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("squeeze_dims", &squeeze_dims));
    squeeze_dims_.insert(squeeze_dims.begin(), squeeze_dims.end());
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->input(0).dtype() != DT_VARIANT,
                errors::InvalidArgument("Squeeze on Variant not supported"));

    auto existing_dims = ctx->input(0).shape().dim_sizes();
    const int existing_dims_size = static_cast<int>(existing_dims.size());
    std::vector<int64_t> new_shape;

    std::unordered_set<int32> wrapped_squeeze_dims;
    wrapped_squeeze_dims.reserve(squeeze_dims_.size());
    // Validate squeeze dims against the input.
    for (int32_t dim : squeeze_dims_) {
      OP_REQUIRES(
          ctx, (dim >= -ctx->input(0).dims() && dim < ctx->input(0).dims()),
          errors::InvalidArgument("Tried to squeeze dim index ", dim,
                                  " for tensor with ", ctx->input(0).dims(),
                                  " dimensions."));
      // If dim is < 0, we wrap around (-1 means the last element).
      if (dim < 0) {
        dim = existing_dims_size + dim;
      }

      wrapped_squeeze_dims.insert(dim);
    }

    for (int i = 0; i < existing_dims_size; ++i) {
      auto existing_dim = existing_dims[i];

      // If squeeze_set is non-empty, only squeeze those dimensions.
      if (!wrapped_squeeze_dims.empty()) {
        if (wrapped_squeeze_dims.count(i) > 0) {
          OP_REQUIRES(ctx, existing_dim == 1,
                      errors::InvalidArgument(
                          "Can not squeeze dim[", i,
                          "], expected a dimension of 1, got ", existing_dim));
        } else {
          // This dimension is not being squeezed.
          new_shape.push_back(existing_dim);
        }
      } else {
        // Copy over all non-1-length dimensions.
        if (existing_dim != 1) {
          new_shape.push_back(existing_dim);
        }
      }
    }

    TensorShape output_shape(new_shape);
    AllocateOutputAndReshapePjrtBuffer(ctx, ctx->input(0), output_shape);
  }

  bool IsExpensive() { return false; }

 private:
  std::unordered_set<int32> squeeze_dims_;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_SHAPE_OPS_H_
