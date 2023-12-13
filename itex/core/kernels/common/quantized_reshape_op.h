/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_COMMON_QUANTIZED_RESHAPE_OP_H_
#define ITEX_CORE_KERNELS_COMMON_QUANTIZED_RESHAPE_OP_H_

#include <string>
#include <vector>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/overflow.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

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
    Tensor output(input.dtype());
    ITEX_CHECK(output.CopyFrom(input, shape));
    context->set_output(0, output);
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

class QuantizedReshapeOp : public ReshapeOp {
 public:
  explicit QuantizedReshapeOp(OpKernelConstruction* c) : ReshapeOp(c) {}

  void Compute(OpKernelContext* ctx) override {
    // This call processes inputs 1 and 2 to write output 0.
    ReshapeOp::Compute(ctx);
    if (!ctx->status().ok()) {
      return;
    }

    const auto& input_min_float_tensor = ctx->input(2);
    const auto& input_min_float_shape = input_min_float_tensor.shape();
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(input_min_float_shape) ||
                    (TensorShapeUtils::IsVector(input_min_float_shape) &&
                     (input_min_float_shape.dim_size(0) == 1)),
                errors::InvalidArgument(
                    "input_min must be a scalar or a vector of 1 element"));
    const float input_min_float = input_min_float_tensor.flat<float>()(0);
    const auto& input_max_float_tensor = ctx->input(3);
    const auto& input_max_float_shape = input_max_float_tensor.shape();
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(input_max_float_shape) ||
                    (TensorShapeUtils::IsVector(input_max_float_shape) &&
                     (input_max_float_shape.dim_size(0) == 1)),
                errors::InvalidArgument(
                    "input_max must be a scalar or a vector of 1 element"));
    const float input_max_float = input_max_float_tensor.flat<float>()(0);

    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &output_min));
    output_min->flat<float>()(0) = input_min_float;

    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &output_max));
    output_max->flat<float>()(0) = input_max_float;
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_QUANTIZED_RESHAPE_OP_H_
