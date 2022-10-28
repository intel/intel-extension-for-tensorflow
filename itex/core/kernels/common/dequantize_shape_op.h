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

#ifndef ITEX_CORE_KERNELS_COMMON_DEQUANTIZE_SHAPE_OP_H_
#define ITEX_CORE_KERNELS_COMMON_DEQUANTIZE_SHAPE_OP_H_

#include <limits>
#include <vector>

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/quantization_util.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename Device, typename OutType>
class DequantizeShapeOp : public OpKernel {
 public:
  explicit DequantizeShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx,
                (input.dtype() == DataType::DT_QUINT8) ||
                    (input.dtype() == DataType::DT_QINT8),
                errors::InvalidArgument(
                    "Input Datatype must be DT_QUINT8 or DT_QINT8 !!!"));

    TensorShape shape = input.shape();
    const int rank = shape.dims();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rank}), &out));
    auto vec = out->vec<OutType>();
    for (int i = 0; i < rank; ++i) {
      int64_t dim_size = shape.dim_size(i);
      if (out->dtype() == DT_INT32) {
        OP_REQUIRES(
            ctx, FastBoundsCheck(dim_size, std::numeric_limits<int32>::max()),
            errors::InvalidArgument("Shape output type is 32-bit ", " but dim ",
                                    i, " is ", dim_size));
      }
      vec(i) = static_cast<OutType>(dim_size);
    }
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_DEQUANTIZE_SHAPE_OP_H_
