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

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"

namespace itex {
template <typename T>
class RangeOp : public OpKernel {
 public:
  explicit RangeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& start_in = context->input(0);
    const Tensor& limit_in = context->input(1);
    const Tensor& delta_in = context->input(2);
    // TODO(itex): Disallow legacy use of length-1 vectors as scalars.
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(start_in.shape()) ||
                    (TensorShapeUtils::IsVector(start_in.shape()) &&
                     start_in.shape().dim_size(0) == 1),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in.shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(limit_in.shape()) ||
                    (TensorShapeUtils::IsVector(limit_in.shape()) &&
                     limit_in.shape().dim_size(0) == 1),
                errors::InvalidArgument("limit must be a scalar, not shape ",
                                        limit_in.shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(delta_in.shape()) ||
                    (TensorShapeUtils::IsVector(delta_in.shape()) &&
                     delta_in.shape().dim_size(0) == 1),
                errors::InvalidArgument("delta must be a scalar, not shape ",
                                        delta_in.shape().DebugString()));
    const T start = start_in.scalar<T>()();
    const T limit = limit_in.scalar<T>()();
    const T delta = delta_in.scalar<T>()();
    OP_REQUIRES(context, delta != 0,
                errors::InvalidArgument("Requires delta != 0: ", delta));
    if (delta > 0) {
      OP_REQUIRES(
          context, start <= limit,
          errors::InvalidArgument(
              "Requires start <= limit when delta > 0: ", start, "/", limit));
    } else {
      OP_REQUIRES(
          context, start >= limit,
          errors::InvalidArgument(
              "Requires start >= limit when delta < 0: ", start, "/", limit));
    }
    int64 size = (std::is_integral<T>::value
                      ? ((std::abs(limit - start) + std::abs(delta) - 1) /
                         std::abs(delta))
                      : std::ceil(std::abs((limit - start) / delta)));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({size}), &out));
    auto flat = out->flat<T>();
    T val = start;
    for (int64 i = 0; i < size; ++i) {
      flat(i) = T(val);
      val += delta;
    }
  }
};

#define REGISTER_RANGE_GPU_KERNEL(TYPE)                      \
  REGISTER_KERNEL_BUILDER(Name("Range")                      \
                              .Device(DEVICE_GPU)            \
                              .HostMemory("start")           \
                              .HostMemory("limit")           \
                              .HostMemory("delta")           \
                              .HostMemory("output")          \
                              .TypeConstraint<TYPE>("Tidx"), \
                          RangeOp<TYPE>);
TF_CALL_float(REGISTER_RANGE_GPU_KERNEL);
TF_CALL_bfloat16(REGISTER_RANGE_GPU_KERNEL);
TF_CALL_int32(REGISTER_RANGE_GPU_KERNEL);
TF_CALL_int64(REGISTER_RANGE_GPU_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_RANGE_GPU_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_RANGE_GPU_KERNEL
}  // namespace itex
