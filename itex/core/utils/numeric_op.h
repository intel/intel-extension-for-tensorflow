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

#ifndef ITEX_CORE_UTILS_NUMERIC_OP_H_
#define ITEX_CORE_UTILS_NUMERIC_OP_H_

#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/types.h"

namespace itex {
// One input and one output, both the same type.
template <class T>
class UnaryOp : public OpKernel {
 public:
  explicit UnaryOp(OpKernelConstruction* context) : OpKernel(context) {
    // TODO(itex): add dataType check in future
    // const DataType dt = DataTypeToEnum<T>::v();
    // OP_REQUIRES_OK(context, context->MatchSignature({dt}, {dt}));
  }
};

// Two inputs and one output, all the same type.
template <class T>
class BinaryOp : public OpKernel {
 public:
  explicit BinaryOp(OpKernelConstruction* context) : OpKernel(context) {
    // TODO(itex): add dataType check in future
    // const DataType dt = DataTypeToEnum<T>::v();
    // OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }
};

// For operations where the input and output are the same shape.
template <class T, class CHILD>
class UnaryElementWiseOp : public UnaryOp<T> {
 public:
  using UnaryOp<T>::UnaryOp;

  void Compute(OpKernelContext* context) override {
    // Output shape is the same as input shape.
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    static_cast<CHILD*>(this)->Operate(context, input, output);
  }
};

// For binary elementwise operations.
template <class T, class CHILD>
class BinaryElementWiseOp : public BinaryOp<T> {
 public:
  using BinaryOp<T>::BinaryOp;

  void Compute(OpKernelContext* context) override {
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);

    if (!context->ValidateInputsAreSameShape()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0, 1}, 0, a.shape(), &output));

    // Dispatch to the descendant's Operate() function.
    switch (a.dims()) {
#define NDIM_CASE(NDIMS)                                                       \
  case NDIMS: {                                                                \
    static_cast<CHILD*>(this)->template Operate<NDIMS>(context, a, b, output); \
    break;                                                                     \
  }

      NDIM_CASE(0);
      NDIM_CASE(1);
      NDIM_CASE(2);
      NDIM_CASE(3);
      NDIM_CASE(4);
      NDIM_CASE(5);
      NDIM_CASE(6);
      NDIM_CASE(7);
      NDIM_CASE(8);
#undef NDIM_CASE

      default:
        context->SetStatus(errors::InvalidArgument(
            "We only handle up to Tensor::dims() up to 8, not ", a.dims()));
        break;
    }
  }
};
}  // namespace itex

#endif  // ITEX_CORE_UTILS_NUMERIC_OP_H_
