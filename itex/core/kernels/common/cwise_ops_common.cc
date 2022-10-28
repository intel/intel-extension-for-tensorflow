/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/kernels/common/cwise_ops_common.h"

namespace itex {
// TODO(itex): skip Signature checking, as cannot get input types and output
// types from OpKernelConstruction in function MatchSignature. It can be done
// after intergrating graph c api.
BinaryOpShared::BinaryOpShared(OpKernelConstruction* ctx, DataType out,
                               DataType in)
    : OpKernel(ctx) {
  // OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
  op_name = ctx->OpName();
  has_attr = ctx->HasAttr("incompatible_shape_error");
  if (has_attr) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("incompatible_shape_error",
                                     &(incompatible_shape_error)));
  }
}

void BinaryOpShared::SetUnimplementedError(OpKernelContext* ctx) {
  ctx->SetStatus(errors::Unimplemented(
      "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
      ctx->input(1).shape().DebugString(), " is not supported yet."));
}

void BinaryOpShared::SetComputeError(OpKernelContext* ctx) {
  // For speed, errors during compute are caught only via boolean flag, with no
  // associated information.  This is sufficient for now, since the only binary
  // ops that have compute errors are integer division and mod, and the only
  // error they produce is zero division.
  const string& op = op_name;
  if ((op == "Div" || op == "Mod" || op == "FloorMod" || op == "FloorDiv") &&
      DataTypeIsInteger(ctx->input_dtype(0))) {
    ctx->CtxFailure(errors::InvalidArgument("Integer division by zero"));
  } else if ((op == "Pow") && DataTypeIsInteger(ctx->input_dtype(0)) &&
             DataTypeIsSigned(ctx->input_dtype(1))) {
    ctx->CtxFailure(errors::InvalidArgument(
        "Integers to negative integer powers are not allowed"));
  } else {
    ctx->CtxFailure(
        errors::Internal("Unexpected error in binary operator "
                         "(only integer div and mod should have errors)"));
  }
}

BinaryOpShared::BinaryOpState::BinaryOpState(OpKernelContext* ctx,
                                             const string& op, bool has_attr,
                                             bool incompatible_shape_error)
    : in0(ctx->input(0)),
      in1(ctx->input(1)),
      bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape())) {
  if (!bcast.IsValid()) {
    if (has_attr && !incompatible_shape_error) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
      result = (op == "NotEqual");
      return;
    }
    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", in0.shape().DebugString(), " vs. ",
        in1.shape().DebugString()));

    return;
  }

  const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
  out_num_elements = output_shape.num_elements();
  in0_num_elements = in0.NumElements();
  in1_num_elements = in1.NumElements();
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {0, 1}, 0, output_shape, &out));

  ndims = static_cast<int>(bcast.x_reshape().size());
}

}  // namespace itex
