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

#include "itex/core/ops/shape_inference_fns.h"
#include "itex/core/ops/utils/logging.h"
#include "itex/core/ops/utils/status.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

void Register_ITEXEinsum() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXEinsum");
    TF_OpDefinitionBuilderAddInput(op_builder, "inputs: N * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "equation: string");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 1");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXEinsum op registration failed: ";
  }
}

void register_equality_comparison_with_cast(
    TF_OpDefinitionBuilder* op_builder) {
  TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "z: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float, half}");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "incompatible_shape_error: bool = true");

  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);
}

void register_comparison_with_cast(TF_OpDefinitionBuilder* op_builder) {
  TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "z: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float, half}");

  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);
}

void Register_EqualWithCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_EqualWithCast");
    register_equality_comparison_with_cast(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_EqualWithCast op registration failed: ";
  }
}

void Register_NotEqualWithCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_NotEqualWithCast");
    register_equality_comparison_with_cast(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_NotEqualWithCast op registration failed: ";
  }
}

void Register_GreaterWithCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_GreaterWithCast");
    register_comparison_with_cast(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_GreaterWithCast op registration failed: ";
  }
}

void Register_GreaterEqualWithCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_GreaterEqualWithCast");
    register_comparison_with_cast(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_GreaterEqualWithCast op registration failed: ";
  }
}

void Register_LessWithCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_LessWithCast");
    register_comparison_with_cast(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_LessWithCast op registration failed: ";
  }
}

void Register_LessEqualWithCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_LessEqualWithCast");
    register_comparison_with_cast(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_LessEqualWithCast op registration failed: ";
  }
}

void Register_FusedAddNOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_FusedAddN");
    TF_OpDefinitionBuilderAddInput(op_builder, "inputs: N * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "sum: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 1");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_FusedAddN op registration failed: ";
  }
}

void Register_ITEXCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("_ITEXCast");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: SrcT");

    TF_OpDefinitionBuilderAddOutput(op_builder, "y: DstT");

    TF_OpDefinitionBuilderAddAttr(op_builder, "SrcT: {half, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "DstT: {half, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Truncate: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXCast op registration failed.";
  }
}

void Register_FusedRandomOP() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedRandom");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: DstT");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: DstT");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {int32, int64}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "DstT: {half,bfloat16,float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "direction: int = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "seed: int = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "seed2: int = 0");
    TF_OpDefinitionBuilderSetIsStateful(op_builder, true);
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_FusedRandom op registration failed.";
  }
}

void Register_ITEXRandomUniformOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXRandomUniform");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape: T");
    TF_OpDefinitionBuilderSetIsStateful(op_builder, true);
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "seed: int = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "seed2: int = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dtype: {half,bfloat16,float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {int32, int64}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXRandomUniform op registration failed: ";
  }
}

void Register_ITEXFusedBinaryOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedBinary");

    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half,bfloat16,float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "input_order: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 3");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedBinary op registration failed: ";
  }
}
