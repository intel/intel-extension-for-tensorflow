/* Copyright (c) 2021 Intel Corporation

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

// Binary ops
void register_binary(TF_OpDefinitionBuilder* op_builder) {
  TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
  TF_OpDefinitionBuilderAddInput(op_builder, "y_meta: uint8");

  TF_OpDefinitionBuilderAddOutput(op_builder, "z: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "z_meta: uint8");

  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, float, bfloat16}");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);
}

void Register_OneDnnAddOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnAdd");
    register_binary(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnAdd op registration failed.";
  }
}

void Register_OneDnnAddV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnAddV2");
    register_binary(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnAddV2 op registration failed.";
  }
}

void Register_OneDnnMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnMul");
    register_binary(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnMul op registration failed.";
  }
}

void Register_OneDnnSubOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnSub");
    register_binary(op_builder);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnSub op registration failed.";
  }
}

// cast op
void Register_OneDnnCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnCast");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: SrcT");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "y: DstT");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "SrcT: {half, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "DstT: {half, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Truncate: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnCast op registration failed.";
  }
}
