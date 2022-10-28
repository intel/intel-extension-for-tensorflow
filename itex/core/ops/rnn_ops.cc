/* Copyright (c) 2022 Intel Corporation

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

#include "itex/core/ops/shape_inference_fns.h"
#include "itex/core/ops/utils/logging.h"
#include "itex/core/ops/utils/status.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace {
constexpr auto kRNNModeAttrs =
    "rnn_mode: {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'} = 'lstm'";
}  // namespace

void Register_ITEXRnnOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("ItexRnn");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_h: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "params: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "dropout_mask: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "recurrent_dropout_mask: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "sequence_lengths: int32");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_h: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_c: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, kRNNModeAttrs);
    TF_OpDefinitionBuilderAddAttr(op_builder, "dropout: float = 0.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "recurrent_dropout: float = 0.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_proj: int = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "var_seq_length: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ItexRnn op registration failed: ";
  }
}

void Register_ITEXRnnGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ItexRnnGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_h: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "params: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "dropout_mask: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "recurrent_dropout_mask: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "sequence_lengths: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "output_h: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "output_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "output_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "output_h_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "output_c_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "input_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "input_h_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "input_c_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "params_backprop: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, kRNNModeAttrs);
    TF_OpDefinitionBuilderAddAttr(op_builder, "dropout: float = 0.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "recurrent_dropout: float = 0.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_proj: int = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "var_seq_length: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ItexRnnGrad op registration failed: ";
  }
}
