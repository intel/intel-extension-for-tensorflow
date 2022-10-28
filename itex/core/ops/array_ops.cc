/* Copyright (c) 2022 Intel Corporation

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

void Register_FusedDequantizeWithReshapeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_FusedDequantizeWithReshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_range: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_range: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape: Tshape");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tshape: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "mode: {'MIN_COMBINED', 'MIN_FIRST', 'SCALED'} = 'SCALED'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "narrow_range: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "axis: int = -1");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dtype: {bfloat16, float} = DT_FLOAT");
    // TODO(itex): Implement dequantize_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_FusedDequantizeWithReshape op registration failed: ";
  }
}
