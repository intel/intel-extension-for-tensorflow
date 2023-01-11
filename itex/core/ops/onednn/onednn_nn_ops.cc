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

#include "itex/core/ops/shape_inference_fns.h"
#include "itex/core/ops/utils/logging.h"
#include "itex/core/ops/utils/padding.h"
#include "itex/core/ops/utils/status.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

// TODO(itex): Develop shape inference strategy. Some ops may fail with
// Tensorflow debug build.

void Register_OneDnnAddNOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnAddN");

    TF_OpDefinitionBuilderAddInput(op_builder, "inputs: N * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "inputs_meta: N * uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "sum: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "sum_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 1");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnAddN op registration failed: ";
  }
}

void Register_OneDnnAvgPoolOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnAvgPool");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnAvgPool op registration failed: ";
  }
}

void Register_OneDnnAvgPoolGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnAvgPoolGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input_shape: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnAvgPoolGrad op registration failed: ";
  }
}

void Register_OneDnnAvgPool3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnAvgPool3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnAvgPool3D op registration failed: ";
  }
}

void Register_OneDnnAvgPool3DGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnAvgPool3DGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input_shape: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnAvgPool3DGrad op registration failed: ";
  }
}

void Register_OneDnnConcatOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConcat");

    TF_OpDefinitionBuilderAddInput(op_builder, "concat_dim: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "values: N * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "concat_dim_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "values_meta: N * uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConcat op registration failed: ";
  }
}

void Register_OneDnnConcatV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConcatV2");

    TF_OpDefinitionBuilderAddInput(op_builder, "values: N * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "axis: Tidx");
    TF_OpDefinitionBuilderAddInput(op_builder, "values_meta: N * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "axis_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tidx: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConcatV2 op registration failed: ";
  }
}

void Register_OneDnnQuantizedConcatV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedConcatV2");

    TF_OpDefinitionBuilderAddInput(op_builder, "values: N * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "axis: Tidx");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_mins:  N * float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_maxes: N * float");
    TF_OpDefinitionBuilderAddInput(op_builder, "values_meta: N * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "axis_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_mins_meta:  N * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_maxes_meta: N * uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_min: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_max: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_min_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_max_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tidx: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConcatV2 op registration failed: ";
  }
}

void Register_OneDnnMaxPoolOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnMaxPool");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnMaxPool op registration failed: ";
  }
}

void Register_OneDnnMaxPoolGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnMaxPoolGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_output: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnMaxPoolGrad op registration failed: ";
  }
}

void Register_OneDnnMaxPool3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnMaxPool3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnMaxPool3D op registration failed: ";
  }
}

void Register_OneDnnMaxPool3DGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnMaxPool3DGrad");
    // TInput is not typo in ITEX, it is used in TF-Proper
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input: TInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_output: TInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "TInput: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnMaxPool3DGrad op registration failed: ";
  }
}

void Register_OneDnnDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_range: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_range: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_range_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_range_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: dtype");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: quantizedtype");
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
        << "_OneDnnDequantize op registration failed: ";
  }
}

void Register_OneDnnConv2DBackpropFilterOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv2DBackpropFilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv2DBackpropFilter op registration failed: ";
  }
}

void Register_OneDnnConv2DBackpropInputWithSliceOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv2DBackpropInputWithSlice");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "size_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv2DBackpropInputWithSlice op registration failed: ";
  }
}

void Register_OneDnnConv2DBackpropInputOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv2DBackpropInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv2DBackpropInput op registration failed: ";
  }
}

// TODO(itex): Refactor these functions to macro
void Register_OneDnnConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv2D op registration failed: ";
  }
}

void Register_OneDnnConv3DBackpropFilterV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv3DBackpropFilterV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv3DBackpropFilterV2 op registration failed: ";
  }
}

void Register_OneDnnConv3DBackpropInputV2WithSliceOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv3DBackpropInputV2WithSlice");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: Tshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "size_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tshape: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv3DBackpropInputV2WithSlice op registration failed: ";
  }
}

void Register_OneDnnConv3DBackpropInputV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv3DBackpropInputV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: Tshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tshape: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv3DBackpropInputV2 op registration failed: ";
  }
}

void Register_OneDnnConv3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv3D op registration failed: ";
  }
}

void Register_OneDnnDepthwiseConv2dNativeBackpropFilterOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnDepthwiseConv2dNativeBackpropFilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnDepthwiseConv2dNativeBackpropFilter op registration "
           "failed: ";
  }
}

void Register_OneDnnDepthwiseConv2dNativeBackpropInputOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnDepthwiseConv2dNativeBackpropInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnDepthwiseConv2dNativeBackpropInput op registration failed: ";
  }
}

void Register_OneDnnDepthwiseConv2dNativeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnDepthwiseConv2dNative");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnDepthwiseConv2dNative op registration failed: ";
  }
}

void Register_OneDnnFusedBatchNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchNorm");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "exponential_avg_factor: float = 1.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchNorm op registration failed: ";
  }
}

void Register_OneDnnFusedBatchNormV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchNormV2");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "data_format: { 'NHWC', 'NCHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "exponential_avg_factor: float = 1.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchNormV2 op registration failed: ";
  }
}

void Register_OneDnnFusedBatchNormV3Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchNormV3");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "exponential_avg_factor: float = 1.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchNormV3 op registration failed: ";
  }
}

void Register_OneDnnFusedBatchNormExOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchNormEx");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance: U");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "side_input: num_side_inputs * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "side_input_meta: num_side_inputs * uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "exponential_avg_factor: float = 1.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_side_inputs: int >= 0 = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "activation_mode: string = \"Identity\"");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchNormEx op registration failed: ";
  }
}

void Register_OneDnnFusedBatchNormGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchNormGrad");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchNormGrad op registration failed: ";
  }
}

void Register_OneDnnFusedBatchNormGradV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchNormGradV2");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "data_format: { 'NHWC', 'NCHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchNormGradV2 op registration failed: ";
  }
}

void Register_OneDnnFusedBatchNormGradV3Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchNormGradV3");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_3_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_5: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_5_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchNormGradV3 op registration failed: ";
  }
}

void Register_OneDnnFusedBatchNormGradExOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchNormGradEx");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_3_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_5: U");
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "side_input_backprop: num_side_inputs * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_5_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(
        op_builder, "side_input_backprop_meta: num_side_inputs * uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "activation_mode: string = \"Identity\"");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_side_inputs: int >= 0 = 0");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchNormGradEx op registration failed: ";
  }
}

void Register_OneDnnFusedConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "inplace_sum: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedConv2D op registration failed: ";
  }
}

void Register_OneDnnFusedConv3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedConv3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedConv3D op registration failed.";
  }
}

void Register_OneDnnFusedDepthwiseConv2dNativeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedDepthwiseConv2dNative");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedDepthwiseConv2dNative op registration failed: ";
  }
}

void Register_OneDnnConv3DBackpropFilterWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv3DBackpropFilterWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv3DBackpropFilterWithBias op registration failed: ";
  }
}

void Register_OneDnnConv2DBackpropFilterWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnConv2DBackpropFilterWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnConv2DBackpropFilterWithBias op registration failed: ";
  }
}

void Register_OneDnnToTfOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnToTf");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    // Temporarily not register with integral datatype like qint8
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "T: {bfloat16, half, float, qint8, quint8, qint32} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnToTf op registration failed: ";
  }
}

void Register_OneDnnFusedBatchMatMulV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedBatchMatMulV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedBatchMatMulV2 op registration failed: ";
  }
}

void Register_OneDnnFusedMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "inplace_sum: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    // TODO(itex): Implement matmul_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedMatMul op registration failed: ";
  }
}

void Register_OneDnnFusedMatMulGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedMatMulGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    // TODO(itex): Implement matmul_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedMatMulGrad op registration failed: ";
  }
}

void Register_OneDnnMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    // TODO(itex): Implement matmul_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnMatMul op registration failed: ";
  }
}

void Register_OneDnnInstanceNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnInstanceNorm");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnInstanceNorm op registration failed: ";
  }
}

void Register_OneDnnFusedInstanceNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnFusedInstanceNorm");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "activation_mode: string = \"Identity\"");
    // Attributes for the LeakyRelu ----------------------------------------- //
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnFusedInstanceNorm op registration failed: ";
  }
}

void Register_OneDnnLayerNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnLayerNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "layer_mean: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "layer_variance: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "layer_mean_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "layer_variance_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "data_format: { 'NHWC', 'NCHW'} = 'NHWC' ");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnLayerNorm op registration failed: ";
  }
}

void Register_OneDnnLayerNormGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnLayerNormGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "data_format: { 'NHWC', 'NCHW'} = 'NHWC' ");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnLayerNormGrad op registration failed: ";
  }
}

void Register_OneDnnMklLayerNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnMklLayerNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnMklLayerNorm op registration failed: ";
  }
}

void Register_OneDnnQuantizedMaxPoolOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedMaxPool");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());

    // TODO(itex): Implement maxpool_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedMaxPool op registration failed: ";
  }
}

void Register_OneDnnQuantizedAvgPoolOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedAvgPool");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 4");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());

    // TODO(itex): Implement maxpool_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedAvgPool op registration failed: ";
  }
}

void Register_OneDnnBatchMatMulV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnBatchMatMulV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    // TODO(itex): Implement matmul_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnBatchMatMulV2 op registration failed: ";
  }
}

void Register_OneDnnQuantizedConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2D op registration failed: ";
  }
}

void Register_OneDnnQuantizedConv2DAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedConv2DAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DAndRequantize op registration failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedConv2DWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBias
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithBias op registration failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithBiasAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedConv2DWithBiasAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBias
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithBiasAndRequantize op registration "
           "failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithBiasAndReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedConv2DWithBiasAndRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBiasAndRelu
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithBiasAndRelu op registration failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithBiasAndReluAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBiasAndRelu
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize op registration "
           "failed: ";
  }
}

void Register_OneDnnQuantizeV2WithQuantizedConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizeV2WithQuantizedConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "input: Tinput");  // fp32 input data type
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_input: float");  // quantizeV2 min input
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_input: float");  // quantizeV2 max input
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tinput: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBiasAndRelu
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "mode: {'MIN_COMBINED', 'MIN_FIRST', 'SCALED'} = 'SCALED'");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "round_mode: {'HALF_AWAY_FROM_ZERO', "
                                  "'HALF_TO_EVEN'} = 'HALF_AWAY_FROM_ZERO'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "narrow_range: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "axis: int = -1");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "ensure_minimum_range: float = 0.01");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizeV2WithQuantizedConv2D op registration "
           "failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithBiasSumAndReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedConv2DWithBiasSumAndRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "summand: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "summand_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBiasAndRelu
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithBiasSumAndRelu op registration failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "summand: Tsummand");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_summand: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_summand: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "summand_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_summand_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_summand_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tsummand: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBiasAndRelu
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "summand: Tsummand");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_summand: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_summand: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "summand_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_summand_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_summand_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tsummand: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBiasAndRelu
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedDepthwiseConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedDepthwiseConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedDepthwiseConv2D op registration failed: ";
  }
}

void Register_OneDnnQuantizedDepthwiseConv2DWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedDepthwiseConv2DWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedDepthwiseConv2DWithBias op registration failed: ";
  }
}

void Register_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedDepthwiseConv2DWithBiasAndRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedDepthwiseConv2DWithBiasAndRelu op registration "
           "failed: ";
  }
}

void Register_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: quantizedtype = DT_QINT32");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedMatMulWithBiasAndReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedMatMulWithBiasAndRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Toutput: quantizedtype = DT_QINT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_weight_const: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedMatMulWithBiasAndRelu op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedMatMulWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedMatMulWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Toutput: quantizedtype = DT_QINT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_weight_const: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedMatMulWithBias op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedMatMulWithBiasAndReluAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedMatMulWithBiasAndReluAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Toutput: quantizedtype = DT_QUINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_weight_const: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedMatMulWithBiasAndReluAndRequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedMatMulWithBiasAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedMatMulWithBiasAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Toutput: quantizedtype = DT_QUINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_weight_const: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedMatMulWithBiasAndRequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedMatMulWithBiasAndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedMatMulWithBiasAndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tbias: {float, bfloat16, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Toutput: {float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_weight_const: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedMatMulWithBiasAndDequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedFusedMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedFusedMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_product: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_product: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_product_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_product_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Targs: {bfloat16, half, float, quantizedtype}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Toutput: quantizedtype");
    // TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedFusedMatMul op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedFusedMatMulAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedFusedMatMulAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_product: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_product: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_product_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_product_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Targs: {bfloat16, half, float, quantizedtype}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Toutput: quantizedtype");
    // TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedFusedMatMulAndRequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedFusedMatMulAndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedFusedMatMulAndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Targs: {bfloat16, half, float, quantizedtype}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Toutput: {bfloat16, half, float} = DT_FLOAT");
    // TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedFusedMatMulAndDequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedBatchMatMulV2AndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedBatchMatMulV2AndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_y: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_y: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_y_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_y_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Toutput: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    // is_filter_const is default false, because two inputs of BMM INT8 maybe
    // data tensor, e.g. in Bert Model
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedBatchMatMulV2AndDequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizedFusedBatchMatMulV2AndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_OneDnnQuantizedFusedBatchMatMulV2AndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_y: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_y: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_y_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_y_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Toutput: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    // is_filter_const is default false, because two inputs of BMM INT8 maybe
    // data tensor, e.g. in Bert Model
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedFusedBatchMatMulV2AndDequantize op "
           "registration failed: ";
  }
}

void Register_OneDnnQuantizeV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizeV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_range: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_range: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_range_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_range_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_min: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_max: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_min_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_max_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "mode: {'MIN_COMBINED', 'MIN_FIRST', 'SCALED'} = 'SCALED'");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "round_mode: {'HALF_AWAY_FROM_ZERO', "
                                  "'HALF_TO_EVEN'} = 'HALF_AWAY_FROM_ZERO'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "narrow_range: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "axis: int = -1");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "ensure_minimum_range: float = 0.01");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dtype: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "classic_asymmetric_algorithm: bool = false");
    // TODO(itex): Implement quantize_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizeV2 op registration failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedConv2DWithDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "out_type: {bfloat16, half, float} = DT_FLOAT");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBias
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithDequantize op registration "
           "failed: ";
  }
}

void Register_OneDnnQuantizedConv2DWithCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedConv2DWithCast");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "min_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "max_freezed_output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output_meta:  uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output_meta:  uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: {bfloat16, half} = DT_HALF");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2DWithBias
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding_list: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedConv2DWithCast op registration "
           "failed: ";
  }
}

void Register_OneDnnReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnRelu op registration failed: ";
  }
}

void Register_OneDnnReluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnReluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "features_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnReluGrad op registration failed.";
  }
}

void Register_OneDnnLeakyReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnLeakyRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnLeakyRelu op registration failed: ";
  }
}

void Register_OneDnnLeakyReluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnLeakyReluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "features_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnLeakyReluGrad op registration failed.";
  }
}

void Register_OneDnnGeluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnGelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "approximate: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnGelu op registration failed: ";
  }
}

void Register_OneDnnGeluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnGeluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "features_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "approximate: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnGeluGrad op registration failed.";
  }
}

void Register_OneDnnPadWithConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnPadWithConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding: {'VALID'}");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnPadWithConv2D op registration failed: ";
  }
}

void Register_OneDnnPadWithConv3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnPadWithConv3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding: {'VALID'}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnPadWithConv3D op registration failed: ";
  }
}

void Register_OneDnnPadWithFusedConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnPadWithFusedConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding: {'VALID'}");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnPadWithFusedConv2D op registration failed: ";
  }
}

void Register_OneDnnPadWithFusedConv3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnPadWithFusedConv3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: num_args * uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "padding: {'VALID'}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnPadWithFusedConv3D op registration failed: ";
  }
}

void Register_OneDnnReshapeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnReshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "tensor: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape: Tshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "tensor_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape_meta: uint8");
    // OneDnn always has plain format output, so there is no meta tensor
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tshape: {int32, int64} = DT_INT32");
    // TODO(itex): investigate the influence of setting unknown shape
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnReshape op registration failed: ";
  }
}

void Register_OneDnnQuantizedReshapeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedReshape");

    TF_OpDefinitionBuilderAddInput(op_builder, "tensor: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape: Tshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_min: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_max: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "tensor_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_min_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_max_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_min: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_max: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: type");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tshape: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedReshape op registration failed: ";
  }
}

void Register_OneDnnQuantizedTransposeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnQuantizedTranspose");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "perm: Tperm");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "perm_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_y: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_y: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tperm: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnQuantizedTranspose op registration failed: ";
  }
}

// _OneDnnSlice output is always plain
void Register_OneDnnSliceOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnSlice");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin: Index");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: Index");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "size_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Index: {int32,int64}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnSlice op registration failed: ";
  }
}

void Register_OneDnnSoftmaxOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnSoftmax");
    TF_OpDefinitionBuilderAddInput(op_builder, "logits: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "logits_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "softmax: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "softmax_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnSoftmax op registration failed: ";
  }
}

void Register_OneDnnTransposeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnTranspose");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "perm: Tperm");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "perm_meta: uint8");
    // OneDnn always has plain format output, so there is no meta tensor
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tperm: {int32, int64} = DT_INT32");
    // TODO(itex): investigate the influence of setting unknown shape
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnTranspose op registration failed: ";
  }
}

void Register_OneDnnIdentityOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnIdentity");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnTranspose op registration failed: ";
  }
}

void Register_OneDnnResizeBilinearOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnResizeBilinear");
    TF_OpDefinitionBuilderAddInput(op_builder, "images: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "images_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "size_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "resized_images: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "resized_images_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "align_corners: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "half_pixel_centers: bool = false");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnResizeBilinear op registration failed: ";
  }
}

void Register_OneDnnResizeBilinearGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnResizeBilinearGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "grads: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "orginal_image: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "grads_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "original_image_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "align_corners: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "half_pixel_centers: bool = false");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnResizeBilinearGrad op registration failed: ";
  }
}

void Register_OneDnnSwishOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnSwish");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 1.0");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnSwish op registration failed: ";
  }
}

void Register_OneDnnMishOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnMish");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features_meta: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations_meta: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnMish op registration failed: ";
  }
}

void Register_OneDnnResizeNearestNeighborOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnResizeNearestNeighbor");
    TF_OpDefinitionBuilderAddInput(op_builder, "images: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "images_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "size_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "resized_images: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "resized_images_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "align_corners: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "half_pixel_centers: bool = false");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnResizeNearestNeighbor op registration failed: ";
  }
}

void Register_OneDnnResizeNearestNeighborGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_OneDnnResizeNearestNeighborGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "grads: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "grads_meta: uint8");
    TF_OpDefinitionBuilderAddInput(op_builder, "size_meta: uint8");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_meta: uint8");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "align_corners: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "half_pixel_centers: bool = false");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnResizeNearestNeighborGrad op registration failed: ";
  }
}
