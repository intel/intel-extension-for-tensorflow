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

#include "external/local_config_tf/include/tensorflow/c/ops.h"
#include "itex/core/ops/shape_inference_fns.h"
#include "itex/core/ops/utils/logging.h"
#include "itex/core/ops/utils/padding.h"
#include "itex/core/ops/utils/status.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

// TODO(itex): Develop shape inference strategy. Some ops may fail with
// Tensorflow debug build.

void Register_ITEXFusedMatMulGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedMatMulGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedMatMulGrad op registration failed: ";
  }
}

void Register_GeluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("Gelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "approximate: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Gelu op registration failed: ";
  }
}

void Register_GeluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("GeluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "approximate: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "GeluGrad op registration failed: ";
  }
}

void Register_ITEXInstanceNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXInstanceNorm");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, half, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXInstanceNorm op registration failed: ";
  }
}

void Register_ITEXFusedInstanceNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedInstanceNorm");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
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
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedInstanceNorm op registration failed: ";
  }
}

void Register_LayerNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("LayerNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "layer_mean: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "layer_variance: U");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "data_format: { 'NHWC', 'NCHW'} = 'NHWC' ");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &layer_norm_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "LayerNorm op registration failed: ";
  }
}

void Register_LayerNormGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("LayerNormGrad");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "data_format: { 'NHWC', 'NCHW'} = 'NHWC' ");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &layer_norm_grad_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "LayerNormGrad op registration failed: ";
  }
}

void Register_ITEXConv3DBackpropFilterWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv3DBackpropFilterWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");

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
        << "_ITEXConv3DBackpropFilterWithBias op registration failed: ";
  }
}

void Register_ITEXConv2DBackpropFilterWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv2DBackpropFilterWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXConv2DBackpropFilterWithBias op registration failed: ";
  }
}

void Register_ITEXConv3DBackpropInputV2WithSliceOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv3DBackpropInputV2WithSlice");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: Tshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tshape: {int32, int64} = DT_INT32");
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
        << "_ITEXConv3DBackpropInputV2WithSlice op registration failed: ";
  }
}

void Register_ITEXConv2DBackpropInputWithSliceOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv2DBackpropInputWithSlice");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXConv2DBackpropInputWithSlice op registration failed: ";
  }
}

void Register_ITEXFusedConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
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
        << "_ITEXFusedConv2D op registration failed: ";
  }
}

void Register_ITEXFusedMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    // TODO(itex): Implement matmul_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedMatMul op registration failed: ";
  }
}

void Register_QuantizedFusedMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedFusedMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_product: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_product: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Targs: {float, bfloat16, half, quantizedtype}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Toutput: quantizedtype");
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
        << "_QuantizedFusedMatMul op "
           "registration failed: ";
  }
}

void Register_QuantizedFusedMatMulAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedFusedMatMulAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_product: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_product: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Targs: {float, bfloat16, half, quantizedtype}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Toutput: quantizedtype");
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
        << "_QuantizedFusedMatMulAndRequantize op "
           "registration failed: ";
  }
}

void Register_QuantizedFusedMatMulAndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedFusedMatMulAndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Targs: {float, bfloat16, half, quantizedtype}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Toutput: {float, bfloat16, half} = DT_FLOAT");
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
        << "_QuantizedFusedMatMulAndDequantize op "
           "registration failed: ";
  }
}

void Register_QuantizedBatchMatMulV2AndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedBatchMatMulV2AndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_y: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_y: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: Toutput");
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
        << "_QuantizedBatchMatMulV2AndDequantize op "
           "registration failed: ";
  }
}

void Register_QuantizedFusedBatchMatMulV2AndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedFusedBatchMatMulV2AndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_y: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_y: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: Toutput");
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
        << "_QuantizedFusedBatchMatMulV2AndDequantize op "
           "registration failed: ";
  }
}

void Register_QuantizedTransposeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedTranspose");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "perm: Tperm");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
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
        << "_QuantizedTranspose op registration failed: ";
  }
}

// TODO(itex): remove this op definition, once this op is upstreamed from
// spr-base to public TF
// V2 means Conv INT8 new API for CPU
void Register_QuantizedConv2DV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tinput: quantizedtype = DT_QUINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tfilter: quantizedtype = DT_QINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tbias: {float, qint32} = DT_QINT32");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Tsummand: {float, quint8, qint8, qint32, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "out_type: {qint8, quint8, qint32, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_inputs: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_outputs: list(type) >= 0");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.0");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_QuantizedConv2DV2 op registration failed: ";
  }
}

// TODO(itex): remove this op definition, once this op is upstreamed from
// spr-base to public TF
// V2 means Conv INT8 new API for CPU
void Register_QuantizedDepthwiseConv2DV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedDepthwiseConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tinput: quantizedtype = DT_QUINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tfilter: quantizedtype = DT_QINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tbias: {float, qint32} = DT_QINT32");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Tsummand: {float, quint8, qint8, qint32, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "out_type: {qint8, quint8, qint32, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_inputs: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_outputs: list(type) >= 0");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.0");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_QuantizedDepthwiseConv2D op registration failed: ";
  }
}

// TODO(itex): remove this op definition, once this op is upstreamed from
// spr-base to public TF
// V2 means Conv INT8 new API for CPU
void Register_QuantizedConv3DV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedConv3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tinput: quantizedtype = DT_QUINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tfilter: quantizedtype = DT_QINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tbias: {float, qint32} = DT_QINT32");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Tsummand: {float, quint8, qint8, qint32, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "out_type: {qint8, quint8, qint32, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_inputs: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_outputs: list(type) >= 0");
    // TODO(itex): avoid hardcode "NDHWC" for _QuantizedConv3D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NDHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.0");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_QuantizedConv3D op registration failed: ";
  }
}

void Register_ITEXQuantizedConcatV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConcatV2");

    TF_OpDefinitionBuilderAddInput(op_builder, "values: N * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "axis: Tidx");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_mins:  N * float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_maxes: N * float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_min: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_max: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tidx: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedConcatV2Op op registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tinput: quantizedtype = DT_QUINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tfilter: quantizedtype = DT_QINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tbias: {float, qint32} = DT_QINT32");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Tsummand: {float, quint8, qint8, qint32, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "out_type: {qint8, quint8, qint32, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_inputs: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_outputs: list(type) >= 0");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.0");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedConv2DV2 op registration failed: ";
  }
}

void Register_ITEXQuantizedDepthwiseConv2DV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedDepthwiseConv2DV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tinput: quantizedtype = DT_QUINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tfilter: quantizedtype = DT_QINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tbias: {float, qint32} = DT_QINT32");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Tsummand: {float, quint8, qint8, qint32, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "out_type: {qint8, quint8, qint32, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_inputs: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_outputs: list(type) >= 0");
    // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.0");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedDepthwiseConv2DV2 op registration failed: ";
  }
}

void Register_ITEXQuantizedConv3DV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv3DV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tinput: quantizedtype = DT_QUINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tfilter: quantizedtype = DT_QINT8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tbias: {float, qint32} = DT_QINT32");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Tsummand: {float, quint8, qint8, qint32, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "out_type: {qint8, quint8, qint32, float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_inputs: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Thost_outputs: list(type) >= 0");
    // TODO(itex): avoid hardcode "NDHWC" for QuantizedConv3D
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NDHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.0");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedConv3DV2 op registration failed: ";
  }
}

// Native format
void Register_ITEXBatchMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXBatchMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXBatchMatMul op registration failed: ";
  }
}

void Register_ITEXBatchMatMulV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXBatchMatMulV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXBatchMatMulV2 op registration failed: ";
  }
}

void Register_ITEXFusedBatchMatMulV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedBatchMatMulV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
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
        << "_ITEXFusedBatchMatMulV2 op registration failed: ";
  }
}

void Register_ITEXConv2DBackpropFilterOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv2DBackpropFilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
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
        << "_ITEXConv2DBackpropFilter op registration failed: ";
  }
}

void Register_ITEXConv3DBackpropFilterV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv3DBackpropFilterV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
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
        << "_ITEXConv3DBackpropFilterV2 op registration failed: ";
  }
}

void Register_ITEXDepthwiseConv2dNativeBackpropFilterOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXDepthwiseConv2dNativeBackpropFilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
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
        << "_ITEXDepthwiseConv2dNativeBackpropFilter op registration "
           "failed: ";
  }
}

void Register_ITEXMatMul() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
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
        << "_ITEXMatMul op registration failed: ";
  }
}

void Register_ITEXQuantizeV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizeV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_range: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_range: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_min: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output_max: float");
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
        << "_ITEXQuantizeV2 op registration failed: ";
  }
}

void Register_ITEXDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_range: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_range: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: dtype");
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
        << "_ITEXDequantize op registration failed: ";
  }
}

void Register_ITEXFusedMatMulWithSumOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedMatMulWithSum");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float, half}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "inplace_sum: bool = false");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedMatMulWithSum op registration failed: ";
  }
}

void Register_ITEXConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
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
        << "_ITEXConv2D op registration failed: ";
  }
}

void Register_ITEXConv2DBackpropInputOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv2DBackpropInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
        << "_ITEXConv2DBackpropInput op registration failed: ";
  }
}

void Register_ITEXConv3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXConv3D op registration failed: ";
  }
}

void Register_ITEXConv3DBackpropInputOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv3DBackpropInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tshape: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXConv3DBackpropInput op registration failed: ";
  }
}

void Register_ITEXConv3DBackpropInputV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXConv3DBackpropInputV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: Tshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
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
        << "_ITEXConv3DBackpropInputV2 op registration failed: ";
  }
}

void Register_ITEXDepthwiseConv2dNativeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXDepthwiseConv2dNative");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
        << "_ITEXDepthwiseConv2dNative op registration failed: ";
  }
}

void Register_ITEXDepthwiseConv2dNativeBackpropInputOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXDepthwiseConv2dNativeBackpropInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
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
        << "_ITEXDepthwiseConv2dNativeBackpropInput op registration failed: ";
  }
}

void Register_ITEXPadWithConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXPadWithConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
        << "_ITEXPadWithConv2D op registration failed: ";
  }
}

void Register_ITEXPadWithConv3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXPadWithConv3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");

    TF_OpDefinitionBuilderAddAttr(op_builder, "padding: {'VALID'}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXPadWithConv3D op registration failed: ";
  }
}

void Register_ITEXFusedConv2DWithSumOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedConv2DWithSum");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
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
        << "_ITEXFusedConv2DWithSum op registration failed: ";
  }
}

void Register_ITEXFusedDepthwiseConv2dNativeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedDepthwiseConv2dNative");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedDepthwiseConv2dNative op registration failed: ";
  }
}

void Register_ITEXPadWithFusedConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXPadWithFusedConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
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
        << "_ITEXPadWithFusedConv2D op registration failed: ";
  }
}

void Register_ITEXPadWithFusedConv3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXPadWithFusedConv3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
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
        << "_ITEXPadWithFusedConv3D op registration failed: ";
  }
}

void Register_ITEXFusedConv3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ITEXFusedConv3D");

  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");

  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "T: {float, double, bfloat16, half}");
  TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
  TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnet3dDataFormatAttrString());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "dilations: list(int) = [1, 1, 1, 1, 1]");
  TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
  TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
  // TODO(itex): unknown_shape_fn -> Conv3DShape?
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status.get());
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << "_ITEXFusedConv3D op registration failed.";
}

void Register_ITEXQuantizedMatMulWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedMatMulWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out: float");
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
        << "_ITEXQuantizedMatMulWithBias op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedMatMulWithBiasAndReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedMatMulWithBiasAndRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out: float");
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
        << "_ITEXQuantizedMatMulWithBiasAndRelu op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedMatMulWithBiasAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedMatMulWithBiasAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out: float");
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
        << "_ITEXQuantizedMatMulWithBiasAndRequantize op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedMatMulWithBiasAndReluAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_ITEXQuantizedMatMulWithBiasAndReluAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_out: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_out: float");
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
        << "_ITEXQuantizedMatMulWithBiasAndReluAndRequantize op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedMatMulWithBiasAndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedMatMulWithBiasAndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Toutput");
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
        << "_ITEXQuantizedMatMulWithBiasAndDequantize op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedFusedMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedFusedMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_product: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_product: float");
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
        << "_ITEXQuantizedFusedMatMul op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedFusedMatMulAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedFusedMatMulAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_product: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_product: float");
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
        << "_ITEXQuantizedFusedMatMulAndRequantize op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedFusedMatMulAndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedFusedMatMulAndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Targs");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_a: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_b: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_b: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Toutput");
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
        << "_ITEXQuantizedFusedMatMulAndDequantize op "
           "registration failed: ";
  }
}

void Register_ITEXAvgPoolOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXAvgPool");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
        << "_ITEXAvgPool op registration failed: ";
  }
}

void Register_ITEXAvgPoolGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXAvgPoolGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input_shape: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
        << "_ITEXAvgPoolGrad op registration failed: ";
  }
}

void Register_ITEXAvgPool3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXAvgPool3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
        << "_ITEXAvgPool3D op registration failed: ";
  }
}

void Register_ITEXAvgPool3DGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXAvgPool3DGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input_shape: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
        << "_ITEXAvgPool3DGrad op registration failed: ";
  }
}

void Register_ITEXMaxPoolOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXMaxPool");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace: uint8");
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
        << "_ITEXMaxPool op registration failed: ";
  }
}

void Register_ITEXMaxPoolGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXMaxPoolGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_output: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

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
        << "ITEXMaxPoolGrad op registration failed: ";
  }
}

void Register_ITEXMaxPoolV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXMaxPoolV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "ksize: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "strides: int32");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace: uint8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXMaxPoolV2 op registration failed: ";
  }
}

void Register_ITEXMaxPoolGradV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXMaxPoolGradV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_output: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "ksize: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "strides: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ITEXMaxPoolGradV2 op registration failed: ";
  }
}

void Register_ITEXMaxPool3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXMaxPool3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace: uint8");

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
        << "_ITEXMaxPool3D op registration failed: ";
  }
}

void Register_ITEXMaxPool3DGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXMaxPool3DGrad");
    // TInput is not typo in ITEX, it is used in TF-Proper
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_input: TInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "orig_output: TInput");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace: uint8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
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
        << "_ITEXMaxPool3DGrad op registration failed: ";
  }
}

void Register_ITEXAddNOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("_ITEXAddN");

    TF_OpDefinitionBuilderAddInput(op_builder, "inputs: N * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "sum: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 1");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXAddN op registration failed: ";
  }
}

void Register_ITEXFusedBatchNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedBatchNorm");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mean: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "variance: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "exponential_avg_factor: float = 1.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedBatchNorm op registration failed: ";
  }
}

void Register_ITEXFusedBatchNormV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ITEXFusedBatchNormV2");

  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "mean: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: U");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
  TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "data_format: { 'NHWC', 'NCHW' } = 'NHWC' ");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "exponential_avg_factor: float = 1.0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);
  TF_RegisterOpDefinition(op_builder, status.get());
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << "_ITEXFusedBatchNormV2 op registration failed: ";
}

void Register_ITEXFusedBatchNormV3Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ITEXFusedBatchNormV3");

  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "mean: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
  TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderAddAttr(
      op_builder,
      "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "exponential_avg_factor: float = 1.0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);
  TF_RegisterOpDefinition(op_builder, status.get());
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << "_ITEXFusedBatchNormV3 op registration failed: ";
}

void Register_ITEXFusedBatchNormExOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ITEXFusedBatchNormEx");

  TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "mean: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "variance: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "side_input: num_side_inputs * T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
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
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);
  TF_RegisterOpDefinition(op_builder, status.get());
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << "_ITEXFusedBatchNormEx op registration failed: ";
}

void Register_ITEXFusedBatchNormGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedBatchNormGrad");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "data_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedBatchNormGrad op registration failed: ";
  }
}

void Register_ITEXFusedBatchNormGradV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedBatchNormGradV2");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
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
        << "_ITEXFusedBatchNormGradV2 op registration failed: ";
  }
}

void Register_ITEXFusedBatchNormGradV3Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedBatchNormGradV3");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_5: U");
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
        << "_ITEXFusedBatchNormGradV3 op registration failed: ";
  }
}

void Register_ITEXFusedBatchNormGradExOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedBatchNormGradEx");

    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_5: U");
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "side_input_backprop: num_side_inputs * T");
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
        << "_ITEXFusedBatchNormGradEx op registration failed: ";
  }
}

void Register_ITEXReluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXReluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXReluGrad op registration failed.";
  }
}

void Register_ITEXLeakyReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXLeakyRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXLeakyRelu op registration failed: ";
  }
}

void Register_ITEXLeakyReluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXLeakyReluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXLeakyReluGrad op registration failed.";
  }
}

void Register_ITEXGeluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("ITEXGelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "approximate: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ITEXGelu op registration failed: ";
  }
}

void Register_ITEXGeluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ITEXGeluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "approximate: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ITEXGeluGrad op registration failed.";
  }
}

void Register_ITEXEluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("_ITEXElu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXElu op registration failed: ";
  }
}

void Register_ITEXEluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXEluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXEluGrad op registration failed.";
  }
}

void Register_ITEXRelu6GradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXRelu6Grad");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "backprops: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXRelu6Grad op registration failed.";
  }
}

void Register_ITEXSwishOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXSwish");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "alpha: float = 1.0");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXSwish op registration failed: ";
  }
}

void Register_ITEXSoftmaxOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXSoftmax");
    TF_OpDefinitionBuilderAddInput(op_builder, "logits: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "softmax: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXSoftmax op registration failed: ";
  }
}

void Register_ITEXMishOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("_ITEXMish");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXMish op registration failed: ";
  }
}

void Register_ITEXFusedAddV2WithSoftmaxOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedAddV2WithSoftmax");
    TF_OpDefinitionBuilderAddInput(op_builder, "logits: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "sum: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "softmax: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXSoftmax op registration failed: ";
  }
}

// For TensorArray serial ops, we all follows semantic of v3 version. For v0,
// v2,  will be handled as v3

// TODO(itex): missed SetIsStateful and ShapeFn
void Register_ITEXTensorArray() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArray");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");
    TF_OpDefinitionBuilderAddOutput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddOutput(op_builder, "flow: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dtype: type");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_dims_of_element_shape: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dims_of_element_shape: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "element_shape: shape = { unknown_rank: true }");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dynamic_size: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "clear_after_read: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "identical_element_shapes: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "tensor_array_name: string = ''");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArray op registration failed: ";
  }
}

// TODO(itex): missed SetIsStateful and ShapeFn
void Register_ITEXTensorArrayGrad() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "grad_handle: resource");
    TF_OpDefinitionBuilderAddOutput(op_builder, "flow_out: float");

    TF_OpDefinitionBuilderAddAttr(op_builder, "source: string");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayGrad op registration failed: ";
  }
}

void Register_ITEXTensorArrayGradWithShape() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayGradWithShape");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape_to_prepend: int32");
    TF_OpDefinitionBuilderAddOutput(op_builder, "grad_handle: resource");
    TF_OpDefinitionBuilderAddOutput(op_builder, "flow_out: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "source: string");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayGradWithShape op registration failed: ";
  }
}

void Register_ITEXTensorArrayWrite() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayWrite");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "index: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "value: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "flow_out: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: type");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayWrite op registration failed: ";
  }
}

void Register_ITEXTensorArrayRead() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayRead");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "index: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "value: dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dtype: type");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayRead op registration failed: ";
  }
}

void Register_ITEXTensorArrayGather() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayGather");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "indices: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "value: dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dtype: type");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_dims_of_element_shape: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dims_of_element_shape: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "element_shape: shape = { unknown_rank: true }");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayGather op registration failed: ";
  }
}

void Register_ITEXTensorArrayPack() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayPack");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "value: dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dtype: type");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_dims_of_element_shape: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dims_of_element_shape: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "element_shape: shape = { unknown_rank: true }");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayPack op registration failed: ";
  }
}

void Register_ITEXTensorArrayScatter() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayScatter");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "indices: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "value: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "flow_out: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: type");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayScatter op registration failed: ";
  }
}

void Register_ITEXTensorArrayUnpack() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayUnpack");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "value: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "flow_out: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: type");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayUnpack op registration failed: ";
  }
}

void Register_ITEXTensorArrayConcat() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayConcat");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "value: dtype");
    TF_OpDefinitionBuilderAddOutput(op_builder, "lengths: int64");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dtype: type");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_dims_of_element_shape: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dims_of_element_shape: list(int) = []");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "element_shape_except0: shape = { unknown_rank: true }");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayConcat op registration failed: ";
  }
}

void Register_ITEXTensorArraySplit() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArraySplit");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "value: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "lengths: int64");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "flow_out: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: type");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArraySplit op registration failed: ";
  }
}

void Register_ITEXTensorArraySize() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArraySize");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "flow_in: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "size: int32");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArraySize op registration failed: ";
  }
}

void Register_ITEXTensorArrayClose() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTensorArrayClose");
    TF_OpDefinitionBuilderAddInput(op_builder, "handle: resource");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTensorArrayClose op registration failed: ";
  }
}

void Register_ITEXResizeBilinearOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXResizeBilinear");
    TF_OpDefinitionBuilderAddInput(op_builder, "images: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: int32");

    TF_OpDefinitionBuilderAddOutput(op_builder, "resized_images: float");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "align_corners: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "half_pixel_centers: bool = false");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXResizeBilinear op registration failed: ";
  }
}

void Register_ITEXResizeBilinearGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXResizeBilinearGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "grads: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "orginal_image: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "align_corners: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "half_pixel_centers: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXResizeBilinearGrad op registration failed: ";
  }
}

void Register_ITEXTransposeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXTranspose");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "perm: Tperm");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, half, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tperm: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXTranspose op registration failed: ";
  }
}

void Register_ITEXGroupNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ITEXGroupNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_groups: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_scale: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_center: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ITEXGroupNorm op registration failed: ";
  }
}

void Register_ITEXLayerNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ITEXLayerNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "layer_mean: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "layer_variance: U");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "data_format: { 'NHWC', 'NCHW'} = 'NHWC' ");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &layer_norm_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ITEXLayerNorm op registration failed: ";
  }
}

void Register_ITEXLayerNormGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ITEXLayerNormGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "y_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_1: U");
    TF_OpDefinitionBuilderAddInput(op_builder, "reserve_space_2: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "x_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "scale_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "offset_backprop: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
    TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_4: U");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "data_format: { 'NHWC', 'NCHW'} = 'NHWC' ");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(
        op_builder, &itex_layer_norm_grad_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ITEXLayerNormGrad op registration failed: ";
  }
}

void Register_ITEXMklLayerNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXMklLayerNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "offset: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_inplace: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXMklLayerNorm op registration failed: ";
  }
}

void Register_ITEXSliceOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXSlice");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "begin: Index");
    TF_OpDefinitionBuilderAddInput(op_builder, "size: Index");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Index: {int32,int64}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXSlice op registration failed: ";
  }
}

void Register_ITEXAccMatMul() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXAccMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Tout");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_BFLOAT16");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: {float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpost: {float, bfloat16} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "is_bf16_math_mode: bool = false");
    // TODO(itex): Implement matmul_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXAccMatMul op registration failed: ";
  }
}

void Register_ITEXFusedAccMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedAccMatMul");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Tpost");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Tout");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_BFLOAT16");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: {float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpost: {float, bfloat16} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "is_bf16_math_mode: bool = false");
    // TODO(itex): Implement matmul_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedAccMatMul op registration failed: ";
  }
}

void Register_ITEXFusedAccMatMulWithSumOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedAccMatMulWithSum");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * Tpost");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: Tout");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_BFLOAT16");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: {float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpost: {float, bfloat16} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "is_bf16_math_mode: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "inplace_sum: bool = false");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedAccMatMulWithSum op registration failed: ";
  }
}

void Register_ITEXQuantizedReshapeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedReshape");

    TF_OpDefinitionBuilderAddInput(op_builder, "tensor: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "shape: Tshape");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_min: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "input_max: float");
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
        << "_ITEXQuantizedReshape op registration failed: ";
  }
}

void Register_ITEXQuantizedTransposeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedTranspose");

    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "perm: Tperm");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
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
        << "_ITEXQuantizedTranspose op registration failed: ";
  }
}

void Register_ITEXFusedAccMatMulGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedAccMatMulGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: Tgrad");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16} = DT_BFLOAT16");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tgrad: {float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedAccMatMulGrad op registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2D op registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DAndRequantize op registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DPerChannelOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DPerChannel");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DPerChannel op registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DWithBias op registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DWithBiasAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DWithBiasAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DWithBiasAndRequantize op registration "
           "failed: ";
  }
}

void Register_ITEXQuantizedConv2DWithBiasAndReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DWithBiasAndRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DWithBiasAndRelu op registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DWithBiasAndReluAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_ITEXQuantizedConv2DWithBiasAndReluAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DWithBiasAndReluAndRequantize op registration "
           "failed: ";
  }
}

void Register_ITEXQuantizedConv2DWithBiasSumAndReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DWithBiasSumAndRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "summand: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DWithBiasSumAndRelu op registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DWithBiasSumAndReluAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_ITEXQuantizedConv2DWithBiasSumAndReluAndRequantize");
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
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DWithBiasSumAndReluAndRequantize op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedConv2DWithBiasSignedSumAndReluAndRequantize() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_ITEXQuantizedConv2DWithBiasSignedSumAndReluAndRequantize");
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
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedConv2DWithBiasSignedSumAndReluAndRequantize op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedDepthwiseConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedDepthwiseConv2D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedDepthwiseConv2D op registration failed: ";
  }
}

void Register_ITEXQuantizedDepthwiseConv2DWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedDepthwiseConv2DWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedDepthwiseConv2DWithBias op registration failed: ";
  }
}

void Register_ITEXQuantizedDepthwiseConv2DWithBiasAndReluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_ITEXQuantizedDepthwiseConv2DWithBiasAndRelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedDepthwiseConv2DWithBiasAndRelu op registration "
           "failed: ";
  }
}

void Register_ITEXQuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_ITEXQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedBatchMatMulV2AndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedBatchMatMulV2AndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_y: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_y: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: Toutput");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Toutput: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    // is_filter_const is default false, because two inputs of BMM INT8 maybe
    // data tensor, e.g. in Bert Model
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedBatchMatMulV2AndDequantize op "
           "registration failed: ";
  }
}

void Register_ITEXQuantizedFusedBatchMatMulV2AndDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(
        "_ITEXQuantizedFusedBatchMatMulV2AndDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T1");
    TF_OpDefinitionBuilderAddInput(op_builder, "y: T2");
    TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_x: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_y: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_y: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: Toutput");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T1: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T2: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T: {bfloat16, float} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Toutput: {bfloat16, float} = DT_FLOAT");
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
        << "_ITEXQuantizedFusedBatchMatMulV2AndDequantize op "
           "registration failed: ";
  }
}

void Register_ITEXGRUOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXGRUCell");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "h_prev: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_c: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "r: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "u: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "c: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "h: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "lbr: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "training: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXGRUCell op registration failed: ";
  }
}

void Register_ITEXAUGRUOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXAUGRUCell");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "h_prev: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "au_x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_c: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "r: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "u: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "c: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "h: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "lbr: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "training: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXAUGRUCell op registration failed: ";
  }
}

void Register_ITEXForwardGRUOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("MklGRU");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "h_prev: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_c: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "h_out: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "h_n: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "lbr: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "training: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "TimeDim: int >= 1");
    TF_OpDefinitionBuilderAddAttr(op_builder, "x_format: string = 'TNC'");

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "MklGRU op registration failed: ";
  }
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXForwardGRU");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "h_prev: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_c: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "h_out: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "h_n: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "lbr: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "training: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "TimeDim: int >= 1");
    TF_OpDefinitionBuilderAddAttr(op_builder, "x_format: string = 'TNC'");

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXForwardGRU op registration failed: ";
  }
}

void Register_ITEXForwardAUGRUOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("MklAUGRU");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "h_prev: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "au_x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_c: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "h_out: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "h_n: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "lbr: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "training: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "TimeDim: int >= 1");
    TF_OpDefinitionBuilderAddAttr(op_builder, "x_format: string = 'TNC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "au_format: string = 'TNC'");

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "MklAUGRU op registration failed: ";
  }
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXForwardAUGRU");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "h_prev: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "au_x: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "w_c: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_ru: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_c: T");

    TF_OpDefinitionBuilderAddOutput(op_builder, "h_out: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "h_n: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "lbr: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "training: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_filter_const: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "TimeDim: int >= 1");
    TF_OpDefinitionBuilderAddAttr(op_builder, "x_format: string = 'TNC'");
    TF_OpDefinitionBuilderAddAttr(op_builder, "au_format: string = 'TNC'");

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXForwardAUGRU op registration failed: ";
  }
}

void Register_QuantizedMaxPool3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedMaxPool3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());

    // TODO(itex): Implement maxpool_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_QuantizedMaxPool3D op registration failed: ";
  }
}

void Register_ITEXQuantizedMaxPool3DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedMaxPool3D");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "ksize: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());

    // TODO(itex): Implement maxpool_shape_fn in the future
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedMaxPool3D op registration failed: ";
  }
}

void Register_ITEXQuantizedMaxPoolOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedMaxPool");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizedMaxPool op registration failed: ";
  }
}

void Register_ITEXQuantizedAvgPoolOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    // TODO(itex) Change back to _ITEXQuantizedAvgPool when public resolve the
    // shape inference issue.
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ITEXQuantizedAvgPool");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "ITEXQuantizedAvgPool op registration failed: ";
  }
}

void Register_ITEXFusedQuantizeV2WithQuantizedConv2DOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizeV2WithQuantizedConv2D");
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
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");
    TF_OpDefinitionBuilderAddOutput(op_builder, "min_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "max_output: float");
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
        << "_ITEXQuantizeV2WithQuantizedConv2D op registration "
           "failed: ";
  }
}

void Register_ITEXFusedQuantizedConv2DWithDequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DWithDequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");

    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "out_type: {bfloat16, half, float} = DT_FLOAT");
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
        << "_ITEXQuantizedConv2DWithDequantize op registration "
           "failed: ";
  }
}

void Register_ITEXFusedQuantizedConv2DWithCastOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedConv2DWithCast");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tinput");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter: Tfilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: Tbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_input: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_filter: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "min_freezed_output: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "max_freezed_output: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_type");

    TF_OpDefinitionBuilderAddAttr(op_builder, "Tinput: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tfilter: quantizedtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tbias: {float, qint32}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_type: {bfloat16, half} = DT_HALF");
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
        << "_ITEXQuantizedConv2DWithCast op registration "
           "failed: ";
  }
}

void Register_ITEXPadWithConv2DBackpropFilterOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXPadWithConv2DBackpropFilter");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetPaddingAttrStringWithExplicit());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXPadWithConv2DBackpropFilter op registration failed: ";
  }
}

void Register_ITEXPadWithConv2DBackpropFilterWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXPadWithConv2DBackpropFilterWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetExplicitPaddingsAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXPadWithConv2DBackpropFilterWithBias op registration failed: ";
  }
}

void Register_ITEXPadWithConv3DBackpropFilterV2Op() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXPadWithConv3DBackpropFilterV2");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXPadWithConv3DBackpropFilterV2 op registration failed: ";
  }
}

void Register_ITEXPadWithConv3DBackpropFilterWithBiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXPadWithConv3DBackpropFilterWithBias");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "filter_sizes: int32");
    TF_OpDefinitionBuilderAddInput(op_builder, "out_backprop: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "paddings: Tpaddings");

    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_grad: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int) >= 5");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "dilations: list(int) = [1, 1, 1, 1, 1]");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tpaddings: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetPaddingAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  GetConvnet3dDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXPadWithConv3DBackpropFilterWithBias op registration failed: ";
  }
}

void Register_QuantizedMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedMatMul");

    // Variable number of inputs depending on fusion. The inputs contain
    // quantized or real tensors. Some of the inputs carry min-max values for
    // quantized tensors.
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    // Variable number of outputs depending on the main output type. For
    // example, quantized output will need additional tensors to carry min-max
    // values. If the output type is real tensor (e.g. Dequantize fusion), the
    // op should produce only single output tensor.
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Thost_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Thost_outputs: list(type) >= 0 = []");
    // The following attributes T1, T2, U, and Tout are members of Tinputs
    // and Toutputs, used here for type constraints in the templatized OpKernel
    // registrations.
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T1: quantizedtype");  // 0-th input
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T2: quantizedtype");  // 1st input
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Tbias: {bfloat16, float, quantizedtype} = DT_FLOAT");
    // Additional inputs' type. Currently, restricting all to be of same type.
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "U: {bfloat16, float, quantizedtype} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "Tout: {bfloat16, float, quantizedtype} = DT_FLOAT");  // 0-th output
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_weight_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    // Attribute for quantization mode of all quantized input tensors.
    // Currently restricting all operands using same quantization mode.
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    // Attribute for activation (0-th output) requnatization mode
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "output_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    // Attributes for the LeakyRelu ----------------------------------------- //
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    // ---------------------------------------------------------------------- //
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_QuantizedMatMul op registration failed: ";
  }
}

void Register_ITEXQuantizedMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedMatMul");

    // Variable number of inputs depending on fusion. The inputs contain
    // quantized or real tensors. Some of the inputs carry min-max values for
    // quantized tensors.
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    // Variable number of outputs depending on the main output type. For
    // example, quantized output will need additional tensors to carry min-max
    // values. If the output type is real tensor (e.g. Dequantize fusion), the
    // op should produce only single output tensor.
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Thost_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Thost_outputs: list(type) >= 0 = []");
    // The following attributes T1, T2, U, and Tout are members of Tinputs
    // and Toutputs, used here for type constraints in the templatized OpKernel
    // registrations.
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T1: quantizedtype");  // 0-th input
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T2: quantizedtype");  // 1st input
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "Tbias: {bfloat16, float, quantizedtype} = DT_FLOAT");
    // Additional inputs' type. Currently, restricting all to be of same type.
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "U: {bfloat16, float, quantizedtype} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "Tout: {bfloat16, float, quantizedtype} = DT_FLOAT");  // 0-th output
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_weight_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_bias_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    // Attribute for quantization mode of all quantized input tensors.
    // Currently restricting all operands using same quantization mode.
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    // Attribute for activation (0-th output) requnatization mode
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "output_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    // Attributes for the LeakyRelu ----------------------------------------- //
    TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
    // ---------------------------------------------------------------------- //
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedMatMul op registration failed: ";
  }
}

void Register_QuantizedFusedBatchNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedFusedBatchNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: input_types");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_types");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {qint8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: {float, qint8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "input_types: list(type)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "out_types: list(type)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "exponential_avg_factor: float = 1.0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "activation_mode: string = \"Identity\"");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_offset_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_mean_const: bool = true");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_QuantizedFusedBatchNorm op registration failed: ";
  }
}

void Register_ITEXQuantizedFusedBatchNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedFusedBatchNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: input_types");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: out_types");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {qint8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: {float, qint8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "input_types: list(type)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "out_types: list(type)");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "exponential_avg_factor: float = 1.0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "activation_mode: string = \"Identity\"");
    TF_OpDefinitionBuilderAddAttr(op_builder, GetConvnetDataFormatAttrString());
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_offset_const: bool = true");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_mean_const: bool = true");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedFusedBatchNorm op registration failed: ";
  }
}

void Register_QuantizedBatchMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_QuantizedBatchMatMul");

    // Variable number of inputs depending on fusion. The inputs contain
    // quantized or real tensors. Some of the inputs carry min-max values for
    // quantized tensors.
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    // Variable number of outputs depending on the main output type. For
    // example, quantized output will need additional tensors to carry min-max
    // values. If the output type is real tensor (e.g. Dequantize fusion), the
    // op should produce only single output tensor.
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Thost_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Thost_outputs: list(type) >= 0 = []");
    // The following attributes T1, T2, U, and Tout are members of Tinputs
    // and Toutputs, used here for type constraints in the templatized OpKernel
    // registrations.
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T1: quantizedtype");  // 0-th input
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T2: quantizedtype");  // 1st input
    // Additional inputs' type. Currently, restricting all to be of same type.
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "U: {bfloat16, float, quantizedtype} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "Tout: {bfloat16, float, quantizedtype} = DT_FLOAT");  // 0-th output
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    // Attribute for quantization mode of all quantized input tensors.
    // Currently restricting all operands using same quantization mode.
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    // Attribute for activation (0-th output) requnatization mode
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "output_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    // ---------------------------------------------------------------------- //
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_QuantizedBatchMatMul op registration failed: ";
  }
}

void Register_ITEXQuantizedBatchMatMulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXQuantizedBatchMatMul");

    // Variable number of inputs depending on fusion. The inputs contain
    // quantized or real tensors. Some of the inputs carry min-max values for
    // quantized tensors.
    TF_OpDefinitionBuilderAddInput(op_builder, "device_inputs: Tdevice_inputs");
    TF_OpDefinitionBuilderAddInput(op_builder, "host_inputs: Thost_inputs");
    // Variable number of outputs depending on the main output type. For
    // example, quantized output will need additional tensors to carry min-max
    // values. If the output type is real tensor (e.g. Dequantize fusion), the
    // op should produce only single output tensor.
    TF_OpDefinitionBuilderAddOutput(op_builder,
                                    "device_outputs: Tdevice_outputs");
    TF_OpDefinitionBuilderAddOutput(op_builder, "host_outputs: Thost_outputs");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Thost_inputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Tdevice_outputs: list(type) >= 0 = []");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "Thost_outputs: list(type) >= 0 = []");
    // The following attributes T1, T2, U, and Tout are members of Tinputs
    // and Toutputs, used here for type constraints in the templatized OpKernel
    // registrations.
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T1: quantizedtype");  // 0-th input
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "T2: quantizedtype");  // 1st input
    // Additional inputs' type. Currently, restricting all to be of same type.
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "U: {bfloat16, float, quantizedtype} = DT_FLOAT");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "Tout: {bfloat16, float, quantizedtype} = DT_FLOAT");  // 0-th output
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_x: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "adj_y: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    // Attribute for quantization mode of all quantized input tensors.
    // Currently restricting all operands using same quantization mode.
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    // Attribute for activation (0-th output) requnatization mode
    TF_OpDefinitionBuilderAddAttr(
        op_builder, "output_quant_mode: {'MIN_FIRST', 'SCALED'} = 'SCALED'");
    // ---------------------------------------------------------------------- //
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXQuantizedBatchMatMul op registration failed: ";
  }
}

void Register_SDPOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ScaledDotProductAttention");
    TF_OpDefinitionBuilderAddInput(op_builder, "query: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "key: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "value: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "atten_mask: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "dropout_mask: bool");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "atten: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "atten_dp: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_mask: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_dropout: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dropout_prob: float = 0.0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ScaledDotProductAttention op registration failed: ";
  }
}

void Register_SDPGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ScaledDotProductAttentionGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "query: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "key: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "value: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "dropout_mask: bool");
    TF_OpDefinitionBuilderAddInput(op_builder, "atten: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "atten_dp: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "output_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "query_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "key_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "value_backprop: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dropout_prob: float = 0.0");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ScaledDotProductAttentionGrad op registration failed: ";
  }
}

void Register_FusedDenseBiasAddGeluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("FusedDenseBiasAddGelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "weights: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "outputs: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "workspace: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "FusedDenseBiasAddGelu op registration failed: ";
  }
}

void Register_FusedDenseBiasAddGeluGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("FusedDenseBiasAddGeluGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "workspace: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "gradients: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "input_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "weights_backprop: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "bias_backprop: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "FusedDenseBiasAddGeluGrad op registration failed: ";
  }
}
