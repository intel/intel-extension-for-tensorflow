/* Copyright (c) 2023 Intel Corporation

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

void Register_Fp8QuantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("Fp8Quantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: in_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_scale: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: int8");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "in_dtype: {float, bfloat16, half}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8Quantize op registration failed: ";
  }
}

void Register_Fp8DequantizeOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("Fp8Dequantize");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "x_scale_inv: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "y: out_dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_dtype: {float, bfloat16, half}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8Dequantize op registration failed: ";
  }
}

void Register_Fp8LayerNormOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("Fp8LayerNorm");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: in_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "gamma: weight_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta: weight_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "z_amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "z_scale: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "z: out_dtype");
    TF_OpDefinitionBuilderAddOutput(op_builder, "mu: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "rsigma: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "in_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "weight_dtype: {float, bfloat16, half}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8LayerNorm op registration failed: ";
  }
}

void Register_Fp8LayerNormGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("Fp8LayerNormGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "dz: grad_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "x: in_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "mu: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "rsigma: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "gamma: weight_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "dz_scale_inv: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dx: out_dtype");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dgamma: weight_dtype");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dbeta: weight_dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "in_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "weight_dtype: {float, bfloat16, half}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "grad_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8LayerNormGrad op registration failed: ";
  }
}

void Register_Fp8GeluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("Fp8Gelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "inp: in_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "gelu_out: int8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "in_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8Gelu op registration failed: ";
  }
}

void Register_Fp8QuantizeDbiasOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("Fp8QuantizeDbias");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: grad_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "quantize_out: int8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dbias: grad_dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "grad_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8QuantizeDbias op registration failed: ";
  }
}

void Register_Fp8QuantizeDbiasDgeluOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("Fp8QuantizeDbiasDgelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: grad_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "gelu_inp: in_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "scale: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dgelu: int8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dbias: grad_dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "in_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "grad_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8QuantizeDbiasDgelu op registration failed: ";
  }
}
