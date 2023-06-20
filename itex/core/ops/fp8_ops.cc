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

void Register_Fp8MatmulOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("Fp8Matmul");
    TF_OpDefinitionBuilderAddInput(op_builder, "src: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "weight: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "bias: sum_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "post_add: sum_dtype");
    TF_OpDefinitionBuilderAddInput(op_builder, "a_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "b_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "c_amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "c_scale: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dst: out_dtype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_a: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_b: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_c: int = -1");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_bias: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "has_post_add: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "sum_dtype: {float, bfloat16, half}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "out_dtype: {float, bfloat16, half, int8}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype_a: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype_b: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "fp8_dtype_c: {'E4M3', 'E5M2', ''} = ''");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8Matmul op registration failed: ";
  }
}

void Register_Fp8ScaledDotProductAttentionOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("Fp8ScaledDotProductAttention");
    TF_OpDefinitionBuilderAddInput(op_builder, "query: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "key: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "value: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "qk_scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "attention_mask: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "dropout_mask: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "q_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "k_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "v_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "attn_amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "attn_scale: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "attn_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "z_amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "z_scale: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "z: int8");
    TF_OpDefinitionBuilderAddOutput(op_builder, "softmax: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "attn: int8");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16, half}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dropout_prob: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_q: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_k: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_v: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_attn: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_z: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_dtype: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8ScaledDotProductAttention op registration failed: ";
  }
}

void Register_Fp8ScaledDotProductAttentionGradOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("Fp8ScaledDotProductAttentionGrad");
    TF_OpDefinitionBuilderAddInput(op_builder, "dz: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "query: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "key: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "value: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "qk_scale: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "softmax: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "dropout_mask: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "attn: int8");
    TF_OpDefinitionBuilderAddInput(op_builder, "dz_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "attn_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "q_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "k_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "v_scale_inv: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "dp_amax: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "dp_scale: float");
    TF_OpDefinitionBuilderAddInput(op_builder, "dp_scale_inv: float");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dq: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dk: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "dv: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16, half}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "dropout_prob: float");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_dz: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_attn: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_q: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_k: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_v: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fp8_meta_index_dp: int");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "fp8_dtype_forward: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "fp8_dtype_backward: {'E4M3', 'E5M2'}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Fp8ScaledDotProductAttentionGrad op registration failed: ";
  }
}
