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

#ifndef ITEX_CORE_OPS_OP_INIT_H_
#define ITEX_CORE_OPS_OP_INIT_H_

// Op definition is different between Intel TF and public TF.
void Register_ITEXQuantizeV2Op();
void Register_ITEXQuantizedMatMulWithBiasAndDequantizeOp();

// Training kernels
void Register_ApplyAdamWithWeightDecayOp();
void Register_FusedApplyAdamOp();
void Register_FusedApplyAdamWithWeightDecayOp();
void Register_FusedApplyMomentumOp();
void Register_FusedResourceApplyAdamOp();
void Register_FusedResourceApplyAdamWithWeightDecayOp();
void Register_FusedResourceApplyMomentumOp();
void Register_ResourceApplyAdamWithWeightDecayOp();

// Unupstreamed ops. These ops are only available in spr-base branch, not in
// TF master.
void Register_QuantizedBatchMatMulOp();
void Register_QuantizedBatchMatMulV2AndDequantizeOp();
void Register_QuantizedFusedBatchMatMulV2AndDequantizeOp();
void Register_QuantizedFusedMatMulOp();
void Register_QuantizedFusedMatMulAndRequantizeOp();
void Register_QuantizedFusedMatMulAndDequantizeOp();
void Register_QuantizedMaxPool3DOp();
void Register_QuantizedTransposeOp();

// TODO(itex): remove this op definition, once this op is upstreamed from
// spr-base to public TF
// V2 means Conv INT8 new API for CPU
void Register_QuantizedConv2DV2Op();
void Register_QuantizedConv3DV2Op();
void Register_QuantizedDepthwiseConv2DV2Op();
void Register_QuantizedMatMulOp();
void Register_QuantizedFusedBatchNormOp();

// Custom kernels
void Register_Conv2DBackpropFilterWithBiasOp();
void Register_Conv2DBackpropInputWithSliceOp();
void Register_Conv3DBackpropFilterWithBiasOp();
void Register_Conv3DBackpropInputV2WithSliceOp();
void Register_FusedBatchNormExGradOp();
void Register_FusedBatchMatMulV2Op();
void Register_FusedConv2DWithSumOp();
void Register_FusedDequantizeWithReshapeOp();
void Register_FusedMatMulGradOp();
void Register_FusedMatMulWithSumOp();
void Register_FusedInstanceNormOp();
void Register_GeluOp();
void Register_GeluGradOp();
void Register_InstanceNormOp();
// There are similar ops called "_FusedConv2D" or in "_FusedMatMul" TF-Proper.
// We use such custom ops in ITEX to enable more features.
void Register_ITEXFusedConv2DOp();
void Register_ITEXFusedConv3DOp();
void Register_ITEXFusedDepthwiseConv2dNativeOp();
void Register_ITEXFusedMatMulOp();
void Register_ITEXFusedQuantizeV2WithQuantizedConv2DOp();
void Register_ITEXFusedBinaryOp();
void Register_ITEXRandomUniformOp();
void Register_ITEXFusedAddV2WithSoftmaxOp();
void Register_LayerNormOp();
void Register_LayerNormGradOp();
void Register_ITEXRnnOp();
void Register_ITEXRnnGradOp();
void Register_OneDnnGraphOp();
void Register_PadWithConv2DOp();
void Register_PadWithConv3DOp();
void Register_PadWithFusedConv2DOp();
void Register_PadWithFusedConv3DOp();
void RegisterRMSPropComputeRMSOp();
void RegisterRMSPropVarUpdateOp();
void Register_SwishOp();

// Native kernels
void Register_ITEXAddNOp();
void Register_ITEXAUGRUOp();
void Register_ITEXAvgPoolOp();
void Register_ITEXAvgPoolGradOp();
void Register_ITEXAvgPool3DOp();
void Register_ITEXAvgPool3DGradOp();
void Register_ITEXBatchMatMulOp();
void Register_ITEXBatchMatMulV2Op();
void Register_ITEXCastOp();
void Register_ITEXConv2DBackpropFilterOp();
void Register_ITEXConv2DBackpropFilterWithBiasOp();
void Register_ITEXConv2DBackpropInputOp();
void Register_ITEXConv2DBackpropInputWithSliceOp();
void Register_ITEXConv2DOp();
void Register_ITEXConv3DBackpropFilterV2Op();
void Register_ITEXConv3DBackpropFilterWithBiasOp();
void Register_ITEXConv3DBackpropInputOp();
void Register_ITEXConv3DBackpropInputV2Op();
void Register_ITEXConv3DBackpropInputV2WithSliceOp();
void Register_ITEXConv3DOp();
void Register_ITEXDepthwiseConv2dNativeBackpropFilterOp();
void Register_ITEXDepthwiseConv2dNativeBackpropInputOp();
void Register_ITEXDepthwiseConv2dNativeOp();
void Register_ITEXDequantizeOp();
void Register_ITEXEluGradOp();
void Register_ITEXEluOp();
void Register_ITEXForwardAUGRUOp();
void Register_ITEXForwardGRUOp();
void Register_ITEXFusedBatchMatMulV2Op();
void Register_ITEXFusedBatchNormExOp();
void Register_ITEXFusedBatchNormExGradOp();
void Register_ITEXFusedBatchNormGradOp();
void Register_ITEXFusedBatchNormGradV2Op();
void Register_ITEXFusedBatchNormGradV3Op();
void Register_ITEXFusedBatchNormOp();
void Register_ITEXFusedBatchNormV2Op();
void Register_ITEXFusedBatchNormV3Op();
void Register_ITEXFusedConv2DWithSumOp();
void Register_ITEXFusedMatMulWithSumOp();
void Register_ITEXFusedInstanceNormOp();
void Register_ITEXGeluGradOp();
void Register_ITEXGeluOp();
void Register_ITEXGRUOp();
void Register_ITEXInstanceNormOp();
void Register_ITEXLayerNormOp();
void Register_ITEXLayerNormGradOp();
void Register_ITEXLeakyReluGradOp();
void Register_ITEXLeakyReluOp();
void Register_ITEXMatMul();
void Register_ITEXMaxPool3DGradOp();
void Register_ITEXMaxPool3DOp();
void Register_ITEXMaxPoolGradOp();
void Register_ITEXMaxPoolOp();
void Register_ITEXMklLayerNormOp();
void Register_ITEXPadWithConv2DOp();
void Register_ITEXPadWithConv3DOp();
void Register_ITEXPadWithFusedConv2DOp();
void Register_ITEXPadWithFusedConv3DOp();
void Register_ITEXPadWithConv2DBackpropFilterOp();
void Register_ITEXPadWithConv2DBackpropFilterWithBiasOp();
void Register_ITEXPadWithConv3DBackpropFilterV2Op();
void Register_ITEXPadWithConv3DBackpropFilterWithBiasOp();
void Register_ITEXQuantizedAvgPoolOp();
void Register_ITEXQuantizedBatchMatMulOp();
void Register_ITEXQuantizedBatchMatMulV2AndDequantizeOp();
void Register_ITEXQuantizedFusedBatchMatMulV2AndDequantizeOp();
void Register_ITEXQuantizedFusedBatchNormOp();
void Register_ITEXQuantizedFusedMatMulOp();
void Register_ITEXQuantizedFusedMatMulAndDequantizeOp();
void Register_ITEXQuantizedFusedMatMulAndRequantizeOp();
void Register_ITEXQuantizedMatMulWithBiasOp();
void Register_ITEXQuantizedMatMulWithBiasAndReluOp();
void Register_ITEXQuantizedMatMulWithBiasAndReluAndRequantizeOp();
void Register_ITEXQuantizedMatMulWithBiasAndRequantizeOp();
void Register_ITEXQuantizedMaxPoolOp();
void Register_ITEXQuantizedMaxPool3DOp();
void Register_ITEXQuantizedReshapeOp();
void Register_ITEXQuantizedTransposeOp();
void Register_ITEXQuantizedConv2DOp();
void Register_ITEXQuantizedConv2DAndRequantizeOp();
void Register_ITEXQuantizedConv2DPerChannelOp();
void Register_ITEXQuantizedConv2DWithBiasOp();
void Register_ITEXQuantizedConv2DWithBiasAndRequantizeOp();
void Register_ITEXQuantizedConv2DWithBiasAndReluOp();
void Register_ITEXQuantizedConv2DWithBiasAndReluAndRequantizeOp();
void Register_ITEXQuantizedConv2DWithBiasSumAndReluOp();
void Register_ITEXQuantizedConv2DWithBiasSumAndReluAndRequantizeOp();
void Register_ITEXQuantizedConv2DWithBiasSignedSumAndReluAndRequantize();
void Register_ITEXQuantizedDepthwiseConv2DWithBiasAndReluOp();
void Register_ITEXQuantizedDepthwiseConv2DWithBiasOp();
void Register_ITEXQuantizedDepthwiseConv2DOp();
void Register_ITEXQuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOp();

void Register_ITEXQuantizedConcatV2Op();
void Register_ITEXQuantizedConv2DV2Op();
void Register_ITEXQuantizedConv3DV2Op();
void Register_ITEXQuantizedDepthwiseConv2DV2Op();
void Register_ITEXQuantizedMatMulOp();

void Register_ITEXRelu6GradOp();
void Register_ITEXRelu6Op();
void Register_ITEXReluGradOp();
void Register_ITEXReluOp();
void Register_ITEXResizeBilinearOp();
void Register_ITEXResizeBilinearGradOp();
void Register_ITEXSliceOp();
void Register_ITEXSoftmaxOp();
void Register_ITEXSwishOp();
void Register_ITEXTransposeOp();

// BF32 native kernels
void Register_ITEXAccMatMul();
void Register_ITEXFusedAccMatMulOp();
void Register_ITEXFusedAccMatMulGradOp();
void Register_ITEXFusedAccMatMulWithSumOp();

// OneDnn kernels
void Register_OneDnnAddNOp();
void Register_OneDnnAvgPoolOp();
void Register_OneDnnAvgPoolGradOp();
void Register_OneDnnAvgPool3DOp();
void Register_OneDnnAvgPool3DGradOp();
void Register_OneDnnBatchMatMulV2Op();
void Register_OneDnnConcatOp();
void Register_OneDnnConcatV2Op();
void Register_OneDnnConv2DBackpropFilterOp();
void Register_OneDnnConv2DBackpropFilterWithBiasOp();
void Register_OneDnnConv2DBackpropInputOp();
void Register_OneDnnConv2DBackpropInputWithSliceOp();
void Register_OneDnnConv2DOp();
void Register_OneDnnConv3DBackpropFilterV2Op();
void Register_OneDnnConv3DBackpropFilterWithBiasOp();
void Register_OneDnnConv3DBackpropInputV2Op();
void Register_OneDnnConv3DBackpropInputV2WithSliceOp();
void Register_OneDnnConv3DOp();
void Register_OneDnnDepthwiseConv2dNativeBackpropFilterOp();
void Register_OneDnnDepthwiseConv2dNativeBackpropInputOp();
void Register_OneDnnDepthwiseConv2dNativeOp();
void Register_OneDnnDequantizeOp();
void Register_OneDnnFusedDequantizeWithReshapeOp();
void Register_OneDnnFusedBatchMatMulV2Op();
void Register_OneDnnFusedBatchNormOp();
void Register_OneDnnFusedBatchNormV2Op();
void Register_OneDnnFusedBatchNormV3Op();
void Register_OneDnnFusedBatchNormExOp();
void Register_OneDnnFusedBatchNormGradOp();
void Register_OneDnnFusedBatchNormGradV2Op();
void Register_OneDnnFusedBatchNormGradV3Op();
void Register_OneDnnFusedBatchNormExGradOp();
void Register_OneDnnFusedConv2DOp();
void Register_OneDnnFusedConv3DOp();
void Register_OneDnnFusedDepthwiseConv2dNativeOp();
void Register_OneDnnFusedInstanceNormOp();
void Register_OneDnnFusedMatMulOp();
void Register_OneDnnFusedMatMulGradOp();
void Register_OneDnnGeluOp();
void Register_OneDnnGeluGradOp();
void Register_OneDnnIdentityOp();
void Register_OneDnnInstanceNormOp();
void Register_OneDnnLayerNormOp();
void Register_OneDnnLayerNormGradOp();
void Register_OneDnnLeakyReluOp();
void Register_OneDnnLeakyReluGradOp();
void Register_OneDnnMatMulOp();
void Register_OneDnnMaxPoolOp();
void Register_OneDnnMaxPoolGradOp();
void Register_OneDnnMaxPool3DOp();
void Register_OneDnnMaxPool3DGradOp();
void Register_OneDnnMklLayerNormOp();
void Register_OneDnnQuantizedAvgPoolOp();
void Register_OneDnnQuantizedBatchMatMulV2AndDequantizeOp();
void Register_OneDnnQuantizedConcatV2Op();
void Register_OneDnnQuantizedConv2DOp();
void Register_OneDnnQuantizedConv2DAndRequantizeOp();
void Register_OneDnnQuantizedConv2DWithBiasOp();
void Register_OneDnnQuantizedConv2DWithBiasAndRequantizeOp();
void Register_OneDnnQuantizedConv2DWithBiasAndReluOp();
void Register_OneDnnQuantizedConv2DWithBiasAndReluAndRequantizeOp();
void Register_OneDnnQuantizedConv2DWithBiasSumAndReluOp();
void Register_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantizeOp();
void Register_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize();
void Register_OneDnnQuantizedDepthwiseConv2DOp();
void Register_OneDnnQuantizedDepthwiseConv2DWithBiasOp();
void Register_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluOp();
void Register_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOp();
void Register_OneDnnQuantizedFusedBatchMatMulV2AndDequantizeOp();
void Register_OneDnnQuantizedFusedMatMulOp();
void Register_OneDnnQuantizedFusedMatMulAndRequantizeOp();
void Register_OneDnnQuantizedFusedMatMulAndDequantizeOp();
void Register_OneDnnQuantizedMatMulWithBiasAndReluOp();
void Register_OneDnnQuantizedMatMulWithBiasOp();
void Register_OneDnnQuantizedMatMulWithBiasAndReluAndRequantizeOp();
void Register_OneDnnQuantizedMatMulWithBiasAndRequantizeOp();
void Register_OneDnnQuantizedMatMulWithBiasAndDequantizeOp();
void Register_OneDnnQuantizedMaxPoolOp();
void Register_OneDnnQuantizedReshapeOp();
void Register_OneDnnQuantizedTransposeOp();
void Register_OneDnnQuantizeV2Op();
void Register_OneDnnQuantizeV2WithQuantizedConv2DOp();
void Register_OneDnnPadWithConv2DOp();
void Register_OneDnnPadWithConv3DOp();
void Register_OneDnnPadWithFusedConv2DOp();
void Register_OneDnnPadWithFusedConv3DOp();
void Register_OneDnnReluOp();
void Register_OneDnnReluGradOp();
void Register_OneDnnReshapeOp();
void Register_OneDnnResizeBilinearOp();
void Register_OneDnnResizeBilinearGradOp();
void Register_OneDnnResizeNearestNeighborOp();
void Register_OneDnnResizeNearestNeighborGradOp();
void Register_OneDnnShapeOp();
void Register_OneDnnSliceOp();
void Register_OneDnnSoftmaxOp();
void Register_OneDnnSwishOp();
void Register_OneDnnToTfOp();
void Register_OneDnnTransposeOp();

// Math ops
void Register_EqualWithCastOp();
void Register_FusedAddNOp();
void Register_FusedRandomOP();
void Register_GreaterEqualWithCastOp();
void Register_GreaterWithCastOp();
void Register_LessEqualWithCastOp();
void Register_LessWithCastOp();
void Register_NotEqualWithCastOp();

// OneDnn math kernels
void Register_OneDnnAddOp();
void Register_OneDnnAddV2Op();
void Register_OneDnnCastOp();
void Register_OneDnnMulOp();
void Register_OneDnnSubOp();

#ifdef __cplusplus
extern "C" {
#endif
void RegisterOps();
#ifdef __cplusplus
}
#endif
#endif  // ITEX_CORE_OPS_OP_INIT_H_
