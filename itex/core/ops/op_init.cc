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

#include "itex/core/ops/op_init.h"

#include <functional>
#include <map>
#include <string>

#include "itex/core/ops/utils/logging.h"
#include "protos/op_def.pb.h"
#include "tensorflow/c/c_api.h"

// Some ops currently are available only in spr-base branch, not in TF master
// branch. We will register those ops in ITEX, before they are upstreamed to TF
// public.
void Register_TFLegacyOp() {
  // Get all ops registered in TF Proper, before ITEX op registration
  TF_Buffer* op_list_buffer = TF_GetAllOpList();
  itex::OpList op_list;
  op_list.ParseFromArray(op_list_buffer->data, op_list_buffer->length);

  std::map<std::string, std::function<void()>> op_register_map = {
      {"_QuantizedBatchMatMul", Register_QuantizedBatchMatMulOp},
      {"_QuantizedBatchMatMulV2AndDequantize",
       Register_QuantizedBatchMatMulV2AndDequantizeOp},
      {"_QuantizedFusedBatchMatMulV2AndDequantize",
       Register_QuantizedFusedBatchMatMulV2AndDequantizeOp},
      {"_QuantizedFusedBatchNorm", Register_QuantizedFusedBatchNormOp},
      {"_QuantizedFusedMatMul", Register_QuantizedFusedMatMulOp},
      {"_QuantizedFusedMatMulAndRequantize",
       Register_QuantizedFusedMatMulAndRequantizeOp},
      {"_QuantizedFusedMatMulAndDequantize",
       Register_QuantizedFusedMatMulAndDequantizeOp},
      {"_QuantizedMatMul", Register_QuantizedMatMulOp},
      {"_QuantizedMaxPool3D", Register_QuantizedMaxPool3DOp},
      {"_QuantizedTranspose", Register_QuantizedTransposeOp}};

  for (auto register_pair : op_register_map) {
    if (std::find_if(op_list.op().begin(), op_list.op().end(), [=](auto op) {
          return op.name() == register_pair.first;
        }) == op_list.op().end()) {
      auto register_func = register_pair.second;
      register_func();
    } else {
      ITEX_LOG(ERROR) << "Op: " << register_pair.first
                      << " is already registered in Tensorflow";
    }
  }

  TF_DeleteBuffer(op_list_buffer);
}

void RegisterOps() {
  // Ops currently are available only in spr-base branch
  Register_TFLegacyOp();

  // Op definition is different between Intel TF and public TF.
  Register_ITEXQuantizeV2Op();
  Register_ITEXQuantizedMatMulWithBiasAndDequantizeOp();

  // Training kernels
  Register_ApplyAdamWithWeightDecayOp();
  Register_ResourceApplyAdamWithWeightDecayOp();
  Register_FusedApplyMomentumOp();
  Register_FusedResourceApplyMomentumOp();
  Register_FusedApplyAdamOp();
  Register_FusedResourceApplyAdamOp();
  Register_FusedApplyAdamWithWeightDecayOp();
  Register_FusedResourceApplyAdamWithWeightDecayOp();

  Register_QuantizedConv2DV2Op();
  Register_QuantizedConv3DV2Op();
  Register_QuantizedDepthwiseConv2DV2Op();

  // Custom kernels
  Register_ITEXFusedAddV2WithSoftmaxOp();
  Register_ITEXTensorArray();
  Register_ITEXTensorArrayGrad();
  Register_ITEXTensorArrayGradWithShape();
  Register_ITEXTensorArrayWrite();
  Register_ITEXTensorArrayRead();
  Register_ITEXTensorArrayGather();
  Register_ITEXTensorArrayPack();
  Register_ITEXTensorArrayUnpack();
  Register_ITEXTensorArrayScatter();
  Register_ITEXTensorArrayConcat();
  Register_ITEXTensorArraySplit();
  Register_ITEXTensorArraySize();
  Register_ITEXTensorArrayClose();
  Register_InstanceNormOp();
  Register_GeluOp();
  Register_GeluGradOp();
  Register_ITEXConv2DBackpropFilterWithBiasOp();
  Register_ITEXConv2DBackpropInputWithSliceOp();
  Register_ITEXConv3DBackpropFilterWithBiasOp();
  Register_ITEXConv3DBackpropInputV2WithSliceOp();
  Register_ITEXFusedBatchNormGradExOp();
  Register_ITEXFusedBatchMatMulV2Op();
  Register_ITEXFusedConv2DOp();
  Register_ITEXFusedConv2DWithSumOp();
  Register_ITEXFusedConv3DOp();
  Register_ITEXFusedDepthwiseConv2dNativeOp();
  Register_ITEXFusedDequantizeWithReshapeOp();
  Register_ITEXFusedInstanceNormOp();
  Register_ITEXFusedMatMulOp();
  Register_ITEXFusedMatMulGradOp();
  Register_ITEXFusedMatMulWithSumOp();
  Register_ITEXFusedQuantizeV2WithQuantizedConv2DOp();
  Register_ITEXFusedQuantizedConv2DWithDequantizeOp();
  Register_ITEXFusedQuantizedConv2DWithCastOp();
  Register_ITEXFusedBinaryOp();
  Register_ITEXMishOp();
  Register_ITEXRandomUniformOp();
  Register_LayerNormOp();
  Register_LayerNormGradOp();
  Register_MishOp();
  Register_ITEXRnnOp();
  Register_ITEXRnnGradOp();
  Register_OneDnnGraphOp();
  Register_PadWithConv2DOp();
  Register_PadWithConv3DOp();
  Register_PadWithFusedConv2DOp();
  Register_PadWithFusedConv3DOp();
  RegisterRMSPropComputeRMSOp();
  RegisterRMSPropVarUpdateOp();
  Register_SwishOp();

  // Native kernels
  Register_ITEXAddNOp();
  Register_ITEXAUGRUOp();
  Register_ITEXAvgPoolOp();
  Register_ITEXAvgPoolGradOp();
  Register_ITEXAvgPool3DOp();
  Register_ITEXAvgPool3DGradOp();
  Register_ITEXBatchMatMulOp();
  Register_ITEXBatchMatMulV2Op();
  Register_ITEXCastOp();
  Register_ITEXConv2DBackpropFilterOp();
  Register_ITEXConv2DBackpropInputOp();
  Register_ITEXConv2DOp();
  Register_ITEXConv3DBackpropFilterV2Op();
  Register_ITEXConv3DBackpropInputOp();
  Register_ITEXConv3DBackpropInputV2Op();
  Register_ITEXConv3DOp();
  Register_ITEXDepthwiseConv2dNativeBackpropFilterOp();
  Register_ITEXDepthwiseConv2dNativeBackpropInputOp();
  Register_ITEXDepthwiseConv2dNativeOp();
  Register_ITEXDequantizeOp();
  Register_ITEXEinsum();
  Register_ITEXEluGradOp();
  Register_ITEXEluOp();
  Register_ITEXForwardAUGRUOp();
  Register_ITEXForwardGRUOp();
  Register_ITEXFusedBatchNormExOp();
  Register_ITEXFusedBatchNormGradOp();
  Register_ITEXFusedBatchNormGradV2Op();
  Register_ITEXFusedBatchNormGradV3Op();
  Register_ITEXFusedBatchNormOp();
  Register_ITEXFusedBatchNormV2Op();
  Register_ITEXFusedBatchNormV3Op();
  Register_ITEXGeluGradOp();
  Register_ITEXGeluOp();
  Register_ITEXGRUOp();
  Register_ITEXInstanceNormOp();
  Register_ITEXLayerNormOp();
  Register_ITEXLayerNormGradOp();
  Register_ITEXLeakyReluGradOp();
  Register_ITEXLeakyReluOp();
  Register_ITEXMatMul();
  Register_ITEXMaxPool3DGradOp();
  Register_ITEXMaxPool3DOp();
  Register_ITEXMaxPoolGradOp();
  Register_ITEXMaxPoolOp();
  Register_ITEXMklLayerNormOp();
  Register_ITEXPadWithConv2DOp();
  Register_ITEXPadWithConv3DOp();
  Register_ITEXPadWithFusedConv2DOp();
  Register_ITEXPadWithFusedConv3DOp();
  Register_ITEXPadWithConv2DBackpropFilterOp();
  Register_ITEXPadWithConv2DBackpropFilterWithBiasOp();
  Register_ITEXPadWithConv3DBackpropFilterV2Op();
  Register_ITEXPadWithConv3DBackpropFilterWithBiasOp();
  Register_ITEXQuantizedAvgPoolOp();
  Register_ITEXQuantizedBatchMatMulOp();
  Register_ITEXQuantizedBatchMatMulV2AndDequantizeOp();
  Register_ITEXQuantizedFusedBatchMatMulV2AndDequantizeOp();
  Register_ITEXQuantizedFusedBatchNormOp();
  Register_ITEXQuantizedFusedMatMulOp();
  Register_ITEXQuantizedFusedMatMulAndDequantizeOp();
  Register_ITEXQuantizedFusedMatMulAndRequantizeOp();
  Register_ITEXQuantizedMatMulWithBiasOp();
  Register_ITEXQuantizedMatMulWithBiasAndReluOp();
  Register_ITEXQuantizedMatMulWithBiasAndReluAndRequantizeOp();
  Register_ITEXQuantizedMatMulWithBiasAndRequantizeOp();
  Register_ITEXQuantizedMaxPoolOp();
  Register_ITEXQuantizedMaxPool3DOp();
  Register_ITEXQuantizedReshapeOp();
  Register_ITEXQuantizedTransposeOp();
  Register_ITEXQuantizedConv2DOp();
  Register_ITEXQuantizedConv2DAndRequantizeOp();
  Register_ITEXQuantizedConv2DPerChannelOp();
  Register_ITEXQuantizedConv2DWithBiasOp();
  Register_ITEXQuantizedConv2DWithBiasAndRequantizeOp();
  Register_ITEXQuantizedConv2DWithBiasAndReluOp();
  Register_ITEXQuantizedConv2DWithBiasAndReluAndRequantizeOp();
  Register_ITEXQuantizedConv2DWithBiasSumAndReluOp();
  Register_ITEXQuantizedConv2DWithBiasSumAndReluAndRequantizeOp();
  Register_ITEXQuantizedConv2DWithBiasSignedSumAndReluAndRequantize();
  Register_ITEXQuantizedDepthwiseConv2DWithBiasAndReluOp();
  Register_ITEXQuantizedDepthwiseConv2DWithBiasOp();
  Register_ITEXQuantizedDepthwiseConv2DOp();
  Register_ITEXQuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOp();
  Register_ITEXRelu6GradOp();
  Register_ITEXRelu6Op();
  Register_ITEXReluGradOp();
  Register_ITEXReluOp();
  Register_ITEXResizeBilinearOp();
  Register_ITEXResizeBilinearGradOp();
  Register_ITEXSliceOp();
  Register_ITEXSoftmaxOp();
  Register_ITEXSwishOp();
  Register_ITEXTransposeOp();

  Register_ITEXQuantizedConcatV2Op();
  Register_ITEXQuantizedConv2DV2Op();
  Register_ITEXQuantizedConv3DV2Op();
  Register_ITEXQuantizedDepthwiseConv2DV2Op();
  Register_ITEXQuantizedMatMulOp();

  // BF32 native kernels
  Register_ITEXAccMatMul();
  Register_ITEXFusedAccMatMulOp();
  Register_ITEXFusedAccMatMulGradOp();
  Register_ITEXFusedAccMatMulWithSumOp();

  // OneDnn kernels
  Register_OneDnnAddNOp();
  Register_OneDnnAvgPoolOp();
  Register_OneDnnAvgPoolGradOp();
  Register_OneDnnAvgPool3DOp();
  Register_OneDnnAvgPool3DGradOp();
  Register_OneDnnBatchMatMulV2Op();
  Register_OneDnnConcatOp();
  Register_OneDnnConcatV2Op();
  Register_OneDnnQuantizedConcatV2Op();
  Register_OneDnnConv2DBackpropFilterOp();
  Register_OneDnnConv2DBackpropFilterWithBiasOp();
  Register_OneDnnConv2DBackpropInputOp();
  Register_OneDnnConv2DBackpropInputWithSliceOp();
  Register_OneDnnConv2DOp();
  Register_OneDnnConv3DBackpropFilterV2Op();
  Register_OneDnnConv3DBackpropFilterWithBiasOp();
  Register_OneDnnConv3DBackpropInputV2Op();
  Register_OneDnnConv3DBackpropInputV2WithSliceOp();
  Register_OneDnnConv3DOp();
  Register_OneDnnDepthwiseConv2dNativeBackpropFilterOp();
  Register_OneDnnDepthwiseConv2dNativeBackpropInputOp();
  Register_OneDnnDepthwiseConv2dNativeOp();
  Register_OneDnnDequantizeOp();
  Register_OneDnnFusedDequantizeWithReshapeOp();
  Register_OneDnnFusedBatchMatMulV2Op();
  Register_OneDnnFusedBatchNormOp();
  Register_OneDnnFusedBatchNormV2Op();
  Register_OneDnnFusedBatchNormV3Op();
  Register_OneDnnFusedBatchNormExOp();
  Register_OneDnnFusedBatchNormGradOp();
  Register_OneDnnFusedBatchNormGradV2Op();
  Register_OneDnnFusedBatchNormGradV3Op();
  Register_OneDnnFusedBatchNormGradExOp();
  Register_OneDnnFusedConv2DOp();
  Register_OneDnnFusedConv3DOp();
  Register_OneDnnFusedDepthwiseConv2dNativeOp();
  Register_OneDnnFusedMatMulOp();
  Register_OneDnnFusedMatMulGradOp();
  Register_OneDnnFusedInstanceNormOp();
  Register_OneDnnInstanceNormOp();
  Register_OneDnnGeluOp();
  Register_OneDnnGeluGradOp();
  Register_OneDnnIdentityOp();
  Register_OneDnnLayerNormOp();
  Register_OneDnnLayerNormGradOp();
  Register_OneDnnLeakyReluOp();
  Register_OneDnnLeakyReluGradOp();
  Register_OneDnnMatMulOp();
  Register_OneDnnMaxPoolOp();
  Register_OneDnnMaxPoolGradOp();
  Register_OneDnnMaxPool3DOp();
  Register_OneDnnMaxPool3DGradOp();
  Register_OneDnnMishOp();
  Register_OneDnnMklLayerNormOp();
  Register_OneDnnQuantizedBatchMatMulV2AndDequantizeOp();
  Register_OneDnnQuantizedConv2DOp();
  Register_OneDnnQuantizedConv2DAndRequantizeOp();
  Register_OneDnnQuantizedConv2DWithBiasOp();
  Register_OneDnnQuantizedConv2DWithBiasAndRequantizeOp();
  Register_OneDnnQuantizedConv2DWithBiasAndReluOp();
  Register_OneDnnQuantizedConv2DWithBiasAndReluAndRequantizeOp();
  Register_OneDnnQuantizedConv2DWithBiasSumAndReluOp();
  Register_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantizeOp();
  Register_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize();
  Register_OneDnnQuantizedDepthwiseConv2DOp();
  Register_OneDnnQuantizedDepthwiseConv2DWithBiasOp();
  Register_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluOp();
  Register_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOp();
  Register_OneDnnQuantizedFusedBatchMatMulV2AndDequantizeOp();
  Register_OneDnnQuantizedFusedMatMulOp();
  Register_OneDnnQuantizedFusedMatMulAndRequantizeOp();
  Register_OneDnnQuantizedFusedMatMulAndDequantizeOp();
  Register_OneDnnQuantizedMatMulWithBiasAndReluOp();
  Register_OneDnnQuantizedMatMulWithBiasOp();
  Register_OneDnnQuantizedMatMulWithBiasAndReluAndRequantizeOp();
  Register_OneDnnQuantizedMatMulWithBiasAndRequantizeOp();
  Register_OneDnnQuantizedMatMulWithBiasAndDequantizeOp();
  Register_OneDnnQuantizedMaxPoolOp();
  Register_OneDnnQuantizedAvgPoolOp();
  Register_OneDnnQuantizedReshapeOp();
  Register_OneDnnQuantizedTransposeOp();
  Register_OneDnnQuantizeV2Op();
  Register_OneDnnQuantizeV2WithQuantizedConv2DOp();
  Register_OneDnnQuantizedConv2DWithDequantizeOp();
  Register_OneDnnQuantizedConv2DWithCastOp();
  Register_OneDnnPadWithConv2DOp();
  Register_OneDnnPadWithConv3DOp();
  Register_OneDnnPadWithFusedConv2DOp();
  Register_OneDnnPadWithFusedConv3DOp();
  Register_OneDnnReluOp();
  Register_OneDnnReluGradOp();
  Register_OneDnnReshapeOp();
  Register_OneDnnResizeBilinearOp();
  Register_OneDnnResizeBilinearGradOp();
  Register_OneDnnResizeNearestNeighborOp();
  Register_OneDnnResizeNearestNeighborGradOp();
  Register_OneDnnShapeOp();
  Register_OneDnnSliceOp();
  Register_OneDnnSoftmaxOp();
  Register_OneDnnSwishOp();
  Register_OneDnnToTfOp();
  Register_OneDnnTransposeOp();

  // Math ops
  Register_EqualWithCastOp();
  Register_NotEqualWithCastOp();
  Register_GreaterWithCastOp();
  Register_GreaterEqualWithCastOp();
  Register_LessWithCastOp();
  Register_LessEqualWithCastOp();
  Register_FusedAddNOp();
  Register_FusedRandomOP();

  // OneDnn math kernels
  Register_OneDnnAddOp();
  Register_OneDnnAddV2Op();
  Register_OneDnnCastOp();
  Register_OneDnnMulOp();
  Register_OneDnnSubOp();
}
