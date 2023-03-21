/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_GRAPH_REMAPPER_CONSTANT_NAMES_H_
#define ITEX_CORE_GRAPH_REMAPPER_CONSTANT_NAMES_H_

// TODO(itex) should be generated automatically.
namespace itex {
namespace graph {

// Placeholder for pattern matcher.
constexpr char kAny[] = "*";

//  Original TensorFlow op names.
constexpr char kAdd[] = "Add";
constexpr char kAddN[] = "AddN";
constexpr char kAddV2[] = "AddV2";
constexpr char kApplyRMSPropComputeRMS[] = "ApplyRMSPropComputeRMS";
constexpr char kApplyRMSPropVarUpdate[] = "ApplyRMSPropVarUpdate";
constexpr char kAssignVariableOp[] = "AssignVariableOp";
constexpr char kBatchMatMulV2[] = "BatchMatMulV2";
constexpr char kBiasAdd[] = "BiasAdd";
constexpr char kBinaryAdd[] = "BinaryAdd";
constexpr char kBinaryMul[] = "BinaryMul";
constexpr char kCast[] = "Cast";
constexpr char kConcatV2[] = "ConcatV2";
constexpr char kConst[] = "Const";
constexpr char kConv2D[] = "Conv2D";
constexpr char kConv2DBackpropFilter[] = "Conv2DBackpropFilter";
constexpr char kConv3D[] = "Conv3D";
constexpr char kConv3DBackpropFilter[] = "Conv3DBackpropFilter";
constexpr char kConv3DBackpropFilterV2[] = "Conv3DBackpropFilterV2";
constexpr char kDequantize[] = "Dequantize";
constexpr char kFill[] = "Fill";
constexpr char kFusedBatchNormV3[] = "FusedBatchNormV3";
constexpr char kGelu[] = "ITEXGelu";
constexpr char kLeakyRelu[] = "LeakyRelu";
constexpr char kMatMul[] = "MatMul";
constexpr char kMean[] = "Mean";
constexpr char kMish[] = "_ITEXMish";
constexpr char kMul[] = "Mul";
constexpr char kPad[] = "Pad";
constexpr char kQuantizeV2[] = "QuantizeV2";
constexpr char kReadVariableOp[] = "ReadVariableOp";
constexpr char kRelu[] = "Relu";
constexpr char kRealDiv[] = "RealDiv";
constexpr char kReshape[] = "Reshape";
constexpr char kResizeNearestNeighbor[] = "ResizeNearestNeighbor";
constexpr char kResizeNearestNeighborGrad[] = "ResizeNearestNeighborGrad";
constexpr char kRsqrt[] = "Rsqrt";
constexpr char kShape[] = "Shape";
constexpr char kSigmoid[] = "Sigmoid";
constexpr char kSlice[] = "Slice";
constexpr char kSoftplus[] = "Softplus";
constexpr char kSplit[] = "Split";
constexpr char kSplitV[] = "SplitV";
constexpr char kSqrt[] = "Sqrt";
constexpr char kSquare[] = "Square";
constexpr char kSquaredDifference[] = "SquaredDifference";
constexpr char kSub[] = "Sub";
constexpr char kSwish[] = "_ITEXSwish";
constexpr char kTanh[] = "Tanh";

// ITEX specific fused op names.
constexpr char kAccMatMul[] = "_ITEXAccMatMul";
constexpr char kAddV2WithSoftmax[] = "_ITEXFusedAddV2WithSoftmax";
constexpr char kConv2DBackpropFilterWithBias[] =
    "_ITEXConv2DBackpropFilterWithBias";
constexpr char kConv2DBackpropInputWithSlice[] =
    "_ITEXConv2DBackpropInputWithSlice";
constexpr char kConv3DBackpropFilterWithBias[] =
    "_ITEXConv3DBackpropFilterWithBias";
constexpr char kConv3DBackpropInputWithSlice[] =
    "_ITEXConv3DBackpropInputV2WithSlice";
constexpr char kDequantizeReshape[] = "_ITEXFusedDequantizeWithReshape";
constexpr char kFusedAccMatMul[] = "_ITEXFusedAccMatMul";
constexpr char kFusedAccMatMulGrad[] = "_ITEXFusedAccMatMulGrad";
constexpr char kFusedAccMatMulWithSum[] = "_ITEXFusedAccMatMulWithSum";
constexpr char kFusedApplyAdam[] = "_FusedApplyAdam";
constexpr char kFusedApplyAdamWithWeightDecay[] =
    "_FusedApplyAdamWithWeightDecay";
constexpr char kFusedAddN[] = "_FusedAddN";
constexpr char kFusedApplyMomentum[] = "_FusedApplyMomentum";
constexpr char kFusedBatchMatMul[] = "_ITEXFusedBatchMatMulV2";
constexpr char kFusedBatchNormEx[] = "_FusedBatchNormEx";
constexpr char kFusedBatchNormGradEx[] = "_ITEXFusedBatchNormGradEx";
constexpr char kFusedBinary[] = "_ITEXFusedBinary";
constexpr char kFusedConv2D[] = "_ITEXFusedConv2D";
constexpr char kFusedConv2DWithSum[] = "_ITEXFusedConv2DWithSum";
constexpr char kFusedConv3D[] = "_ITEXFusedConv3D";
constexpr char kFusedDepthwiseConv2dNative[] =
    "_ITEXFusedDepthwiseConv2dNative";
constexpr char kFusedMatMul[] = "_ITEXFusedMatMul";
constexpr char kFusedMatMulWithSum[] = "_ITEXFusedMatMulWithSum";
constexpr char kFusedMatMulGrad[] = "_ITEXFusedMatMulGrad";
constexpr char kFusedInstanceNorm[] = "_ITEXFusedInstanceNorm";
constexpr char kFusedRandom[] = "_ITEXFusedRandom";
constexpr char kFusedResourceApplyAdam[] = "_FusedResourceApplyAdam";
constexpr char kFusedResourceApplyAdamWithWeightDecay[] =
    "_FusedResourceApplyAdamWithWeightDecay";
constexpr char kFusedResourceApplyMomentum[] = "_FusedResourceApplyMomentum";
constexpr char kInstanceNorm[] = "_ITEXInstanceNorm";
constexpr char kLayerNorm[] = "ITEXLayerNorm";
constexpr char kPadWithConv2D[] = "_ITEXPadWithConv2D";
constexpr char kPadWithConv3D[] = "_ITEXPadWithConv3D";
constexpr char kPadWithFusedConv2D[] = "_ITEXPadWithFusedConv2D";
constexpr char kPadWithFusedConv3D[] = "_ITEXPadWithFusedConv3D";
constexpr char kQuantizeV2WithQuantizedConv2D[] =
    "_ITEXQuantizeV2WithQuantizedConv2D";
constexpr char kFusedQuantizedConv2DWithDequantize[] =
    "_ITEXQuantizedConv2DWithDequantize";
constexpr char kFusedQuantizedConv2DWithCast[] = "_ITEXQuantizedConv2DWithCast";

// TODO(itex): This op may be duplicated, remove it in future if possible.
constexpr char kPadConv3d[] = "_ITEXConv3D";

// Legacy op names from Intel TensorFlow.
constexpr char kMklLayerNorm[] = "_MklLayerNorm";

// Misc constant names.
constexpr int kMissingIndex = -1;
constexpr char kDataFormat[] = "data_format";
constexpr char kIsTraining[] = "is_training";

}  // namespace graph
}  // namespace itex
#endif  // ITEX_CORE_GRAPH_REMAPPER_CONSTANT_NAMES_H_
