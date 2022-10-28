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

constexpr char kAny[] = "*";

constexpr char kAddV2[] = "AddV2";
constexpr char kAddN[] = "AddN";
constexpr char kApplyRMSPropComputeRMS[] = "ApplyRMSPropComputeRMS";
constexpr char kApplyRMSPropVarUpdate[] = "ApplyRMSPropVarUpdate";
constexpr char kAssignVariableOp[] = "AssignVariableOp";
constexpr char kBiasAdd[] = "BiasAdd";
constexpr char kBatchMatMulV2[] = "BatchMatMulV2";
constexpr char kBinaryAdd[] = "BinaryAdd";
constexpr char kCast[] = "Cast";
constexpr char kConcatV2[] = "ConcatV2";
constexpr char kConst[] = "Const";
constexpr char kConv2DBackpropFilter[] = "Conv2DBackpropFilter";
constexpr char kConv2DBackpropFilterWithBias[] = "Conv2DBackpropFilterWithBias";
constexpr char kConv3DBackpropFilter[] = "Conv3DBackpropFilter";
constexpr char kConv3DBackpropFilterV2[] = "Conv3DBackpropFilterV2";
constexpr char kConv3DBackpropFilterWithBias[] = "Conv3DBackpropFilterWithBias";
constexpr char kConv3D[] = "Conv3D";
constexpr char kDequantize[] = "Dequantize";
constexpr char kFusedBatchNormV3[] = "FusedBatchNormV3";
constexpr char kLeakyRelu[] = "LeakyRelu";
constexpr char kMatMul[] = "MatMul";
constexpr char kMean[] = "Mean";
constexpr char kMul[] = "Mul";
constexpr char kFill[] = "Fill";
constexpr char kPad[] = "Pad";
constexpr char kQuantizeV2[] = "QuantizeV2";
constexpr char kReadVariableOp[] = "ReadVariableOp";
constexpr char kRelu[] = "Relu";
constexpr char kReshape[] = "Reshape";
constexpr char kRealDiv[] = "RealDiv";
constexpr char kResizeNearestNeighbor[] = "ResizeNearestNeighbor";
constexpr char kResizeNearestNeighborGrad[] = "ResizeNearestNeighborGrad";
constexpr char kRsqrt[] = "Rsqrt";
constexpr char kSlice[] = "Slice";
constexpr char kSub[] = "Sub";
constexpr char kSigmoid[] = "Sigmoid";
constexpr char kSplit[] = "Split";
constexpr char kSplitV[] = "SplitV";
constexpr char kSqrt[] = "Sqrt";
constexpr char kSquare[] = "Square";
constexpr char kSquaredDifference[] = "SquaredDifference";
constexpr char kSwish[] = "Swish";
constexpr char kTanh[] = "Tanh";

constexpr char kFusedBatchMatMulV2[] = "_FusedBatchMatMulV2";
constexpr char kInstanceNorm[] = "InstanceNorm";
constexpr char kFusedInstanceNorm[] = "FusedInstanceNorm";
constexpr char kITEXFusedMatMulWithSum[] = "_FusedMatMulWithSum";
constexpr char kITEXFusedMatMul[] = "_ITEXFusedMatMul";
constexpr char kLayerNorm[] = "LayerNorm";
constexpr char kMklLayerNorm[] = "_MklLayerNorm";
constexpr char kPadConv3d[] = "_ITEXConv3D";

}  // namespace graph
}  // namespace itex
#endif  // ITEX_CORE_GRAPH_REMAPPER_CONSTANT_NAMES_H_
