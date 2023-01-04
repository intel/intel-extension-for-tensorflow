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

#include "itex/core/graph/native_layout/native_layout.h"

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/attr_value_util.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace graph {

namespace {
namespace protobuf = ::google::protobuf;

const std::vector<NativeFormatInfo>* GetCPUNativeFormatInfo() {
  static std::vector<NativeFormatInfo> rinfo{
      {"_FusedBatchNormEx", "_ITEXFusedBatchNormEx", CopyAttrsAll,
       RewriteFusedBatchNormEx},
      {"_PadWithConv2D", "_ITEXPadWithConv2D", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"_PadWithConv3D", "_ITEXPadWithConv3D", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"_PadWithFusedConv2D", "_ITEXPadWithFusedConv2D",
       CopyAttrsAllCheckConstFilter, RewriteFusedConv},
      {"_PadWithFusedConv3D", "_ITEXPadWithFusedConv3D",
       CopyAttrsAllCheckConstFilter, RewriteFusedConv},
      {"AddN", "_ITEXAddN", CopyAttrsAll, AlwaysRewrite},
      {"AvgPool", "_ITEXAvgPool", CopyAttrsAll, RewritePool},
      {"AvgPool3D", "_ITEXAvgPool3D", CopyAttrsAll, RewritePool},
      {"AvgPoolGrad", "_ITEXAvgPoolGrad", CopyAttrsAll, AlwaysRewrite},
      {"AvgPool3DGrad", "_ITEXAvgPool3DGrad", CopyAttrsAll, AlwaysRewrite},
      {"BatchMatMul", "_ITEXBatchMatMul", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"BatchMatMulV2", "_ITEXBatchMatMulV2", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"Cast", "_ITEXCast", CopyAttrsCast, RewriteNativeCast},
      {"Conv2D", "_ITEXConv2D", CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"Conv2DBackpropFilter", "_ITEXConv2DBackpropFilter", CopyAttrsAll,
       RewriteBackwardDataType},
      {"Conv2DBackpropInput", "_ITEXConv2DBackpropInput", CopyAttrsAll,
       RewriteConv2DBackprop},
      {"Conv3D", "_ITEXConv3D", CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"Conv3DBackpropFilterV2", "_ITEXConv3DBackpropFilterV2", CopyAttrsAll,
       RewriteBackwardDataType},
      {"Conv3DBackpropInput", "_ITEXConv3DBackpropInput", CopyAttrsAll,
       RewriteBackwardDataType},
      {"Conv3DBackpropInputV2", "_ITEXConv3DBackpropInputV2", CopyAttrsAll,
       RewriteBackwardDataType},
      {"DepthwiseConv2dNative", "_ITEXDepthwiseConv2dNative",
       CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"DepthwiseConv2dNativeBackpropFilter",
       "_ITEXDepthwiseConv2dNativeBackpropFilter", CopyAttrsAll,
       RewriteBackwardDataType},
      {"DepthwiseConv2dNativeBackpropInput",
       "_ITEXDepthwiseConv2dNativeBackpropInput", CopyAttrsAll,
       RewriteBackwardDataType},
      {"Einsum", "_ITEXEinsum", CopyAttrsAll, AlwaysRewrite},
      {"Elu", "_ITEXElu", CopyAttrsAll, AlwaysRewrite},
      {"EluGrad", "_ITEXEluGrad", CopyAttrsAll, RewriteBackwardDataType},
      {"FusedBatchNorm", "_ITEXFusedBatchNorm", CopyAttrsAll, AlwaysRewrite},
      {"FusedBatchNormV2", "_ITEXFusedBatchNormV2", CopyAttrsAll,
       AlwaysRewrite},
      {"FusedBatchNormV3", "_ITEXFusedBatchNormV3", CopyAttrsAll,
       AlwaysRewrite},
      {"FusedBatchNormGrad", "_ITEXFusedBatchNormGrad", CopyAttrsAll,
       RewriteBackwardDataType},
      {"FusedBatchNormGradV2", "_ITEXFusedBatchNormGradV2", CopyAttrsAll,
       RewriteBackwardDataType},
      {"FusedBatchNormGradV3", "_ITEXFusedBatchNormGradV3", CopyAttrsAll,
       RewriteFusedBatchNormGradV3},
      {"FusedInstanceNorm", "_ITEXFusedInstanceNorm", CopyAttrsAll,
       AlwaysRewrite},
      {"Gelu", "_ITEXGelu", CopyAttrsAll, AlwaysRewrite},
      {"GeluGrad", "_ITEXGeluGrad", CopyAttrsAll, RewriteBackwardDataType},
      {"GRUBlockCell", "_ITEXGRUCell", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"InstanceNorm", "_ITEXInstanceNorm", CopyAttrsAll, AlwaysRewrite},
      {"LayerNorm", "_ITEXLayerNorm", CopyAttrsAll, RewriteLayerNorm},
      {"LayerNormGrad", "_ITEXLayerNormGrad", CopyAttrsAll,
       RewriteLayerNormGrad},
      {"LeakyRelu", "_ITEXLeakyRelu", CopyAttrsAll, AlwaysRewrite},
      {"LeakyReluGrad", "_ITEXLeakyReluGrad", CopyAttrsAll,
       RewriteBackwardDataType},
      {"MatMul", "_ITEXMatMul", CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"MaxPool", "_ITEXMaxPool", CopyAttrsAll, RewritePool},
      {"MaxPool3D", "_ITEXMaxPool3D", CopyAttrsAll, RewritePool},
      {"MaxPoolGrad", "_ITEXMaxPoolGrad", CopyAttrsAll, RewriteMaxPoolGrad},
      {"MaxPool3DGrad", "_ITEXMaxPool3DGrad", CopyAttrsAll, RewriteMaxPoolGrad},
      {"Mish", "_ITEXMish", CopyAttrsAll, AlwaysRewrite},
      {"RandomUniform", "_ITEXRandomUniform", CopyAttrsAll, AlwaysRewrite},
      {"Relu", "_ITEXRelu", CopyAttrsAll, AlwaysRewrite},
      {"Relu6", "_ITEXRelu6", CopyAttrsAll, AlwaysRewrite},
      {"Relu6Grad", "_ITEXRelu6Grad", CopyAttrsAll, RewriteBackwardDataType},
      {"ReluGrad", "_ITEXReluGrad", CopyAttrsAll, RewriteBackwardDataType},
      {"ResizeBilinear", "_ITEXResizeBilinear", CopyAttrsAll, RewriteResize},
      {"ResizeBilinearGrad", "_ITEXResizeBilinearGrad", CopyAttrsAll,
       RewriteResize},
      {"Slice", "_ITEXSlice", CopyAttrsAll, AlwaysRewrite},
      {"Softmax", "_ITEXSoftmax", CopyAttrsAll, AlwaysRewrite},
      {"Swish", "_ITEXSwish", CopyAttrsAll, AlwaysRewrite},
      {"Transpose", "_ITEXTranspose", CopyAttrsAll, AlwaysRewrite},

      // Remapper can generate these Ops directly, but the attribute
      // "is_filter_const" is set by layout pass, which affects weight cache.
      // Thus, these Ops are rewritten as themselves.
      // TODO(yifeng): Remove this quick fix after formal solution is done.
      {"_ITEXAccMatMul", "_ITEXAccMatMul", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"_ITEXFusedAccMatMul", "_ITEXFusedAccMatMul",
       CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"_ITEXFusedAccMatMulWithSum", "_ITEXFusedAccMatMulWithSum",
       CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"_ITEXFusedBatchMatMulV2", "_ITEXFusedBatchMatMulV2",
       CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"_ITEXFusedDepthwiseConv2dNative", "_ITEXFusedDepthwiseConv2dNative",
       CopyAttrsAllCheckConstFilter, RewriteFusedConv},
      {"_ITEXFusedConv2D", "_ITEXFusedConv2D", CopyAttrsAllCheckConstFilter,
       RewriteFusedConv},
      {"_ITEXFusedConv2DWithSum", "_ITEXFusedConv2DWithSum",
       CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"_ITEXFusedConv3D", "_ITEXFusedConv3D", CopyAttrsAllCheckConstFilter,
       RewriteFusedConv},
      {"_ITEXFusedMatMul", "_ITEXFusedMatMul", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"_ITEXFusedMatMulWithSum", "_ITEXFusedMatMulWithSum",
       CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"_ITEXGRUCell", "_ITEXGRUCell", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"_ITEXAUGRUCell", "_ITEXAUGRUCell", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      // Intel-TF ops. Usually these ops should always be rewritten.
      // This part is for compatibility of legacy Intel-TF models, it will be
      // removed in future.
      {"_FusedMatMul", "_ITEXFusedMatMul", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"_MklFusedBatchMatMulV2", "_ITEXFusedBatchMatMulV2",
       CopyAttrsAllCheckConstFilter, AlwaysRewrite},
      {"_MklLayerNorm", "_ITEXMklLayerNorm", CopyAttrsAll, AlwaysRewrite},
      {"MklGRU", "_ITEXForwardGRU", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},
      {"MklAUGRU", "_ITEXForwardAUGRU", CopyAttrsAllCheckConstFilter,
       AlwaysRewrite},

      // INT8 op
      {"Dequantize", "_ITEXDequantize", CopyAttrsAll, RewriteQuantize},
      {"QuantizedAvgPool", "ITEXQuantizedAvgPool", CopyAttrsAll, AlwaysRewrite},
      // Disable concat rewrite, since we don't support concat int8 with
      // different scales
      // {"QuantizedConcatV2", "_ITEXQuantizedConcatV2", CopyAttrsAll,
      //  AlwaysRewrite},
      {"QuantizedConv2D", "_ITEXQuantizedConv2D", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedConv2DAndRequantize", "_ITEXQuantizedConv2DAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DPerChannel", "_ITEXQuantizedConv2DPerChannel",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBias", "_ITEXQuantizedConv2DWithBias",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBiasAndRequantize",
       "_ITEXQuantizedConv2DWithBiasAndRequantize", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedConv2DWithBiasAndRelu", "_ITEXQuantizedConv2DWithBiasAndRelu",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBiasAndReluAndRequantize",
       "_ITEXQuantizedConv2DWithBiasAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBiasSumAndRelu",
       "_ITEXQuantizedConv2DWithBiasSumAndRelu", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedConv2DWithBiasSumAndReluAndRequantize",
       "_ITEXQuantizedConv2DWithBiasSumAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
       "_ITEXQuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedDepthwiseConv2D", "_ITEXQuantizedDepthwiseConv2D",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedDepthwiseConv2DWithBias",
       "_ITEXQuantizedDepthwiseConv2DWithBias", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedDepthwiseConv2DWithBiasAndRelu",
       "_ITEXQuantizedDepthwiseConv2DWithBiasAndRelu", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
       "_ITEXQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedMatMulWithBias", "_ITEXQuantizedMatMulWithBias",
       CopyAttrsQuantizedMatMul, AlwaysRewrite},
      {"QuantizedMatMulWithBiasAndRelu", "_ITEXQuantizedMatMulWithBiasAndRelu",
       CopyAttrsQuantizedMatMul, AlwaysRewrite},
      {"QuantizedMatMulWithBiasAndReluAndRequantize",
       "_ITEXQuantizedMatMulWithBiasAndReluAndRequantize",
       CopyAttrsQuantizedMatMul, AlwaysRewrite},
      {"QuantizedMatMulWithBiasAndDequantize",
       "_ITEXQuantizedMatMulWithBiasAndDequantize", CopyAttrsQuantizedMatMul,
       AlwaysRewrite},
      {"QuantizedMatMulWithBiasAndRequantize",
       "_ITEXQuantizedMatMulWithBiasAndRequantize", CopyAttrsQuantizedMatMul,
       AlwaysRewrite},
      {"QuantizedMaxPool", "_ITEXQuantizedMaxPool", CopyAttrsAll,
       AlwaysRewrite},
      {"QuantizedMaxPool3D", "_ITEXQuantizedMaxPool3D", CopyAttrsAll,
       AlwaysRewrite},
      {"QuantizedReshape", "_ITEXQuantizedReshape", CopyAttrsAll,
       RewriteQuantizeReshape},
      {"QuantizeV2", "_ITEXQuantizeV2", CopyAttrsQuantize, RewriteQuantize},

      {"_QuantizedBatchMatMul", "_ITEXQuantizedBatchMatMul", CopyAttrsAll,
       AlwaysRewrite},
      {"_QuantizedBatchMatMulV2AndDequantize",
       "_ITEXQuantizedBatchMatMulV2AndDequantize", CopyAttrsAll, AlwaysRewrite},
      {"_QuantizedConv2D", "_ITEXQuantizedConv2DV2", CopyAttrsAll,
       AlwaysRewrite},
      {"_QuantizedConv3D", "_ITEXQuantizedConv3DV2", CopyAttrsAll,
       AlwaysRewrite},
      {"_QuantizedDepthwiseConv2D", "_ITEXQuantizedDepthwiseConv2DV2",
       CopyAttrsAll, AlwaysRewrite},
      {"_QuantizedFusedBatchMatMulV2AndDequantize",
       "_ITEXQuantizedFusedBatchMatMulV2AndDequantize", CopyAttrsAll,
       AlwaysRewrite},
      {"_QuantizedFusedBatchNorm", "_ITEXQuantizedFusedBatchNorm", CopyAttrsAll,
       AlwaysRewrite},
      {"_QuantizedFusedMatMul", "_ITEXQuantizedFusedMatMul",
       CopyAttrsQuantizedMatMul, AlwaysRewrite},
      {"_QuantizedFusedMatMulAndDequantize",
       "_ITEXQuantizedFusedMatMulAndDequantize", CopyAttrsQuantizedMatMul,
       AlwaysRewrite},
      {"_QuantizedFusedMatMulAndRequantize",
       "_ITEXQuantizedFusedMatMulAndRequantize", CopyAttrsQuantizedMatMul,
       AlwaysRewrite},
      {"_QuantizedMatMul", "_ITEXQuantizedMatMul", CopyAttrsAll, AlwaysRewrite},
      {"_QuantizedTranspose", "_ITEXQuantizedTranspose", CopyAttrsAll,
       AlwaysRewrite}};
  return &rinfo;
}

const std::vector<NativeFormatInfo>* GetGPUNativeFormatInfo() {
  static std::vector<NativeFormatInfo> rinfo{
      {"TensorArray", "_ITEXTensorArray", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayV2", "_ITEXTensorArray", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayV3", "_ITEXTensorArray", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayGrad", "_ITEXTensorArrayGrad", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayGradV2", "_ITEXTensorArrayGrad", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayGradV3", "_ITEXTensorArrayGrad", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayGradWithShape", "_ITEXTensorArrayGradWithShape",
       CopyAttrsForTensorArray, AlwaysRewrite},
      {"TensorArrayWrite", "_ITEXTensorArrayWrite", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayWriteV2", "_ITEXTensorArrayWrite", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayWriteV3", "_ITEXTensorArrayWrite", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayRead", "_ITEXTensorArrayRead", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayReadV2", "_ITEXTensorArrayRead", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayReadV3", "_ITEXTensorArrayRead", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayPack", "_ITEXTensorArrayPack", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayGather", "_ITEXTensorArrayGather", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayGatherV2", "_ITEXTensorArrayGather", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayGatherV3", "_ITEXTensorArrayGather", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayConcat", "_ITEXTensorArrayConcat", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayConcatV2", "_ITEXTensorArrayConcat", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayConcatV3", "_ITEXTensorArrayConcat", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayUnPack", "_ITEXTensorArrayUnPack", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayScatter", "_ITEXTensorArrayScatter", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayScatterV2", "_ITEXTensorArrayScatter",
       CopyAttrsForTensorArray, AlwaysRewrite},
      {"TensorArrayScatterV3", "_ITEXTensorArrayScatter",
       CopyAttrsForTensorArray, AlwaysRewrite},
      {"TensorArraySplit", "_ITEXTensorArraySplit", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArraySplitV2", "_ITEXTensorArraySplit", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArraySplitV3", "_ITEXTensorArraySplit", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArraySize", "_ITEXTensorArraySize", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArraySizeV2", "_ITEXTensorArraySize", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArraySizeV3", "_ITEXTensorArraySize", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayClose", "_ITEXTensorArrayClose", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayCloseV2", "_ITEXTensorArrayClose", CopyAttrsForTensorArray,
       AlwaysRewrite},
      {"TensorArrayCloseV3", "_ITEXTensorArrayClose", CopyAttrsForTensorArray,
       AlwaysRewrite},
  };
  return &rinfo;
}

}  // namespace

const NativeFormatInfo* CheckForNodeNativeFormat(
    const char* device_name, const utils::MutableNodeView& node_view) {
  NodeDef& node_def = *(node_view.node());

  if (!IsLayoutRewriteSupportedDataType(node_def)) return nullptr;

  // We now check if rewrite rule applies for this op. If rewrite rule passes
  // for this op, then we rewrite it to Native op.
  // Find matching NativeFormatInfo and then check that rewrite rule applies.
  const std::vector<NativeFormatInfo>* rinfo;
  if (absl::StrContains("CPU", device_name)) {
    rinfo = GetCPUNativeFormatInfo();
  } else if (absl::StrContains("GPU", device_name)) {
    rinfo = GetGPUNativeFormatInfo();
  } else if (absl::StrContains("XPU", device_name)) {
    rinfo = GetGPUNativeFormatInfo();
  } else {
    ITEX_LOG(WARNING) << "invalid device name, expected CPU/GPU/XPU, got "
                      << device_name;
    return nullptr;
  }

  for (auto ri = rinfo->cbegin(); ri != rinfo->cend(); ++ri) {
    if (node_def.op() == ri->name && ri->rewrite_rule(node_view)) {
      return &*ri;
    }
  }

  // Else return not found.
  return nullptr;
}

// Rewrites input node to a new node specified by its matching rewrite info.
//
// Method first searches matching rewrite info for input node and then
// uses that info to rewrite.
//
// Input node may be deleted in case of rewrite. Attempt to use the node
// after the call can result in undefined behaviors.
Status RewriteNode(NativeFormatContext* ctx, const int node_index,
                   const NativeFormatInfo* ri) {
  const auto* node_view = ctx->graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  NodeDef new_node_def;
  // Let's copy all inputs (TF tensors) of original node to new node.
  for (int idx = 0; idx < node_view->NumRegularFanins(); idx++) {
    new_node_def.add_input(node_def->input(idx));
  }

  new_node_def.set_name(node_def->name());
  new_node_def.set_op(ri->new_name);
  new_node_def.set_device(node_def->device());

  ri->copy_attrs(node_view, &new_node_def);

  // TODO(yifeng): Remove this check after formal solution
  // for is_filter_const setting is done.
  if (ri->name == ri->new_name) {
    bool is_filter_const = false;
    ITEX_CHECK(
        TryGetNodeAttr(new_node_def, "is_filter_const", &is_filter_const));
  }

  SetConstFilterAttr(node_view, &new_node_def, ctx->nodes_to_preserve);

  // Incoming data edges from 'orig_node' node to new 'new_node' node are
  // already copied in BuildNode. We need to handle control edges now.
  for (int idx = 0; idx < node_view->NumControllingFanins(); idx++) {
    new_node_def.add_input(
        node_def->input(node_view->NumRegularFanins() + idx));
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();

  // apply mutation
  Status status;
  mutation->AddNode(std::move(new_node_def), &status);
  TF_ABORT_IF_ERROR(std::move(status));
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

Status RunNativeLayout(const char* device_name, const GrapplerItem& item,
                       const GraphDef& graph_def, GraphDef* optimized_graph) {
  Status status;
  GraphDef multable_graph_def = graph_def;
  NativeFormatContext ctx(item, &multable_graph_def, &status);

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_ABORT_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  // Skip nodes that were invalidated
  int num_nodes = multable_graph_def.node_size();

  ITEX_VLOG(1) << "NativeLayoutPass: Start to rewrite nodes.";

  for (int node_index = num_nodes - 1; node_index >= 0; --node_index) {
    const auto* node_view = ctx.graph_view.GetNode(node_index);
    const auto* node_def = node_view->node();

    // Check if node can run on current optimizer device.
    if (!NodeIsOnDevice(device_name, node_def)) continue;

    const NativeFormatInfo* ri = nullptr;
    // We will first search if node is to be rewritten.
    if ((ri = CheckForNodeNativeFormat(device_name, *node_view)) != nullptr) {
      const string& node_name = node_def->name();
      const string& op_name = node_def->op();

      if (RewriteNode(&ctx, node_index, ri) == Status::OK()) {
        ITEX_VLOG(2) << "NativeLayoutPass: rewrote node " << node_name
                     << " with op " << op_name
                     << " for Native layout optimization.";
      } else {
        ITEX_VLOG(2) << "NativeLayoutPass: found node " << node_name
                     << " with op " << op_name << " but rewrite failed.";
      }
    }
  }

  *optimized_graph = std::move(multable_graph_def);
  return Status::OK();
}

}  // namespace graph
}  // namespace itex
