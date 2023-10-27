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

#include "itex/core/graph/onednn_layout/onednn_layout.h"

#include <cstring>
#include <utility>

#include "google/protobuf/text_format.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/attr_value_util.h"
#include "itex/core/utils/node_def_util.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace graph {

namespace {
namespace protobuf = ::google::protobuf;

int OpDefOutputPorts(const NodeDef* node_def) {
  OpDef op_def = GetOpDef(*node_def);
  int num_output = 0;
  // TF opdef may use 1 argdef to represent several input/output tensors. And we
  // need to handle type representations.
  for (auto output_arg : op_def.output_arg()) {
    int n = 1;  // default value with single tensor
    if (!output_arg.number_attr().empty()) {
      n = node_def->attr().at(output_arg.number_attr()).i();
    } else if (!output_arg.type_list_attr().empty()) {
      n = node_def->attr().at(output_arg.type_list_attr()).list().type_size();
    }

    ITEX_CHECK_GE(n, 0);
    num_output += n;
  }

  return num_output;
}

/// Maintain info about nodes to rewrite.
/// Add related info here if new rule is supported.
static const std::vector<RewriteInfo>* GetRewriteInfo() {
  static std::vector<RewriteInfo> rinfo{
      // Proper OP
      {"Add", "_OneDnnAdd", CopyAttrsAll, RewriteBinary},
      {"AddN", "_OneDnnAddN", CopyAttrsAll, RewriteWithBlockInput},
      {"AddV2", "_OneDnnAddV2", CopyAttrsAll, RewriteBinary},
      {"AvgPool", "_OneDnnAvgPool", CopyAttrsAll, RewriteOneDnnPool},
      {"AvgPool3D", "_OneDnnAvgPool3D", CopyAttrsAll, RewriteOneDnnPool},
      {"AvgPool3DGrad", "_OneDnnAvgPool3DGrad", CopyAttrsAll, AlwaysRewrite},
      {"AvgPoolGrad", "_OneDnnAvgPoolGrad", CopyAttrsAll, AlwaysRewrite},
      {"BatchMatMulV2", "_OneDnnBatchMatMulV2", CopyAttrsAll,
       RewriteWithBlockInput},
      {"Cast", "_OneDnnCast", CopyAttrsCast, RewriteCast},
      {"Concat", "_OneDnnConcat", CopyAttrsAll, RewriteWithBlockInput},
      {"ConcatV2", "_OneDnnConcatV2", CopyAttrsAll, RewriteWithBlockInput},
      {"Conv2D", "_OneDnnConv2D", CopyAttrsAll, RewriteOneDnnConv},
      {"Conv2DBackpropFilter", "_OneDnnConv2DBackpropFilter", CopyAttrsAll,
       RewriteConv2DBackprop},
      {"Conv2DBackpropInput", "_OneDnnConv2DBackpropInput", CopyAttrsAll,
       RewriteConv2DBackprop},
      {"Conv3D", "_OneDnnConv3D", CopyAttrsAll, RewriteOneDnnConv},
      {"Conv3DBackpropFilterV2", "_OneDnnConv3DBackpropFilterV2", CopyAttrsAll,
       RewriteBackwardDataType},
      {"Conv3DBackpropInputV2", "_OneDnnConv3DBackpropInputV2", CopyAttrsAll,
       RewriteBackwardDataType},
      {"DepthwiseConv2dNative", "_OneDnnDepthwiseConv2dNative", CopyAttrsAll,
       RewriteOneDnnConv},
      {"DepthwiseConv2dNativeBackpropFilter",
       "_OneDnnDepthwiseConv2dNativeBackpropFilter", CopyAttrsAll,
       RewriteBackwardDataType},
      {"DepthwiseConv2dNativeBackpropInput",
       "_OneDnnDepthwiseConv2dNativeBackpropInput", CopyAttrsAll,
       RewriteBackwardDataType},
      {"Dequantize", "_OneDnnDequantize", CopyAttrsAll, RewriteQuantize},
      {"FusedBatchNorm", "_OneDnnFusedBatchNorm", CopyAttrsAll,
       RewriteWithBlockInput},
      {"FusedBatchNormGrad", "_OneDnnFusedBatchNormGrad", CopyAttrsAll,
       RewriteBackwardDataType},
      {"FusedBatchNormGradV2", "_OneDnnFusedBatchNormGradV2", CopyAttrsAll,
       RewriteBackwardDataType},
      {"FusedBatchNormGradV3", "_OneDnnFusedBatchNormGradV3", CopyAttrsAll,
       RewriteBackwardDataType},
      {"FusedBatchNormV2", "_OneDnnFusedBatchNormV2", CopyAttrsAll,
       RewriteWithBlockInput},
      {"FusedBatchNormV3", "_OneDnnFusedBatchNormV3", CopyAttrsAll,
       RewriteWithBlockInput},
      {"Gelu", "_OneDnnGelu", CopyAttrsAll, RewriteWithBlockInput},
      {"GeluGrad", "_OneDnnGeluGrad", CopyAttrsAll, RewriteBackwardDataType},
      {"Identity", "_OneDnnIdentity", CopyAttrsAll, RewriteWithBlockInput},
      {"LayerNorm", "_OneDnnLayerNorm", CopyAttrsAll, RewriteLayerNorm},
      {"LayerNormGrad", "_OneDnnLayerNormGrad", CopyAttrsAll,
       RewriteLayerNormGrad},
      {"LeakyRelu", "_OneDnnLeakyRelu", CopyAttrsAll, RewriteWithBlockInput},
      {"LeakyReluGrad", "_OneDnnLeakyReluGrad", CopyAttrsAll,
       RewriteBackwardDataType},
      {"MatMul", "_OneDnnMatMul", CopyAttrsAll, RewriteMatMul},
      {"MaxPool", "_OneDnnMaxPool", CopyAttrsAll, RewriteOneDnnPool},
      {"MaxPool3D", "_OneDnnMaxPool3D", CopyAttrsAll, RewriteOneDnnPool},
      {"MaxPool3DGrad", "_OneDnnMaxPool3DGrad", CopyAttrsAll,
       RewriteMaxPoolGrad},
      {"MaxPoolGrad", "_OneDnnMaxPoolGrad", CopyAttrsAll, RewriteMaxPoolGrad},
      {"Mul", "_OneDnnMul", CopyAttrsAll, RewriteBinary},
      {onednngrap_op_name, _onednngrap_op_name, CopyAttrsOneDnnGraph,
       AlwaysRewrite},
      {"QuantizedConcatV2", "_OneDnnQuantizedConcatV2", CopyAttrsAll,
       AlwaysRewrite},
      {"Relu", "_OneDnnRelu", CopyAttrsAll, RewriteWithBlockInput},
      {"ReluGrad", "_OneDnnReluGrad", CopyAttrsAll, RewriteBackwardDataType},
      {"Reshape", "_OneDnnReshape", CopyAttrsAll, RewriteWithBlockInput},
      {"ResizeBilinear", "_OneDnnResizeBilinear", CopyAttrsAll, RewriteResize},
      {"ResizeBilinearGrad", "_OneDnnResizeBilinearGrad", CopyAttrsAll,
       RewriteResize},
      {"ResizeNearestNeighbor", "_OneDnnResizeNearestNeighbor", CopyAttrsAll,
       RewriteResize},
      {"ResizeNearestNeighborGrad", "_OneDnnResizeNearestNeighborGrad",
       CopyAttrsAll, RewriteResize},
      {"Shape", "_OneDnnShape", CopyAttrsAll, RewriteWithBlockInput},
      {"Slice", "_OneDnnSlice", CopyAttrsAll, RewriteWithBlockInput},
      {"Softmax", "_OneDnnSoftmax", CopyAttrsAll, RewriteWithBlockInput},
      {"Sub", "_OneDnnSub", CopyAttrsAll, RewriteBinary},
      {"Transpose", "_OneDnnTranspose", CopyAttrsAll, RewriteWithBlockInput},
      {"_FusedBatchNormEx", "_OneDnnFusedBatchNormEx", CopyAttrsAll,
       RewriteFusedBatchNormEx},
      // Intel-TF ops. Usually these ops should always be rewritten.
      // This part is for compatibility of legacy Intel-TF models, it will be
      // removed in future.
      {"_FusedMatMul", "_OneDnnFusedMatMul", CopyAttrsAll, RewriteMatMul},
      {"_MklFusedBatchMatMulV2", "_OneDnnFusedBatchMatMulV2", CopyAttrsAll,
       AlwaysRewrite},
      {"_MklLayerNorm", "_OneDnnMklLayerNorm", CopyAttrsAll, AlwaysRewrite},

      // ITEX OP
      {"ITEXGelu", "_OneDnnGelu", CopyAttrsAll, RewriteWithBlockInput},
      {"ITEXGeluGrad", "_OneDnnGeluGrad", CopyAttrsAll,
       RewriteBackwardDataType},
      {"ITEXLayerNorm", "_OneDnnLayerNorm", CopyAttrsAll, RewriteLayerNorm},
      {"ITEXLayerNormGrad", "_OneDnnLayerNormGrad", CopyAttrsAll,
       RewriteLayerNormGrad},
      {"_ITEXConv2DBackpropFilterWithBias",
       "_OneDnnConv2DBackpropFilterWithBias", CopyAttrsAll,
       RewriteConv2DBackprop},
      {"_ITEXConv2DBackpropInputWithSlice",
       "_OneDnnConv2DBackpropInputWithSlice", CopyAttrsAll,
       RewriteConv2DBackprop},
      {"_ITEXConv3DBackpropFilterWithBias",
       "_OneDnnConv3DBackpropFilterWithBias", CopyAttrsAll,
       RewriteBackwardDataType},
      {"_ITEXConv3DBackpropInputV2WithSlice",
       "_OneDnnConv3DBackpropInputV2WithSlice", CopyAttrsAll,
       RewriteBackwardDataType},
      {"_ITEXFusedBatchMatMulV2", "_OneDnnFusedBatchMatMulV2", CopyAttrsAll,
       RewriteWithBlockInput},
      {"_ITEXFusedBatchNormGradEx", "_OneDnnFusedBatchNormGradEx", CopyAttrsAll,
       RewriteFusedBatchNormExGrad},
      {"_ITEXFusedConv2D", "_OneDnnFusedConv2D", CopyAttrsAll,
       RewriteOneDnnFusedConv},
      {"_ITEXFusedConv2DWithSum", "_OneDnnFusedConv2D", CopyAttrsAll,
       RewriteOneDnnFusedConv},
      {"_ITEXFusedConv3D", "_OneDnnFusedConv3D", CopyAttrsAll,
       RewriteOneDnnFusedConv},
      {"_ITEXFusedDepthwiseConv2dNative", "_OneDnnFusedDepthwiseConv2dNative",
       CopyAttrsAll, RewriteOneDnnFusedConv},
      {"_ITEXFusedInstanceNorm", "_OneDnnFusedInstanceNorm", CopyAttrsAll,
       RewriteWithBlockInput},
      {"_ITEXFusedMatMul", "_OneDnnFusedMatMul", CopyAttrsAll, RewriteMatMul},
      {"_ITEXFusedMatMulGrad", "_OneDnnFusedMatMulGrad", CopyAttrsAll,
       AlwaysRewrite},
      {"_ITEXFusedMatMulWithSum", "_OneDnnFusedMatMul", CopyAttrsAll,
       RewriteMatMul},
      {"_ITEXInstanceNorm", "_OneDnnInstanceNorm", CopyAttrsAll,
       RewriteWithBlockInput},
      {"_ITEXMish", "_OneDnnMish", CopyAttrsAll, RewriteWithBlockInput},
      {"_ITEXPadWithConv2D", "_OneDnnPadWithConv2D", CopyAttrsAll,
       RewriteOneDnnConv},
      {"_ITEXPadWithConv3D", "_OneDnnPadWithConv3D", CopyAttrsAll,
       RewriteOneDnnConv},
      {"_ITEXPadWithFusedConv2D", "_OneDnnPadWithFusedConv2D", CopyAttrsAll,
       RewriteOneDnnFusedConv},
      {"_ITEXPadWithFusedConv3D", "_OneDnnPadWithFusedConv3D", CopyAttrsAll,
       RewriteOneDnnFusedConv},
      {"_ITEXSwish", "_OneDnnSwish", CopyAttrsAll, RewriteWithBlockInput},

      // INT8 OP
      // Old Conv and MatMul OP
      {"QuantizedConv2D", "_OneDnnQuantizedConv2D", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedConv2DAndRequantize", "_OneDnnQuantizedConv2DAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBias", "_OneDnnQuantizedConv2DWithBias",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBiasAndRelu",
       "_OneDnnQuantizedConv2DWithBiasAndRelu", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedConv2DWithBiasAndReluAndRequantize",
       "_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBiasAndRequantize",
       "_OneDnnQuantizedConv2DWithBiasAndRequantize", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
       "_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedConv2DWithBiasSumAndRelu",
       "_OneDnnQuantizedConv2DWithBiasSumAndRelu", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedConv2DWithBiasSumAndReluAndRequantize",
       "_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedDepthwiseConv2D", "_OneDnnQuantizedDepthwiseConv2D",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedDepthwiseConv2DWithBias",
       "_OneDnnQuantizedDepthwiseConv2DWithBias", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"QuantizedDepthwiseConv2DWithBiasAndRelu",
       "_OneDnnQuantizedDepthwiseConv2DWithBiasAndRelu",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
       "_OneDnnQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"QuantizedMatMulWithBias", "_OneDnnQuantizedMatMulWithBias",
       CopyAttrsQuantizedMatMul, AlwaysRewrite},
      {"QuantizedMatMulWithBiasAndDequantize",
       "_OneDnnQuantizedMatMulWithBiasAndDequantize", CopyAttrsQuantizedMatMul,
       AlwaysRewrite},
      {"QuantizedMatMulWithBiasAndRelu",
       "_OneDnnQuantizedMatMulWithBiasAndRelu", CopyAttrsQuantizedMatMul,
       AlwaysRewrite},
      {"QuantizedMatMulWithBiasAndReluAndRequantize",
       "_OneDnnQuantizedMatMulWithBiasAndReluAndRequantize",
       CopyAttrsQuantizedMatMul, AlwaysRewrite},
      {"QuantizedMatMulWithBiasAndRequantize",
       "_OneDnnQuantizedMatMulWithBiasAndRequantize", CopyAttrsQuantizedMatMul,
       AlwaysRewrite},
      // New Conv and MatMul OP
      {"_ITEXQuantizedConv2D", "_OneDnnQuantizedConv2D",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"_ITEXQuantizedConv2DAndRequantize",
       "_OneDnnQuantizedConv2DAndRequantize", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithBias", "_OneDnnQuantizedConv2DWithBias",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithBiasAndRelu",
       "_OneDnnQuantizedConv2DWithBiasAndRelu", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithBiasAndReluAndRequantize",
       "_OneDnnQuantizedConv2DWithBiasAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithBiasAndRequantize",
       "_OneDnnQuantizedConv2DWithBiasAndRequantize", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
       "_OneDnnQuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithBiasSumAndRelu",
       "_OneDnnQuantizedConv2DWithBiasSumAndRelu", CopyAttrsQuantizedConv2D,
       AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithBiasSumAndReluAndRequantize",
       "_OneDnnQuantizedConv2DWithBiasSumAndReluAndRequantize",
       CopyAttrsQuantizedConv2D, AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithCast", "_OneDnnQuantizedConv2DWithCast",
       CopyAttrsAll, AlwaysRewrite},
      {"_ITEXQuantizedConv2DWithDequantize",
       "_OneDnnQuantizedConv2DWithDequantize", CopyAttrsAll, AlwaysRewrite},
      {"_ITEXQuantizedMatMulWithBiasAndDequantize",
       "_OneDnnQuantizedMatMulWithBiasAndDequantize", CopyAttrsQuantizedMatMul,
       AlwaysRewrite},
      {"_ITEXQuantizeV2WithQuantizedConv2D",
       "_OneDnnQuantizeV2WithQuantizedConv2D", CopyAttrsAll, AlwaysRewrite},
      {"_QuantizedBatchMatMulV2AndDequantize",
       "_OneDnnQuantizedBatchMatMulV2AndDequantize", CopyAttrsAll,
       AlwaysRewrite},
      {"_QuantizedFusedBatchMatMulV2AndDequantize",
       "_OneDnnQuantizedFusedBatchMatMulV2AndDequantize", CopyAttrsAll,
       AlwaysRewrite},
      {"_QuantizedFusedMatMul", "_OneDnnQuantizedFusedMatMul", CopyAttrsAll,
       AlwaysRewrite},
      {"_QuantizedFusedMatMulAndDequantize",
       "_OneDnnQuantizedFusedMatMulAndDequantize", CopyAttrsAll, AlwaysRewrite},
      {"_QuantizedFusedMatMulAndRequantize",
       "_OneDnnQuantizedFusedMatMulAndRequantize", CopyAttrsAll, AlwaysRewrite},
      // Other new INT8 op
      {"ITEXQuantizedAvgPool", "_OneDnnQuantizedAvgPool", CopyAttrsAll,
       AlwaysRewrite},
      {"QuantizedAvgPool", "_OneDnnQuantizedAvgPool", CopyAttrsAll,
       AlwaysRewrite},
      {"QuantizedMaxPool", "_OneDnnQuantizedMaxPool", CopyAttrsAll,
       AlwaysRewrite},
      {"QuantizedReshape", "_OneDnnQuantizedReshape", CopyAttrsAll,
       AlwaysRewrite},
      {"QuantizeV2", "_OneDnnQuantizeV2", CopyAttrsQuantize, RewriteQuantize},
      {"_ITEXFusedDequantizeWithReshape", "_OneDnnFusedDequantizeWithReshape",
       CopyAttrsAll, AlwaysRewrite},
      {"_ITEXQuantizeV2", "_OneDnnQuantizeV2", CopyAttrsQuantize,
       RewriteQuantize},
      {"_QuantizedTranspose", "_OneDnnQuantizedTranspose", CopyAttrsAll,
       AlwaysRewrite},
  };

  return &rinfo;
}  // GetRewriteInfo

}  // namespace

void GetDummyOneDnnTensorNode(const NodeDef& input, NodeDef* dummy) {
  static uint64 index = 0;
  if (dummy->op() == "HostConst") return;
  // We use a tensor of shape {8} and value 0,0,0,0,0,0,0,0 to represent
  // dummy OneDNN tensor. 8 = 2*size_t.
  const DataType dt = DataTypeToEnum<uint8>::v();
  TensorProto proto;
  proto.set_dtype(dt);
  uint8 zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  proto.set_tensor_content(string(reinterpret_cast<char*>(&zero), 8));
  TensorShape dummy_shape({8});
  dummy_shape.AsProto(proto.mutable_tensor_shape());

  dummy->set_name(input.name() + "_DMT_" + std::to_string(index));
  dummy->set_op("HostConst");
  dummy->set_device(input.device());

  auto* attr = dummy->mutable_attr();
  SetAttrValue(proto, &(*attr)["value"]);
  SetAttrValue(dt, &(*attr)["dtype"]);
  ++index;
}

// Add dummy oneDNN meta data node `dummy` between `input` and `new_node`.
// A new control edge will be added between `input` and `dummy` to fix frame
// dependency in loop statement:
/*
     input         input
       |             | \
       |       ->    | dummy
       |             |  |
     new_node      new_node
*/
void AddDummyOneDnnNode(utils::Mutation* mutation, const NodeDef& input,
                        NodeDef* new_node) {
  Status status;
  NodeDef dummy;

  ITEX_DCHECK(new_node);
  GetDummyOneDnnTensorNode(input, &dummy);
  dummy.add_input(AsControlDependency(input));
  new_node->add_input(GetInputName(&dummy, 0));
  mutation->AddNode(std::move(dummy), &status);
  TF_ABORT_IF_ERROR(status);
}

// This function is similar to the "AddDummyOneDnnNode". The difference is it
// applies to the _OneDnnGraph node which is already in the graph. It has
// already 2*N inputs. Here we need AddOrUpdateRegularFanin() to replace the
// input, instead of add_input() to add input.
void UpdateDummyOneDnnNode(utils::Mutation* mutation, const NodeDef& input,
                           utils::MutableNodeView* node_view,
                           int update_index) {
  Status status;
  NodeDef dummy;

  GetDummyOneDnnTensorNode(input, &dummy);
  dummy.add_input(AsControlDependency(input));

  TensorId output(dummy.name(), 0);
  mutation->AddOrUpdateRegularFanin(
      const_cast<utils::MutableNodeView*>(node_view), update_index, output);
  mutation->AddNode(std::move(dummy), &status);
  TF_ABORT_IF_ERROR(status);
}

// Rewrites input node to a new node specified by its matching rewrite info.
//
// Method first searches matching rewrite info for input node and then
// uses that info to rewrite.
//
// Input node may be deleted in case of rewrite. Attempt to use the node
// after the call can result in undefined behaviors.
Status RewriteNode(OneDnnLayoutContext* ctx, const int node_index,
                   const RewriteInfo* ri) {
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

  // Add workspace inputs if needed.
  NodeDef* ws_input_node_def = AddWorkspace(node_view, &new_node_def);

  // Let's now setup all OneDNN inputs to a new node.
  // Number of OneDNN inputs must be same as number of TF inputs.
  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();

  for (int idx = 0; idx < node_view->NumRegularFanins(); idx++) {
    const auto* input_node_view = node_view->GetRegularFanin(idx).node_view();
    const auto* input_node_def = input_node_view->node();

    if (!IsOneDnnLayoutDependentOp(input_node_def->op())) {
      // If we have not visited the node and rewritten it, then we need
      // to create a dummy node that will feed a dummy OneDNN tensor to this
      // node.
      AddDummyOneDnnNode(mutation, *input_node_def, &new_node_def);
    } else {
      // If this is an MKL op, then it will generate an edge that will receive
      // OneDNN tensor from a node.
      int input_node_output_size = OpDefOutputPorts(input_node_def);
      const int out_slot_meta = ParseTensorName(node_def->input(idx)).index() +
                                input_node_output_size / 2;
      new_node_def.add_input(GetInputName(input_node_def, out_slot_meta));
    }
  }

  // Set up ws meta tensor.
  if (ws_input_node_def != nullptr) {
    AddDummyOneDnnNode(mutation, *ws_input_node_def, &new_node_def);
    ITEX_VLOG(3) << "Workspace: Add workspace OneDnn meta tensor between ["
                 << ws_input_node_def->op() << "] and [" << new_node_def.op()
                 << "], while rewriting [" << node_def->op() << "]";
  }

  ri->copy_attrs(node_view, &new_node_def);

  // Incoming data edges from 'orig_node' node to new 'new_node' node are
  // already copied in BuildNode. We need to handle control edges now.
  for (int idx = 0; idx < node_view->NumControllingFanins(); idx++) {
    new_node_def.add_input(
        node_def->input(node_view->NumRegularFanins() + idx));
  }

  ITEX_VLOG(4) << "Rewritten node: " << new_node_def.DebugString();

  // apply mutation
  Status status;
  mutation->AddNode(std::move(new_node_def), &status);
  TF_ABORT_IF_ERROR(std::move(status));
  TF_ABORT_IF_ERROR(mutation->Apply());

  return Status::OK();
}

const RewriteInfo* CheckForNodeRewrite(
    const utils::MutableNodeView& node_view) {
  NodeDef& node_def = *(node_view.node());
  // TODO(itex): Enable quantized.
  if (node_def.op() != onednngrap_op_name) {
    // First check if node along with its type is supported by OneDNN.
    // Do not rewrite an op if types are not supported.
    // E.g., OneDnnRelu does not support INT32.
    if (!IsLayoutRewriteSupportedDataType(node_def)) return nullptr;
  }

  // We now check if rewrite rule applies for this op. If rewrite rule passes
  // for this op, then we rewrite it to OneDNN op.
  // Find matching RewriteInfo and then check that rewrite rule applies.
  const std::vector<RewriteInfo>* rinfo = GetRewriteInfo();
  for (auto ri = rinfo->cbegin(); ri != rinfo->cend(); ++ri) {
    if (node_def.op().compare(ri->name) == 0 && ri->rewrite_rule(node_view)) {
      return &*ri;
    }
  }

  // Else return not found.
  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
//              Insert conversion nodes
///////////////////////////////////////////////////////////////////////////////
Status InsertConversionNode(OneDnnLayoutContext* ctx, const int node_index) {
  // node index of _OneDnnToTf.
  static int conversion_node_idx = 0;

  const auto* node_view = ctx->graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  auto* mutation = ctx->graph_view.GetMutationBuilder();

  // Check whether dst is PlainLayoutOp
  bool dst_is_plain_op = (IsPlainLayoutOp(node_def->op()));
  if (!dst_is_plain_op) return Status::OK();

  for (int idx = 0; idx < node_view->NumRegularFanins(); idx++) {
    const auto* input_node_view = node_view->GetRegularFanin(idx).node_view();
    const auto* input_node_def = input_node_view->node();

    bool src_is_onednn_op = (IsOneDnnLayoutDependentOp(input_node_def->op()));

    // Check if src is OneDnnLayoutDependentOp
    if (!src_is_onednn_op) continue;

    if (input_node_def->op() == _onednngrap_op_name) continue;

    TypeAttrId input_type_attr =
        ctx->node_type_map.GetInputTypeAttr(*node_def, idx);

    // Calculate the output slot of previous op
    // Get an OneDNN tensor slot from the Tf tensor slot
    TensorId input = ParseTensorName(node_def->input(idx));
    int out_slot = input.index();
    TypeAttrId output_type_attr =
        ctx->node_type_map.GetOutputTypeAttr(*input_node_def, out_slot);

    DataType in = GetDataType(*node_def, input_type_attr);
    DataType out = GetDataType(*input_node_def, output_type_attr);

    // TODO(itex): Do we really need to check the input and output datatype
    // is the same.
    if (in != out) {
      string err_msg = "T attribute of " + input_node_def->name() + " and " +
                       node_def->name() +
                       " do not match. _OneDnnToTf node will not be inserted.";
      return Status(TF_Code::TF_INVALID_ARGUMENT, err_msg.c_str());
    }

    NodeDef conversion_node;
    string conversion_node_name =
        "OneDnn2Tf_" + std::to_string(conversion_node_idx++);
    conversion_node.set_name(conversion_node_name);
    conversion_node.set_op("_OneDnnToTf");
    conversion_node.set_device(input_node_def->device());
    conversion_node.add_input(node_def->input(idx));

    // distance between output data slot and meta slot is num_outputs / 2
    int out_slot_meta =
        out_slot + ctx->node_type_map.GetOutputSize(*input_node_def) / 2;
    TensorId meta_input(input.node(), out_slot_meta);
    conversion_node.add_input(meta_input.ToString());

    auto* attr = conversion_node.mutable_attr();
    string data_format;
    if (GetNodeAttr(*input_node_def, "data_format", &data_format) ==
            Status::OK() &&
        (data_format == ToString(FORMAT_NHWC) ||
         data_format == ToString(FORMAT_NCHW))) {
      SetAttrValue(data_format, &(*attr)["data_format"]);
    }
    SetAttrValue(in, &(*attr)["T"]);

    if (ITEX_VLOG_IS_ON(4)) {
      string before_conversion;
      protobuf::TextFormat::PrintToString(*input_node_def, &before_conversion);
      ITEX_VLOG(4) << "Original Before Node" << before_conversion;

      string after_conversion;
      protobuf::TextFormat::PrintToString(*node_def, &after_conversion);
      ITEX_VLOG(4) << "Original After Node" << after_conversion;

      string conversion;
      protobuf::TextFormat::PrintToString(conversion_node, &conversion);
      ITEX_VLOG(4) << "Conversion Node" << conversion;
    }

    // add edge from output of conversion_node to the dest node. Since
    // conversion_node has only 1 output, the src_output of conversion_node is
    // 0.
    Status status;
    mutation->AddNode(std::move(conversion_node), &status);
    TF_ABORT_IF_ERROR(std::move(status));
    TensorId output(conversion_node_name, 0);
    mutation->AddOrUpdateRegularFanin(
        const_cast<utils::MutableNodeView*>(node_view), idx, output);
  }
  // TODO(itex): Investigate why mutation->Apply() will cause
  // node_view->NumRegularFanins() = 0
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

// Since ITEX block-layout op and OneDnnGraph op don't recognize layout of each
// other, there should be a conversion node between them. This condition is
// different from the conversion node between block op and plain op, since the
// latter node is also a block node.
//
// 1) Conversion node between block and plain node.
//
//    block node
//        |
//  conversion node
//        |
//    plain node
//
// 2) Conversion node between block and OneDnnGraph node.
//
//    block node
//        |
//  conversion node      dummy meta node
//               \            /
//             onednngraph node
Status InsertConversionForLLGANode(OneDnnLayoutContext* ctx,
                                   const int node_index) {
  // node index of _OneDnnToTf.
  static int conversion_node_idx = 0;

  const auto* node_view = ctx->graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();

  // Check whether dst is PlainLayoutOp
  bool dst_is_llga_op = (node_def->op() == _onednngrap_op_name);
  if (!dst_is_llga_op) return Status::OK();

  // Here we have "_OneDnnGraph" node, and half of inputs are meta nodes
  for (int idx = 0; idx < node_view->NumRegularFanins() / 2; idx++) {
    const auto* input_node_view = node_view->GetRegularFanin(idx).node_view();
    const auto* input_node_def = input_node_view->node();

    bool src_is_onednn_op = (IsOneDnnLayoutDependentOp(input_node_def->op()) &&
                             input_node_def->op() != _onednngrap_op_name);

    // Check if src is OneDnnLayoutDependentOp
    if (!src_is_onednn_op) continue;

    TypeAttrId input_type_attr =
        ctx->node_type_map.GetInputTypeAttr(*node_def, idx);

    // Calculate the output slot of previous op
    // Get an OneDNN tensor slot from the Tf tensor slot
    TensorId input = ParseTensorName(node_def->input(idx));
    int out_slot = input.index();
    TypeAttrId output_type_attr =
        ctx->node_type_map.GetOutputTypeAttr(*input_node_def, out_slot);

    DataType in = GetDataType(*node_def, input_type_attr);
    DataType out = GetDataType(*input_node_def, output_type_attr);

    // TODO(itex): Do we really need to check the input and output datatype
    // is the same.
    if (in != out) {
      string err_msg = "T attribute of " + input_node_def->name() + " and " +
                       node_def->name() +
                       " do not match. _OneDnnToTf node will not be inserted.";
      return Status(TF_Code::TF_INVALID_ARGUMENT, err_msg.c_str());
    }

    NodeDef conversion_node;
    // Here the conversion node name has "LLGA" to distinguish from normal
    // conversion node
    string conversion_node_name =
        "OneDnn2Tf_LLGA_" + std::to_string(conversion_node_idx++);
    conversion_node.set_name(conversion_node_name);
    conversion_node.set_op("_OneDnnToTf");
    conversion_node.set_device(input_node_def->device());
    conversion_node.add_input(node_def->input(idx));

    // distance between output data slot and meta slot is num_outputs / 2
    int out_slot_meta =
        out_slot + ctx->node_type_map.GetOutputSize(*input_node_def) / 2;
    TensorId meta_input(input.node(), out_slot_meta);
    conversion_node.add_input(meta_input.ToString());

    auto* attr = conversion_node.mutable_attr();
    string data_format;
    if (GetNodeAttr(*input_node_def, "data_format", &data_format) ==
            Status::OK() &&
        (data_format == ToString(FORMAT_NHWC) ||
         data_format == ToString(FORMAT_NCHW))) {
      SetAttrValue(data_format, &(*attr)["data_format"]);
    }

    SetAttrValue(in, &(*attr)["T"]);

    if (ITEX_VLOG_IS_ON(4)) {
      string before_conversion;
      protobuf::TextFormat::PrintToString(*input_node_def, &before_conversion);
      ITEX_VLOG(4) << "Original Before Node" << before_conversion;

      string after_conversion;
      protobuf::TextFormat::PrintToString(*node_def, &after_conversion);
      ITEX_VLOG(4) << "Original After Node" << after_conversion;

      string conversion;
      protobuf::TextFormat::PrintToString(conversion_node, &conversion);
      ITEX_VLOG(4) << "Conversion Node" << conversion;
    }

    // add edge from output of conversion_node to the dest node. Since
    // conversion_node has only 1 output, the src_output of conversion_node is
    // 0.

    // Add dummy node is needed for _OneDnnGraph op only
    UpdateDummyOneDnnNode(mutation, *input_node_def,
                          const_cast<utils::MutableNodeView*>(node_view),
                          idx + node_view->NumRegularFanins() / 2);

    Status status;
    mutation->AddNode(std::move(conversion_node), &status);
    TF_ABORT_IF_ERROR(status);
    TensorId output(conversion_node_name, 0);
    mutation->AddOrUpdateRegularFanin(
        const_cast<utils::MutableNodeView*>(node_view), idx, output);
  }
  // TODO(itex): Investigate why mutation->Apply() will cause
  // node_view->NumRegularFanins() = 0
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////
//              Convert Meta Node from Const to HostConst
///////////////////////////////////////////////////////////////////////////////
Status ConvertMetaNodeFromConstToHostConst(OneDnnLayoutContext* ctx,
                                           const int node_index) {
  auto* node_view = ctx->graph_view.GetNode(node_index);
  auto* node_def = node_view->node();

  // These nodes actually are wrong converted to op "Const" by Tensorflow Proper
  // constant-folding optimizer, we need to restore the op type.
  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  // "_DMT_" is the special name to distinguish meta node
  if (IsConstant(*node_def) && node_def->name().find("_DMT_") != string::npos) {
    mutation->UpdateNodeOp(node_view, "HostConst");
  }
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////
//              Mark end node of OneDnn Graph partition
///////////////////////////////////////////////////////////////////////////////
Status MarkOneDnnGraphEndNode(OneDnnLayoutContext* ctx, const int node_index) {
  const auto* node_view = ctx->graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def->op() != _onednngrap_op_name) return Status::OK();
  // Can not use NumRegularFanouts here since the output edges of the last
  // node is not doubled.
  std::vector<DataType> Tout;
  ITEX_CHECK_OK(GetNodeAttr(*node_def, "Tout", &Tout));
  int regular_fanouts_size = Tout.size();
  std::vector<bool> is_end_node(regular_fanouts_size, false);
  for (int idx = 0; idx < regular_fanouts_size; idx++) {
    const auto& output_node_view_list = node_view->GetRegularFanout(idx);
    for (size_t i = 0; i < output_node_view_list.size(); i++) {
      const auto* output_node_view = output_node_view_list[i].node_view();
      const auto* output_node_def = output_node_view->node();

      if (output_node_def->op() != _onednngrap_op_name) {
        is_end_node[idx] = true;
        break;
      }
    }
  }

  auto* mutation = ctx->graph_view.GetMutationBuilder();
  AttrValue is_end_node_attr;
  SetAttrValue(is_end_node, &is_end_node_attr);
  mutation->AddOrUpdateNodeAttr(const_cast<utils::MutableNodeView*>(node_view),
                                "is_end_node", is_end_node_attr);
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////
//              Run function for the pass
///////////////////////////////////////////////////////////////////////////////
Status RunOneDnnLayout(OptimizerContext* opt_ctx, const GrapplerItem& item,
                       const GraphDef& graph_def, GraphDef* optimized_graph) {
  Status status;
  GraphDef multable_graph_def = graph_def;
  OneDnnLayoutContext ctx(item, &multable_graph_def, &status);

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_ABORT_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  // Skip nodes that were invalidated
  int num_nodes = multable_graph_def.node_size();

  ITEX_VLOG(1) << "OneDnnLayoutPass: Start to rewrite nodes.";

  for (int node_index = 0; node_index < num_nodes; ++node_index) {
    const auto* node_view = ctx.graph_view.GetNode(node_index);
    const auto* node_def = node_view->node();

    // Check if node can run on current optimizer device.
    if (!NodeIsOnDevice(opt_ctx->device_name, node_def)) continue;

    // Check if node is fp16 and supported on device.
    if (NodeIsOnCpu(node_def) &&
        GetDataTypeFromAttr(*node_def, "T") == DT_HALF &&
        !port::HasCpuFP16Support())
      continue;

    // Don't rewrite fetch node because layout will insert `OneDnnToTf` op
    // behind it and break the fetch node dependency.
    // TODO(itex): Rewrite fetch nodes if meeting performance regression.
    if (ctx.nodes_to_preserve.count(node_def->name()) > 0) continue;

    const RewriteInfo* ri = nullptr;
    // We will first search if node is to be rewritten.
    if ((ri = CheckForNodeRewrite(*node_view)) != nullptr) {
      string node_name = node_def->name();
      string op_name = node_def->op();
      ITEX_VLOG(1) << "OneDnnLayoutPass: Scheduled node " << node_name
                   << " with OP " << op_name << " for rewrite using"
                   << " layout optimization.";

      if (RewriteNode(&ctx, node_index, ri) == Status::OK()) {
        ITEX_VLOG(2) << "OneDnnLayoutPass: rewrote node " << node_name
                     << " with op " << op_name
                     << " for OneDNN layout optimization.";
      } else {
        ITEX_VLOG(2) << "OneDnnLayoutPass: found node " << node_name
                     << " with op " << op_name << " but rewrite failed.";
      }
    }
  }

#define RUN_LAYOUT_FUNC(ctx, func)                                      \
  do {                                                                  \
    TF_ABORT_IF_ERROR(                                                  \
        ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {})); \
    TF_ABORT_IF_ERROR(ctx.node_type_map.Clear());                       \
    TF_ABORT_IF_ERROR(ctx.node_type_map.Init(*ctx.graph_view.graph())); \
    for (int node_index = ctx.graph_view.graph()->node_size() - 1;      \
         node_index >= 0; --node_index) {                               \
      TF_ABORT_IF_ERROR(func(&ctx, node_index));                        \
    }                                                                   \
  } while (0)

  // Run necessary post functors after rewriting nodes.
  RUN_LAYOUT_FUNC(ctx, InsertConversionNode);
  RUN_LAYOUT_FUNC(ctx, InsertConversionForLLGANode);
  RUN_LAYOUT_FUNC(ctx, ConvertMetaNodeFromConstToHostConst);
  RUN_LAYOUT_FUNC(ctx, MarkOneDnnGraphEndNode);

#undef RUN_LAYOUT_FUNC

  *optimized_graph = std::move(multable_graph_def);
  return Status::OK();
}

}  // namespace graph
}  // namespace itex
