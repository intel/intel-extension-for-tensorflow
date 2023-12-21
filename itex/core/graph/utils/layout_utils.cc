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

#include "itex/core/graph/utils/layout_utils.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/op_def_util.h"

namespace itex {
namespace graph {

//////////////////////////////////////////////////////////////////////////
// Rewrite functions
//////////////////////////////////////////////////////////////////////////

bool AlwaysRewrite(const utils::MutableNodeView& node_view) { return true; }

// Rewrite when inputs have block layout ops.
bool RewriteWithBlockInput(const utils::MutableNodeView& node_view) {
  // Check whether inputs have block ops
  int num_inputs = node_view.NumRegularFanins();
  for (int i = 0; i < num_inputs; ++i) {
    const NodeDef* input_node_def =
        node_view.GetRegularFanin(i).node_view()->node();
    string input_node_op = input_node_def->op();
    if (IsOneDnnLayoutDependentOp(input_node_op)) return true;
  }

  return false;
}

// Rewrite rule for binary ops
bool RewriteBinary(const utils::MutableNodeView& node_view) {
  return NodeIsOnGpu(node_view.node()) && RewriteWithBlockInput(node_view);
}

// Rewrite rule for Cast op:
//   1. Only rewrite if data type can be optimized by oneDNN
//   2. Only rewrite if predecessor is oneDNN op for layout propagation
bool RewriteCast(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());

  // Do not rewrite on GPU
  if (NodeIsOnGpu(&node_def)) return false;

  DataType T;

  ITEX_CHECK_OK(GetNodeAttr(node_def, "SrcT", &T));
  if (!(T == DataType::DT_FLOAT || T == DataType::DT_BFLOAT16 ||
        T == DataType::DT_HALF))
    return false;

  ITEX_CHECK_OK(GetNodeAttr(node_def, "DstT", &T));
  if (!(T == DataType::DT_FLOAT || T == DataType::DT_BFLOAT16 ||
        T == DataType::DT_HALF))
    return false;

  return RewriteWithBlockInput(node_view);
}

bool RewriteBackwardDataType(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  DataType T;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "T", &T));

  if (T != DataType::DT_FLOAT && T != DataType::DT_BFLOAT16)
    return false;
  else
    return true;
}

bool RewriteLayerNorm(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  string data_format;
  // for layernorm, it only supports LDNC(NHWC)
  ITEX_CHECK_OK(GetNodeAttr(node_def, "data_format", &data_format));

  if (data_format == "NCHW") return false;
  return true;
}

bool RewriteLayerNormGrad(const utils::MutableNodeView& node_view) {
  if (!RewriteLayerNorm(node_view)) return false;
  if (!RewriteBackwardDataType(node_view)) return false;
  return true;
}

bool RewriteFusedBatchNormEx(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  int num_side_inputs;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "num_side_inputs", &num_side_inputs));
  string activation_mode;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "activation_mode", &activation_mode));
  if (NodeIsOnGpu(node_view.node())) return true;
  bool is_bn_add = (num_side_inputs != 0) || (activation_mode != "Relu");
  // CPU did not support this fusion and remapper should not rewrite, add
  // defence code here.
  ITEX_CHECK(!is_bn_add);
  return true;
}

bool RewriteFusedBatchNormExGrad(const utils::MutableNodeView& node_view) {
  if (!RewriteBackwardDataType(node_view)) return false;

  const NodeDef& node_def = *(node_view.node());
  string activation_mode;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "activation_mode", &activation_mode));
  if (activation_mode != "ReluGrad") return false;
  return true;
}

static const std::vector<string>* GetPotentialOneDnnOpList() {
  static std::vector<string> onednn_op_list{
      "Add",          "AddN",     "AvgPool",        "Cast",
      "Concat",       "Conv",     "FusedBatchNorm", "Gelu",
      "InstanceNorm", "Identity", "LayerNorm",      "MatMul",
      "Mish",         "Mul",      "MaxPool",        "Quantize",
      "RealDiv",      "Relu",     "Reshape",        "Resize",
      "Slice",        "Sub",      "Swish",
  };
  return &onednn_op_list;
}

bool RewriteOneDnnConv(const utils::MutableNodeView& node_view) {
  // If next node of conv is bn, we enforce conv + bn to plain format
  for (auto const& fanout : node_view.GetRegularFanouts()) {
    if (fanout.size() < 1) continue;
    for (auto const& fanout_i : fanout) {
      auto const* fanout_node_view = fanout_i.node_view();
      if (!fanout_node_view) continue;
      auto const& crt_op_name = fanout_node_view->node()->op();
      const std::string onednn_op = "FusedBatchNorm";
      if (crt_op_name.find(onednn_op) != string::npos) {
        return false;
      }
    }
  }

  // RewriteWithBlockInput
  for (int i = 0; i < node_view.NumRegularFanins(); ++i) {
    const NodeDef* input_node_def =
        node_view.GetRegularFanin(i).node_view()->node();
    if (IsOneDnnLayoutDependentOp(input_node_def->op())) {
      return true;
    }
  }

  const std::vector<string>* potential_onednn_op = GetPotentialOneDnnOpList();
  if (node_view.NumRegularFanouts() < 1) return false;
  for (auto const& fanout : node_view.GetRegularFanouts()) {
    if (fanout.size() < 1) continue;
    for (auto const& fanout_i : fanout) {
      auto const* fanout_node_view = fanout_i.node_view();
      if (!fanout_node_view) continue;
      auto const& crt_op_name = fanout_node_view->node()->op();
      for (auto onednn_op : *potential_onednn_op) {
        if (crt_op_name.find(onednn_op) != string::npos) {
          if (onednn_op == "Conv" || onednn_op == "MatMal") {
            return true;
          } else {
            if (fanout_node_view->NumRegularFanouts() < 1) continue;
            for (auto const& next_op_output :
                 fanout_node_view->GetRegularFanouts()) {
              if (next_op_output.size() < 1) continue;
              for (auto const& next_op_output_i : next_op_output) {
                auto const* next_op_output_node_view =
                    next_op_output_i.node_view();
                if (!next_op_output_node_view) continue;
                auto const& next_op_name =
                    next_op_output_node_view->node()->op();
                for (auto onednn_op : *potential_onednn_op) {
                  if (next_op_name.find(onednn_op) != string::npos) {
                    return true;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}

bool RewriteFusedConv(const utils::MutableNodeView& node_view) {
  // OneDnn currently doesn't support all fusions that grappler fuses
  // together with Conv2D/Conv3D (ex. batchnorm). We rewrite
  // _FusedITEXConv2D/_FusedITEXConv3D only if it includes those we support.
  const NodeDef& node_def = *(node_view.node());
  std::vector<string> fused_ops;
  PostOpUtil post_op_util;

  ITEX_CHECK_OK(GetNodeAttr(node_def, "fused_ops", &fused_ops));

  return post_op_util.AddOps(fused_ops);
}

bool RewriteOneDnnFusedConv(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  std::vector<string> fused_ops;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "fused_ops", &fused_ops));
  for (auto& post_op : fused_ops) {
    if (post_op == "FusedBatchNorm") return false;
  }

  return RewriteFusedConv(node_view) && RewriteOneDnnConv(node_view);
}

bool RewriteMatMul(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());

  // Temporarily rewrite MatMul-like ops for CPU unconditionally.
  // TODO(itex): Remove this condition once MatMul blocked format is
  // supported on CPU.
  if (NodeIsOnCpu(&node_def)) return true;
  if (NodeIsOnGpu(&node_def)) return false;

  // Deal with input data
  bool trans_a;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "transpose_a", &trans_a));
  if (trans_a) return false;

  bool trans_b;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "transpose_b", &trans_b));
  if (trans_b) return false;

  return true;
}

bool RewriteConv2DBackprop(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());

  // Check padding type.
  // TODO(itex): Remove this limitation once it's supported.
  string padding;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "padding", &padding));
  if (padding == "EXPLICIT") return false;

  if (!RewriteBackwardDataType(node_view)) return false;

  return true;
}

bool RewritePool(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());

  // Check batch-wise or depth-wise pooling.
  string data_format_str;
  TensorFormat data_format;
  std::vector<int32> ksize, strides;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "ksize", &ksize));
  ITEX_CHECK_OK(GetNodeAttr(node_def, "strides", &strides));
  ITEX_CHECK_OK(GetNodeAttr(node_def, "data_format", &data_format_str));
  ITEX_CHECK(FormatFromString(data_format_str, &data_format));

  if (GetTensorDim(ksize, data_format, 'N') != 1 ||
      GetTensorDim(strides, data_format, 'N') != 1 ||
      GetTensorDim(ksize, data_format, 'C') != 1 ||
      GetTensorDim(strides, data_format, 'C') != 1)
    return false;
  return true;
}

bool RewriteOneDnnPool(const utils::MutableNodeView& node_view) {
  return RewritePool(node_view) && RewriteWithBlockInput(node_view);
}

bool RewriteMaxPoolGrad(const utils::MutableNodeView& node_view) {
  // Input1 of MaxPoolGrad should be _OneDnnMaxPool or _ITEXMaxPool
  const auto& regular_fanin_1 = node_view.GetRegularFanin(1);
  const auto* maxpool_node_view = regular_fanin_1.node_view();
  string op_name = maxpool_node_view->node()->op();
  if (!(op_name.substr(0, 7) == "_OneDnn" || op_name.substr(0, 5) == "_ITEX"))
    return false;
  if (op_name.find("MaxPool") == std::string::npos) return false;

  // Output0 of _OneDnnMaxPool/_ITEXMaxPool should be a tensor referenced by
  // MaxPoolGrad.
  for (auto fanout : maxpool_node_view->GetRegularFanout(0)) {
    if (fanout.node_view()->node_index() == node_view.node_index()) return true;
  }
  return false;
}

bool RewriteRandomUniform(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());

  // CPU _ITEXRandomUniform doesn't support fp16.
  DataType T;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "dtype", &T));
  if (NodeIsOnCpu(&node_def) && T == DT_HALF) return false;
  return true;
}

bool RewriteQuantize(const utils::MutableNodeView& node_view) {
  // TODO(itex): in intel-tf constant input data is not rewrite. Not sure
  // why there is such setting. In plugin, we allow such rewrite to enable
  // python UT

  // TODO(itex): Actually, in intel-tf proper, only narrow_range = True is
  // enabled. We could correct this error in the future.

  const NodeDef& node_def = *(node_view.node());

  // Quantize mode check
  string mode_string;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "mode", &mode_string));
  if (mode_string == "MIN_COMBINED") {
    ITEX_VLOG(2) << "MIN_COMBINED are not supported yet";
    return false;
  }

  // oneDNN doesn't support reorder primitive with zeropoint attributes
  if (mode_string == "MIN_FIRST" && node_def.op() == "Dequantize" &&
      NodeIsOnGpu(node_view.node())) {
    ITEX_VLOG(2) << "GPU Dequantize with MIN_FRIST mode are not supported yet";
    return false;
  }

  // Round mode check
  string round_mode_string;
  if (TryGetNodeAttr(node_def, "round_mode", &round_mode_string)) {
    // Only Quantize op has round mode attr, Dequantize op doesn't have
    if (mode_string == "SCALED" && !(round_mode_string == "HALF_TO_EVEN")) {
      ITEX_VLOG(2)
          << "SCALED mode only supports HALF_TO_EVEN round mode"
          << "This case is not optimized by OneDnn, thus using Eigen op"
          << "for Quantize op ";
      return false;
    }
  }

  return true;
}

bool RewriteResize(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());

  bool align_corners;
  ITEX_CHECK_OK(GetNodeAttr(node_def, "align_corners", &align_corners));

  bool half_pixel_centers;
  ITEX_CHECK_OK(
      GetNodeAttr(node_def, "half_pixel_centers", &half_pixel_centers));

  if (align_corners == false && half_pixel_centers == true) {
    return true;
  }

  return false;
}

// Rewrite rule for Cast op:
//   1. Only rewrite if data type can be optimized by oneDNN
bool RewriteNativeCast(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  DataType T;

  ITEX_CHECK_OK(GetNodeAttr(node_def, "SrcT", &T));
  if (!(T == DataType::DT_FLOAT || T == DataType::DT_BFLOAT16 ||
        T == DataType::DT_HALF))
    return false;

  ITEX_CHECK_OK(GetNodeAttr(node_def, "DstT", &T));
  if (!(T == DataType::DT_FLOAT || T == DataType::DT_BFLOAT16 ||
        T == DataType::DT_HALF))
    return false;

  return true;
}

bool RewriteQuantizeReshape(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  DataType T;

  ITEX_CHECK_OK(GetNodeAttr(node_def, "T", &T));
  if (T != DataType::DT_QINT8) return false;

  return true;
}

//////////////////////////////////////////////////////////////////////////
// Op-specific functions to copy attributes from old node to new node
//////////////////////////////////////////////////////////////////////////

// _OneDnnCast has another attr T.
void CopyAttrsCast(const utils::MutableNodeView* orig_node_view,
                   NodeDef* new_node) {
  CopyAttrsAll(orig_node_view, new_node);

  DataType DstT;
  ITEX_CHECK_OK(GetNodeAttr(*(orig_node_view->node()), "DstT", &DstT));

  // Layout pass always check datatype by attr name T, So we need add T
  // attribution for _OneDnnCast.
  auto* new_attr = new_node->mutable_attr();
  SetAttrValue(DstT, &(*new_attr)["T"]);
}

void CopyAttrsAll(const utils::MutableNodeView* orig_node_view,
                  NodeDef* new_node) {
  CopyAllAttrs(*(orig_node_view->node()), new_node);
}

void CopyAttrsForTensorArray(const utils::MutableNodeView* orig_node_view,
                             NodeDef* new_node) {
  CopyAttrsAll(orig_node_view, new_node);

  // Check and set filter attribute.
  auto* new_attr = new_node->mutable_attr();

  PartialTensorShape partial_shape;
  if (TryGetNodeAttr(*new_node, "element_shape", &partial_shape) ||
      TryGetNodeAttr(*new_node, "element_shape_except0", &partial_shape)) {
    int32_t rank = partial_shape.dims();
    SetAttrValue(rank, &(*new_attr)["num_dims_of_element_shape"]);
    if (rank == -1) {
      std::vector<int64_t> dims;
      SetAttrValue(dims, &(*new_attr)["dims_of_element_shape"]);
    } else {
      std::vector<int64_t> dims(rank);
      for (int32_t i = 0; i < rank; ++i) {
        dims[i] = static_cast<int64_t>(partial_shape.dim_size(i));
      }
      SetAttrValue(dims, &(*new_attr)["dims_of_element_shape"]);
    }
  }
}

// Function to copy attributes of OneDnnGraph
void CopyAttrsOneDnnGraph(const utils::MutableNodeView* orig_node_view,
                          NodeDef* new_node) {
  CopyAttrsAll(orig_node_view, new_node);

  const NodeDef* orig_node = orig_node_view->node();
  std::vector<DataType> Tin;
  std::vector<DataType> Tout;
  ITEX_CHECK_OK(GetNodeAttr(*orig_node, "Tin", &Tin));
  ITEX_CHECK_OK(GetNodeAttr(*orig_node, "Tout", &Tout));

  std::vector<DataType> Tin_meta(Tin.size(), DT_UINT8);
  std::vector<DataType> Tout_meta(Tout.size(), DT_UINT8);
  std::vector<bool> is_end_node(Tout.size(), false);

  auto* attr = new_node->mutable_attr();
  SetAttrValue(Tin_meta, &(*attr)["Tin_meta"]);
  SetAttrValue(Tout_meta, &(*attr)["Tout_meta"]);
  SetAttrValue(is_end_node, &(*attr)["is_end_node"]);
}

void CopyAttrsQuantizedConv2D(const utils::MutableNodeView* orig_node_view,
                              NodeDef* new_node) {
  // QuantizdConv2D filter is always const, no need to further check
  CopyAttrsAll(orig_node_view, new_node);

  // Get all attributes from old node.
  const NodeDef* orig_node_def = orig_node_view->node();
  DataType out_type;
  ITEX_CHECK_OK(GetNodeAttr(*orig_node_def, "out_type", &out_type));

  // Add attributes to new node.
  auto* new_attr = new_node->mutable_attr();

  // TODO(itex): avoid hardcode "NHWC" for QuantizedConv2D
  string data_format("NHWC");
  SetAttrValue(data_format, &(*new_attr)["data_format"]);

  // Tbias is only valid for quantized op meet 2 requirment
  // 1. fused with BiasAdd
  // 2. fused with Requantize or Dequantize
  DataType Tbias;
  if (TryGetNodeAttr(*orig_node_def, "Tbias", &Tbias)) {
    SetAttrValue(Tbias, &(*new_attr)["Tbias"]);
  }
}

void CopyAttrsQuantizedMatMul(const utils::MutableNodeView* orig_node_view,
                              NodeDef* new_node) {
  // QuantizdMatMul filter is always const, no need to further check
  CopyAttrsAll(orig_node_view, new_node);

  // Get all attributes from old node.
  const NodeDef* orig_node_def = orig_node_view->node();
  DataType out_type;
  ITEX_CHECK_OK(GetNodeAttr(*orig_node_def, "Toutput", &out_type));

  // Add attributes to new node.
  auto* new_attr = new_node->mutable_attr();

  DataType Tbias;
  if (TryGetNodeAttr(*orig_node_def, "Tbias", &Tbias)) {
    SetAttrValue(Tbias, &(*new_attr)["Tbias"]);
  }
}

void CopyAttrsQuantize(const utils::MutableNodeView* orig_node_view,
                       NodeDef* new_node) {
  CopyAttrsAll(orig_node_view, new_node);

  // Get all attributes from old node.
  const NodeDef* orig_node_def = orig_node_view->node();
  // Add attributes to new node.
  auto* new_attr = new_node->mutable_attr();

  DataType dtype = DataType::DT_FLOAT;
  if (!HasNodeAttr(*orig_node_def, "dtype")) {
    // QuantizeV2 condition
    SetAttrValue(dtype, &(*new_attr)["dtype"]);
  }

  bool is_onednn_graph_int8_graph = true;
  // For oneDNN Graph INT8 pb, QuantizeV2's outputs are always Dequantize
  for (auto fanout : orig_node_view->GetRegularFanout(0)) {
    auto* fanout_node_view = fanout.node_view();
    if (fanout_node_view->node()->op() != "Dequantize") {
      is_onednn_graph_int8_graph = false;
    }
  }

  SetAttrValue(is_onednn_graph_int8_graph,
               &(*new_attr)["classic_asymmetric_algorithm"]);
}

// Check whether opname with type T is registered as oneDNN operator
// that can accept input tensors in oneDNN layout.
//
// @input: name of the op
// @return: true if opname is registered as OneDNN-layout dependent op;
// false otherwise

/////////////////////////////////////////////////////////////////////
//  OneDnnLayoutDependentOp:        Input:  Data Tensor + Meta Tensor
//                                  Output: Data Tensor + Meta Tensor
//  OneDnnLayoutPartialDependentOp  Input:  Data Tensor + Meta Tensor
//                                  Output: Data Tensor
//  PlainLayoutOp                   Input:  Data Tensor
//                                  Output: Data Tensor
////////////////////////////////////////////////////////////////////

bool IsOneDnnLayoutPartialDependentOp(const string& op_name) {
  // PartialDependent op means that op can have OneDnn layout input, but
  // plain(Eigen) layout output only
  static const std::unordered_set<string> PartialDependentOp = {
      "_OneDnnFusedDequantizeWithReshape",
      "_OneDnnQuantizedReshape",
      "_OneDnnQuantizedTranspose",
      "_OneDnnReshape",
      "_OneDnnShape",
      "_OneDnnToTf",
      "_OneDnnTranspose"};
  return PartialDependentOp.find(op_name) != PartialDependentOp.end();
}

// Dependent op means that op can have both OneDnn layout input and output
bool IsOneDnnLayoutDependentOp(const string& op_name) {
  return op_name.substr(0, 7) == "_OneDnn" &&
         !IsOneDnnLayoutPartialDependentOp(op_name);
}

// PlainLayout op means normal Eigen op. Input and output don't have meta
// tensor.
bool IsPlainLayoutOp(const string& op_name) {
  return !(op_name.substr(0, 7) == "_OneDnn");
}

bool IsQuantizedOp(const string& op_name) {
  static const std::unordered_set<string> QuantizedOp = {
      "Dequantize",
      "QuantizedAvgPool",
      "QuantizedConcatV2",
      "QuantizedConv2D",
      "QuantizedConv2DPerChannel",
      "QuantizedMaxPool",
      "QuantizedReshape",
      "QuantizeV2",
  };
  return QuantizedOp.find(op_name) != QuantizedOp.end();
}

// Some INT8 ops are only registered by intel tensorflow or ITEX. No need to
// check these ops.
bool IsDataTypeExemptOp(const string& op_name) {
  static const std::unordered_set<string> DataTypeExemptOp = {
      "_ITEXFusedDequantizeWithReshape",
      "ITEXQuantizedAvgPool",
      "QuantizedConcatV2",
      "QuantizedConv2DAndRequantize",
      "QuantizedConv2DWithBias",
      "QuantizedConv2DWithBiasAndRequantize",
      "QuantizedConv2DWithBiasAndRelu",
      "QuantizedConv2DWithBiasAndReluAndRequantize",
      "QuantizedConv2DWithBiasSumAndRelu",
      "QuantizedConv2DWithBiasSumAndReluAndRequantize",
      "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
      "QuantizedDepthwiseConv2D",
      "QuantizedDepthwiseConv2DWithBias",
      "QuantizedDepthwiseConv2DWithBiasAndRelu",
      "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
      "QuantizedMatMulWithBiasAndRelu",
      "QuantizedMatMulWithBias",
      "QuantizedMatMulWithBiasAndReluAndRequantize",
      "QuantizedMatMulWithBiasAndRequantize",
      "QuantizedMatMulWithBiasAndDequantize",
      // Below ops are registered in TF, but ITEX can always rewrite them.
      "Cast",
      "QuantizedReshape",
      "Shape",

      // New INT8 ops
      "_ITEXQuantizedConv2D",
      "_ITEXQuantizedConv2DAndRequantize",
      "_ITEXQuantizedConv2DWithBias",
      "_ITEXQuantizedMatMulWithBiasAndDequantize",
      "_ITEXQuantizedConv2DWithBiasAndRequantize",
      "_ITEXQuantizedConv2DWithBiasAndRelu",
      "_ITEXQuantizedConv2DWithBiasAndReluAndRequantize",
      "_ITEXQuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
      "_ITEXQuantizedConv2DWithBiasSumAndRelu",
      "_ITEXQuantizedConv2DWithBiasSumAndReluAndRequantize",
      "_ITEXQuantizedConv2DWithDequantize",
      "_ITEXQuantizedConv2DWithCast",
      "_ITEXQuantizeV2",
      "_ITEXQuantizeV2WithQuantizedConv2D",
      "_QuantizedBatchMatMul",
      "_QuantizedBatchMatMulV2AndDequantize",
      "_QuantizedConv2D",
      "_QuantizedConv3D",
      "_QuantizedDepthwiseConv2D",
      "_QuantizedFusedBatchMatMulV2AndDequantize",
      "_QuantizedFusedBatchNorm",
      "_QuantizedFusedMatMul",
      "_QuantizedFusedMatMulAndDequantize",
      "_QuantizedFusedMatMulAndRequantize",
      "_QuantizedMatMul",
      "_QuantizedTranspose",
  };

  return DataTypeExemptOp.find(op_name) != DataTypeExemptOp.end();
}

bool IsLayoutRewriteSupportedDataType(const NodeDef& node_def) {
  string op_name = node_def.op();

  // Op only used by ITEX.
  if (IsDataTypeExemptOp(op_name)) return true;
  if (IsTensorArray(node_def)) return true;

  // Quantized op used by both normal TF and intel ITEX. Need to check whether
  // they are supported before rewritting.
  if (IsQuantizedOp(op_name)) {
    if (op_name == "QuantizeV2" || op_name == "Dequantize" ||
        op_name == "QuantizedMaxPool" || op_name == "QuantizedAvgPool" ||
        op_name == "QuantizedConcatV2") {
      DataType T;
      AttrSlice attr_list(node_def);
      ITEX_CHECK_OK(GetNodeAttr(attr_list, "T", &T));
      return (T == DataType::DT_QINT8 || T == DataType::DT_QUINT8);
    } else if (op_name == "QuantizedConv2D" ||
               op_name == "QuantizedConv2DPerChannel") {
      DataType Tinput;
      DataType Tfilter;
      AttrSlice attr_list(node_def);
      ITEX_CHECK_OK(GetNodeAttr(attr_list, "Tinput", &Tinput));
      ITEX_CHECK_OK(GetNodeAttr(attr_list, "Tfilter", &Tfilter));
      return ((Tinput == DataType::DT_QINT8 || Tinput == DataType::DT_QUINT8) &&
              (Tfilter == DataType::DT_QINT8));
    } else {
      ITEX_LOG(FATAL) << "unsuppported quantized type" << op_name;
    }
  }

  // Prevent rewritting if current op doesn't have attr `T`. Should bypass op
  // without `T` if want to rewrite it.
  DataType T;
  AttrSlice attr_list(node_def);
  if (!TryGetNodeAttr(attr_list, "T", &T)) {
    return false;
  }

  // Handle custom ops here since it may not follow oneDNN op definition rule.
  // TODO(itex): Use standard solution to unify all custom ops instead of
  // simple condition check.
  if (IsRandomUniform(node_def)) {
    ITEX_CHECK_OK(GetNodeAttr(attr_list, "dtype", &T));
  }

  return (T == DataType::DT_FLOAT || T == DataType::DT_BFLOAT16 ||
          T == DataType::DT_HALF);
}

OpDef GetOpDef(const NodeDef& node_def) {
  static FunctionLibraryDefinition function_lib =
      FunctionLibraryDefinition(GraphDef());
  OpDef op_def;
  Status status = function_lib.LookUpOpDef(node_def.op(), &op_def);

  TF_ABORT_IF_ERROR(status);

  return op_def;
}

void CopyAllAttrs(const NodeDef& orig_node, NodeDef* new_node) {
  string name;
  AttrSlice attr_list(orig_node);

  auto iter = attr_list.begin();
  OpDef op_def = GetOpDef(*new_node);

  while (iter != attr_list.end()) {
    name = iter->first;
    auto attr = iter->second;

    // Check OpDef first to exclude undefined attr in `new_node`.
    if (FindAttrMutable(name, &op_def) != nullptr) {
      AddNodeAttr(name, attr, new_node);
    }
    ++iter;
  }
}

void AdjustInputOrder(NodeDef* new_node) {
  auto tmp_input = new_node->input(1);
  new_node->set_input(1, new_node->input(2));
  new_node->set_input(2, tmp_input);
  AddNodeAttr("Tpaddings", DT_INT32, new_node);
}

string GetInputName(const NodeDef* input, const int out_slot) {
  if (out_slot == 0)
    return input->name();
  else
    return input->name() + ":" + std::to_string(out_slot);
}

static const std::vector<WorkSpaceInfo>* GetWorkspaceInfo() {
  static std::vector<WorkSpaceInfo> wsinfo{
      {"MaxPoolGrad", 1, 1}, {"MaxPool3DGrad", 1, 1}, {"MaxPoolGradV2", 1, 1}};
  return &wsinfo;
}

NodeDef* AddWorkspace(const itex::graph::utils::MutableNodeView* ori_node_view,
                      NodeDef* new_node_def) {
  NodeDef* input_node_def = nullptr;
  const std::vector<WorkSpaceInfo>* wsinfo = GetWorkspaceInfo();
  for (auto it = wsinfo->cbegin(); it != wsinfo->cend(); ++it) {
    // Add workspace edge between rewritten fwd node and rewritten bwd node.
    if (ori_node_view->node()->op().compare(it->bwd_op) == 0) {
      auto* fanin_node_def =
          ori_node_view->GetRegularFanin(it->bwd_slot).node_view()->node();

      // Add workspace directly, legality will be checked in
      // `RewriteMaxPoolGrad()` later.
      new_node_def->add_input(GetInputName(fanin_node_def, it->ws_fwd_slot));
      input_node_def = fanin_node_def;
      ITEX_VLOG(3) << "Workspace: Add workspace edge between ["
                   << fanin_node_def->op() << "] and [" << new_node_def->op()
                   << "], while rewriting [" << ori_node_view->node()->op()
                   << "]";
      break;
    }
  }
  return input_node_def;
}

}  // namespace graph
}  // namespace itex
