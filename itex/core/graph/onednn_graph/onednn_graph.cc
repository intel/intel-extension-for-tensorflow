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

#include "itex/core/graph/onednn_graph/onednn_graph.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/utils/graph_common_utils.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/attr_value_util.h"
#include "itex/core/utils/device_name_utils.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/mutex.h"
#include "itex/core/utils/onednn/onednn_graph_util.h"
#include "itex/core/utils/quantization_util.h"

namespace itex {
namespace graph {

namespace {

static mutex mu;

using TranslationMap =
    std::map<const std::string,
             const std::function<Status(const OneDnnGraphContext* ctx,
                                        const int node_index,
                                        const utils::MutableNodeView* node_view,
                                        dnnl::graph::op** onednn_graph_node)>>;

// These ops are unlikely to appear in oneDNN Graph INT8 partitions. To reduce
// the possibliliy in creating oneDNN Graph ops, we won't map those ops by
// default.
static const std::unordered_set<std::string> non_int8_candidate_set = {
    {"FusedBatchNormGradV3", "LayerNormGrad", "ITEXLayerNormGrad",
     "MaxPoolGrad", "ReluGrad", "GeluGrad", "ITEXGeluGrad", "ResizeBilinear",
     "Select"}};

// Input and output index in LLGA op and TF op maybe different, e.g. diff_dst
// input in TF BNGrad is 0, while in LLGA BNGrad is 1. Thus, we need to map the
// corresponding input/output index. Op not included in the map means, TF and
// LLGA inputs and outputs sequence are the same. Notes, the number of TF
// inputs are allowed to be more than LLGA. In this circumstance, some TF inputs
// will not pass to the new "OneDnnGraph" node. If the TF outputs outnumbers
// LLGA, we should examine whether the abandoned output tensors are not used by
// other TF ops. On the other hand, TF inputs and outputs should not be less
// than LLGA.
static const std::unordered_map<std::string, std::vector<int>>
    tf_llga_input_map = {
        // vector index is llga index, the corresponding value is tf index, i.e.
        // a[llga_index] = tf_index
        {"Conv2DBackpropInputStatic", {2, 1}},
        {"Conv2DBackpropInputDynamic", {2, 1, 0}},
        {"Conv2DBackpropFilterStatic", {0, 2}},
        {"Conv2DBackpropFilterDynamic", {0, 2, 1}},
        {"FusedBatchNormV3Training", {0, 3, 4, 1, 2}},
        {"FusedBatchNormGradV3", {1, 0, 3, 4, 2}},
        {"GeluGrad", {1, 0}},
        {"ITEXGeluGrad", {1, 0}},
        {"LayerNormGrad", {1, 0, 3, 4, 2}},
        {"ITEXLayerNormGrad", {1, 0, 3, 4, 2}},
        {"MaxPoolGrad", {0, 2}},
        {"QuantizeV2", {0}},
        {"ReluGrad", {1, 0}},
        {"Reshape", {0}},
        {"Min", {0}},
        {"Max", {0}},
        {"Mean", {0}},
        {"ResizeBilinear", {0}},
        {"Sum", {0}},
        {"Transpose", {0}},
        {"Dequantize", {0}}};

static const std::unordered_map<std::string, std::vector<int>>
    tf_llga_output_map = {
        // vector index is llga index, the corresponding value is tf index, i.e.
        // a[llga_index] = tf_index
        {"FusedBatchNormV3Inference", {0}},
        {"FusedBatchNormV3Training", {0, 1, 2, 3, 4}},
        {"FusedBatchNormGradV3", {0, 1, 2}},
        {"LayerNormTraining", {0, 1, 2}},
        {"LayerNormInference", {0}},
        {"LayerNormGrad", {0, 1, 2}},
        {"ITEXLayerNormGrad", {0, 1, 2}},
        {"QuantizeV2", {0}},
        {"Dequantize", {0}}};

// Single TF op may be corresponding several LLGA op, e.g. TF "FusedBatchNormV3"
// corresponding 2 LLGA ops "BatchNormForwardTraining" and
// "BatchNormForwardInference". So we have to distinguish these two ops.
string GetOpInLLGAStyle(const utils::MutableNodeView* node_view) {
  const auto* node_def = node_view->node();
  std::string op = node_def->op();
  if (op == "FusedBatchNormV3") {
    bool is_training;
    TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "is_training", &is_training));
    if (is_training) {
      op = "FusedBatchNormV3Training";
    } else {
      op = "FusedBatchNormV3Inference";
    }
  } else if (op == "Conv2DBackpropInput") {
    const NodeDef* size_node =
        node_view->GetRegularFanin(0).node_view()->node();
    bool is_input_sizes_constant = IsAnyConst(*size_node);
    if (is_input_sizes_constant) {
      op = "Conv2DBackpropInputStatic";
    } else {
      op = "Conv2DBackpropInputDynamic";
    }
  } else if (op == "Conv2DBackpropFilter") {
    const NodeDef* size_node =
        node_view->GetRegularFanin(1).node_view()->node();
    bool is_input_sizes_constant = IsAnyConst(*size_node);
    if (is_input_sizes_constant) {
      op = "Conv2DBackpropFilterStatic";
    } else {
      op = "Conv2DBackpropFilterDynamic";
    }
  } else if (op == "LayerNorm" || op == "ITEXLayerNorm") {
    bool is_training = true;
    // TODO(itex): investigate why Layernorm op doesn't have "is_training" attr.
    TryGetNodeAttr(*node_def, "is_training", &is_training);
    if (is_training) {
      op = "LayerNormTraining";
    } else {
      op = "LayerNormInference";
    }
  } else if (op == "ConcatV2") {
    op = "Concat";
  }

  ITEX_VLOG(2) << "TF op: " << node_def->op()
               << " has 1 corresponding LLGA op: " << op;

  return op;
}

bool CheckDepthwiseINT8Pattern(const OneDnnGraphContext* ctx,
                               const utils::MutableNodeView* node_view) {
  auto* reshape_node_down_view = node_view->GetRegularFanin(1).node_view();
  if (reshape_node_down_view->node()->op() != "Reshape") {
    return false;
  }
  auto* dq_node_view = reshape_node_down_view->GetRegularFanin(0).node_view();
  if (dq_node_view->node()->op() != "Dequantize") {
    return false;
  }
  auto* q_node_view = dq_node_view->GetRegularFanin(0).node_view();
  if (q_node_view->node()->op() != "QuantizeV2") {
    return false;
  }
  auto* reshape_node_up_view = q_node_view->GetRegularFanin(0).node_view();
  if (reshape_node_up_view->node()->op() != "Reshape") {
    return false;
  }
  return true;
}

// This function is to check whether INT8 graph meets the required pattern. If
// not, emit warning message to disable constant folding
void CheckINT8Pattern(const OneDnnGraphContext* ctx,
                      const utils::MutableNodeView* node_view) {
  auto* node = node_view->node();

  auto* input_node_0_deq_view = node_view->GetRegularFanin(0).node_view();
  auto* input_node_0_deq = input_node_0_deq_view->node();
  if (input_node_0_deq->op() != "Dequantize") {
    return;
  }

  auto* input_node_0_q_view =
      input_node_0_deq_view->GetRegularFanin(0).node_view();
  auto* input_node_0_q = input_node_0_q_view->node();
  if (input_node_0_q->op() != "QuantizeV2") {
    return;
  }

  auto* input_node_1_deq_view = node_view->GetRegularFanin(1).node_view();
  auto* input_node_1_deq = input_node_1_deq_view->node();
  if (input_node_1_deq->op() == "Dequantize") {
    // Valid INT8 pattern for Conv/MatMul/BatchMatMul
    return;
  }

  bool is_valid_depthwise_int8_pattern =
      CheckDepthwiseINT8Pattern(ctx, node_view);

  if (!is_valid_depthwise_int8_pattern) {
    ITEX_LOG(ERROR)
        << "Unsupported INT8 pattern detected! Model performance may be "
           "damaged. Please disable constant folding pass to get best "
           "performance. You can do it by \"export "
           "ITEX_TF_CONSTANT_FOLDING=0\"";
    ITEX_LOG(WARNING) << "Node: " << node->op() << " " << node->name()
                      << " will not be converted into INT8 format";
  }
}

struct LayerParams {
  bool is_conv;
  bool is_maxpool;
};

// LLGA op requires fixed number of output, in other words, if some outputs of
// Op are constant folded, this op cannot be rewritten to LLGA op
bool IsOpOutputFolded(const OneDnnGraphContext* ctx,
                      const utils::MutableNodeView* node_view) {
  const auto* node_def = node_view->node();
  int expected_output_num;
  int actual_output_num;
  bool is_folded;

  std::string op = GetOpInLLGAStyle(node_view);
  auto iter = tf_llga_output_map.find(op);
  if (iter != tf_llga_output_map.end()) {
    expected_output_num = iter->second.size();
  } else {
    expected_output_num = ctx->node_type_map.GetOutputSize(*node_def);
  }

  actual_output_num = 0;
  for (auto output_nodes : node_view->GetRegularFanouts()) {
    if (output_nodes.size() != 0) actual_output_num++;
  }

  // We allow TF op output is more than LLGA op.
  is_folded = (expected_output_num > actual_output_num);

  if (is_folded) {
    ITEX_VLOG(2) << "Node: " << node_def->name()
                 << " has folded outputs, so it is not rewritten to LLGA op"
                 << ", expected_output_num: " << expected_output_num
                 << ", actual_output_num: " << actual_output_num;
  }
  return is_folded;
}

void GetShapeFromConstShapeNode(const NodeDef* node,
                                std::vector<int64_t>* shape_value,
                                bool* is_success, DataType dt) {
  if (!IsAnyConst(*node)) {
    *is_success = false;
    return;
  }

  Tensor shape_tensor;
  TensorProto tensor_proto = node->attr().at("value").tensor();

  if (!shape_tensor.FromProto(tensor_proto)) {
    *is_success = false;
    return;
  }

  switch (dt) {
    case DT_INT64:
      for (int i = 0; i < shape_tensor.NumElements(); ++i) {
        shape_value->push_back(shape_tensor.flat<int64>()(i));
      }
      *is_success = true;
      break;

    case DT_INT32:
      for (int i = 0; i < shape_tensor.NumElements(); ++i) {
        shape_value->push_back(shape_tensor.flat<int32>()(i));
      }
      *is_success = true;
      break;

    default:
      *is_success = false;
      break;
  }

  return;
}

void GetShapeFromConstDataNode(const NodeDef* node,
                               std::vector<int64_t>* shape_value,
                               bool* is_success) {
  if (!IsAnyConst(*node)) {
    *is_success = false;
    return;
  }

  Tensor data_tensor;
  TensorProto tensor_proto = node->attr().at("value").tensor();

  if (!data_tensor.FromProto(tensor_proto)) {
    *is_success = false;
    return;
  }

  for (int i = 0; i < data_tensor.dims(); ++i) {
    shape_value->push_back(data_tensor.dim_size(i));
  }
  *is_success = true;
  return;
}

std::vector<int64_t> GetReshapeTargetShape(
    const utils::MutableNodeView* node_view) {
  if (node_view->node()->op() != "Pack") return {};

  std::vector<int64_t> target_shape;

  for (int index = 0; index < node_view->NumRegularFanins(); ++index) {
    const NodeDef* size_node =
        node_view->GetRegularFanin(index).node_view()->node();
    bool is_input_sizes_constant = IsAnyConst(*size_node);
    if (is_input_sizes_constant) {
      Tensor input_shape_tensor;
      TensorProto tensor_proto = size_node->attr().at("value").tensor();

      bool load_success = input_shape_tensor.FromProto(tensor_proto);
      if (load_success && input_shape_tensor.NumElements() == 1) {
        // Currently, we only support condition in Bert where each input
        // contains only 1 dimension
        target_shape.push_back(input_shape_tensor.flat<int32>()(0));
      } else {
        return {};
      }
    } else {
      target_shape.push_back(-1);
    }
  }

  if (std::count(target_shape.begin(), target_shape.end(), -1) > 1) {
    return {};
  } else {
    return target_shape;
  }
}

// TODO(itex): investigate why it has multiple definition issues when move to
// quantize_util.h
void AdjustInputMinMaxRange(float input_min_range, float input_max_range,
                            float* adjust_min_range, float* adjust_max_range,
                            float ensure_minimum_range_ = 0.01) {
  ITEX_CHECK_GE(input_max_range, input_min_range)
      << "input_max_range must be larger than input_min_range.";

  *adjust_min_range = std::min(0.0f, input_min_range);
  // When the minimum and maximum ranges are too close together, nudge them
  // apart by a small value so that they are slightly different. This helps
  // us avoid creating ill-formed buffers where all quantized values map to
  // the same float number. These kinds of buffers cause problems for
  // downstream ops when they need to do calculations on them.
  // We pick the value by making sure that zero is not more than 100x the
  // overall range from the maximum, so that the value can be easily
  // represented when we promote the quantized value to a higher
  // intermediate bit depth, since that's a common requirement.
  const float epsilon =
      std::max(1.0f, std::max(fabsf(input_min_range), fabsf(input_max_range))) *
      ensure_minimum_range_;
  *adjust_max_range =
      std::max(0.0f, std::max(input_max_range, *adjust_min_range + epsilon));
}

//////////////////////////////////////////////////////////////////////////
// Contraction
//////////////////////////////////////////////////////////////////////////
Status SetAttr(const utils::MutableNodeView* node_view,
               dnnl::graph::op** onednn_graph_node, const LayerParams& params) {
  const NodeDef* node_def = node_view->node();

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::vector<int32> tf_ksize;
  std::vector<int32> tf_explicit_paddings;
  std::string tf_padding_type;
  std::string tf_data_format;

  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "strides", &tf_strides));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "padding", &tf_padding_type));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "data_format", &tf_data_format));
  if (params.is_conv)
    TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "dilations", &tf_dilations));
  else
    TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "ksize", &tf_ksize));

  bool is_3d = (tf_data_format.size() == 5);
  int dims = is_3d ? 5 : 4;
  bool is_channel_last = (tf_data_format[dims - 1] == 'C');

  // Strides in the batch and channel dimension is not supported
  if (tf_strides[0] != 1 || tf_strides[is_channel_last ? dims - 1 : 1] != 1) {
    delete *onednn_graph_node;
    *onednn_graph_node = nullptr;
    return Status::OK();
  }

  std::vector<int64_t> strides(dims - 2);
  std::vector<int64_t> dilations(dims - 2);
  std::vector<int64_t> ksize(dims - 2);

  ExtractSpatialDims(is_channel_last, tf_strides, &strides);
  if (params.is_conv)
    ExtractSpatialDims(is_channel_last, tf_dilations, &dilations);
  else
    ExtractSpatialDims(is_channel_last, tf_ksize, &ksize);

  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::strides, strides);
  if (params.is_conv) {
    (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::dilations, dilations);
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::weights_format, std::string("XIO"));
    if (node_def->op() == "DepthwiseConv2dNative") {
      // We can get group size based on weight shape
      NodeDef* weight_node;
      auto* weight_node_view = node_view->GetRegularFanin(1).node_view();

      if (weight_node_view->node()->op() == "Dequantize") {
        // INT8 case with multiplier = 1
        weight_node_view = weight_node_view->GetRegularFanin(0)
                               .node_view()
                               ->GetRegularFanin(0)
                               .node_view();
        weight_node = weight_node_view->node();
      } else if (weight_node_view->node()->op() == "Reshape") {
        // INT8 case with multiplier != 1
        auto* dq_node_view = weight_node_view->GetRegularFanin(0).node_view();
        if (dq_node_view->node()->op() != "Dequantize") {
          delete *onednn_graph_node;
          *onednn_graph_node = nullptr;
          return Status::OK();
        }
        auto* q_node_view = dq_node_view->GetRegularFanin(0).node_view();
        if (q_node_view->node()->op() != "QuantizeV2") {
          delete *onednn_graph_node;
          *onednn_graph_node = nullptr;
          return Status::OK();
        }
        auto* reshape_node_view = q_node_view->GetRegularFanin(0).node_view();
        if (reshape_node_view->node()->op() != "Reshape") {
          delete *onednn_graph_node;
          *onednn_graph_node = nullptr;
          return Status::OK();
        }
        weight_node_view = reshape_node_view->GetRegularFanin(0).node_view();
        weight_node = weight_node_view->node();
      } else {
        // non INT8 case
        weight_node = weight_node_view->node();
      }

      if (weight_node_view->NumRegularFanouts() != 1) {
        // We have to change the shape of Depthwise weight node, thus it
        // shouldn't have other output
        delete *onednn_graph_node;
        *onednn_graph_node = nullptr;
        return Status::OK();
      }

      bool is_success;
      std::vector<int64_t> shape_value;
      GetShapeFromConstDataNode(weight_node, &shape_value, &is_success);

      if (!is_success) {
        delete *onednn_graph_node;
        *onednn_graph_node = nullptr;
        return Status::OK();
      }

      // The 3rd dim of weight tensor is group size
      (*onednn_graph_node)
          ->set_attr(dnnl::graph::op::attr::groups, shape_value[2]);
    } else {
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::groups, int64_t{1});
    }
  } else {
    if (params.is_maxpool) {
      // No dilation attr for tf pool op, so use default val here.
      if (is_3d) {
        (*onednn_graph_node)
            ->set_attr(dnnl::graph::op::attr::dilations,
                       std::vector<int64_t>{1, 1, 1});
      } else {
        (*onednn_graph_node)
            ->set_attr(dnnl::graph::op::attr::dilations,
                       std::vector<int64_t>{1, 1});
      }
    }
    (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::kernel, ksize);
  }
  if (is_channel_last) {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::data_format, std::string("NXC"));
  } else {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  }

  // pads_begin/pads_end are required in op_schema, but they will be
  // ignored when auto_pad attribute is specified.
  std::vector<int64_t> dummy_pad(dims - 2, 0);
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::pads_begin, dummy_pad);
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::pads_end, dummy_pad);
  if (tf_padding_type == "SAME") {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::auto_pad, std::string("SAME_UPPER"));
  } else if (tf_padding_type == "VALID") {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::auto_pad, std::string("VALID"));
  } else {
    if (!HasNodeAttr(*node_def, "explicit_paddings"))
      return errors::InvalidArgument("Invalid padding format");
    TF_ABORT_IF_ERROR(
        GetNodeAttr(*node_def, "explicit_paddings", &tf_explicit_paddings));

    std::vector<int64_t> pads_begin(dims - 2);
    std::vector<int64_t> pads_end(dims - 2);
    ExtractSpatialPadDims(is_channel_last, tf_explicit_paddings, &pads_begin,
                          &pads_end);
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::pads_begin, pads_begin);
    (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::pads_end, pads_end);
  }

  return Status::OK();
}

[[maybe_unused]] bool IsConv2DSizeTensor(const Tensor& shape_tensor) {
  if (shape_tensor.dtype() != DT_INT32) return false;
  // Conv input and filter both have 4 dimensions, thus the shape tensor has 4
  // elements
  if (shape_tensor.NumElements() != 4) return false;
  return true;
}

void SetStaticShapeAttr(const OneDnnGraphContext* ctx,
                        const utils::MutableNodeView* node_view,
                        dnnl::graph::op** onednn_graph_node) {
  auto* node_def = node_view->node();

  int size_input_index;
  if (node_def->op() == "Conv2DBackpropInput") {
    size_input_index = 0;
  } else if (node_def->op() == "Conv2DBackpropFilter" ||
             node_def->op() == "Sum" || node_def->op() == "Mean" ||
             node_def->op() == "Min" || node_def->op() == "Max" ||
             node_def->op() == "Reshape" ||
             node_def->op() == "ResizeBilinear" ||
             node_def->op() == "Transpose") {
    size_input_index = 1;
  } else if (node_def->op() == "ConcatV2") {
    size_input_index = node_view->NumRegularFanins() - 1;
  } else {
    delete *onednn_graph_node;
    *onednn_graph_node = nullptr;
    return;
  }

  const NodeDef* size_node =
      node_view->GetRegularFanin(size_input_index).node_view()->node();
  bool is_input_sizes_constant = IsAnyConst(*size_node);

  // Only with const size node, we set attribute for LLGA op, otherwise we
  // pass additional shape tensor in the runtime to inform LLGA op with shape
  // information
  if (is_input_sizes_constant) {
    std::vector<int64_t> size_value;
    bool is_success;
    DataType dt = GetDataType(*node_def, ctx->node_type_map.GetInputTypeAttr(
                                             *node_def, size_input_index));
    GetShapeFromConstShapeNode(size_node, &size_value, &is_success, dt);

    if (!is_success) {
      // TODO(itex): do we have better way to check, instead of setting many
      // nullptr
      delete *onednn_graph_node;
      *onednn_graph_node = nullptr;
      return;
    }

    // Special case for Depthwise weight. We need to change the value of
    // second Reshape from (KH, KW, IC, K) to (KH, KW, 1, K*IC).
    if (node_def->op() == "Reshape") {
      bool is_input_dequantize = false;
      bool is_output_depthwise = false;

      auto* fanin_view = node_view->GetRegularFanin(0).node_view();
      if (fanin_view->node()->op() == "Dequantize") {
        is_input_dequantize = true;
      }

      if (node_view->NumRegularFanouts() == 1) {
        auto* fanout_view = node_view->GetRegularFanout(0)[0].node_view();
        if (fanout_view->node()->op() == "DepthwiseConv2dNative") {
          is_output_depthwise = true;
        }
      }

      if (is_input_dequantize && is_output_depthwise &&
          size_value.size() == 4) {
        int output_channel = size_value[2] * size_value[3];
        size_value[2] = 1;
        size_value[3] = output_channel;
      }
    }

    // TODO(itex): check whether other llga op allow -1 as shape info.
    if (node_def->op() == "Reshape" &&
        std::count(size_value.begin(), size_value.end(), -1) > 1) {
      delete *onednn_graph_node;
      *onednn_graph_node = nullptr;
      return;
    }

    if (size_value.size() == 0) {
      delete *onednn_graph_node;
      *onednn_graph_node = nullptr;
      return;
    }

    if (node_def->op() == "Conv2DBackpropInput") {
      (*onednn_graph_node)
          ->set_attr(dnnl::graph::op::attr::dst_shape, size_value);
    } else if (node_def->op() == "Conv2DBackpropFilter") {
      (*onednn_graph_node)
          ->set_attr(dnnl::graph::op::attr::weights_shape, size_value);
    } else if (node_def->op() == "Min" || node_def->op() == "Max" ||
               node_def->op() == "Sum" || node_def->op() == "Mean") {
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::axes, size_value);
    } else if (node_def->op() == "Reshape") {
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::shape, size_value);
    } else if (node_def->op() == "Transpose") {
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::order, size_value);
    } else if (node_def->op() == "ResizeBilinear") {
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::sizes, size_value);
    } else if (node_def->op() == "ConcatV2") {
      (*onednn_graph_node)
          ->set_attr(dnnl::graph::op::attr::axis, size_value[0]);
    }
  } else {
    if (node_def->op() == "Reshape") {
      // Special handling when reshape input is Pack, and with less than 1
      // unknown dimension
      auto* input_node_view = node_view->GetRegularFanin(1).node_view();
      std::vector<int64_t> target_shape;
      target_shape = GetReshapeTargetShape(input_node_view);
      if (target_shape.size() != 0) {
        (*onednn_graph_node)
            ->set_attr(dnnl::graph::op::attr::shape, target_shape);
      } else {
        delete *onednn_graph_node;
        *onednn_graph_node = nullptr;
        return;
      }
    } else {
      // TODO(itex): check whether other llga op allow -1 as shape info.
      delete *onednn_graph_node;
      *onednn_graph_node = nullptr;
      return;
    }
  }
}

Status TranslateConv(const OneDnnGraphContext* ctx, const int node_index,
                     const utils::MutableNodeView* node_view,
                     dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::Convolution, node_def->name());

  TF_ABORT_IF_ERROR(
      SetAttr(node_view, onednn_graph_node, LayerParams{true, false}));
  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  CheckINT8Pattern(ctx, node_view);

  return Status::OK();
}

Status TranslateConv2DBackpropInput(const OneDnnGraphContext* ctx,
                                    const int node_index,
                                    const utils::MutableNodeView* node_view,
                                    dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::ConvolutionBackwardData,
      node_def->name());

  TF_ABORT_IF_ERROR(
      SetAttr(node_view, onednn_graph_node, LayerParams{true, false}));

  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  SetStaticShapeAttr(ctx, node_view, onednn_graph_node);

  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  CheckINT8Pattern(ctx, node_view);

  return Status::OK();
}

Status TranslateConv2DBackpropFilter(const OneDnnGraphContext* ctx,
                                     const int node_index,
                                     const utils::MutableNodeView* node_view,
                                     dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::ConvolutionBackwardWeights,
      node_def->name());

  TF_ABORT_IF_ERROR(
      SetAttr(node_view, onednn_graph_node, LayerParams{true, false}));
  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  SetStaticShapeAttr(ctx, node_view, onednn_graph_node);

  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  CheckINT8Pattern(ctx, node_view);

  return Status::OK();
}

Status TranslateMatMul(const OneDnnGraphContext* ctx, const int node_index,
                       const utils::MutableNodeView* node_view,
                       dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();

  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::MatMul, node_def->name());

  bool transpose_a, transpose_b;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "transpose_a", &transpose_a));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "transpose_b", &transpose_b));
  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::transpose_a,
                 static_cast<bool>(transpose_a));
  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::transpose_b,
                 static_cast<bool>(transpose_b));

  CheckINT8Pattern(ctx, node_view);

  return Status::OK();
}

Status TranslateBatchMatMulV2(const OneDnnGraphContext* ctx,
                              const int node_index,
                              const utils::MutableNodeView* node_view,
                              dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::MatMul, node_def->name());

  bool transpose_a, transpose_b;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "adj_x", &transpose_a));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "adj_y", &transpose_b));
  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::transpose_a,
                 static_cast<bool>(transpose_a));
  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::transpose_b,
                 static_cast<bool>(transpose_b));

  CheckINT8Pattern(ctx, node_view);

  return Status::OK();
}
//////////////////////////////////////////////////////////////////////////
// BatchNorm
//////////////////////////////////////////////////////////////////////////
Status TranslateBNGrad(const OneDnnGraphContext* ctx, const int node_index,
                       const utils::MutableNodeView* node_view,
                       dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  // We should rewrite both of or none of BN forward and backward
  const auto* bn_forward_node_view = node_view->GetRegularFanin(3).node_view();
  if (IsOpOutputFolded(ctx, bn_forward_node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  bool is_training;
  std::string tf_data_format;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "is_training", &is_training));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "data_format", &tf_data_format));
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::BatchNormTrainingBackward,
      node_def->name());

  float epsilon;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "epsilon", &epsilon));
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::epsilon, epsilon);
  if (tf_data_format == "NCHW") {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  } else if (tf_data_format == "NHWC") {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::data_format, std::string("NXC"));
  } else {
    // Currently, only supports 2D BN
    delete *onednn_graph_node;
    *onednn_graph_node = nullptr;
    return Status::OK();
  }
  return Status::OK();
}

Status TranslateBN(const OneDnnGraphContext* ctx, const int node_index,
                   const utils::MutableNodeView* node_view,
                   dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  // We should rewrite both of or none of BN forward and backward
  for (auto fanout : node_view->GetRegularFanout(3)) {
    auto* fanout_view = fanout.node_view();
    if (IsOpOutputFolded(ctx, fanout_view)) {
      onednn_graph_node = nullptr;
      return Status::OK();
    }
  }

  auto* node_def = node_view->node();
  bool is_training;
  std::string tf_data_format;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "is_training", &is_training));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "data_format", &tf_data_format));
  if (is_training) {
    *onednn_graph_node = new dnnl::graph::op(
        node_index, dnnl::graph::op::kind::BatchNormForwardTraining,
        node_def->name());
  } else {
    *onednn_graph_node = new dnnl::graph::op(
        node_index, dnnl::graph::op::kind::BatchNormInference,
        node_def->name());
  }

  if (is_training) {
    float exponential_avg_factor;
    TF_RETURN_IF_ERROR(GetNodeAttr(*node_def, "exponential_avg_factor",
                                   &exponential_avg_factor));
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::momentum, 1 - exponential_avg_factor);
  }

  float epsilon;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "epsilon", &epsilon));
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::epsilon, epsilon);
  if (tf_data_format == "NCHW") {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  } else if (tf_data_format == "NHWC") {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::data_format, std::string("NXC"));
  } else {
    delete *onednn_graph_node;
    *onednn_graph_node = nullptr;
    return Status::OK();
  }
  return Status::OK();
}

// TODO(itex): investigate why LayerNormGrad is not rewrite to LLGA op
Status TranslateLN(const OneDnnGraphContext* ctx, const int node_index,
                   const utils::MutableNodeView* node_view,
                   dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  bool is_training;
  std::string tf_data_format;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "is_training", &is_training));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "data_format", &tf_data_format));
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::LayerNorm, node_def->name());

  if (is_training) {
    (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::keep_stats, true);
  } else {
    (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::keep_stats, false);
  }

  // TODO(itex): support more axis option, currently OneDnn only supports
  // last axis
  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::begin_norm_axis, int64_t{-1});

  float epsilon;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "epsilon", &epsilon));
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::epsilon, epsilon);

  return Status::OK();
}

Status TranslateLNGrad(const OneDnnGraphContext* ctx, const int node_index,
                       const utils::MutableNodeView* node_view,
                       dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  bool is_training;
  std::string tf_data_format;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "is_training", &is_training));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "data_format", &tf_data_format));
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::LayerNormBackward, node_def->name());

  // TODO(itex): support more axis option, currently OneDnn only supports
  // last axis
  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::begin_norm_axis, int64_t{-1});

  float epsilon;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "epsilon", &epsilon));
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::epsilon, epsilon);

  return Status::OK();
}

Status TranslateReshape(const OneDnnGraphContext* ctx, const int node_index,
                        const utils::MutableNodeView* node_view,
                        dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();

  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::StaticReshape, node_def->name());

  SetStaticShapeAttr(ctx, node_view, onednn_graph_node);

  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  // special zero is turn on, then 0 means same as input shape in that
  // dimension
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::special_zero, false);

  return Status::OK();
}

Status TranslateTranspose(const OneDnnGraphContext* ctx, const int node_index,
                          const utils::MutableNodeView* node_view,
                          dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();

  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::StaticTranspose, node_def->name());

  SetStaticShapeAttr(ctx, node_view, onednn_graph_node);

  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  return Status::OK();
}

[[maybe_unused]] Status TranslateResize(const OneDnnGraphContext* ctx,
                                        const int node_index,
                                        const utils::MutableNodeView* node_view,
                                        dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();

  bool align_corners;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "align_corners", &align_corners));
  bool half_pixel_centers;
  TF_ABORT_IF_ERROR(
      GetNodeAttr(*node_def, "half_pixel_centers", &half_pixel_centers));

  if ((align_corners && half_pixel_centers) ||
      (!align_corners && !half_pixel_centers)) {
    // oneDNN Graph's interpolate attr align_cornes and half_pixel_centers are
    // mutually exclusive.
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::Interpolate, node_def->name());

  SetStaticShapeAttr(ctx, node_view, onednn_graph_node);

  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  SetStaticShapeAttr(ctx, node_view, onednn_graph_node);

  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::mode, std::string("bilinear"));

  if (half_pixel_centers) {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::coordinate_transformation_mode,
                   std::string("half_pixel"));
  } else {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::coordinate_transformation_mode,
                   std::string("align_corners"));
  }

  return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
// Pooling
//////////////////////////////////////////////////////////////////////////
Status TranslateMaxPool(const OneDnnGraphContext* ctx, const int node_index,
                        const utils::MutableNodeView* node_view,
                        dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::MaxPool, node_def->name());

  TF_ABORT_IF_ERROR(
      SetAttr(node_view, onednn_graph_node, LayerParams{false, true}));
  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::rounding_type, std::string("floor"));
  return Status::OK();
}

Status TranslateAvgPool(const OneDnnGraphContext* ctx, const int node_index,
                        const utils::MutableNodeView* node_view,
                        dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::AvgPool, node_def->name());
  // TODO(itex): Set exclude_pad
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::exclude_pad, false);

  TF_ABORT_IF_ERROR(
      SetAttr(node_view, onednn_graph_node, LayerParams{false, false}));
  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::rounding_type, std::string("floor"));
  return Status::OK();
}

Status TranslateMaxPoolGrad(const OneDnnGraphContext* ctx, const int node_index,
                            const utils::MutableNodeView* node_view,
                            dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::MaxPoolBackward, node_def->name());

  TF_ABORT_IF_ERROR(
      SetAttr(node_view, onednn_graph_node, LayerParams{false, true}));
  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  return Status::OK();
}

// Status TranslateAvgPoolGrad(const OneDnnGraphContext* ctx, const int
// node_index,
//                             const utils::MutableNodeView* node_view,
//                             dnnl::graph::op** onednn_graph_node) {
//   if (IsOpOutputFolded(ctx, node_view)) {
//     onednn_graph_node = nullptr;
//     return Status::OK();
//   }

//   auto* node_def = node_view->node();
//   *onednn_graph_node = new dnnl::graph::op(
//       node_index, dnnl::graph::op::kind::AvgPoolBackward,
//       node_def->name());

//   TF_ABORT_IF_ERROR(SetAttr(node_view, onednn_graph_node, false, false));
//   return Status::OK();
// }

//////////////////////////////////////////////////////////////////////////
// Activation
//////////////////////////////////////////////////////////////////////////
Status TranslateEltwise(const OneDnnGraphContext* ctx, const int node_index,
                        const utils::MutableNodeView* node_view,
                        dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  using kind = dnnl::graph::op::kind;

  static std::map<std::string, kind> TF_LLGA_op_map = {
      {"Elu", kind::Elu},
      {"Gelu", kind::GELU},
      {"ITEXGelu", kind::GELU},
      {"GeluGrad", kind::GELUBackward},
      {"ITEXGeluGrad", kind::GELUBackward},
      {"LeakyRelu", kind::LeakyReLU},
      {"_ITEXMish", kind::Mish},
      {"Sigmoid", kind::Sigmoid},
      {"Relu", kind::ReLU},
      {"ReluGrad", kind::ReLUBackward},
      {"Relu6", kind::Clamp},
      // #ifndef ITEX_ONEDNN_3_0
      //       {"Rsqrt", kind::Rsqrt},
      // #endif
      {"Square", kind::Square},
      {"Tanh", kind::Tanh}};

  auto* node_def = node_view->node();
  auto it = TF_LLGA_op_map.find(node_def->op());
  if (it != TF_LLGA_op_map.end()) {
    *onednn_graph_node =
        new dnnl::graph::op(node_index, it->second, node_def->name());

    if (node_def->op() == "LeakyRelu") {
      float alpha;
      TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "alpha", &alpha));
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::alpha, alpha);
    } else if (node_def->op() == "Relu6") {
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::min, 0.0f);
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::max, 6.0f);
    } else if (node_def->op() == "Elu") {
      (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::alpha, 1.0f);
    }
    return Status::OK();
  } else {
    onednn_graph_node = nullptr;
    return Status::OK();
  }
}

Status TranslateSoftmax(const OneDnnGraphContext* ctx, const int node_index,
                        const utils::MutableNodeView* node_view,
                        dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::SoftMax, node_def->name());
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::axis, int64_t{-1});
  return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
// Binary
//////////////////////////////////////////////////////////////////////////
Status TranslateAddN(const OneDnnGraphContext* ctx, const int node_index,
                     const utils::MutableNodeView* node_view,
                     dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  // LLGA currently only support Add op, not support AddN op. So we only
  // rewrite TF AddN op with 2 inputs to LLGA op.
  if (node_view->NumRegularFanins() != 2) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::Add, node_def->name());
  return Status::OK();
}

Status TranslateBinary(const OneDnnGraphContext* ctx, const int node_index,
                       const utils::MutableNodeView* node_view,
                       dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  using kind = dnnl::graph::op::kind;

  static std::map<std::string, kind> TF_LLGA_op_map = {
      {"Add", kind::Add},
      {"AddV2", kind::Add},
      {"Mul", kind::Multiply},
      {"SquaredDifference", kind::SquaredDifference},
      {"Sub", kind::Subtract}};

  // TODO(itex): Add scalar sanity check, if encountering shape inference issue
  // caused by both input's are scalar tensors
  auto* node_def = node_view->node();
  auto it = TF_LLGA_op_map.find(node_def->op());
  if (it != TF_LLGA_op_map.end()) {
    *onednn_graph_node =
        new dnnl::graph::op(node_index, it->second, node_def->name());
    return Status::OK();
  } else {
    onednn_graph_node = nullptr;
    return Status::OK();
  }
}

Status TranslateBiasAdd(const OneDnnGraphContext* ctx, const int node_index,
                        const utils::MutableNodeView* node_view,
                        dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::BiasAdd, node_def->name());
  return Status::OK();
}

Status TranslateBiasAddGrad(const OneDnnGraphContext* ctx, const int node_index,
                            const utils::MutableNodeView* node_view,
                            dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::BiasAddBackward, node_def->name());
  return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
// Quantize/Dequantize
//////////////////////////////////////////////////////////////////////////
Status GetQuantizeMinMaxValue(const utils::MutableNodeView* node_view,
                              Tensor* input_min_range, Tensor* input_max_range,
                              bool* find_const_min_max) {
  auto* min_fanin = node_view->GetRegularFanin(1).node_view()->node();
  if (!IsAnyConst(*min_fanin)) {
    if (IsEnter(*min_fanin)) {
      min_fanin = node_view->GetRegularFanin(1)
                      .node_view()
                      ->GetRegularFanin(0)
                      .node_view()
                      ->node();
    } else {
      *find_const_min_max = false;
      return Status::OK();
    }
  }
  if (!input_min_range->FromProto(min_fanin->attr().at("value").tensor())) {
    return errors::InvalidArgument("Cannot parse constant value from ",
                                   min_fanin->name());
  }

  auto* max_fanin = node_view->GetRegularFanin(2).node_view()->node();
  if (!IsAnyConst(*max_fanin)) {
    if (IsEnter(*max_fanin)) {
      max_fanin = node_view->GetRegularFanin(2)
                      .node_view()
                      ->GetRegularFanin(0)
                      .node_view()
                      ->node();
    } else {
      *find_const_min_max = false;
      return Status::OK();
    }
  }
  if (!input_max_range->FromProto(max_fanin->attr().at("value").tensor())) {
    return errors::InvalidArgument("Cannot parse constant value from ",
                                   max_fanin->name());
  }

  *find_const_min_max = true;
  return Status::OK();
}

Status SetScaleAndZp(const OneDnnGraphContext* ctx,
                     const utils::MutableNodeView* node_view,
                     dnnl::graph::op** onednn_graph_node, const DataType& T,
                     const std::string& mode, int axis,
                     QuantizeMode quan_mode) {
  auto* node_def = node_view->node();

  auto* input_node_view = node_view->GetRegularFanin(0).node_view();

  if (IsAnyMaxPool(*(input_node_view->node()))) {
    // Maxpool cases, to ensure input/output scale are the same
    // In some situation, INC may not insert QDQ before MaxPool
    auto* dq_node_view = input_node_view->GetRegularFanin(0).node_view();
    if (dq_node_view->node()->op() == "Dequantize") {
      auto* q_node_view = dq_node_view->GetRegularFanin(0).node_view();
      if (q_node_view->node()->op() == "QuantizeV2") {
        TF_ABORT_IF_ERROR(SetScaleAndZp(ctx, q_node_view, onednn_graph_node, T,
                                        mode, axis, quan_mode));
        return Status::OK();
      }
    }
  }

  (*onednn_graph_node)
      ->set_attr(dnnl::graph::op::attr::axis, static_cast<int64_t>(axis));

  Tensor input_min_range, input_max_range;
  bool find_const_min_max = false;
  TF_ABORT_IF_ERROR(GetQuantizeMinMaxValue(
      node_view, &input_min_range, &input_max_range, &find_const_min_max));

  if (!find_const_min_max) {
    delete *onednn_graph_node;
    *onednn_graph_node = nullptr;
    return Status::OK();
  }

  int num_slices = 1;
  if (axis > -1) {
    num_slices = input_min_range.NumElements();
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::qtype, std::string("per_channel"));
  } else {
    (*onednn_graph_node)
        ->set_attr(dnnl::graph::op::attr::qtype, std::string("per_tensor"));
  }

  std::vector<float> min_range(num_slices);
  std::vector<float> max_range(num_slices);

  // We need to do input range adjust for both quantized and dequantize op
  // when creating llga q/dq op. The reason why ITEX only requires Quantize
  // does adjust is because the calculation happens in runtime execution. And
  // it will pass the adjusted min/max to dequantize op, so dequantize doesn't
  // required to do so. But here we are in graph optimization stage, both
  // quantize and dequantize can see the unadjusted min/max input.
  if (num_slices == 1) {
    const float min_range_before_adjust =
        input_min_range.template flat<float>()(0);
    const float max_range_before_adjust =
        input_max_range.template flat<float>()(0);
    AdjustInputMinMaxRange(min_range_before_adjust, max_range_before_adjust,
                           &min_range[0], &max_range[0]);
  } else {
    auto min_ranges_before_adjust = input_min_range.template flat<float>();
    auto max_ranges_before_adjust = input_max_range.template flat<float>();
    for (int i = 0; i < num_slices; ++i) {
      AdjustInputMinMaxRange(min_ranges_before_adjust(i),
                             max_ranges_before_adjust(i), &min_range[i],
                             &max_range[i]);
    }
  }

  // Calculating scales and zeropoints for quantization.
  std::vector<float> scale_factor(num_slices, 0);
  std::vector<int32> zero_points(num_slices, 0);

  switch (T) {
    case DT_QINT8:
      GetScaleAndZeropointAndAlignMinMax<qint8>(
          min_range.data(), max_range.data(), quan_mode,
          QuantDequantFlag::Dequantize, num_slices, scale_factor.data(),
          zero_points.data());
      break;
    case DT_QUINT8:
      GetScaleAndZeropointAndAlignMinMax<quint8>(
          min_range.data(), max_range.data(), quan_mode,
          QuantDequantFlag::Dequantize, num_slices, scale_factor.data(),
          zero_points.data());
      break;

    default:
      ITEX_LOG(FATAL) << "unsupported int8 datatype " << T << " of node "
                      << node_def->op() << " " << node_def->name();
      break;
  }

  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::scales, scale_factor);

  std::vector<int64_t> zero_points_int64(num_slices, 0);
  std::transform(zero_points.begin(), zero_points.end(),
                 zero_points_int64.begin(),
                 [](int32 v) -> int64_t { return static_cast<int64_t>(v); });

  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::zps, zero_points_int64);
  return Status::OK();
}

// TODO(itex): merge quantize/requantize to a single function
Status TranslateQuantizeV2(const OneDnnGraphContext* ctx, const int node_index,
                           const utils::MutableNodeView* node_view,
                           dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();

  // For oneDNN Graph INT8 pb, QuantizeV2's outputs are always Dequantize
  for (auto fanout : node_view->GetRegularFanout(0)) {
    auto* fanout_node_view = fanout.node_view();
    if (fanout_node_view->node()->op() != "Dequantize") {
      onednn_graph_node = nullptr;
      return Status::OK();
    }
  }

  DataType T;
  std::string mode;
  std::string round_mode;
  float ensure_minimum_range;
  int axis;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "T", &T));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "mode", &mode));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "round_mode", &round_mode));
  TF_ABORT_IF_ERROR(
      GetNodeAttr(*node_def, "ensure_minimum_range", &ensure_minimum_range));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "axis", &axis));

  QuantizeMode quan_mode;
  if (mode == "SCALED") {
    quan_mode = QuantizeMode::SCALED;
  } else if (mode == "MIN_FIRST") {
    quan_mode = QuantizeMode::MIN_FIRST;
  } else {
    // Unsupported quantized mode
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::Quantize, node_def->name());

  TF_ABORT_IF_ERROR(SetScaleAndZp(ctx, node_view, onednn_graph_node, T, mode,
                                  axis, quan_mode));

  return Status::OK();
}

Status TranslateDequantize(const OneDnnGraphContext* ctx, const int node_index,
                           const utils::MutableNodeView* node_view,
                           dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  // For oneDNN Graph INT8 pb, Dequantize's input is always QuantizeV2
  const NodeDef* input_node_node =
      node_view->GetRegularFanin(0).node_view()->node();
  if (input_node_node->op() != "QuantizeV2") {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::Dequantize, node_def->name());

  DataType T;
  std::string mode;
  int64_t axis;
  DataType dtype;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "T", &T));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "mode", &mode));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "axis", &axis));
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "dtype", &dtype));

  QuantizeMode quan_mode;
  if (mode == "SCALED") {
    quan_mode = QuantizeMode::SCALED;
  } else if (mode == "MIN_FIRST") {
    quan_mode = QuantizeMode::MIN_FIRST;
  } else {
    ITEX_LOG(FATAL) << "unsupported quantize mode: " << node_def->op() << " "
                    << node_def->name();
  }

  auto* quantize_node_view = node_view->GetRegularFanin(0).node_view();

  if (quantize_node_view->node()->op() == "QuantizeV2") {
    TF_ABORT_IF_ERROR(SetScaleAndZp(ctx, quantize_node_view, onednn_graph_node,
                                    T, mode, axis, quan_mode));
  } else {
    TF_ABORT_IF_ERROR(SetScaleAndZp(ctx, node_view, onednn_graph_node, T, mode,
                                    axis, quan_mode));
  }

  return Status::OK();
}

Status TranslateCast(const OneDnnGraphContext* ctx, const int node_index,
                     const utils::MutableNodeView* node_view,
                     dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();

  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::TypeCast, node_def->name());
  return Status::OK();
}

Status TranslateReduce(const OneDnnGraphContext* ctx, const int node_index,
                       const utils::MutableNodeView* node_view,
                       dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  // TODO(yunfei): check why we need to hardcode disable these ops
  if (node_view->node()->name() == "cls/predictions/Sum_1" ||
      node_view->node()->name() == "cls/seq_relationship/Sum" ||
      node_view->node()->name() == "cls/predictions/Sum_2") {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx->graph_properties.GetInputProperties(node_def->name(), &props));
  if (props.size() != 2) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  bool input_invalid = props[0].shape().unknown_rank() ||
                       IsScalar(props[0].shape()) || Is1D(props[0].shape());

  // TODO(itex): remove this workaround when oneDNN Graph fix their shape
  // inference bugs for full reduce reduction operation
  if (!input_invalid && !props[0].shape().unknown_rank()) {
    auto* size_view = node_view->GetRegularFanin(1).node_view();
    auto* size_node = size_view->node();
    std::vector<int64_t> size_value;

    bool is_success;
    DataType dt = GetDataType(
        *node_def, ctx->node_type_map.GetInputTypeAttr(*node_def, 1));
    GetShapeFromConstShapeNode(size_node, &size_value, &is_success, dt);
    if (is_success && size_value.size() == props[0].shape().dim().size()) {
      input_invalid = true;
    }
  }

  if (input_invalid) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  if (node_def->op() == "Min") {
    *onednn_graph_node = new dnnl::graph::op(
        node_index, dnnl::graph::op::kind::ReduceMin, node_def->name());
  } else if (node_def->op() == "Max") {
    *onednn_graph_node = new dnnl::graph::op(
        node_index, dnnl::graph::op::kind::ReduceMax, node_def->name());
  } else if (node_def->op() == "Sum") {
    *onednn_graph_node = new dnnl::graph::op(
        node_index, dnnl::graph::op::kind::ReduceSum, node_def->name());
  } else if (node_def->op() == "Mean") {
    *onednn_graph_node = new dnnl::graph::op(
        node_index, dnnl::graph::op::kind::ReduceMean, node_def->name());
  } else {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  SetStaticShapeAttr(ctx, node_view, onednn_graph_node);

  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  bool keep_dims;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "keep_dims", &keep_dims));
  (*onednn_graph_node)->set_attr(dnnl::graph::op::attr::keep_dims, keep_dims);

  return Status::OK();
}

[[maybe_unused]] Status TranslateConcat(const OneDnnGraphContext* ctx,
                                        const int node_index,
                                        const utils::MutableNodeView* node_view,
                                        dnnl::graph::op** onednn_graph_node) {
  if (IsOpOutputFolded(ctx, node_view)) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::Concat, node_def->name());

  SetStaticShapeAttr(ctx, node_view, onednn_graph_node);

  if (*onednn_graph_node == nullptr) {
    onednn_graph_node = nullptr;
    return Status::OK();
  }

  return Status::OK();
}

// #ifndef ITEX_ONEDNN_3_0
// Status TranslateSelect(const OneDnnGraphContext* ctx, const int node_index,
//                        const utils::MutableNodeView* node_view,
//                        dnnl::graph::op** onednn_graph_node) {
//   if (IsOpOutputFolded(ctx, node_view)) {
//     onednn_graph_node = nullptr;
//     return Status::OK();
//   }

//   auto* node_def = node_view->node();
//   *onednn_graph_node = new dnnl::graph::op(
//       node_index, dnnl::graph::op::kind::Select, node_def->name());

//   return Status::OK();
// }
// #endif

Status TranslateWildcard(const OneDnnGraphContext* ctx, const int node_index,
                         const utils::MutableNodeView* node_view,
                         dnnl::graph::op** onednn_graph_node) {
  auto* node_def = node_view->node();
  *onednn_graph_node = new dnnl::graph::op(
      node_index, dnnl::graph::op::kind::Wildcard, node_def->name());
  return Status::OK();
}

Status TranslateUnhandled(const OneDnnGraphContext* ctx, const int node_index,
                          const utils::MutableNodeView* node_view,
                          dnnl::graph::op** onednn_graph_node) {
  *onednn_graph_node = nullptr;
  return Status::OK();
}

// TODO(itex): refactor TranslateXXX function code into a macro

const TranslationMap& getTranslationMap() {
  static const TranslationMap tf_to_onednn_graph_op_translation_map{
      ////// contraction
      {"Conv2D", TranslateConv},
      {"Conv3D", TranslateConv},
      {"_ITEXConv3D", TranslateConv},
      {"DepthwiseConv2dNative", TranslateConv},
      {"Conv2DBackpropInput", TranslateConv2DBackpropInput},
      {"Conv2DBackpropFilter", TranslateConv2DBackpropFilter},
      {"MatMul", TranslateMatMul},
      {"BatchMatMul", TranslateBatchMatMulV2},
      {"BatchMatMulV2", TranslateBatchMatMulV2},
      ////// BN
      // Note: we only support V3 of FusedBatchNorm and FusedBatchNormGrad
      {"FusedBatchNormV3", TranslateBN},
      {"FusedBatchNormGradV3", TranslateBNGrad},
      {"LayerNorm", TranslateLN},
      {"ITEXLayerNorm", TranslateLN},
      {"LayerNormGrad", TranslateLNGrad},
      {"ITEXLayerNormGrad", TranslateLNGrad},
      ////// pool
      {"AvgPool", TranslateAvgPool},
      {"MaxPool", TranslateMaxPool},
      {"MaxPool3D", TranslateMaxPool},
      // TODO(itex): Enable Avgpoolgrad, once align with LLGA in op
      // definition.
      // {"AvgPoolGrad", TranslateAvgPoolGrad},
      {"MaxPoolGrad", TranslateMaxPoolGrad},
      ////// activation
      {"Elu", TranslateEltwise},
      // TODO(itex): check why this op is missing in oneDNN master
      // #ifndef ITEX_ONEDNN_3_0
      //       {"Rsqrt", TranslateEltwise},
      // #endif
      {"Relu6", TranslateEltwise},
      {"LeakyRelu", TranslateEltwise},
      // Disable LLGA Square, before we root cause the Bert training NAN issue
      // {"Square", TranslateEltwise},
      {"Sigmoid", TranslateEltwise},
      {"Tanh", TranslateEltwise},
      {"Relu", TranslateEltwise},
      {"ReluGrad", TranslateEltwise},
      {"Gelu", TranslateEltwise},
      {"ITEXGelu", TranslateEltwise},
      {"GeluGrad", TranslateEltwise},
      {"ITEXGeluGrad", TranslateEltwise},
      {"_ITEXMish", TranslateEltwise},
      {"Reshape", TranslateReshape},
      {"Transpose", TranslateTranspose},
      {"Softmax", TranslateSoftmax},
      ////// binary
      {"Add", TranslateBinary},
      {"AddV2", TranslateBinary},
      {"Sub", TranslateBinary},
      {"Mul", TranslateBinary},
      {"SquaredDifference", TranslateBinary},

      // TODO(itex): enable the op mapping
      // {"ResizeBilinear", TranslateResize},

      {"AddN", TranslateAddN},
      {"BiasAdd", TranslateBiasAdd},
      {"BiasAddGrad", TranslateBiasAddGrad},
      {"QuantizeV2", TranslateQuantizeV2},
      {"Dequantize", TranslateDequantize},
      {"Cast", TranslateCast},
      {"Min", TranslateReduce},
      {"Max", TranslateReduce},
      {"Mean", TranslateReduce},
      {"Sum", TranslateReduce},

      ////// variadic input op
      // TODO(itex): Enable concat op, once root cause the crash with input num
      // > 64 and hang issue in TLT.
      // {"ConcatV2", TranslateConcat},

      ////// conditional op
      // TODO(itex): enable it once graph compiler & boolean datatype is merged
      // to oneDNN master
      // #ifndef ITEX_ONEDNN_3_0
      //       {"Select", TranslateSelect},
      // #endif

      {"Wildcard", TranslateWildcard},
      {"Unhandled", TranslateUnhandled}};

  return tf_to_onednn_graph_op_translation_map;
}

int GetLLGANumInput(const utils::MutableNodeView* node_view) {
  std::string op = GetOpInLLGAStyle(node_view);

  if (tf_llga_input_map.find(op) != tf_llga_input_map.end()) {
    return tf_llga_input_map.at(op).size();
  } else if (op == "Concat") {
    // TODO(itex): make another rule for variadic inputs op like Concat
    return node_view->NumRegularFanins() - 1;
  } else {
    return node_view->NumRegularFanins();
  }
}

int GetLLGANumOutput(const utils::MutableNodeView* node_view) {
  std::string op = GetOpInLLGAStyle(node_view);

  if (tf_llga_output_map.find(op) != tf_llga_output_map.end()) {
    return tf_llga_output_map.at(op).size();
  } else {
    return node_view->GetRegularFanouts().size();
  }
}

int GetTFInputIndexInLLGASequence(const utils::MutableNodeView* node_view,
                                  int llga_index) {
  std::string op = GetOpInLLGAStyle(node_view);

  if (tf_llga_input_map.find(op) != tf_llga_input_map.end()) {
    return tf_llga_input_map.at(op)[llga_index];
  } else {
    return llga_index;
  }
}

int GetTFOutputIndexInLLGASequence(const utils::MutableNodeView* node_view,
                                   int llga_index) {
  std::string op = GetOpInLLGAStyle(node_view);

  if (tf_llga_output_map.find(op) != tf_llga_output_map.end()) {
    return tf_llga_output_map.at(op)[llga_index];
  } else {
    return llga_index;
  }
}

class LLGAEdgeManager {
 public:
  uint64_t GetOneDnnGraphTensorId(const SafeTensorId& tid) {
    const auto it = edge_id_map_.find(tid);
    if (it == edge_id_map_.end()) {
      edge_id_map_.insert({tid, edge_id_});
      id_edge_map_.insert({edge_id_, tid});
      return edge_id_++;
    } else {
      return it->second;
    }
  }

  void UpdateTensorId(const SafeTensorId& tid, int64_t edge_id) {
    edge_id_map_.insert({tid, edge_id});
    id_edge_map_.insert({edge_id, tid});
  }

  void UpdateEdgeManager(const LLGAEdgeManager& manager) {
    for (auto it : manager.edge_id_map_) {
      this->edge_id_map_[it.first] = it.second;
      this->id_edge_map_[it.second] = it.first;
    }
  }

  // TODO(itex): investigate how to set this function const
  SafeTensorId* FindLLGATensorId(uint64_t logical_tensor_id) {
    auto it = id_edge_map_.find(logical_tensor_id);
    if (it != id_edge_map_.end()) {
      return &(it->second);
    } else {
      return nullptr;
    }
  }

 private:
  std::map<SafeTensorId, uint64_t> edge_id_map_;
  std::map<uint64_t, SafeTensorId> id_edge_map_;
  int64_t edge_id_ = 0;
};

struct AdditionalArgs {
  std::map<int, int> depthwise_weight_map;
};

int GetRegularFaninIndex(const utils::MutableNodeView* from_node,
                         const utils::MutableNodeView* to_node,
                         const int from_index) {
  for (int to_index = 0; to_index < to_node->NumRegularFanins(); to_index++) {
    auto& regular_fanin = to_node->GetRegularFanin(to_index);
    if (regular_fanin.node_view()->node_index() == from_node->node_index() &&
        regular_fanin.index() == from_index)
      return to_index;
  }
  return -1;
}

bool IsOneDnnGraphSupportedDataType(const NodeDef& node_def) {
  // Update this map when OneDnn Graph supports more datatype
  static std::unordered_set<DataType> float_datatype = {DT_FLOAT, DT_BFLOAT16,
                                                        DT_HALF};
  static std::unordered_set<DataType> int8_datatype = {DT_QINT8, DT_QUINT8};
  DataType T;
  AttrSlice attr_list(node_def);
  if (TryGetNodeAttr(attr_list, "T", &T)) {
    if (node_def.op() == "QuantizeV2" || node_def.op() == "Dequantize") {
      if (int8_datatype.find(T) == int8_datatype.end()) return false;
    } else {
      if (float_datatype.find(T) == float_datatype.end()) return false;
    }
  }

  // Cast op
  DataType SrcT, DstT;
  if (TryGetNodeAttr(attr_list, "SrcT", &SrcT) &&
      TryGetNodeAttr(attr_list, "DstT", &DstT)) {
    if (float_datatype.find(SrcT) == float_datatype.end()) return false;
    if (float_datatype.find(DstT) == float_datatype.end()) return false;
  }
  return true;
}

[[maybe_unused]] bool IsConstantInput(const utils::MutableNodeView* node_view,
                                      const int input_index) {
  const NodeDef* input_node =
      node_view->GetRegularFanin(input_index).node_view()->node();
  return IsAnyConst(*input_node);
}

}  // namespace

// Note: this function only handles LLGA graph, adding input/output for LLGA
// ops. The function not changes the TF graph.

// Some corner cases:
// 1. When an edge has multi outputs, they should all be selected
//    (if they are not supported by onednn graph, set them as Wildcard)
// 2. The other input of Add op.
Status SelectNode(OneDnnGraphContext* ctx, int num_nodes,
                  const TranslationMap& tf_to_onednn_graph_op_translation_map,
                  std::unordered_set<std::string>* wildcard_nodes,
                  std::unordered_set<std::string>* rewrite_nodes,
                  bool is_wildcard, dnnl::graph::graph* graph_ctx,
                  LLGAEdgeManager* edge_manager,
                  AdditionalArgs* additional_args,
                  bool onednn_graph_all_type_flag) {
  ITEX_VLOG(2) << "====== Start selecting nodes, is_wildcard = " << is_wildcard;

  for (int f_node = 0; f_node < num_nodes; f_node++) {
    const auto* f_node_view = ctx->graph_view.GetNode(f_node);
    const auto* f_node_def = f_node_view->node();
    ITEX_VLOG(2) << "FRAMEWORK NODE OP " << f_node_def->op();
    ITEX_VLOG(2) << "FRAMEWORK NODE NAME " << f_node_def->name();

    // TODO(itex): Add control edge check.
    // if (HasControlFaninOrFanout(*f_node_view)) continue;

    if (!is_wildcard) {
      // Layout rewrite pass does not rewrite preserved nodes, neither does LLGA
      // pass. Currently, LLGA only works with Layout pass ON.
      if (ctx->nodes_to_preserve.count(f_node_def->name()) > 0) continue;

      // No need to add oneDNN Graph ops by default, if they are not possible in
      // INT8 partitions.
      if (!onednn_graph_all_type_flag &&
          non_int8_candidate_set.find(f_node_def->op()) !=
              non_int8_candidate_set.end())
        continue;
    }

    if (is_wildcard) {
      // Skip if current node is not wildcard node
      if (wildcard_nodes->find(f_node_def->name()) == wildcard_nodes->end())
        continue;

      // Skip if current node is rewrite to LLGA non-trivial node
      if (rewrite_nodes->find(f_node_def->name()) != rewrite_nodes->end())
        continue;
    }
    // Convert fw node to onednn graph node
    const std::function<Status(const OneDnnGraphContext* ctx,
                               const int node_index,
                               const utils::MutableNodeView* node_view,
                               dnnl::graph::op** onednn_graph_node)>* op_func;
    dnnl::graph::op* onednn_graph_node = nullptr;
    if (!is_wildcard) {
      if (tf_to_onednn_graph_op_translation_map.find(f_node_def->op()) !=
              tf_to_onednn_graph_op_translation_map.end() &&
          IsOneDnnGraphSupportedDataType(*f_node_def)) {
        op_func = &(tf_to_onednn_graph_op_translation_map.at(f_node_def->op()));
      } else {
        op_func = &(tf_to_onednn_graph_op_translation_map.at("Unhandled"));
      }
    } else {
      op_func = &(tf_to_onednn_graph_op_translation_map.at("Wildcard"));
    }

    try {
      TF_ABORT_IF_ERROR(
          (*op_func)(ctx, f_node, f_node_view, &onednn_graph_node));
    } catch (const std::exception& e) {
      return errors::InvalidArgument(
          "Error occurs in create LLGA node: " + f_node_def->name() + " (" +
          f_node_def->op() + ")\n" + f_node_def->DebugString() + "\n" +
          "what(): " + e.what());
    }
    if (onednn_graph_node == nullptr) continue;

    if (!is_wildcard) {
      rewrite_nodes->insert(f_node_def->name());
    }

    if (!is_wildcard && f_node_def->op() == "DepthwiseConv2dNative") {
      auto* weight_node_view = f_node_view->GetRegularFanin(1).node_view();
      if (weight_node_view->node()->op() == "Dequantize") {
        // INT8 case
        weight_node_view = weight_node_view->GetRegularFanin(0)
                               .node_view()
                               ->GetRegularFanin(0)
                               .node_view();
      }
      additional_args->depthwise_weight_map.insert(
          {f_node_view->node_index(), weight_node_view->node_index()});
    }

    ITEX_VLOG(2) << "=== Selecting Node: Name " << f_node_def->name() << ", Op "
                 << f_node_def->op() << ", is wildcard: " << is_wildcard;

    // Wildcard op are actually not rewritten to LLGA op, they are TF ops.
    size_t llga_input_num = is_wildcard ? f_node_view->NumRegularFanins()
                                        : GetLLGANumInput(f_node_view);

    // TODO(itex): Add shape inference check.
    // Handle inputs
    // const auto& in_props =
    // ctx->graph_properties.GetInputProperties(f_node_def->name());
    for (size_t idx_llga_in = 0; idx_llga_in < llga_input_num; idx_llga_in++) {
      int idx_tf_in =
          is_wildcard ? idx_llga_in
                      : GetTFInputIndexInLLGASequence(f_node_view, idx_llga_in);
      DataType dt = GetDataType(
          *f_node_def,
          ctx->node_type_map.GetInputTypeAttr(*f_node_def, idx_tf_in));

      // In this step, unique_logical_tensor_id is corresponding to original TF
      // edges
      uint64_t unique_logical_tensor_id = edge_manager->GetOneDnnGraphTensorId(
          ParseTensorName(f_node_def->input(idx_tf_in)));

      // Some LLGA op requires rank information when building graph, such as
      // Reshape.
      int32_t ndims = -1;

      auto input_logical_tensor = dnnl::graph::logical_tensor(
          unique_logical_tensor_id, GetOneDnnGraphDataType(dt), ndims,
          dnnl::graph::logical_tensor::layout_type::undef);
      onednn_graph_node->add_input(input_logical_tensor);
      ITEX_VLOG(2) << "Input tensor(" << idx_tf_in << "): "
                   << ParseTensorName(f_node_def->input(idx_tf_in)).ToString()
                   << ", (tensor_id=" << unique_logical_tensor_id << ")";

      // Always add wildcard now
      if (is_wildcard) continue;
      auto* in_node_def =
          f_node_view->GetRegularFanin(idx_tf_in).node_view()->node();

      wildcard_nodes->insert(in_node_def->name());
      ITEX_VLOG(2) << " -- wildcard select " << in_node_def->name() << " "
                   << in_node_def->op();
    }
    // Handle outputs
    // const auto& out_props =
    // ctx->graph_properties.GetOutputProperties(f_node_def->name());

    // Wildcard op are actually not rewritten to LLGA op, they are TF ops.
    size_t llga_output_num = is_wildcard
                                 ? f_node_view->GetRegularFanouts().size()
                                 : GetLLGANumOutput(f_node_view);
    for (size_t idx_llga_out = 0; idx_llga_out < llga_output_num;
         idx_llga_out++) {
      int idx_tf_out = is_wildcard ? idx_llga_out
                                   : GetTFOutputIndexInLLGASequence(
                                         f_node_view, idx_llga_out);
      DataType dt = GetDataType(
          *f_node_def,
          ctx->node_type_map.GetOutputTypeAttr(*f_node_def, idx_tf_out));

      // Here pair <from_tensor, llga_logical_id> is in the order of LLGA op
      // definition
      uint64_t unique_logical_tensor_id = edge_manager->GetOneDnnGraphTensorId(
          SafeTensorId(f_node_def->name(), idx_llga_out));

      auto output_logical_tensor = dnnl::graph::logical_tensor(
          unique_logical_tensor_id, GetOneDnnGraphDataType(dt), -1,
          dnnl::graph::logical_tensor::layout_type::undef);

      onednn_graph_node->add_output(output_logical_tensor);
      ITEX_VLOG(2) << "Output tensor(" << idx_llga_out << "): "
                   << SafeTensorId(f_node_def->name(), idx_llga_out).ToString()
                   << ", (tensor_id=" << unique_logical_tensor_id << ")";

      // If out_offset has more than one output node, make sure the output is
      // onednn graph node or wildcard.
      /*
           node
           /  \
         node (wildcard)
      */
      if (is_wildcard) continue;
      // if (f_node_view->GetRegularFanout(idx_tf_out).size() == 1) continue;
      for (size_t out_idx = 0;
           out_idx < f_node_view->GetRegularFanout(idx_tf_out).size();
           out_idx++) {
        auto* to_node =
            f_node_view->GetRegularFanout(idx_tf_out)[out_idx].node_view();
        auto* to_node_def = to_node->node();

        wildcard_nodes->insert(to_node_def->name());
        ITEX_VLOG(2) << " -- wildcard select " << to_node_def->name() << " "
                     << to_node_def->op();
      }
    }

    // Select nodes
    DeviceNameUtils::ParsedName name;
    if (!DeviceNameUtils::ParseFullName(f_node_def->device(), &name) ||
        !name.has_type) {
      continue;
    }

    try {
      graph_ctx->add_op(*onednn_graph_node);
      delete onednn_graph_node;
    } catch (const std::exception& e) {
      return errors::InvalidArgument(
          "Error occurs in insert LLGA node in the graph: " +
          f_node_def->name() + " (" + f_node_def->op() + ")\n" +
          f_node_def->DebugString() + "\n" + "what(): " + e.what());
    }
    ITEX_VLOG(2) << "Node: " << f_node_def->name()
                 << " op: " << f_node_def->op()
                 << " add to LLGA graph, is_wildcard: " << is_wildcard;
  }
  return Status::OK();
}

Status FuseFwPartitionWithLLGA(
    OneDnnGraphContext* ctx,
    dnnl::graph::partition& p,  // NOLINT(runtime/references)
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete,
    LLGAEdgeManager* edge_manager, LLGAEdgeManager* edge_manager_tmp,
    AdditionalArgs* additional_args, bool onednn_graph_all_type_flag) {
  auto* mutation = ctx->graph_view.GetMutationBuilder();

  ITEX_VLOG(2) << "IN REWRITE ";
  absl::flat_hash_set<int> f_nodes;
  size_t nodes_no = p.get_ops_num();

  ITEX_VLOG(2) << "rewrite partition id: " << p.get_id();

  // TODO(itex): figure out why removing Q / DQ check will cause OOB model
  // failure
  bool find_quantize_dequantize = false;
  for (size_t l_index = 0; l_index < nodes_no; l_index++) {
    auto node_index = p.get_ops()[l_index];
    const auto* f_node_view = ctx->graph_view.GetNode(node_index);
    const auto* f_node_def = f_node_view->node();

    if (f_node_def->op() == "QuantizeV2" || f_node_def->op() == "Dequantize") {
      find_quantize_dequantize = true;
      break;
    }
  }

  if (!onednn_graph_all_type_flag & !find_quantize_dequantize) {
    ITEX_VLOG(2) << "oneDNN Graph partition doesn't contain INT8 op, won't "
                    "rewrite this partition to ";
    return Status::OK();
  }

  bool find_all_binary_on_CPU = true;
  for (size_t l_index = 0; l_index < nodes_no; l_index++) {
    auto node_index = p.get_ops()[l_index];
    const auto* f_node_view = ctx->graph_view.GetNode(node_index);
    const auto* f_node_def = f_node_view->node();

    if (!IsAnyBinary(*f_node_def) || !NodeIsOnCpu(f_node_def)) {
      find_all_binary_on_CPU = false;
      break;
    }
  }

  if (find_all_binary_on_CPU) {
    ITEX_VLOG(2)
        << "oneDNN Graph partition doesn't contain non-binary op on CPU, "
           "won't rewrite this partition to ";
    return Status::OK();
  }

  if (nodes_no == 0) return Status::OK();
  ITEX_VLOG(2) << "NUMBER OF OPS IN PARTITION " << p.get_ops_num();
  for (size_t l_index = 0; l_index < nodes_no; l_index++) {
    auto f_index = p.get_ops()[l_index];
    f_nodes.emplace(f_index);

    // switch last dimension of depthwiseconv2d weight
    auto iter = additional_args->depthwise_weight_map.find(f_index);
    if (iter != additional_args->depthwise_weight_map.end()) {
      int index = iter->second;
      auto* weight_node_view = ctx->graph_view.GetNode(index);
      NodeDef* weight_node = weight_node_view->node();

      if (!IsAnyConst(*weight_node)) {
        // Condition when we change llga reshape value, instead of itex const
        // value
        continue;
      }

      std::vector<int64_t> shape_value;
      Tensor data_tensor;
      TensorProto tensor_proto = weight_node->attr().at("value").tensor();

      data_tensor.FromProto(tensor_proto);
      for (int i = 0; i < data_tensor.dims(); ++i) {
        shape_value.push_back(data_tensor.dim_size(i));
      }

      int dims = shape_value.size();
      std::swap(shape_value[dims - 2], shape_value[dims - 1]);

      ITEX_CHECK_OK(data_tensor.BitcastFrom(data_tensor, data_tensor.dtype(),
                                            TensorShape{shape_value}));

      AttrValue value;
      TensorProto* value_proto = value.mutable_tensor();
      data_tensor.AsProtoTensorContent(value_proto);
      mutation->AddOrUpdateNodeAttr(weight_node_view, "value", value);
    }
  }

  std::vector<string> framework_ops;

  ITEX_VLOG(2) << "Original TF op within the partition ";
  for (auto node_index : f_nodes) {
    const auto* f_node_view = ctx->graph_view.GetNode(node_index);
    const auto* f_node_def = f_node_view->node();
    ITEX_VLOG(2) << "Op " << f_node_def->op()
                 << ", Name: " << f_node_def->name();
    framework_ops.push_back(f_node_def->op());
  }

  std::vector<std::string> in_edges;  // input edges of fused-node
  std::vector<std::string>
      in_control_edges;                // control input edges of fused-node
  std::vector<DataType> in_datatypes;  // datatypes for input tensors
  std::vector<std::vector<std::pair<int, int>>>
      out_nodes;                        // to_nodes, to_index
  std::vector<int> out_control_index;   // control out index of fused-node
  std::vector<DataType> out_datatypes;  // datatypes for output tensors
  std::vector<int64> input_edge_ids;
  std::vector<int64> output_edge_ids;
  std::vector<bool> is_constant_input_edge;
  std::vector<bool> candidate_inplace_input_edge;

  NodeDef onednn_graph_node;
  // f_index indicates the last node in the partition(always the last)
  // The new oneDNN Graph node name will be original last node name + "_LLGA"
  auto f_index = *std::max_element(f_nodes.begin(), f_nodes.end());
  const auto* f_node_view = ctx->graph_view.GetNode(f_index);
  const auto* last_node_def = f_node_view->node();
  onednn_graph_node.set_op("OneDnnGraph");
  string llga_op_name = last_node_def->name() + "_LLGA";
  onednn_graph_node.set_name(llga_op_name);
  onednn_graph_node.set_device(last_node_def->device());

  ITEX_VLOG(2) << "Generate LLGA node: " << llga_op_name;

  // handle input
  ITEX_VLOG(2) << "Handle inputs";
  std::vector<dnnl::graph::logical_tensor> input_logical_tensors =
      p.get_input_ports();
  ITEX_VLOG(2) << "partition input number: " << input_logical_tensors.size();

  // All inputs are defaultly can be inplaced
  candidate_inplace_input_edge.resize(input_logical_tensors.size());

  static std::set<std::string> contraction_nodes = {
      "Conv2D",
      "Conv3D",
      "DepthwiseConv2dNative",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "Conv3DBackpropFilterV2",
      "Conv3DBackpropInputV2",
      "DepthwiseConv2dNativeBackpropFilter",
      "DepthwiseConv2dNativeBackpropInput",
      "MatMul",
      "BatchMatMul",
      "BatchMatMulV2"};

  // TODO(itex): relax the restrction here to allow non-contraction inplace
  bool has_contraction_node = false;
  for (auto op : framework_ops) {
    if (contraction_nodes.find(op) != contraction_nodes.end()) {
      has_contraction_node = true;
      break;
    }
  }

  ITEX_VLOG(2) << "op: " << onednn_graph_node.name()
               << (has_contraction_node ? " has " : " hasn't ")
               << "contraction nodes";

  std::fill(candidate_inplace_input_edge.begin(),
            candidate_inplace_input_edge.end(), has_contraction_node);

  for (int i = 0; i < input_logical_tensors.size(); ++i) {
    dnnl::graph::logical_tensor input_logical_tensor = input_logical_tensors[i];
    size_t input_logical_tensor_id = input_logical_tensor.get_id();

    SafeTensorId* tid =
        edge_manager_tmp->FindLLGATensorId(input_logical_tensor_id);
    if (tid == nullptr) {
      tid = edge_manager->FindLLGATensorId(input_logical_tensor_id);
    }
    string input_node_name = tid->node();
    int input_node_index = tid->index();

    auto* input_node_view = ctx->graph_view.GetNode(input_node_name);
    auto* input_node_def = input_node_view->node();

    if (IsAnyConst(*input_node_def)) {
      candidate_inplace_input_edge[i] = false;
      ITEX_VLOG(2) << "llga_op_name's " << i << "th input: " << tid->ToString()
                   << " cannot be inplaced, because it is const node";
    }

    // TODO(yunfei): enable binary inplace once fix bug
    if (nodes_no == 1 && IsAnyBinary(*last_node_def)) {
      candidate_inplace_input_edge[i] = false;
      ITEX_VLOG(2) << "llga_op cannot be inplaced, because it is Binary node";
    }

    for (auto fanout : input_node_view->GetRegularFanout(input_node_index)) {
      int node_index = fanout.node_index();
      // The input tensor may be used by other ops
      if (!f_nodes.count(node_index)) {
        candidate_inplace_input_edge[i] = false;
        ITEX_VLOG(2) << "llga_op_name's " << i
                     << "th input: " << tid->ToString()
                     << " cannot be inplaced, because is has outputs outside "
                        "of this partition";
        break;
      }
    }

    ITEX_CHECK_EQ(f_nodes.count(input_node_view->node_index()), 0)
        << "LLGA partition input should not be in the partition";

    in_edges.push_back(tid->ToString());
    input_edge_ids.push_back(input_logical_tensor_id);

    if (IsAnyConst(*input_node_def)) {
      is_constant_input_edge.push_back(true);
    } else if (IsEnter(*input_node_def)) {
      // if input is enter, it can propogate const information
      string enter_input_tensor_name = input_node_def->input(0);
      TensorId enter_input_tensorid = ParseTensorName(enter_input_tensor_name);
      auto* enter_input_node_view =
          ctx->graph_view.GetNode(enter_input_tensorid.node());
      auto* enter_input_node_def = enter_input_node_view->node();

      if (IsAnyConst(*enter_input_node_def)) {
        is_constant_input_edge.push_back(true);
      } else {
        is_constant_input_edge.push_back(false);
      }
    } else if (GetOptimizerConfigFlags().enable_optimize_aggressive) {
      // Aggressive optimization
      if (IsReadVariableOp(*input_node_def)) {
        // if input is Readvariable, and the variable is const, it can be set
        // constant property
        string arg_tensor_name = input_node_def->input(0);
        TensorId arg_tensorid = ParseTensorName(arg_tensor_name);
        auto* arg_node_view = ctx->graph_view.GetNode(arg_tensorid.node());
        auto* arg_node_def = arg_node_view->node();

        // if _Arg doesn't have outputs, other than current ReadVariable, that
        // means _Arg value cannot be modifid, then it can be regarded as Const.
        // TODO(itex): we may relax restriction to no outputs are variable
        // modification ops, such as AssignAddVariable
        if (IsArg(*arg_node_def) && arg_node_view->NumRegularFanouts() == 1) {
          is_constant_input_edge.push_back(true);
        } else {
          is_constant_input_edge.push_back(false);
        }
      } else {
        is_constant_input_edge.push_back(false);
      }
    } else {
      is_constant_input_edge.push_back(false);
    }

    onednn_graph_node.add_input(tid->ToString());

    ITEX_VLOG(2) << "Input tensor " << tid->ToString()
                 << ", tensor_id=" << input_logical_tensor_id;

    // datatype still requires old map
    SafeTensorId* old_tid =
        edge_manager->FindLLGATensorId(input_logical_tensor_id);
    string old_input_node_name = old_tid->node();
    int old_input_node_index = old_tid->index();

    auto* old_input_node_view = ctx->graph_view.GetNode(old_input_node_name);
    auto* old_input_node_def = old_input_node_view->node();

    in_datatypes.push_back(GetDataType(
        *old_input_node_def, ctx->node_type_map.GetOutputTypeAttr(
                                 *old_input_node_def, old_input_node_index)));
  }

  // handle output
  ITEX_VLOG(2) << "Handle outputs";
  std::vector<dnnl::graph::logical_tensor> output_logical_tensors =
      p.get_output_ports();
  ITEX_VLOG(2) << "partition output number: " << output_logical_tensors.size();

  for (int i = 0; i < output_logical_tensors.size(); ++i) {
    dnnl::graph::logical_tensor output_logical_tensor =
        output_logical_tensors[i];
    size_t output_logical_tensor_id = output_logical_tensor.get_id();

    // auto it = id_edge_map_tmp.find(output_logical_tensor_id);
    // if (it == id_edge_map_tmp.end()) {
    //   it = id_edge_map.find(output_logical_tensor_id);
    // }

    SafeTensorId* tid =
        edge_manager_tmp->FindLLGATensorId(output_logical_tensor_id);
    if (tid == nullptr) {
      tid = edge_manager->FindLLGATensorId(output_logical_tensor_id);
    }
    string output_node_name = tid->node();
    int output_node_index = tid->index();
    auto* output_node_view = ctx->graph_view.GetNode(output_node_name);

    std::vector<std::pair<int, int>> out_nodes_port;
    for (auto fanout : output_node_view->GetRegularFanout(output_node_index)) {
      auto* fanout_node_view = fanout.node_view();
      int out_node_index = fanout_node_view->node_index();
      if (f_nodes.find(out_node_index) != f_nodes.end()) continue;
      // ITEX_CHECK_EQ(f_nodes.count(out_node_index), 0)
      //     << "LLGA partition output should not be in the partition";
      int to_offset = fanout.index();
      out_nodes_port.emplace_back(out_node_index, to_offset);

      SafeTensorId new_onednn_graph_tid(onednn_graph_node.name(), i);
      mutation->AddOrUpdateRegularFanin(fanout_node_view, to_offset,
                                        new_onednn_graph_tid);
      ITEX_VLOG(2) << "Output tensor " << new_onednn_graph_tid.ToString()
                   << ", tensor_id=" << output_logical_tensor_id;
    }

    output_edge_ids.push_back(output_logical_tensor_id);
    out_nodes.push_back(out_nodes_port);

    // datatype still requires old map
    // auto old_it = id_edge_map.find(output_logical_tensor_id);

    SafeTensorId* old_tid =
        edge_manager->FindLLGATensorId(output_logical_tensor_id);
    string old_output_node_name = old_tid->node();
    int old_output_node_index = old_tid->index();

    auto* old_output_node_view = ctx->graph_view.GetNode(old_output_node_name);
    auto* old_output_node_def = old_output_node_view->node();

    out_datatypes.push_back(
        GetDataType(*old_output_node_def,
                    ctx->node_type_map.GetOutputTypeAttr(
                        *old_output_node_def, old_output_node_index)));

    // update edge_id_map_tmp and id_edge_map_tmp, where edge points to the new
    // LLGA op. SafeTensorId old_tid(output_node_name, output_node_index);
    SafeTensorId new_tid(llga_op_name, i);
    // TransferOneDnnGraphTensorId(old_tid, new_tid);
    // edge_id_map_tmp.insert({new_tid, output_logical_tensor_id});
    // id_edge_map_tmp.insert({output_logical_tensor_id, new_tid});

    edge_manager_tmp->UpdateTensorId(new_tid, output_logical_tensor_id);
  }

  // // handle control inputs and outputs
  for (int l_index = nodes_no - 1; l_index >= 0; l_index--) {
    auto f_index = p.get_ops()[l_index];
    ITEX_VLOG(2) << "=== Fusing " << l_index << "th node in the partition";

    const auto* node_view = ctx->graph_view.GetNode(f_index);
    const auto* node = node_view->node();
    ITEX_VLOG(2) << "FUSED NODE " << node->name();

    // TODO(itex): Currently, OneDnn Graph and ITEX bridge only supports fusion
    // with 1 output
    // Prepare control input edges for fused node
    for (const auto& in_control_view : node_view->GetControllingFanins()) {
      int in_control_node_index = in_control_view.node_index();
      if (f_nodes.find(in_control_node_index) != f_nodes.end()) continue;
      // If in_control_node is not in partition, then current control input edge
      // should be the input edge of fused node.
      in_control_edges.push_back(in_control_view.node_view()->node()->name());
    }

    // Prepare control output edges for fused node
    for (const auto& out_control_view : node_view->GetControlledFanouts()) {
      int out_control_node_index = out_control_view.node_index();
      if (f_nodes.find(out_control_node_index) != f_nodes.end()) continue;
      // If out_control_node is not out partition, then current control output
      // edge should be the output edge of fused node.
      out_control_index.push_back(out_control_view.node_index());
    }
  }

  // Add control edge for new node
  for (auto f_in_edge : in_control_edges) {
    onednn_graph_node.add_input(AsControlDependency(f_in_edge));
  }

  auto* attr = onednn_graph_node.mutable_attr();
  auto attr_in = gtl::ArraySlice<DataType>(in_datatypes);
  SetAttrValue(attr_in, &(*attr)["Tin"]);
  auto attr_out = gtl::ArraySlice<DataType>(out_datatypes);
  SetAttrValue(attr_out, &(*attr)["Tout"]);
  SetAttrValue(static_cast<int32>(p.get_id()), &(*attr)["partition_id"]);
  SetAttrValue(input_edge_ids, &(*attr)["input_edge_ids"]);
  SetAttrValue(is_constant_input_edge, &(*attr)["is_constant_input_edge"]);
  SetAttrValue(candidate_inplace_input_edge,
               &(*attr)["candidate_inplace_input_edge"]);
  SetOneDnnGraphPartition(std::move(p));

  SetAttrValue(framework_ops, &(*attr)["framework_ops"]);

  Status status;

  // For output nodes which have input control edge to the nodes within the LLGA
  // partition. If the pointed node is not the last node, the node will be
  // removed. So those output node should remove the control edge to the
  // original node, instead they should point to the new OneDnnGraph node.
  for (auto f_out_index : out_control_index) {
    auto* out_node_view = ctx->graph_view.GetNode(f_out_index);
    auto control_fanins = out_node_view->GetControllingFanins();
    for (size_t i = 0; i < control_fanins.size(); ++i) {
      auto fanin_node_index = control_fanins[i].node_view()->node_index();
      auto fanin_name = control_fanins[i].node_view()->node()->name();
      if (std::find(f_nodes.begin(), f_nodes.end(), fanin_node_index) !=
          f_nodes.end()) {
        mutation->RemoveControllingFanin(out_node_view, fanin_name);
      }
    }
    mutation->AddControllingFanin(out_node_view, onednn_graph_node.name());
  }

  SetAttrValue(output_edge_ids, &(*attr)["output_edge_ids"]);
  mutation->AddNode(std::move(onednn_graph_node), &status);
  TF_ABORT_IF_ERROR(std::move(status));
  TF_ABORT_IF_ERROR(mutation->Apply());

  if (f_index < invalidated_nodes->size()) (*invalidated_nodes)[f_index] = true;
  for (auto index : f_nodes) {
    (*nodes_to_delete)[index] = true;
  }
  ITEX_VLOG(2) << "REWRITE SUCCESS";
  return Status::OK();
}

Status AddRetNode(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));

  static int ret_node_idx = 0;
  std::unordered_map<std::string, NodeDef*> name_to_nodes;
  int num_nodes = ctx->graph_view.graph()->node_size();
  for (int idx = 0; idx < num_nodes; idx++) {
    auto* node_view = ctx->graph_view.GetNode(idx);
    auto* node_def = node_view->node();
    name_to_nodes[node_def->name()] = node_def;
  }

  auto* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  for (const auto& str : ctx->fetch_tensors) {
    NodeDef ret_node;
    TensorId input_slot = ParseTensorName(str);
    NodeDef* input = name_to_nodes[std::string(input_slot.node())];
    // Skip ops that do not have outputs
    if (ctx->node_type_map.GetOutputSize(*input) == 0) continue;
    // Create ret node
    ret_node.set_op("_Retval");
    ret_node.set_name("_RetNode_" + std::to_string(ret_node_idx));
    ret_node.set_device(input->device());
    if (input_slot.index() == 0)
      ret_node.add_input(std::string(input_slot.node()));
    else
      ret_node.add_input(input_slot.ToString());

    auto* attr = ret_node.mutable_attr();
    DataType dt = GetDataType(*input, ctx->node_type_map.GetOutputTypeAttr(
                                          *input, input_slot.index()));
    SetAttrValue(dt, &(*attr)["T"]);
    SetAttrValue(ret_node_idx++, &(*attr)["index"]);

    mutation->AddNode(std::move(ret_node), &status);
    TF_ABORT_IF_ERROR(std::move(status));
  }

  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

// Remove AssignAddVariableOp, AssignSubVariableOp and AssignVariableOp before
// Cast.
Status RemoveAssignXVariable(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));
  ITEX_VLOG(2) << "Remove Assign X VariableOp to break circle.";
  auto* mutation = ctx->graph_view.GetMutationBuilder();
  int num_nodes = ctx->graph_view.graph()->node_size();
  for (int idx = 0; idx < num_nodes; idx++) {
    auto* node_view = ctx->graph_view.GetNode(idx);
    auto* node_def = node_view->node();
    if (node_def->name() == "AssignAddVariableOp" ||
        node_def->name() == "AssignSubVariableOp" ||
        node_def->name() == "AssignVariableOp") {
      auto control_fanouts = node_view->GetControlledFanouts();
      if (control_fanouts.size() != 1) {
        return Status::OK();
      }
      for (int i = 0; i < control_fanouts.size(); ++i) {
        auto out_node_view = control_fanouts[i].node_view();
        auto* out_node_def = out_node_view->node();
        if (out_node_def->name() != "Cast") {
          return Status::OK();
        }
      }
      for (int i = 0; i < node_view->NumRegularFanins(); ++i) {
        auto* input_node_view = node_view->GetRegularFanin(i).node_view();
        const NodeDef* input_node = input_node_view->node();
        ITEX_VLOG(2) << "input node of AssignAddVariableOp "
                     << input_node->name();
        ITEX_VLOG(2) << "regular input num of this node "
                     << input_node_view->NumRegularFanins();
        mutation->RemoveNode(input_node_view);
      }
      for (int i = 0; i < control_fanouts.size(); ++i) {
        auto out_node_view = control_fanouts[i].node_view();
        mutation->RemoveControllingFanin(out_node_view, node_def->name());
      }
      mutation->RemoveNode(node_view);
      ITEX_VLOG(2) << "Remove AssignAddVariableOp done.";
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

Status RemoveRetNode(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));

  auto* mutation = ctx->graph_view.GetMutationBuilder();
  int num_nodes = ctx->graph_view.graph()->node_size();
  for (int idx = 0; idx < num_nodes; idx++) {
    auto* node_view = ctx->graph_view.GetNode(idx);
    auto* node_def = node_view->node();
    if (node_def->name().substr(0, 9) == "_RetNode_")
      mutation->RemoveNode(node_view);
  }

  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

//         Const                       Const
//           |                        /     \
//           Q          =>           Q      Q
//        /     \                    |      |
//     DQ        DQ                  DQ     DQ
Status DuplicateQuantize(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));

  auto* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  ITEX_VLOG(2) << "Duplicate Quantize when 1) input is const, 2) multiple "
                  "dequantize outputs";
  int num_nodes = ctx->graph_view.graph()->node_size();
  for (int idx = 0; idx < num_nodes; idx++) {
    auto* node_view = ctx->graph_view.GetNode(idx);
    auto* node_def = node_view->node();

    // TODO(itex): handle cast + quantize condition
    if (node_def->op() == "QuantizeV2" &&
        IsAnyConst(*(node_view->GetRegularFanin(0).node_view()->node()))) {
      // Duplicate Quantize node to the number of output Dequantize
      for (int i = 1; i < node_view->GetRegularFanout(0).size(); i++) {
        NodeDef quant_node;
        quant_node.set_op(node_def->op());
        quant_node.set_name(node_def->name() + "_duplicate_" +
                            std::to_string(i));
        quant_node.set_device(node_def->device());
        quant_node.add_input(node_def->input(0));
        quant_node.add_input(node_def->input(1));
        quant_node.add_input(node_def->input(2));
        // Duplicate the attributes
        auto* attr = quant_node.mutable_attr();
        auto& src_attr = node_def->attr();
        (*attr)["T"] = src_attr.at("T");
        (*attr)["mode"] = src_attr.at("mode");
        (*attr)["round_mode"] = src_attr.at("round_mode");
        (*attr)["axis"] = src_attr.at("axis");
        (*attr)["narrow_range"] = src_attr.at("narrow_range");
        (*attr)["ensure_minimum_range"] = src_attr.at("ensure_minimum_range");

        // Connect the output correctly
        SafeTensorId unique_id(quant_node.name(), 0);

        mutation->AddNode(std::move(quant_node), &status);
        auto* output_node_view = node_view->GetRegularFanout(0)[i].node_view();

        // TODO(itex): check whether this works
        // int in_idx = fanout_node_view->GetRegularFanout(0)[i].index();
        int in_idx = GetRegularFaninIndex(node_view, output_node_view, 0);
        ITEX_VLOG(2) << output_node_view->node()->name() << " "
                     << std::to_string(in_idx);
        mutation->AddOrUpdateRegularFanin(output_node_view, in_idx, unique_id);
        TF_ABORT_IF_ERROR(std::move(status));
      }
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

Status DuplicateDequantize(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));

  auto* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  ITEX_VLOG(2) << "DECOMPOSE DEQUANT";
  int num_nodes = ctx->graph_view.graph()->node_size();
  for (int idx = 0; idx < num_nodes; idx++) {
    auto* node_view = ctx->graph_view.GetNode(idx);
    auto* node_def = node_view->node();

    // TODO(intel-tf): refactor these code, here we 1) duplicate dequantize +
    // cast, or 2) duplicate quantize
    if (node_def->op() == "Dequantize" &&
        node_view->GetRegularFanin(0).node_view()->node()->op() ==
            "QuantizeV2") {
      if (node_view->GetRegularFanout(0).size() == 1) {
        // Duplicate a new Dequantize and Cast node.
        auto* fanout_node_view = node_view->GetRegularFanout(0)[0].node_view();
        auto* fanout_node_def = fanout_node_view->node();
        if (fanout_node_def->op() != "Cast") continue;
        for (int i = 1; i < fanout_node_view->GetRegularFanout(0).size(); i++) {
          NodeDef dequant_node;
          dequant_node.set_op(node_def->op());
          dequant_node.set_name(node_def->name() + "_duplicate_" +
                                std::to_string(i));
          dequant_node.set_device(node_def->device());
          dequant_node.add_input(node_def->input(0));
          dequant_node.add_input(node_def->input(1));
          dequant_node.add_input(node_def->input(2));
          // Duplicate the attributes
          auto* new_deq_attr = dequant_node.mutable_attr();
          auto& old_deq_attr = node_def->attr();
          (*new_deq_attr)["T"] = old_deq_attr.at("T");
          (*new_deq_attr)["mode"] = old_deq_attr.at("mode");
          (*new_deq_attr)["axis"] = old_deq_attr.at("axis");
          (*new_deq_attr)["dtype"] = old_deq_attr.at("dtype");
          (*new_deq_attr)["narrow_range"] = old_deq_attr.at("narrow_range");

          NodeDef cast_node;
          cast_node.set_op(fanout_node_def->op());
          cast_node.set_name(fanout_node_def->name() + "_duplicate_" +
                             std::to_string(i));
          cast_node.set_device(fanout_node_def->device());
          cast_node.add_input(dequant_node.name());
          // Duplicate the attributes
          auto* new_cast_attr = cast_node.mutable_attr();
          auto& old_cast_attr = fanout_node_def->attr();
          (*new_cast_attr)["DstT"] = old_cast_attr.at("DstT");
          (*new_cast_attr)["SrcT"] = old_cast_attr.at("SrcT");
          (*new_cast_attr)["Truncate"] = old_cast_attr.at("Truncate");
          // Connect the output correctly
          SafeTensorId unique_id(cast_node.name(), 0);

          mutation->AddNode(std::move(dequant_node), &status);
          mutation->AddNode(std::move(cast_node), &status);

          auto* output_node_view =
              fanout_node_view->GetRegularFanout(0)[i].node_view();

          // TODO(itex): check whether this works
          // int in_idx = fanout_node_view->GetRegularFanout(0)[i].index();
          int in_idx =
              GetRegularFaninIndex(fanout_node_view, output_node_view, 0);
          ITEX_VLOG(2) << output_node_view->node()->name() << " "
                       << std::to_string(in_idx);
          mutation->AddOrUpdateRegularFanin(output_node_view, in_idx,
                                            unique_id);
          TF_ABORT_IF_ERROR(std::move(status));
        }
      } else {
        // Duplicate a new Dequantize node from the existing dequant node.
        for (int i = 1; i < node_view->GetRegularFanout(0).size(); i++) {
          NodeDef dequant_node;
          dequant_node.set_op(node_def->op());
          dequant_node.set_name(node_def->name() + "_duplicate_" +
                                std::to_string(i));
          dequant_node.set_device(node_def->device());
          dequant_node.add_input(node_def->input(0));
          dequant_node.add_input(node_def->input(1));
          dequant_node.add_input(node_def->input(2));
          // Duplicate the attributes
          auto* attr = dequant_node.mutable_attr();
          auto& src_attr = node_def->attr();
          (*attr)["T"] = src_attr.at("T");
          (*attr)["mode"] = src_attr.at("mode");
          (*attr)["axis"] = src_attr.at("axis");
          (*attr)["dtype"] = src_attr.at("dtype");
          (*attr)["narrow_range"] = src_attr.at("narrow_range");
          // Connect the output correctly
          SafeTensorId unique_id(dequant_node.name(), 0);

          mutation->AddNode(std::move(dequant_node), &status);
          auto* output_node_view =
              node_view->GetRegularFanout(0)[i].node_view();

          // TODO(itex): check whether this works
          // int in_idx = fanout_node_view->GetRegularFanout(0)[i].index();
          int in_idx = GetRegularFaninIndex(node_view, output_node_view, 0);
          ITEX_VLOG(2) << output_node_view->node()->name() << " "
                       << std::to_string(in_idx);
          mutation->AddOrUpdateRegularFanin(output_node_view, in_idx,
                                            unique_id);
          TF_ABORT_IF_ERROR(std::move(status));
        }
      }
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

Status InsertReshapeForDepthwise(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));

  auto* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  int num_nodes = ctx->graph_view.graph()->node_size();
  for (int idx = 0; idx < num_nodes; idx++) {
    auto* node_view = ctx->graph_view.GetNode(idx);
    auto* node_def = node_view->node();

    if (node_def->op() == "DepthwiseConv2dNative") {
      NodeDef* weight_node;
      auto* weight_node_view = node_view->GetRegularFanin(1).node_view();
      if (weight_node_view->node()->op() == "Dequantize") {
        // INT8 case with multiplier = 1
        weight_node_view = weight_node_view->GetRegularFanin(0)
                               .node_view()
                               ->GetRegularFanin(0)
                               .node_view();
        weight_node = weight_node_view->node();
        if (weight_node_view->NumRegularFanouts() != 1) {
          continue;
        }

        bool is_success;
        std::vector<int64_t> shape_value;
        GetShapeFromConstDataNode(weight_node, &shape_value, &is_success);
        if (!is_success) {
          continue;
        }

        auto& src_attr = node_def->attr();

        NodeDef const_up_node;
        const_up_node.set_op("HostConst");
        const_up_node.set_name(node_def->name() + "_up_const");
        const_up_node.set_device(node_def->device());

        AttrValue const_up_attr_type;
        const_up_attr_type.set_type(DT_INT32);
        AttrValue const_up_attr_value;
        TensorProto* const_up_t = const_up_attr_value.mutable_tensor();

        Tensor const_up_value_tensor = Tensor(DT_INT32, TensorShape({3}));
        int32* const_up_value_tensor_ptr =
            static_cast<int32*>(const_up_value_tensor.data());
        const_up_value_tensor_ptr[0] = shape_value[0];
        const_up_value_tensor_ptr[1] = shape_value[1];
        const_up_value_tensor_ptr[2] = shape_value[2] * shape_value[3];

        const_up_value_tensor.AsProtoTensorContent(const_up_t);
        const_up_node.mutable_attr()->insert({"dtype", const_up_attr_type});
        const_up_node.mutable_attr()->insert({"value", const_up_attr_value});

        NodeDef reshape_up_node;
        reshape_up_node.set_op("Reshape");
        reshape_up_node.set_name(node_def->name() + "_up_reshape");
        reshape_up_node.set_device(node_def->device());
        reshape_up_node.add_input(weight_node->name());
        reshape_up_node.add_input(const_up_node.name());

        auto* reshape_up_attr = reshape_up_node.mutable_attr();
        (*reshape_up_attr)["T"] = src_attr.at("T");
        SetAttrValue(DT_INT32, &(*reshape_up_attr)["Tshape"]);

        TensorId reshape_up_id = ParseTensorName(reshape_up_node.name());
        mutation->AddNode(std::move(const_up_node), &status);
        TF_ABORT_IF_ERROR(status);
        mutation->AddNode(std::move(reshape_up_node), &status);
        TF_ABORT_IF_ERROR(status);

        auto* q_node_view = node_view->GetRegularFanin(1)
                                .node_view()
                                ->GetRegularFanin(0)
                                .node_view();
        mutation->AddOrUpdateRegularFanin(q_node_view, 0, reshape_up_id);

        NodeDef const_down_node;
        const_down_node.set_op("HostConst");
        const_down_node.set_name(node_def->name() + "_down_const");
        const_down_node.set_device(node_def->device());

        AttrValue const_down_attr_type;
        const_down_attr_type.set_type(DT_INT32);
        AttrValue const_down_attr_value;
        TensorProto* const_down_t = const_down_attr_value.mutable_tensor();

        Tensor const_down_value_tensor = Tensor(DT_INT32, TensorShape({4}));
        int32* const_down_value_tensor_ptr =
            static_cast<int32*>(const_down_value_tensor.data());
        const_down_value_tensor_ptr[0] = shape_value[0];
        const_down_value_tensor_ptr[1] = shape_value[1];
        const_down_value_tensor_ptr[2] = shape_value[2];
        const_down_value_tensor_ptr[3] = shape_value[3];

        const_down_value_tensor.AsProtoTensorContent(const_down_t);
        const_down_node.mutable_attr()->insert({"dtype", const_down_attr_type});
        const_down_node.mutable_attr()->insert(
            {"value", const_down_attr_value});

        NodeDef reshape_down_node;
        reshape_down_node.set_op("Reshape");
        reshape_down_node.set_name(node_def->name() + "_down_reshape");
        reshape_down_node.set_device(node_def->device());
        string deq_name =
            node_view->GetRegularFanin(1).node_view()->node()->name();

        reshape_down_node.add_input(deq_name);
        reshape_down_node.add_input(const_down_node.name());

        auto* reshape_down_attr = reshape_down_node.mutable_attr();
        (*reshape_down_attr)["T"] = src_attr.at("T");
        SetAttrValue(DT_INT32, &(*reshape_down_attr)["Tshape"]);

        TensorId reshape_down_id = ParseTensorName(reshape_down_node.name());
        mutation->AddNode(std::move(const_down_node), &status);
        TF_ABORT_IF_ERROR(status);
        mutation->AddNode(std::move(reshape_down_node), &status);
        TF_ABORT_IF_ERROR(status);
        mutation->AddOrUpdateRegularFanin(node_view, 1, reshape_down_id);
      }
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

Status SeparateQuantizeAndDequantize(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));

  auto* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  ITEX_VLOG(2)
      << "Separate QuantizeAndDequantizeV4 to QuantizeV2 and Dequantize pair";

  int num_nodes = ctx->graph_view.graph()->node_size();
  for (int idx = 0; idx < num_nodes; idx++) {
    auto* node_view = ctx->graph_view.GetNode(idx);
    auto* node_def = node_view->node();

    // TODO(itex): handle Cast + QuantizeAndDequantizeV4 condition
    if (node_def->op() == "QuantizeAndDequantizeV4" &&
        IsAnyConst(*(node_view->GetRegularFanin(1).node_view()->node())) &&
        IsAnyConst(*(node_view->GetRegularFanin(2).node_view()->node()))) {
      DataType T;
      TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "T", &T));
      if (T != DT_FLOAT) continue;

      // Create quantize node
      NodeDef quant_node;
      quant_node.set_op("QuantizeV2");
      quant_node.set_name(node_def->name() + "_quantize");
      quant_node.set_device(node_def->device());
      quant_node.add_input(node_def->input(0));
      quant_node.add_input(node_def->input(1));
      quant_node.add_input(node_def->input(2));

      // Set QuantizeV2 attributes
      auto* new_q_attr = quant_node.mutable_attr();
      auto& src_attr = node_def->attr();
      if (src_attr.at("signed_input").b()) {
        (*new_q_attr)["T"].set_type(DT_QINT8);
      } else {
        (*new_q_attr)["T"].set_type(DT_QUINT8);
      }
      // TODO(itex): Check the correctness here. QuantizeAndDequantizeV4 doesn't
      // have attr "mode". And the its kernel implementation is SCALED style.
      SetAttrValue("SCALED", &(*new_q_attr)["mode"]);
      (*new_q_attr)["round_mode"] = src_attr.at("round_mode");
      (*new_q_attr)["axis"] = src_attr.at("axis");
      (*new_q_attr)["narrow_range"] = src_attr.at("narrow_range");
      // TODO(itex): Check the correctness here. QuantizeAndDequantizeV4 doesn't
      // have attr "ensure_minimum_range". So we here provide a small value for
      // Quantize op
      SetAttrValue(0.00001, &(*new_q_attr)["ensure_minimum_range"]);

      // Create dequantize node
      NodeDef dequant_node;
      dequant_node.set_op("Dequantize");
      dequant_node.set_name(node_def->name() + "_dequantize");
      dequant_node.set_device(node_def->device());
      dequant_node.add_input(quant_node.name());
      dequant_node.add_input(quant_node.name() + ":1");
      dequant_node.add_input(quant_node.name() + ":2");
      // Set Dequantize attributes
      auto* new_deq_attr = dequant_node.mutable_attr();
      if (src_attr.at("signed_input").b()) {
        (*new_deq_attr)["T"].set_type(DT_QINT8);
      } else {
        (*new_deq_attr)["T"].set_type(DT_QUINT8);
      }
      SetAttrValue("SCALED", &(*new_deq_attr)["mode"]);
      (*new_deq_attr)["axis"] = src_attr.at("axis");
      (*new_deq_attr)["dtype"] = src_attr.at("T");
      (*new_deq_attr)["narrow_range"] = src_attr.at("narrow_range");

      // Connect the output correctly
      SafeTensorId unique_id(dequant_node.name(), 0);

      mutation->AddNode(std::move(quant_node), &status);
      mutation->AddNode(std::move(dequant_node), &status);

      for (int i = 0; i < node_view->GetRegularFanout(0).size(); i++) {
        auto* output_node_view = node_view->GetRegularFanout(0)[i].node_view();

        // TODO(itex): check whether this works
        // int in_idx = fanout_node_view->GetRegularFanout(0)[i].index();
        int in_idx = GetRegularFaninIndex(node_view, output_node_view, 0);
        ITEX_VLOG(2) << output_node_view->node()->name() << " "
                     << std::to_string(in_idx);
        mutation->AddOrUpdateRegularFanin(output_node_view, in_idx, unique_id);
        TF_ABORT_IF_ERROR(std::move(status));
      }
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

// Change the input order of Conv2DBackpropInput
Status RunPrePass(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));
  auto* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  int num_nodes = ctx->graph_view.graph()->node_size();
  for (int idx = 0; idx < num_nodes; idx++) {
    auto* node_view = ctx->graph_view.GetNode(idx);
    auto* node_def = node_view->node();
    if (node_def->op() != "Conv2DBackpropInput" &&
        node_def->op() != "Conv2DBackpropFilter")
      continue;

    int input_index = node_def->op() == "Conv2DBackpropInput" ? 0 : 1;

    auto* input_node_view = node_view->GetRegularFanin(input_index).node_view();
    auto* input_node_def = input_node_view->node();
    if (!IsShapeN(*input_node_def)) continue;
  }

  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

Status RunRewritePass(OneDnnGraphContext* ctx) {
  TF_ABORT_IF_ERROR(ctx->node_type_map.Clear());
  TF_ABORT_IF_ERROR(ctx->node_type_map.Init(*ctx->graph_view.graph()));

  int num_nodes = ctx->graph_view.graph()->node_size();
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);

#ifdef INTEL_CPU_ONLY
  dnnl::graph::graph graph_ctx{dnnl::graph::engine::kind::cpu};
#else
  dnnl::graph::graph graph_ctx{dnnl::graph::engine::kind::gpu};
#endif

  const TranslationMap& tf_to_onednn_graph_op_translation_map =
      getTranslationMap();
  std::unordered_set<std::string> wildcard_nodes;
  std::unordered_set<std::string> rewrite_nodes;
  AdditionalArgs addtional_args;

  // TODO(itex): Infer shapes.
  // TF_ABORT_IF_ERROR(ctx->graph_properties.InferStatically(
  //     /*assume_valid_feeds=*/false,
  //     /*aggressive_shape_inference=*/false,
  //     /*include_input_tensor_values=*/true,
  //     /*include_output_tensor_values=*/false));

  bool onednn_graph_all_type_flag =
      GetOptimizerConfigFlags().enable_onednn_graph_all_type;

  // Tranverse graph, select onednn graph nodes and mark wildcard nodes.
  ITEX_VLOG(2) << "BEFORE SELECT NODE ";
  LLGAEdgeManager edge_manager;
  TF_ABORT_IF_ERROR(
      SelectNode(ctx, num_nodes, tf_to_onednn_graph_op_translation_map,
                 &wildcard_nodes, &rewrite_nodes, false, &graph_ctx,
                 &edge_manager, &addtional_args, onednn_graph_all_type_flag));

  // Tranverse graph, select wildcard nodes.
  TF_ABORT_IF_ERROR(
      SelectNode(ctx, num_nodes, tf_to_onednn_graph_op_translation_map,
                 &wildcard_nodes, &rewrite_nodes, true, &graph_ctx,
                 &edge_manager, &addtional_args, onednn_graph_all_type_flag));

  graph_ctx.finalize();

  auto l_partition_list =
      graph_ctx.get_partitions(dnnl::graph::partition::policy::fusion);
  static int count = 0;
  LLGAEdgeManager edge_manager_tmp;
  for (auto& it : l_partition_list) {
    if (it.is_supported()) {
      count++;
      ITEX_VLOG(2) << "Number of Partitions = " << count;
      TF_ABORT_IF_ERROR(FuseFwPartitionWithLLGA(
          ctx, it, &invalidated_nodes, &nodes_to_delete, &edge_manager,
          &edge_manager_tmp, &addtional_args, onednn_graph_all_type_flag));
    }
  }

  edge_manager.UpdateEdgeManager(edge_manager_tmp);

  auto* mutation = ctx->graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(ctx->graph_view.GetNode(i));
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());
  return Status::OK();
}

void DumpLLGAGraph(const GraphDef& graph_def, const std::string prefix) {
  // 1970-01-01 00:00:00
  const auto start = std::chrono::time_point<std::chrono::system_clock>{};
  const auto current = std::chrono::system_clock::now();
  const auto duration = current - start;
  std::string hash_time = std::to_string(duration.count());
  std::string dump_file_name = prefix + hash_time + ".pbtxt";
  std::ofstream dump_graph(dump_file_name);
  dump_graph << graph_def.DebugString();
  dump_graph.close();
  ITEX_VLOG(4) << "Dump graph to: " << dump_file_name;
}

Status RunOneDnnGraph(const GrapplerItem& item, const GraphDef& graph_def,
                      GraphDef* optimized_graph) {
  // TODO(itex): Remove the lock, when LLGA modify their all thread unsafe
  // data structure, such as "pass_manager". Seems LLGA already fix the error
  mutex_lock m(&mu);

  // Enable oneDNN Graph compiler backend
  bool onednn_graph_compiler_backend_flag =
      GetOptimizerConfigFlags().enable_onednn_graph_compiler_backend;
  if (!onednn_graph_compiler_backend_flag) {
    setenv("_DNNL_DISABLE_COMPILER_BACKEND", "1", 0);
  }

  // Enable oneDNN Graph dnnl backend
  bool onednn_graph_dnnl_backend_flag =
      GetOptimizerConfigFlags().enable_onednn_graph_dnnl_backend;
  if (!onednn_graph_dnnl_backend_flag) {
    setenv("_DNNL_DISABLE_DNNL_BACKEND", "1", 0);
  }

  Status status;
  GraphDef multable_graph_def = graph_def;
  OneDnnGraphContext ctx(item, &multable_graph_def, &status);
  TF_ABORT_IF_ERROR(std::move(status));

  if (ITEX_VLOG_IS_ON(4)) {
    ITEX_VLOG(4) << "graph node before LLGA: "
                 << ctx.graph_view.graph()->node_size();

    DumpLLGAGraph(graph_def, "graph_before_LLGA_");
  }

  // TODO(itex): shape inference currently only used in verify scalar tensor
  // for LLGA Mul. Remove this shape inference function, once LLGA supports
  // scalar tensor.
  if (!ctx.inferred_graph_properties) {
    TF_RETURN_IF_ERROR(ctx.graph_properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/false,
        /*include_input_tensor_values=*/true,
        /*include_output_tensor_values=*/false));
    ctx.inferred_graph_properties = true;
  }

  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(RunPrePass(&ctx));

  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(AddRetNode(&ctx));

  // Separate QuantizeAndDequantizeV4 into QuantizeV2 and Dequantize
  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(SeparateQuantizeAndDequantize(&ctx));

  // Split the dequantize node with >1 outputs into two dequant node so
  // that the  onednngraph quantization patterns match.
  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(DuplicateDequantize(&ctx));

  // Split Quantize node, if it has multiple dequantize node. This situation
  // often happens when multiple conv/mm share the same weight.
  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(DuplicateQuantize(&ctx));

  // Insert Reshape before & after Q / DQ pair, when depthwise weight K = 1
  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(InsertReshapeForDepthwise(&ctx));

  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(RemoveAssignXVariable(&ctx));

  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(RunRewritePass(&ctx));

  TF_ABORT_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
  TF_ABORT_IF_ERROR(RemoveRetNode(&ctx));

  *optimized_graph = std::move(multable_graph_def);

  if (ITEX_VLOG_IS_ON(4)) {
    ITEX_VLOG(4) << "graph node after LLGA: "
                 << ctx.graph_view.graph()->node_size();
    DumpLLGAGraph(*optimized_graph, "graph_after_LLGA_");
  }
  return Status::OK();
}

}  // namespace graph
}  // namespace itex
