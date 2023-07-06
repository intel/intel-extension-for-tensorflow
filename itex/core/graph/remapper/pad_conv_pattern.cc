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

#include "itex/core/graph/remapper/constant_names.h"
#include "itex/core/graph/remapper/fusion.h"
#include "itex/core/graph/remapper/remapper.h"
#include "itex/core/graph/utils/layout_utils.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/pattern_utils.h"
#include "itex/core/graph/utils/utils.h"

/*
Merge pad into conv2d or conv3d as an attribute, remain quantize and dequantize.
Before:                                    After:
                input2  const
                   \     /
                     pad                                       input2
                      |                                            |
                  quantize                                    quantize
                      |                                            |
    input1        dequantize                  input1          dequantize
          \      /                                  \         /
        conv2d|conv3d                       conv2dwithpad|conv3dwithpad
*/
namespace itex {
namespace graph {
class PadConv : public Fusion {
 public:
  explicit PadConv(bool is_conv3d) : Fusion(), is_conv3d_(is_conv3d) {
    is_partial_ = true;
    using utils::NodeStatus;
    using utils::OpTypePattern;

    OpTypePattern input1 = {kAny, "input1", NodeStatus::kRemain};
    OpTypePattern input2 = {kAny, "input2", NodeStatus::kRemain};
    OpTypePattern constv = {kConst, "constv", NodeStatus::kRemain};
    OpTypePattern pad = {kPad, "pad", NodeStatus::kRemove};
    OpTypePattern const_min = {kConst, "const_min", NodeStatus::kRemain};
    OpTypePattern const_max = {kConst, "const_max", NodeStatus::kRemain};
    OpTypePattern quantize = {kQuantizeV2, "quantize", NodeStatus::kRemain};
    OpTypePattern dequantize = {kDequantize, "dequantize", NodeStatus::kRemain};
    OpTypePattern conv = {is_conv3d_ ? kConv3D : kConv2D, "conv_output",
                          NodeStatus::kReplace};

    pad.AddInput(input2);
    pad.AddInput(constv);
    quantize.AddInput(pad);
    quantize.AddInput(const_min);
    quantize.AddInput(const_max);
    dequantize.AddNSameInput(quantize);
    conv.AddInput(dequantize);
    conv.AddInput(input1);

    pattern_ = InternalPattern(std::move(conv));
  }

  ~PadConv() {}
  std::string Name() override {
    return is_conv3d_ ? "pad-conv3d" : "pad-conv2d";
  }
  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    auto& graph_view = ctx->graph_view;
    auto* conv_node_def = graph_view.GetNode(node_index)->node();

    if (is_conv3d_) {
      if (!IsConv3D(*conv_node_def)) return ret;
    } else {
      if (!IsConv2D(*conv_node_def)) return ret;
    }

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);
    if (ret.Empty()) return ret;

    if (NodeIsOnGpu(conv_node_def)) return ret.ToEmpty();

    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    // dtype nodedef
    auto* constv_node =
        ctx->graph_view.GetNode(properties.map.at("constv"))->node();
    auto* pad_node = ctx->graph_view.GetNode(properties.map.at("pad"))->node();
    auto* conv_node =
        ctx->graph_view.GetNode(properties.map.at("conv_output"))->node();
    NodeDef fused_node;
    if (is_conv3d_) {
      // Conv3D op doesn't have explicit padding attr
      fused_node.set_op("_ITEXConv3D");
    } else {
      fused_node.set_op("Conv2D");
    }

    fused_node.set_name(conv_node->name());
    fused_node.set_device(conv_node->device());
    fused_node.add_input(conv_node->input(0));
    fused_node.add_input(conv_node->input(1));

    CopyAllAttrs(*conv_node, &fused_node);

    // set pad as a new attribute of conv2d|conv3d
    Tensor const_tensor;
    std::vector<int> pad_value;
    if (constv_node != nullptr && constv_node->op() == "Const" &&
        const_tensor.FromProto(constv_node->attr().at("value").tensor())) {
      int length = const_tensor.NumElements();
      for (int i = 0; i < length; i++) {
        pad_value.push_back(const_tensor.flat<int32>()(i));
      }
    }
    // check name and attr
    auto* new_attr = fused_node.mutable_attr();
    // (*new_attr)["padding"] = "EXPLICIT";
    SetAttrValue("EXPLICIT", &(*new_attr)["padding"]);
    SetAttrValue(pad_value, &(*new_attr)["explicit_paddings"]);

    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    Status status;
    mutation->AddNode(std::move(fused_node), &status);
    TF_RETURN_IF_ERROR(status);

    // change the input node of quantize from pad to (the former node of pad)
    auto* quantize_view =
        ctx->graph_view.GetNode(properties.map.at("quantize"));
    TensorId fanin = ParseTensorName(pad_node->input(0));
    mutation->AddOrUpdateRegularFanin(quantize_view, 0, fanin);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }

 protected:
  bool is_conv3d_;
};

class PadConv3d : public PadConv {
 public:
  PadConv3d() : PadConv(true) {}
};

class PadConv2d : public PadConv {
 public:
  PadConv2d() : PadConv(false) {}
};

REGISTER_FUSION(PadConv3d)
REGISTER_FUSION(PadConv2d)
}  // namespace graph
}  // namespace itex
