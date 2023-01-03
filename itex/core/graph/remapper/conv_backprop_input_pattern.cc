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
#include "itex/core/graph/utils/pattern_utils.h"
#include "itex/core/graph/utils/utils.h"

namespace itex {
namespace graph {

class PadWithConvBackpropFilterFusion : public Fusion {
 public:
  PadWithConvBackpropFilterFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;

    // It supports multiple types of conv.
    std::string conv_repr =
        absl::StrJoin({kConv2DBackpropFilter, kConv2DBackpropFilterWithBias,
                       kConv3DBackpropFilter, kConv3DBackpropFilterV2,
                       kConv3DBackpropFilterWithBias},
                      "|");

    // Note. We do not delete the pad here. Due to the pad will be used as the
    // input for conv forward.
    OpTypePattern input = {kPad, "pad", NodeStatus::kRemain};
    OpTypePattern filter_sizes = {kAny, "filter_sizes", NodeStatus::kRemain};
    OpTypePattern out_backprop = {kAny, "out_backprop", NodeStatus::kRemain};
    OpTypePattern conv = {conv_repr, "conv", NodeStatus::kReplace};

    conv.AddInput(input).AddInput(filter_sizes).AddInput(out_backprop);

    pattern_ = InternalPattern(std::move(conv));
  }

  ~PadWithConvBackpropFilterFusion() {}

  std::string Name() override { return "pad-with-conv_backprop_filter"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    if (ctx->remap_level == 0) {
      // Only work in second remapper iteration.
      return ret;
    }
    auto& graph_view = ctx->graph_view;

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);

    if (ret.Empty()) {
      return ret;
    }

    const NodeDef* conv = ret.GetNode(&graph_view, "conv");
    string padding_str;
    TF_ABORT_IF_ERROR(GetNodeAttr(*conv, "padding", &padding_str));
    if (padding_str != "VALID") return ret.ToEmpty();

    // Only fuse contraction with INT32 padding.
    // TODO(itex): support INT64 padding in future.
    const NodeDef* pad = ret.GetNode(&graph_view, "pad");
    if (!HasDataType(pad, DT_INT32, "Tpaddings")) return ret.ToEmpty();

    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* pad = properties.GetNode(&graph_view, "pad");
    const NodeDef* conv = properties.GetNode(&graph_view, "conv");

    std::string conv_op_name = conv->op();
    int is_itex_prefix = conv_op_name.find("_ITEX");
    if (is_itex_prefix != -1) {
      conv_op_name = conv_op_name.substr(5);
    }

    // All the new conv has a prefix of PadWith, the others are the same.
    std::string op = "_ITEXPadWith" + conv_op_name;

    NodeDef fused_op;
    fused_op.set_name(conv->name());
    fused_op.set_op(op);
    fused_op.set_device(conv->device());
    fused_op.add_input(pad->input(0));
    fused_op.add_input(conv->input(1));
    fused_op.add_input(conv->input(2));
    fused_op.add_input(pad->input(1));

    CopyAllAttrs(*conv, &fused_op);
    DataType paddings_type;
    TF_ABORT_IF_ERROR(GetNodeAttr(*pad, "Tpaddings", &paddings_type));
    AddNodeAttr("Tpaddings", paddings_type, &fused_op);

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }
};
REGISTER_FUSION(PadWithConvBackpropFilterFusion)
}  // namespace graph
}  // namespace itex
