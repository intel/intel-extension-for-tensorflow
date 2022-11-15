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

class Conv2DWithMishFusion : public Fusion {
 public:
  Conv2DWithMishFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern input = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern weights = {kConst, "weights", NodeStatus::kRemain};
    OpTypePattern bn_offset = {kConst, "bn_offset", NodeStatus::kRemain};
    // Conv2d
    OpTypePattern conv2d = {kFusedConv2D, "conv2d", NodeStatus::kRemove};
    // Mish Act
    OpTypePattern mish = {kMish, "mish", NodeStatus::kReplace};

    conv2d.AddInput(input).AddInput(weights).AddInput(bn_offset);
    mish.AddInput(conv2d);
    pattern_ = InternalPattern(std::move(mish));
  }

  ~Conv2DWithMishFusion() {}

  std::string Name() override { return "contraction_with_mish"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    auto& graph_view = ctx->graph_view;
    const auto* mish_node_def = graph_view.GetNode(node_index)->node();
    const auto* conv2d_node_view =
        graph_view.GetNode(node_index)->GetRegularFanin(0).node_view();
    NodeDef* conv2d_node_def = conv2d_node_view->node();

    if (!HasDataType(mish_node_def, DT_FLOAT) &&
        !HasDataType(mish_node_def, DT_BFLOAT16) &&
        !(NodeIsOnGpu(mish_node_def) && HasDataType(mish_node_def, DT_HALF)))
      return ret;

    if (IsInPreserveSet(*ctx, conv2d_node_def)) return ret;

    // verify the "fused_ops" attr of the FusedConv2D node
    std::vector<string> fused_ops;
    TryGetNodeAttr(*conv2d_node_def, "fused_ops", &fused_ops);
    if (!(fused_ops.size() == 1 && fused_ops[0] == "BiasAdd")) {
      return ret;
    }

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);
    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* mish = graph_view.GetNode(properties.map.at("mish"))->node();
    const NodeDef* conv2d =
        graph_view.GetNode(properties.map.at("conv2d"))->node();
    NodeDef fused_op;
    fused_op.set_name(mish->name());
    fused_op.set_op(kFusedConv2D);
    fused_op.set_device(mish->device());
    fused_op.add_input(conv2d->input(0));
    fused_op.add_input(conv2d->input(1));
    fused_op.add_input(conv2d->input(2));

    CopyAllAttrs(*conv2d, &fused_op);
    SetFusedOpAttributes(&fused_op, {"BiasAdd", "Mish"}, 1);

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    ITEX_VLOG(2) << "Fuse Contraction + Mish: "
                 << " Contraction= " << conv2d->name()
                 << " Mish= " << mish->name();
    return Status::OK();
  }
};
REGISTER_FUSION(Conv2DWithMishFusion)

}  // namespace graph
}  // namespace itex
