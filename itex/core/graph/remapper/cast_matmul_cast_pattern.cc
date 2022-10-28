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

namespace itex {
namespace graph {

class CastBf16MatMulCast32 : public Fusion {
 public:
  CastBf16MatMulCast32() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern matmul = {kMatMul, "matmul", NodeStatus::kRemove};
    OpTypePattern output = {kCast, "output", NodeStatus::kReplace};
    OpTypePattern cast1 = {kCast, "bf16scr1", NodeStatus::kRemove};
    OpTypePattern cast2 = {kCast, "bf16scr2", NodeStatus::kRemove};

    matmul.AddInput(cast1);
    matmul.AddInput(cast2);
    output.AddInput(matmul);

    pattern_ = InternalPattern(std::move(output));
  }

  ~CastBf16MatMulCast32() {}

  std::string Name() override { return "cast-bf16matmul-cast"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    auto& graph_view = ctx->graph_view;
    auto* cast_node_def = graph_view.GetNode(node_index)->node();
    if (!IsCast(*cast_node_def)) return ret;
    DataType dst_dtype = GetDataTypeFromAttr(*cast_node_def, "DstT");
    DataType src_dtype = GetDataTypeFromAttr(*cast_node_def, "SrcT");
    if ((dst_dtype != DT_FLOAT) || (src_dtype != DT_BFLOAT16)) return ret;
    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);
    if (ret.Empty()) return ret;

    NodeDef* cast1_node_def =
        graph_view.GetNode(ret.map.at("bf16scr1"))->node();
    NodeDef* cast2_node_def =
        graph_view.GetNode(ret.map.at("bf16scr2"))->node();

    DataType src1_dtype = GetDataTypeFromAttr(*cast1_node_def, "SrcT");
    DataType src2_dtype = GetDataTypeFromAttr(*cast2_node_def, "SrcT");

    if ((src1_dtype != DT_FLOAT) || (src2_dtype != DT_FLOAT))
      return ret.ToEmpty();
    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    auto* output_node =
        ctx->graph_view.GetNode(properties.map.at("output"))->node();
    auto* matmul_node =
        ctx->graph_view.GetNode(properties.map.at("matmul"))->node();
    auto* src1_node =
        ctx->graph_view.GetNode(properties.map.at("bf16scr1"))->node();
    auto* src2_node =
        ctx->graph_view.GetNode(properties.map.at("bf16scr2"))->node();

    NodeDef fused_node;
    fused_node.set_name(output_node->name());
    fused_node.set_op("_ITEXAccMatMul");
    fused_node.set_device(matmul_node->device());

    CopyAllAttrs(*matmul_node, &fused_node);
    AddNodeAttr("is_bf16_math_mode", true, &fused_node);

    auto* new_attr = fused_node.mutable_attr();
    DataType MatmulType;
    if (TryGetNodeAttr(fused_node, "T", &MatmulType)) {
      SetAttrValue(DT_FLOAT, &(*new_attr)["T"]);
      SetAttrValue(DT_FLOAT, &(*new_attr)["Tout"]);
      SetAttrValue(DT_FLOAT, &(*new_attr)["Tpost"]);
    }

    fused_node.add_input(src1_node->input(0));
    fused_node.add_input(src2_node->input(0));

    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    Status status;
    mutation->AddNode(std::move(fused_node), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    return Status::OK();
  }
};
REGISTER_FUSION(CastBf16MatMulCast32)

}  // namespace graph
}  // namespace itex
