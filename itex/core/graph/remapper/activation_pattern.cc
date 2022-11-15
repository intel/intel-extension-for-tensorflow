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
#include "itex/core/graph/utils/pattern_utils.h"
#include "itex/core/graph/utils/utils.h"

namespace itex {
namespace graph {

class SigmoidWithMulFusion : public Fusion {
 public:
  SigmoidWithMulFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern input = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern sigmoid = {kSigmoid, "sigmoid", NodeStatus::kRemove};
    OpTypePattern mul = {kMul, "mul_to_swish", NodeStatus::kReplace};

    sigmoid.AddInput(input);
    mul.AddInput(sigmoid).AddInput(input);

    pattern_ = InternalPattern(std::move(mul));
  }

  ~SigmoidWithMulFusion() {}

  std::string Name() override { return "sigmoid-with-mul"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    auto& graph_view = ctx->graph_view;
    auto* mul_node_def = graph_view.GetNode(node_index)->node();
    if (!HasDataType(mul_node_def, DT_FLOAT) &&
        !HasDataType(mul_node_def, DT_BFLOAT16) &&
        !(NodeIsOnGpu(mul_node_def) && HasDataType(mul_node_def, DT_HALF)))
      return ret;

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);

    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* mul =
        graph_view.GetNode(properties.map.at("mul_to_swish"))->node();
    const NodeDef* sigmoid =
        graph_view.GetNode(properties.map.at("sigmoid"))->node();
    NodeDef fused_op;
    fused_op.set_name(mul->name());
    fused_op.set_op(kSwish);
    fused_op.set_device(mul->device());
    fused_op.add_input(sigmoid->input(0));

    auto* attr = fused_op.mutable_attr();
    (*attr)["T"] = mul->attr().at("T");

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }
};
REGISTER_FUSION(SigmoidWithMulFusion)

// Fuse Sigmoid(alpha) and Mul into Swish
/*
            mul
            / \
      sigmoid  |          swish
        |      |   =>       |
        mul    |          input
       /  \   /
    const*  input   swish.attr.at("alpha") = const.val
*/
// *) const must be a scalar with float datatype.
class SigmoidAlphaWithMulFusion : public Fusion {
 public:
  SigmoidAlphaWithMulFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern input = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern constant = {kConst, "const", NodeStatus::kRemove};
    OpTypePattern mul_to_sigmoid = {kMul, "mul_to_sigmoid",
                                    NodeStatus::kRemove};
    OpTypePattern sigmoid = {kSigmoid, "sigmoid", NodeStatus::kRemove};
    OpTypePattern mul = {kMul, "mul_to_swish", NodeStatus::kReplace};

    mul_to_sigmoid.AddInput(constant).AddInput(input);
    sigmoid.AddInput(mul_to_sigmoid);
    mul.AddInput(sigmoid).AddInput(input);

    pattern_ = InternalPattern(std::move(mul));
  }

  ~SigmoidAlphaWithMulFusion() {}

  std::string Name() override { return "sigmoid-alpha-with-mul"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    auto& graph_view = ctx->graph_view;
    auto* mul_node_def = graph_view.GetNode(node_index)->node();
    if (!HasDataType(mul_node_def, DT_FLOAT) &&
        !HasDataType(mul_node_def, DT_BFLOAT16) &&
        !(NodeIsOnGpu(mul_node_def) && HasDataType(mul_node_def, DT_HALF)))
      return ret;
    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);

    if (!ret.Empty()) {
      const NodeDef* constant = graph_view.GetNode(ret.map.at("const"))->node();
      Tensor const_val;
      const_val.FromProto(constant->attr().at("value").tensor());
      DataType const_dtype = GetDataTypeFromAttr(*constant, "dtype");
      if (!(const_val.shape().dims() == 0) ||
          !(const_dtype == DT_BFLOAT16 || const_dtype == DT_HALF ||
            const_dtype == DT_DOUBLE || const_dtype == DT_FLOAT))
        ret.ToEmpty();
    }
    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* mul =
        graph_view.GetNode(properties.map.at("mul_to_swish"))->node();
    const NodeDef* input =
        graph_view.GetNode(properties.map.at("input"))->node();
    const NodeDef* constant =
        graph_view.GetNode(properties.map.at("const"))->node();
    NodeDef fused_op;
    fused_op.set_name(mul->name());
    fused_op.set_op(kSwish);
    fused_op.set_device(mul->device());
    fused_op.add_input(input->name());

    float alpha_value = 0;
    DataType const_dtype = GetDataTypeFromAttr(*constant, "dtype");
    Tensor const_tensor;
    const_tensor.FromProto(constant->attr().at("value").tensor());
    if (const_dtype == DT_BFLOAT16) {
      alpha_value = static_cast<float>(const_tensor.flat<Eigen::bfloat16>()(0));
    } else if (const_dtype == DT_HALF) {
      alpha_value = static_cast<float>(const_tensor.flat<Eigen::half>()(0));
    } else if (const_dtype == DT_DOUBLE) {
      alpha_value = static_cast<float>(const_tensor.flat<double>()(0));
    } else if (const_dtype == DT_FLOAT) {
      alpha_value = const_tensor.flat<float>()(0);
    } else {
      ITEX_CHECK(false);
    }

    auto* attr = fused_op.mutable_attr();
    SetAttrValue(alpha_value, &(*attr)["alpha"]);
    SetAttrValue(mul->attr().at("T"), &(*attr)["T"]);

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }
};
REGISTER_FUSION(SigmoidAlphaWithMulFusion)

/*

       Input
       |   \
      |   SoftPlus                Input
     |      \                      |
     \     Tanh        ===>      Mish
      \     |                     |
       \   |                    Output
        Mul
         \
      Output

*/
class MishFusion : public Fusion {
 public:
  MishFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern input = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern softplus = {kSoftplus, "softplus", NodeStatus::kRemove};
    OpTypePattern tanh = {kTanh, "tanh", NodeStatus::kRemove};
    OpTypePattern mul = {kMul, "mul", NodeStatus::kReplace};

    softplus.AddInput(input);
    tanh.AddInput(softplus);
    mul.AddInput(tanh).AddInput(input);
    pattern_ = InternalPattern(std::move(mul));
  }

  ~MishFusion() {}

  std::string Name() override { return "Mish"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    auto& graph_view = ctx->graph_view;
    auto* mul_node_def = graph_view.GetNode(node_index)->node();
    if (!HasDataType(mul_node_def, DT_FLOAT) &&
        !HasDataType(mul_node_def, DT_BFLOAT16) &&
        !(NodeIsOnGpu(mul_node_def) && HasDataType(mul_node_def, DT_HALF)))
      return ret;

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);
    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* mul = graph_view.GetNode(properties.map.at("mul"))->node();
    const NodeDef* softplus =
        graph_view.GetNode(properties.map.at("softplus"))->node();
    NodeDef fused_op;
    fused_op.set_name(mul->name());
    fused_op.set_op(kMish);
    fused_op.set_device(mul->device());
    fused_op.add_input(softplus->input(0));

    auto* attr = fused_op.mutable_attr();
    (*attr)["T"] = mul->attr().at("T");

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    ITEX_VLOG(2) << "Fuse Mish Activation: "
                 << " Softplus= " << softplus->name()
                 << " Mul= " << mul->name();
    return Status::OK();
  }
};
REGISTER_FUSION(MishFusion)

}  // namespace graph
}  // namespace itex
