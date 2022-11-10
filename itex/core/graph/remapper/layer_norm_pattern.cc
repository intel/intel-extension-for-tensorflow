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
#include "itex/core/graph/utils/symbolic_shapes.h"
#include "itex/core/graph/utils/utils.h"

namespace itex {
namespace graph {

class LayerNormFusionBase : public Fusion {
 public:
  LayerNormFusionBase() : Fusion() { is_partial = true; }

  ~LayerNormFusionBase() {}

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    auto* processed_input_node =
        graph_view.GetNode(properties.map.at("processed_input"))->node();
    auto* scale_node = graph_view.GetNode(properties.map.at("gamma"))->node();
    auto* output_node = graph_view.GetNode(properties.map.at("output"))->node();

    // TODO(yifeng): Remove this workaround when custom pattern is not needed.
    bool is_custom_pattern = (properties.map.count("fused_batch_norm") == 0);

    NodeDef fused_node;
    fused_node.set_name(output_node->name());
    if (is_custom_pattern) {
      fused_node.set_op(kMklLayerNorm);
    } else {
      fused_node.set_op(kLayerNorm);
    }
    fused_node.set_device(output_node->device());
    fused_node.add_input(processed_input_node->input(0));
    fused_node.add_input(scale_node->name());
    fused_node.add_input(output_node->input(0));

    auto* attr = fused_node.mutable_attr();
    auto& src_attr = output_node->attr();
    (*attr)["T"] = src_attr.at("T");

    // TODO(itex): the format will be only NHWC now, which is not the same with
    // FusedBatchNormV3.
    if (!is_custom_pattern) {
      AddNodeAttr("data_format", StringPiece("NHWC"), &fused_node);
      AddNodeAttr("U", DT_FLOAT, &fused_node);
    }

    // Set epsilon for layernorm transformerlt
    if (properties.map.find("epsilon") != properties.map.end()) {
      NodeDef* epsilon_node =
          ctx->graph_view.GetNode(properties.map.at("epsilon"))->node();

      Tensor const_tensor;
      float epsilon_value = 0.0001;
      if (epsilon_node != nullptr && epsilon_node->op() == "Const" &&
          const_tensor.FromProto(epsilon_node->attr().at("value").tensor())) {
        if (GetDataTypeFromAttr(*output_node, "T") == DT_BFLOAT16) {
          epsilon_value =
              static_cast<float>(const_tensor.flat<Eigen::bfloat16>()(0));
        } else if (GetDataTypeFromAttr(*output_node, "T") == DT_HALF) {
          epsilon_value =
              static_cast<float>(const_tensor.flat<Eigen::half>()(0));
        } else {
          epsilon_value = const_tensor.flat<float>()(0);
        }
      }
      SetAttrValue(epsilon_value, &(*attr)["epsilon"]);
    }

    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    Status status;
    mutation->AddNode(std::move(fused_node), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    return Status::OK();
  }

 protected:
  bool CheckInputOutputShape(RemapperContext* ctx, int input_node_index,
                             int output_node_index) const {
    auto input_properties = GetOutputProperties(ctx, input_node_index);
    auto output_properties = GetOutputProperties(ctx, output_node_index);

    return !input_properties.empty() && !output_properties.empty() &&
           ShapesSymbolicallyEqual(input_properties[0].shape(),
                                   output_properties[0].shape()) &&
           Rank(input_properties[0].shape()) >= 2 &&
           Rank(input_properties[0].shape()) <= 3;
  }
};

class LayerNormFusion : public LayerNormFusionBase {
 public:
  LayerNormFusion() : LayerNormFusionBase() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern input = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern pre_shape = {kAny, "pre_shae", NodeStatus::kRemain};
    OpTypePattern dims_fill_scale = {kAny, "dims_fill_scale",
                                     NodeStatus::kRemain};
    OpTypePattern unit_gamma = {kConst, "unit_gamma", NodeStatus::kRemain};
    OpTypePattern dims_fill_offset = {kAny, "dims_fill_offset",
                                      NodeStatus::kRemain};
    OpTypePattern zero_beta = {kConst, "zero_beta", NodeStatus::kRemain};
    OpTypePattern empty = {kConst, "empty", NodeStatus::kRemain};

    OpTypePattern processed_input = {kReshape, "processed_input",
                                     NodeStatus::kRemove};
    OpTypePattern fill_scale = {kFill, "fill_scale", NodeStatus::kRemove};
    OpTypePattern fill_offset = {kFill, "fill_offset", NodeStatus::kRemove};
    OpTypePattern fused_batch_norm = {kFusedBatchNormV3, "fused_batch_norm",
                                      NodeStatus::kRemove};

    OpTypePattern post_shape = {kAny, "post_shape", NodeStatus::kRemain};
    OpTypePattern post_reshape = {kAny, "post_reshape", NodeStatus::kRemove};
    OpTypePattern gamma = {kAny, "gamma", NodeStatus::kRemain};
    OpTypePattern scale = {kMul, "scale", NodeStatus::kRemove};
    OpTypePattern beta = {kAny, "beta", NodeStatus::kRemain};
    OpTypePattern output = {kAddV2, "output", NodeStatus::kReplace};

    processed_input.AddInput(input).AddInput(pre_shape);
    fill_scale.AddInput(dims_fill_scale).AddInput(unit_gamma);
    fill_offset.AddInput(dims_fill_offset).AddInput(zero_beta);
    fused_batch_norm.AddInput(processed_input)
        .AddInput(fill_scale)
        .AddInput(fill_offset)
        .AddInput(empty)
        .AddInput(empty);
    post_reshape.AddInput(fused_batch_norm).AddInput(post_shape);
    scale.AddInput(post_reshape).AddInput(gamma);
    output.AddInput(beta).AddInput(scale);

    pattern_ = InternalPattern(std::move(output));
  }

  ~LayerNormFusion() {}

  std::string Name() override { return "layernorm"; }

  MatchedProperties Check(RemapperContext* ctx, int node_index) const override {
    auto& graph_view = ctx->graph_view;
    MatchedProperties ret =
        FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);

    bool is_ok =
        !ret.Empty() && CheckIsTraining(ctx, ret.map.at("fused_batch_norm")) &&
        CheckMeanAndVariance(ctx, ret.map.at("empty")) &&
        CheckInputOutputShape(ctx, ret.map.at("input"), ret.map.at("output"));

    if (!is_ok) return ret.ToEmpty();
    return ret;
  }

 private:
  bool CheckIsTraining(RemapperContext* ctx, int node_index) const {
    // LayerNorm uses FusedBatchNorm in training mode.
    auto& graph_view = ctx->graph_view;
    NodeDef* fused_batch_norm_node = graph_view.GetNode(node_index)->node();
    bool is_training = false;
    return TryGetNodeAttr(*fused_batch_norm_node, kIsTraining, &is_training) &&
           is_training;
  }

  bool CheckMeanAndVariance(RemapperContext* ctx, int node_index) const {
    // FusedBatchNorm node should have mean/variance as empty constant
    auto& graph_view = ctx->graph_view;
    NodeDef* empty_const_node = graph_view.GetNode(node_index)->node();
    Tensor const_tensor;

    return empty_const_node != nullptr && empty_const_node->op() == kConst &&
           const_tensor.FromProto(
               empty_const_node->attr().at("value").tensor()) &&
           const_tensor.NumElements() == 0;
  }
};

class LayerNormFusionTransformerLT : public LayerNormFusionBase {
 public:
  LayerNormFusionTransformerLT() : LayerNormFusionBase() {
    using utils::NodeStatus;
    using utils::OpTypePattern;

    OpTypePattern input = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern indices_mean = {kAny, "indices_mean", NodeStatus::kRemain};
    OpTypePattern mean = {kMean, "mean", NodeStatus::kRemove};
    OpTypePattern processed_input = {kSub, "processed_input",
                                     NodeStatus::kRemove};
    OpTypePattern sub_mean = {kSub, "sub_mean", NodeStatus::kRemove};
    OpTypePattern square = {kSquare, "square", NodeStatus::kRemove};
    OpTypePattern mean_sqare = {kMean, "mean_square", NodeStatus::kRemove};
    OpTypePattern add_epsilon = {kAddV2, "add_epsilon", NodeStatus::kRemove};
    OpTypePattern rqsrt = {kRsqrt, "rqsrt", NodeStatus::kRemove};
    OpTypePattern scale = {kMul, "scale", NodeStatus::kRemove};
    OpTypePattern indices_var = {kAny, "indices_var", NodeStatus::kRemain};
    OpTypePattern epsilon = {kConst, "epsilon", NodeStatus::kRemain};
    OpTypePattern gamma = {kAny, "gamma", NodeStatus::kRemain};
    OpTypePattern mul = {kMul, "mul", NodeStatus::kRemove};
    OpTypePattern beta = {kAny, "beta", NodeStatus::kRemain};
    OpTypePattern output = {kAddV2, "output", NodeStatus::kReplace};

    mean.AddInput(input).AddInput(indices_mean);
    processed_input.AddInput(input).AddInput(mean);

    sub_mean.AddInput(input).AddInput(mean);
    square.AddInput(sub_mean);
    mean_sqare.AddInput(square).AddInput(indices_var);
    add_epsilon.AddInput(mean_sqare).AddInput(epsilon);
    rqsrt.AddInput(add_epsilon);
    scale.AddInput(rqsrt).AddInput(gamma);

    mul.AddInput(processed_input).AddInput(scale);
    output.AddInput(beta).AddInput(mul);

    pattern_ = InternalPattern(std::move(output));
  }

  ~LayerNormFusionTransformerLT() {}

  std::string Name() override { return "layernorm-for-TransformerLT"; }

  MatchedProperties Check(RemapperContext* ctx, int node_index) const override {
    auto& graph_view = ctx->graph_view;
    MatchedProperties ret = FillProperties(
        &graph_view, graph_view.GetNode(node_index), pattern_, false);

    bool is_ok = !ret.Empty() && CheckInputOutputShape(ctx, ret.map.at("input"),
                                                       ret.map.at("output"));

    if (!is_ok) return ret.ToEmpty();
    return ret;
  }
};

REGISTER_FUSION(LayerNormFusion)
REGISTER_FUSION(LayerNormFusionTransformerLT)
}  // namespace graph
}  // namespace itex
