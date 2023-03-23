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
#include "itex/core/utils/op_kernel.h"

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
    auto* beta_node = graph_view.GetNode(properties.map.at("beta"))->node();
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
    fused_node.add_input(beta_node->name());

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

// This class has 2 inheritance class. Their patterns are almost the same,
// except switching 2 inputs in some binary ops.
class LayerNormFusionDistilBase : public Fusion {
 public:
  LayerNormFusionDistilBase() : Fusion() { is_partial = true; }

  ~LayerNormFusionDistilBase() {}

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    NodeDef* output_node =
        ctx->graph_view.GetNode(properties.map.at("output"))->node();
    NodeDef* input_node =
        ctx->graph_view.GetNode(properties.map.at("input"))->node();
    NodeDef* gamma_node =
        ctx->graph_view.GetNode(properties.map.at("gamma"))->node();
    NodeDef* beta_node =
        ctx->graph_view.GetNode(properties.map.at("beta"))->node();
    NodeDef* epsilon_node =
        ctx->graph_view.GetNode(properties.map.at("epsilon"))->node();

    NodeDef fused_node;
    fused_node.set_op(kLayerNorm);
    fused_node.set_device(output_node->device());
    fused_node.add_input(input_node->name());
    fused_node.add_input(gamma_node->name());
    fused_node.add_input(beta_node->name());

    auto* attr = fused_node.mutable_attr();
    auto& src_attr = output_node->attr();
    (*attr)["T"] = src_attr.at("T");
    SetAttrValue(DT_FLOAT, &(*attr)["U"]);
    SetAttrValue(false, &(*attr)["is_training"]);

    Tensor const_tensor;
    float epsilon_value = 0;
    if (epsilon_node != nullptr && epsilon_node->op() == "Const" &&
        const_tensor.FromProto(epsilon_node->attr().at("value").tensor())) {
      if (GetDataTypeFromAttr(*output_node, "T") == DT_BFLOAT16) {
        epsilon_value =
            static_cast<float>(const_tensor.flat<Eigen::bfloat16>()(0));
      } else if (GetDataTypeFromAttr(*output_node, "T") == DT_HALF) {
        epsilon_value = static_cast<float>(const_tensor.flat<Eigen::half>()(0));
      } else {
        epsilon_value = const_tensor.flat<float>()(0);
      }
    }
    SetAttrValue(epsilon_value, &(*attr)["epsilon"]);
    SetAttrValue(data_format, &(*attr)["data_format"]);

    fused_node.set_name(output_node->name());
    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    Status status;
    mutation->AddNode(std::move(fused_node), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    return Status::OK();
  }

  MatchedProperties Check(RemapperContext* ctx, int node_index) const override {
    MatchedProperties ret;
    ret = CheckIsLayerNorm(ctx, node_index);

    if (!ret.Empty()) {
      NodeDef* gamma_node =
          ctx->graph_view.GetNode(ret.map.at("gamma"))->node();
      NodeDef* beta_node = ctx->graph_view.GetNode(ret.map.at("beta"))->node();

      // When gamma and beta is bfloat16 or float16 data type, must change them
      // to float32.
      if (GetDataTypeFromAttr(*gamma_node, "dtype") != DT_FLOAT) {
        ReplaceF16NodeWithF32(gamma_node);
        ReplaceF16NodeWithF32(beta_node);
      }
    }
    return ret;
  }

 protected:
  mutable string data_format = "";

  inline bool HasControlFaninOrFanout(
      const utils::MutableNodeView& node_view) const {
    return node_view.NumControllingFanins() > 0 ||
           node_view.NumControlledFanouts() > 0;
  }

  MatchedProperties CheckIsLayerNorm(RemapperContext* ctx,
                                     int node_index) const {
    auto& graph_view = ctx->graph_view;
    MatchedProperties ret = FillProperties(
        &graph_view, graph_view.GetNode(node_index), pattern_, true);

    bool is_ok =
        !ret.Empty() &&
        CheckGammaAndBeta(ctx, ret.map.at("gamma"), ret.map.at("beta")) &&
        CheckMean(ctx, ret.map.at("mean1"), ret.map.at("r_indices1"));
    if (!is_ok) return ret.ToEmpty();
    return ret;
  }

  bool CheckGammaAndBeta(RemapperContext* ctx, int gamma_index,
                         int beta_index) const {
    // Check if gamma and beta constants have the same shape
    NodeDef* gamma_node = ctx->graph_view.GetNode(gamma_index)->node();
    NodeDef* beta_node = ctx->graph_view.GetNode(beta_index)->node();
    if (!gamma_node || !beta_node) {
      ITEX_VLOG(0) << "Unexpected error to retrieve gamma or beta node";
      return false;
    }
    Tensor gamma_tensor, beta_tensor;
    gamma_tensor.FromProto(gamma_node->attr().at("value").tensor());
    beta_tensor.FromProto(beta_node->attr().at("value").tensor());
    if (!gamma_tensor.IsSameSize(beta_tensor)) return false;

    return true;
  }

  bool CheckMean(RemapperContext* ctx, int mean_index,
                 int r_indciex_index) const {
    auto& graph_view = ctx->graph_view;
    NodeDef* mean1_node = graph_view.GetNode(mean_index)->node();

    bool keep_dims = false;
    if (!mean1_node || !TryGetNodeAttr(*mean1_node, "keep_dims", &keep_dims) ||
        !keep_dims) {
      return false;
    }
    DataType dtype = GetDataTypeFromAttr(*mean1_node, "T");
    // Allow bfloat16 and float16 data type
    if (dtype != DT_FLOAT && dtype != DT_BFLOAT16 && dtype != DT_HALF)
      return false;

    // Get the reduction axes for mean node to check if the
    // mean computation complies with layer normalization
    NodeDef* mean_axis_node = ctx->graph_view.GetNode(r_indciex_index)->node();
    if (!mean_axis_node) {
      ITEX_VLOG(2) << "Unexpected error to retrieve reduction axis node";
      return false;
    }

    Tensor mean_axis_tensor;
    mean_axis_tensor.FromProto(mean_axis_node->attr().at("value").tensor());
    dtype = mean_axis_tensor.dtype();
    if (dtype != DT_INT32 && dtype != DT_INT64) return false;

    return (dtype == DT_INT32) ? IsLayerNormReduction<int32>(mean_axis_tensor)
                               : IsLayerNormReduction<int64>(mean_axis_tensor);
  }

  // Helper function to check if the input axes data conforms with layer
  // normalization's mean computation for specified data format
  template <typename T>
  bool IsLayerNormReduction(const Tensor& axes_data) const {
    const int axis_num = axes_data.NumElements();

    // In this model, there is only 1 circumstance for mean; Data tensor:
    // 3-D, reduction axis: 2 or -1
    if (axis_num == 1) {
      if (axes_data.flat<T>()(0) == static_cast<T>(2) ||
          axes_data.flat<T>()(0) == static_cast<T>(-1)) {
        // TODO(itex): here nhwc is a bit confusing. Actually 3-D is tnc, and
        // seems more close to nchw?
        data_format = "NHWC";
        return true;
      }
    }

    return false;
  }

  // Helper function for LayerNorm bfloat16/float16 fusion. Because gamma and
  // beta must be float, so we need to cast them from bfloat16/float16 to float.
  void ReplaceF16NodeWithF32(NodeDef* node) const {
    const TensorProto& node_val = node->attr().at("value").tensor();
    DataType node_type = GetDataTypeFromAttr(*node, "dtype");
    if (node_type != DT_BFLOAT16 && node_type != DT_HALF) return;

    Tensor raw_tensor = Tensor(node_type, node_val.tensor_shape());
    raw_tensor.FromProto(node_val);

    const Eigen::ThreadPoolDevice d =
        OpKernelContext::eigen_cpu_device_singleton();

    Tensor cast_node_t = Tensor(DT_FLOAT, node_val.tensor_shape());

    if (node_type == DT_BFLOAT16) {
      cast_node_t.flat<float>().device(d) =
          raw_tensor.flat<Eigen::bfloat16>().template cast<float>();
    } else if (node_type == DT_HALF) {
      cast_node_t.flat<float>().device(d) =
          raw_tensor.flat<Eigen::half>().template cast<float>();
    }

    AttrValue attr_tensor;
    TensorProto* cast_node_val = attr_tensor.mutable_tensor();
    cast_node_t.AsProtoTensorContent(cast_node_val);

    (*node->mutable_attr())["dtype"].set_type(DT_FLOAT);
    (*node->mutable_attr())["value"].mutable_tensor()->Swap(cast_node_val);
  }
};

class LayerNormFusionDistil1 : public LayerNormFusionDistilBase {
 public:
  LayerNormFusionDistil1() : LayerNormFusionDistilBase() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern input = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern mean1 = {kMean, "mean1", NodeStatus::kRemove};
    OpTypePattern r_indices1 = {kConst, "r_indices1", NodeStatus::kRemain};

    OpTypePattern squareddiff = {kSquaredDifference, "squareddiff",
                                 NodeStatus::kRemove};
    OpTypePattern r_indices0 = {kConst, "r_indices0", NodeStatus::kRemain};
    OpTypePattern mean0 = {kMean, "mean0", NodeStatus::kRemove};

    OpTypePattern epsilon = {kConst, "epsilon", NodeStatus::kRemain};
    OpTypePattern gamma = {kConst, "gamma", NodeStatus::kRemain};
    OpTypePattern add = {kAddV2, "add", NodeStatus::kRemove};
    OpTypePattern rsqrt = {kRsqrt, "rsqrt", NodeStatus::kRemove};

    OpTypePattern mul1 = {kMul, "mul1", NodeStatus::kRemove};
    OpTypePattern mul0 = {kMul, "mul0", NodeStatus::kRemove};
    OpTypePattern sub0 = {kSub, "sub0", NodeStatus::kRemove};
    OpTypePattern beta = {kConst, "beta", NodeStatus::kRemain};
    OpTypePattern mul2 = {kMul, "mul2", NodeStatus::kRemove};

    OpTypePattern output = {kAddV2, "output", NodeStatus::kReplace};

    mean1.AddInput(input).AddInput(r_indices1);
    squareddiff.AddInput(input).AddInput(mean1);
    mean0.AddInput(squareddiff).AddInput(r_indices0);
    add.AddInput(mean0).AddInput(epsilon);
    rsqrt.AddInput(add);

    mul1.AddInput(rsqrt).AddInput(gamma);
    mul0.AddInput(mul1).AddInput(input);
    mul2.AddInput(mean1).AddInput(mul1);
    sub0.AddInput(beta).AddInput(mul2);

    output.AddInput(mul0).AddInput(sub0);

    pattern_ = InternalPattern(std::move(output));
  }

  ~LayerNormFusionDistil1() {}

  std::string Name() override { return "layernorm-distil-1"; }
};

class LayerNormFusionDistil2 : public LayerNormFusionDistilBase {
 public:
  LayerNormFusionDistil2() : LayerNormFusionDistilBase() {
    is_partial = true;
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern input = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern mean1 = {kMean, "mean1", NodeStatus::kRemove};
    OpTypePattern r_indices1 = {kConst, "r_indices1", NodeStatus::kRemain};

    OpTypePattern squareddiff = {kSquaredDifference, "squareddiff",
                                 NodeStatus::kRemove};
    OpTypePattern r_indices0 = {kConst, "r_indices0", NodeStatus::kRemain};
    OpTypePattern mean0 = {kMean, "mean0", NodeStatus::kRemove};

    OpTypePattern epsilon = {kConst, "epsilon", NodeStatus::kRemain};
    OpTypePattern gamma = {kConst, "gamma", NodeStatus::kRemain};
    OpTypePattern add = {kAddV2, "add", NodeStatus::kRemove};
    OpTypePattern rsqrt = {kRsqrt, "rsqrt", NodeStatus::kRemove};

    OpTypePattern mul1 = {kMul, "mul1", NodeStatus::kRemove};
    OpTypePattern mul0 = {kMul, "mul0", NodeStatus::kRemove};
    OpTypePattern sub0 = {kSub, "sub0", NodeStatus::kRemove};
    OpTypePattern beta = {kConst, "beta", NodeStatus::kRemain};
    OpTypePattern mul2 = {kMul, "mul2", NodeStatus::kRemove};

    OpTypePattern output = {kAddV2, "output", NodeStatus::kReplace};

    mean1.AddInput(input).AddInput(r_indices1);
    squareddiff.AddInput(mean1).AddInput(input);
    mean0.AddInput(squareddiff).AddInput(r_indices0);
    add.AddInput(mean0).AddInput(epsilon);
    rsqrt.AddInput(add);

    mul1.AddInput(rsqrt).AddInput(gamma);
    mul0.AddInput(input).AddInput(mul1);
    mul2.AddInput(mean1).AddInput(mul1);
    sub0.AddInput(beta).AddInput(mul2);

    output.AddInput(mul0).AddInput(sub0);

    pattern_ = InternalPattern(std::move(output));
  }

  ~LayerNormFusionDistil2() {}

  std::string Name() override { return "layernorm-distil-2"; }
};

REGISTER_FUSION(LayerNormFusion)
REGISTER_FUSION(LayerNormFusionTransformerLT)
REGISTER_FUSION(LayerNormFusionDistil1)
REGISTER_FUSION(LayerNormFusionDistil2)
}  // namespace graph
}  // namespace itex
