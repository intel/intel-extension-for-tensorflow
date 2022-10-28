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
#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/pattern_utils.h"
#include "itex/core/graph/utils/symbolic_shapes.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/op_kernel.h"

namespace itex {
namespace graph {

// Make up InstcanNorm fusion.
class InstanceNormFusion : public Fusion {
 public:
  InstanceNormFusion() : Fusion() {
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
    squareddiff.AddInput(input).AddInput(mean1);
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

  ~InstanceNormFusion() {}

  std::string Name() override { return "instancenorm"; }

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
    fused_node.set_op(kInstanceNorm);
    fused_node.set_device(output_node->device());
    fused_node.add_input(input_node->name());
    fused_node.add_input(gamma_node->name());
    fused_node.add_input(beta_node->name());

    auto* attr = fused_node.mutable_attr();
    auto& src_attr = output_node->attr();
    (*attr)["T"] = src_attr.at("T");
    SetAttrValue(DT_FLOAT, &(*attr)["U"]);

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
    ret = CheckIsInstanceNorm(ctx, node_index);

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

  MatchedProperties CheckIsInstanceNorm(RemapperContext* ctx,
                                        int node_index) const {
    auto& graph_view = ctx->graph_view;
    MatchedProperties ret =
        FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);
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
      ITEX_VLOG(2) << "Unexpected error to retrieve gamma or beta node";
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
    // mean computation complies with instance normalization
    NodeDef* mean_axis_node = ctx->graph_view.GetNode(r_indciex_index)->node();
    if (!mean_axis_node) {
      ITEX_VLOG(2) << "Unexpected error to retrieve reduction axis node";
      return false;
    }

    Tensor mean_axis_tensor;
    mean_axis_tensor.FromProto(mean_axis_node->attr().at("value").tensor());
    dtype = mean_axis_tensor.dtype();
    if (dtype != DT_INT32 && dtype != DT_INT64) return false;

    return (dtype == DT_INT32)
               ? IsInstanceNormReduction<int32>(mean_axis_tensor)
               : IsInstanceNormReduction<int64>(mean_axis_tensor);
  }

  // Helper function to check if the input axes data conforms with instance
  // normalization's mean computation for specified data format
  template <typename T>
  bool IsInstanceNormReduction(const Tensor& axes_data) const {
    const int axis_num = axes_data.NumElements();

    // Mean reduction axes for instance norm are expected to be:
    // NCHW - [2,3]; NHWC - [1,2]; NCDHW - [2,3,4]; NDHWC - [1,2,3]
    if (axis_num == 2 || axis_num == 3) {
      if (axes_data.flat<T>()(0) == static_cast<T>(2) &&
          axes_data.flat<T>()(1) == static_cast<T>(3)) {
        bool is_correct_reduction =
            axis_num == 3 ? axes_data.flat<T>()(2) == static_cast<T>(4) : true;
        data_format = (axis_num == 2) ? "NCHW" : "NCDHW";
        return is_correct_reduction;
      }
      if (axes_data.flat<T>()(0) == static_cast<T>(1) &&
          axes_data.flat<T>()(1) == static_cast<T>(2)) {
        bool is_correct_reduction =
            axis_num == 3 ? axes_data.flat<T>()(2) == static_cast<T>(3) : true;
        data_format = (axis_num == 2) ? "NHWC" : "NDHWC";
        return is_correct_reduction;
      }
    }

    return false;
  }

  // Helper function for InstanceNorm bfloat16/float16 fusion. Because gamma and
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

class InstanceNormLeakyRelu : public InstanceNormFusion {
 public:
  InstanceNormLeakyRelu() : InstanceNormFusion() {
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

    OpTypePattern add2 = {kAddV2, "add2", NodeStatus::kRemove};

    OpTypePattern output = {kLeakyRelu, "output", NodeStatus::kReplace};

    mean1.AddInput(input).AddInput(r_indices1);
    squareddiff.AddInput(input).AddInput(mean1);
    mean0.AddInput(squareddiff).AddInput(r_indices0);
    add.AddInput(mean0).AddInput(epsilon);
    rsqrt.AddInput(add);

    mul1.AddInput(rsqrt).AddInput(gamma);
    mul0.AddInput(input).AddInput(mul1);
    mul2.AddInput(mean1).AddInput(mul1);
    sub0.AddInput(beta).AddInput(mul2);

    add2.AddInput(mul0).AddInput(sub0);
    output.AddInput(add2);

    pattern_ = InternalPattern(std::move(output));
  }

  ~InstanceNormLeakyRelu() {}

  std::string Name() override { return "InstanceNorm+LeakyRelu"; }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    NodeDef* output_node =
        ctx->graph_view.GetNode(properties.map.at("output"))->node();
    NodeDef* add2_node =
        ctx->graph_view.GetNode(properties.map.at("add2"))->node();
    NodeDef* input_node =
        ctx->graph_view.GetNode(properties.map.at("input"))->node();
    NodeDef* gamma_node =
        ctx->graph_view.GetNode(properties.map.at("gamma"))->node();
    NodeDef* beta_node =
        ctx->graph_view.GetNode(properties.map.at("beta"))->node();
    NodeDef* epsilon_node =
        ctx->graph_view.GetNode(properties.map.at("epsilon"))->node();

    NodeDef fused_node;
    fused_node.set_op(kFusedInstanceNorm);
    fused_node.set_device(output_node->device());
    fused_node.add_input(input_node->name());
    fused_node.add_input(gamma_node->name());
    fused_node.add_input(beta_node->name());

    auto* attr = fused_node.mutable_attr();
    auto& src_attr = output_node->attr();
    (*attr)["T"] = src_attr.at("T");
    SetAttrValue(DT_FLOAT, &(*attr)["U"]);

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

    SetAttrValue(output_node->op(), &(*attr)["activation_mode"]);

    if (output_node->op() == "LeakyRelu") {
      auto& activation_attr = output_node->attr();
      (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
    }

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
    ret = CheckIsInstanceNorm(ctx, node_index);

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
      NodeDef* activation_node =
          ctx->graph_view.GetNode(ret.map.at("output"))->node();
      if (!IsLeakyRelu(*activation_node) && !IsRelu(*activation_node))
        return ret.ToEmpty();
      if (IsLeakyRelu(*activation_node) && NodeIsOnGpu(activation_node))
        return ret.ToEmpty();
      if (!HasDataType(activation_node, DT_FLOAT) &&
          !HasDataType(activation_node, DT_BFLOAT16) &&
          !(HasDataType(activation_node, DT_HALF) &&
            NodeIsOnGpu(activation_node)))
        return ret.ToEmpty();
    }
    return ret;
  }
};

class InstanceNormRelu : public InstanceNormLeakyRelu {
 public:
  InstanceNormRelu() : InstanceNormLeakyRelu() {
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

    OpTypePattern add2 = {kAddV2, "add2", NodeStatus::kRemove};

    OpTypePattern output = {kRelu, "output", NodeStatus::kReplace};

    mean1.AddInput(input).AddInput(r_indices1);
    squareddiff.AddInput(input).AddInput(mean1);
    mean0.AddInput(squareddiff).AddInput(r_indices0);
    add.AddInput(mean0).AddInput(epsilon);
    rsqrt.AddInput(add);

    mul1.AddInput(rsqrt).AddInput(gamma);
    mul0.AddInput(input).AddInput(mul1);
    mul2.AddInput(mean1).AddInput(mul1);
    sub0.AddInput(beta).AddInput(mul2);

    add2.AddInput(mul0).AddInput(sub0);
    output.AddInput(add2);

    pattern_ = InternalPattern(std::move(output));
  }

  ~InstanceNormRelu() {}

  std::string Name() override { return "InstanceNorm+Relu"; }
};

REGISTER_FUSION(InstanceNormFusion)
REGISTER_FUSION(InstanceNormLeakyRelu)
REGISTER_FUSION(InstanceNormRelu)
}  // namespace graph
}  // namespace itex
