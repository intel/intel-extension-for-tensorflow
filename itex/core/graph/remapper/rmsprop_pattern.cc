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

template <typename Type>
bool OneOfDataTypes(Type dtype, Type expected) {
  return dtype == expected;
}

template <typename Type, typename... Args>
bool OneOfDataTypes(Type dtype, Type expected, Args... args) {
  return dtype == expected || OneOfDataTypes(dtype, args...);
}

template <typename Type, typename... Args>
bool OneOfDataTypes(const NodeDef* node, Type expected, Args... args) {
  DataType dtype = GetDataTypeFromAttr(*node, "T");
  return OneOfDataTypes(dtype, expected, args...);
}

class RMSpropComputeRMSFusion : public Fusion {
 public:
  RMSpropComputeRMSFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern rms = {kAny, "rms", NodeStatus::kRemain};
    OpTypePattern read_rms = {kReadVariableOp, "rms_var", NodeStatus::kRemain};
    OpTypePattern rho = {kAny, "rho", NodeStatus::kRemain};
    OpTypePattern mul_rms_rho = {kMul, "mul_rms_rho", NodeStatus::kRemove};
    OpTypePattern grad = {kAny, "grad", NodeStatus::kRemain};
    OpTypePattern square = {kSquare, "grad_square", NodeStatus::kRemove};
    OpTypePattern one_minus_rho = {kAny, "one_minus_rho", NodeStatus::kRemain};
    OpTypePattern mul_one_minus_rho_square = {kMul, "mul_one_minus_rho_square",
                                              NodeStatus::kRemove};
    OpTypePattern new_rms = {kAddV2, "new_rms", NodeStatus::kReplace};

    read_rms.AddInput(rms);
    mul_rms_rho.AddInput(rho).AddInput(read_rms);
    square.AddInput(grad);
    mul_one_minus_rho_square.AddInput(square).AddInput(one_minus_rho);
    new_rms.AddInput(mul_rms_rho).AddInput(mul_one_minus_rho_square);
    pattern_ = InternalPattern(std::move(new_rms));
  }

  ~RMSpropComputeRMSFusion() {}

  std::string Name() override { return "rmsprop-compute-rms"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    auto& graph_view = ctx->graph_view;
    MatchedProperties ret = FillProperties(
        &graph_view, graph_view.GetNode(node_index), pattern_, false);

    if (ret.Empty()) return ret;

    if (!CheckRMSAndRHO(&graph_view, &ret)) {
      return ret.ToEmpty();
    }

    // TODO(itex) Enable it on CPU.
    auto* new_rms = ret.GetNode(&graph_view, "new_rms");
    if (NodeIsOnCpu(new_rms)) {
      return ret.ToEmpty();
    }

    if (!OneOfDataTypes(new_rms, DT_FLOAT, DT_HALF, DT_BFLOAT16)) {
      return ret.ToEmpty();
    }

    return ret;
  }

  Status Update(RemapperContext* ctx /** in and out **/,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* output =
        graph_view.GetNode(properties.map.at("new_rms"))->node();
    const NodeDef* rms_var =
        graph_view.GetNode(properties.map.at("rms_var"))->node();
    const NodeDef* rho = graph_view.GetNode(properties.map.at("rho"))->node();
    const NodeDef* square = properties.GetNode(&graph_view, "grad_square");

    NodeDef fused_op;
    fused_op.set_name(output->name());
    fused_op.set_op(kApplyRMSPropComputeRMS);
    fused_op.set_device(output->device());
    fused_op.add_input(rms_var->name());
    fused_op.add_input(rho->name());
    fused_op.add_input(square->input(0));

    auto* attr = fused_op.mutable_attr();
    auto& src_attr = output->attr();
    (*attr)["T"] = src_attr.at("T");

    if (src_attr.find("_class") != src_attr.end()) {
      (*attr)["_class"] = src_attr.at("_class");
    }

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    return status;
  }

 private:
  bool CheckRMSAndRHO(utils::MutableGraphView* graph_view,
                      MatchedProperties* properties) const {
    auto const* rho_view = graph_view->GetNode(properties->map.at("rho"));
    auto const* one_minus_rho_view =
        graph_view->GetNode(properties->map.at("one_minus_rho"));

    // Due to the rho is any in the pattern, the inpus of mul_rms_rho maybe not
    // the same with we want. We need to check and exchange them.
    if (one_minus_rho_view->node()->op() != "Sub") return false;

    bool parent_are_rho = false;
    for (int i = 0; i < one_minus_rho_view->NumRegularFanins(); i++) {
      auto const& fanin = one_minus_rho_view->GetRegularFanin(i);
      if (fanin.node_index() == rho_view->node_index()) {
        parent_are_rho = true;
      }
    }
    if (!parent_are_rho) {
      std::swap(properties->map.at("rho"), properties->map.at("rms_var"));
      auto* rms_var_view = graph_view->GetNode(properties->map.at("rms_var"));
      properties->map.at("rms") = rms_var_view->GetRegularFanin(0).node_index();
    }

    return true;
  }
};

class RMSpropVarUpdateFusion : public Fusion {
 public:
  RMSpropVarUpdateFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;
    OpTypePattern grad = {kAny, "grad", NodeStatus::kRemain};
    OpTypePattern read_new_rms = {kReadVariableOp, "read_new_rms",
                                  NodeStatus::kRemain};
    OpTypePattern sqrt = {kSqrt, "sqrt", NodeStatus::kRemove};
    OpTypePattern epsilon = {kConst, "epsilon", NodeStatus::kRemain};
    OpTypePattern add_epsilon = {kAddV2, "add_epsilon", NodeStatus::kRemove};
    OpTypePattern learning_rate = {kReadVariableOp, "lr", NodeStatus::kRemain};
    OpTypePattern mul_lr_grad = {kMul, "mul_lr_grad", NodeStatus::kRemove};
    OpTypePattern real_div = {kRealDiv, "real_div", NodeStatus::kRemove};
    OpTypePattern weight = {kAny, "weight", NodeStatus::kRemain};
    OpTypePattern weight_var = {kReadVariableOp, "weight_var",
                                NodeStatus::kRemain};
    OpTypePattern sub = {kSub, "sub", NodeStatus::kReplace};

    sqrt.AddInput(read_new_rms);
    add_epsilon.AddInput(sqrt).AddInput(epsilon);
    mul_lr_grad.AddInput(learning_rate).AddInput(grad);
    real_div.AddInput(mul_lr_grad).AddInput(add_epsilon);
    weight_var.AddInput(weight);
    sub.AddInput(weight_var).AddInput(real_div);
    pattern_ = InternalPattern(std::move(sub));
  }

  ~RMSpropVarUpdateFusion() {}

  std::string Name() override { return "rmsprop-var-update"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    auto& graph_view = ctx->graph_view;
    MatchedProperties ret = FillProperties(
        &graph_view, graph_view.GetNode(node_index), pattern_, false);

    if (ret.Empty()) {
      return ret;
    }

    // TODO(itex) Enable it on CPU.
    auto* sub = ret.GetNode(&graph_view, "sub");
    if (NodeIsOnCpu(sub)) {
      return ret.ToEmpty();
    }

    if (!OneOfDataTypes(sub, DT_FLOAT, DT_HALF, DT_BFLOAT16)) {
      return ret.ToEmpty();
    }

    return ret;
  }

  Status Update(RemapperContext* ctx /** in and out **/,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* var =
        graph_view.GetNode(properties.map.at("weight_var"))->node();
    const NodeDef* output =
        graph_view.GetNode(properties.map.at("sub"))->node();
    const NodeDef* rms =
        graph_view.GetNode(properties.map.at("read_new_rms"))->node();
    const NodeDef* lr = graph_view.GetNode(properties.map.at("lr"))->node();
    const NodeDef* epsilon =
        graph_view.GetNode(properties.map.at("epsilon"))->node();
    const NodeDef* mul_lr_grad =
        graph_view.GetNode(properties.map.at("mul_lr_grad"))->node();

    NodeDef fused_op;
    fused_op.set_name(output->name());
    fused_op.set_op(kApplyRMSPropVarUpdate);
    fused_op.set_device(output->device());
    fused_op.add_input(var->name());
    fused_op.add_input(rms->name());
    fused_op.add_input(lr->name());
    fused_op.add_input(epsilon->name());

    int grad_index = 0;
    if (mul_lr_grad->input(0) == lr->name()) {
      grad_index = 1;
    }
    fused_op.add_input(mul_lr_grad->input(grad_index));

    auto* attr = fused_op.mutable_attr();
    auto& src_attr = output->attr();
    (*attr)["T"] = src_attr.at("T");

    if (src_attr.find("_class") != src_attr.end()) {
      (*attr)["_class"] = src_attr.at("_class");
    }

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    return status;
  }
};

REGISTER_FUSION(RMSpropComputeRMSFusion);
REGISTER_FUSION(RMSpropVarUpdateFusion);
}  // namespace graph
}  // namespace itex
