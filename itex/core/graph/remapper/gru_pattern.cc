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
using utils::OpTypePattern;
using NodeStatus = utils::NodeStatus;
using InputPair = std::vector<std::pair<std::string, int>>;
const NodeStatus Remove = NodeStatus::kRemove;
const NodeStatus Remain = NodeStatus::kRemain;
const NodeStatus Replace = NodeStatus::kReplace;

/*-------------------------------------------------------------------
   Genric MatmuBiasAd Activation Pattern for matching
---------------------------------------------------------------------*/
struct MatMulBiasAct {
  OpTypePattern mm_param, Matmul, bias_data, bias, act;

 public:
  MatMulBiasAct() = default;
  MatMulBiasAct(const MatMulBiasAct&) = default;
  MatMulBiasAct(std::string name, std::string activation,
                const OpTypePattern& inp, NodeStatus last_node_status) {
    create(name, activation, inp, last_node_status);
  }
  void create(std::string name, std::string activation,
              const OpTypePattern& inp, NodeStatus last_node_status) {
    mm_param = {kAny, name + "/mm_param", Remain};
    Matmul = {kMatMul, name + "/mm", Remove, {inp, mm_param}};
    bias_data = {kAny, name + "/bias_data", Remain};
    bias = {kBiasAdd, name + "/bias", Remove, {Matmul, bias_data}};
    act = {activation, name + "/activation", last_node_status, {bias}};
  }
};

class GruFusion : public Fusion {
 protected:
  InputPair inp_labels;
  std::string FusionName;
  std::vector<std::string> NodesWithType;

 public:
  GruFusion() : Fusion() {
    OpTypePattern axis = {kAny, "concat_axis", Remain};
    OpTypePattern ident = {kAny, "identity", Remain};
    OpTypePattern tarV3 = {kAny, "tarV3", Remain};
    OpTypePattern concat = {
        kConcatV2, "concat_gru", Remove, {tarV3, ident, axis}};

    MatMulBiasAct ru_gates("ru_gates", kSigmoid, concat, Remove);

    OpTypePattern const2 = {kAny, "Const_split", Remain};
    OpTypePattern split = {kSplit, "ru_split", Remove, {const2, ru_gates.act}};
    OpTypePattern mul = {kMul, "lbrmul", Remove, {split, ident}};
    OpTypePattern concat1 = {kConcatV2, "concat1", Remove, {tarV3, mul, axis}};

    MatMulBiasAct c_gate("c_gate", kTanh, concat1, Remove);

    OpTypePattern split_d = {
        kSplit, "ru_split", Remove, {const2, ru_gates.act}};

    OpTypePattern const1 = {kAny, "Const_1", Remain, {}};
    OpTypePattern sub1 = {kSub, "ns_sub1", Remove, {const1, split_d}};
    OpTypePattern mul2 = {kMul, "ns_mul2", Remove, {split_d, ident}};
    OpTypePattern mul1 = {kMul, "ns_mul1", Remove, {sub1, c_gate.act}};
    OpTypePattern addV2 = {kAddV2, "output", Replace, {mul2, mul1}};

    inp_labels = {
        {concat.label, 0},           // x
        {concat.label, 1},           // h_prev
        {ru_gates.Matmul.label, 1},  // w_ru
        {c_gate.Matmul.label, 1},    // w_c
        {ru_gates.bias.label, 1},    // b_ru
        {c_gate.bias.label, 1},      // b_c
    };

    NodesWithType = {addV2.label,   mul1.label, mul2.label,  sub1.label,
                     concat1.label, mul.label,  split.label, concat.label};

    FusionName = "_ITEXGRUCell";
    pattern_ = InternalPattern(std::move(addV2));
  }

  ~GruFusion() {}

  std::string Name() override { return "gru"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    auto& graph_view = ctx->graph_view;

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);
    if (!ret.Empty()) {
      if (!IsValidTypes(ctx, ret, {DT_FLOAT, DT_BFLOAT16})) ret.ToEmpty();
    }
    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    std::vector<string> attrs = {"T"};

    ChangeFanoutPort(ctx, properties, "output", 3);

    const NodeDef* output =
        graph_view.GetNode(properties.map.at("output"))->node();
    NodeDef fused_op;
    fused_op.set_name(output->name());
    fused_op.set_op(FusionName);
    fused_op.set_device(output->device());
    AddFusionInputs(ctx, properties, &fused_op);
    CopyFusionAttrs(&fused_op, *output, attrs);
    auto* attr = fused_op.mutable_attr();
    SetAttrValue(false, &(*attr)["lbr"]);
    SetAttrValue(false, &(*attr)["training"]);

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }

  /*-------------------------------------------------------------------
    AddFusionInputs from mathed nodes
  ---------------------------------------------------------------------*/
  inline void AddFusionInputs(RemapperContext* ctx,
                              const MatchedProperties& properties,
                              NodeDef* fused_op) const {
    const GraphDef* graph = ctx->graph_view.graph();
    for (const auto& label_inp : inp_labels) {
      int index = properties.map.at(label_inp.first);
      const NodeDef& inp_node = graph->node(index);
      fused_op->add_input(inp_node.input(label_inp.second));
    }
  }

  /*-------------------------------------------------------------------
    Copy all attributes from orig_node to fused_node
  ---------------------------------------------------------------------*/
  inline void CopyFusionAttrs(
      NodeDef* fused_op, const NodeDef& orig_node,
      const std::vector<std::string>& attr_names) const {
    auto* attr = fused_op->mutable_attr();
    auto& src_attr = orig_node.attr();
    for (int i = 0; i < attr_names.size(); ++i) {
      (*attr)[attr_names[i]] = src_attr.at(attr_names[i]);
    }
  }

  /*--------------------------------------------------------
  ----------------------------------------------------------*/
  inline Status ChangeFanoutPort(RemapperContext* ctx,
                                 const MatchedProperties& properties,
                                 std::string out_node, int new_pid) const {
    // GRUBlock cell output is the last index
    int node_index = properties.map.at(out_node);
    const auto* output = ctx->graph_view.GetNode(node_index);
    const auto& fouts = output->GetRegularFanout(0);
    if (fouts.size() == 0) {
      return Status::OK();
    }

    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    std::string out_name = output->node()->name();
    for (int i = 0; i < fouts.size(); ++i) {
      auto fnode = fouts[i].node_view();
      int m_indx = -1;
      for (int j = 0; j < fnode->NumRegularFanins(); ++j) {
        auto fin = fnode->GetRegularFanin(j).node_view();
        if (fin == output) {
          m_indx = j;
          break;
        }
      }
      ITEX_CHECK((m_indx >= 0));
      mutation->AddOrUpdateRegularFanin(fnode, m_indx, {out_name, new_pid});
    }
    Status status;
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }

  /*-------------------------------------------------------------------
    Validate Type of all nodes with types
  ---------------------------------------------------------------------*/
  inline bool IsValidTypes(RemapperContext* ctx,
                           const MatchedProperties& properties,
                           std::vector<DataType> valid_types) const {
    for (int i = 0; i < valid_types.size(); ++i) {
      bool valid = true;
      for (int j = 0; j < NodesWithType.size(); ++j) {
        NodeDef* nd =
            ctx->graph_view.GetNode(properties.map.at(NodesWithType[j]))
                ->node();
        if (!HasDataType(nd, valid_types[i])) {
          valid = false;
          break;
        }
      }
      if (valid) return true;
    }
    return false;
  }
};

class AuGruFusion : public GruFusion {
 public:
  AuGruFusion() {
    OpTypePattern axis = {kAny, "concat_axis", Remain};
    OpTypePattern ident = {kAny, "identity", Remain};
    OpTypePattern tarV3 = {kAny, "tarV3", Remain};
    OpTypePattern concat = {
        kConcatV2, "concat_gru", Remove, {tarV3, ident, axis}};

    MatMulBiasAct ru_gates("ru_gates", kSigmoid, concat, Remove);

    OpTypePattern const2 = {kAny, "Const_split", Remain};
    OpTypePattern split = {kSplit, "ru_split", Remove, {const2, ru_gates.act}};
    OpTypePattern mul = {kMul, "lbrmul", Remove, {split, ident}};
    OpTypePattern concat1 = {kConcatV2, "concat1", Remove, {tarV3, mul, axis}};

    MatMulBiasAct c_gate("c_gate", kTanh, concat1, Remove);

    OpTypePattern split_x = {
        kSplit, "ru_split", Remove, {const2, ru_gates.act}};

    OpTypePattern attn_in = {kAny, "attn_in", Remain};
    OpTypePattern constA = {kAny, "attn_1.0", Remain};
    OpTypePattern sub_att = {kSub, "attn_sub", Remove, {constA, attn_in}};
    OpTypePattern attn = {kMul, "au_atten", Remove, {split_x, sub_att}};
    OpTypePattern split_d = attn;

    OpTypePattern const1 = {kAny, "Const_1", Remain, {}};
    OpTypePattern sub1 = {kSub, "ns_sub1", Remove, {const1, split_d}};
    OpTypePattern mul2 = {kMul, "ns_mul2", Remove, {split_d, ident}};
    OpTypePattern mul1 = {kMul, "ns_mul1", Remove, {sub1, c_gate.act}};
    OpTypePattern addV2 = {kAddV2, "output", Replace, {mul2, mul1}};

    inp_labels = {
        {concat.label, 0},           // x
        {concat.label, 1},           // h_prev
        {sub_att.label, 1},          // au_x
        {ru_gates.Matmul.label, 1},  // w_ru
        {c_gate.Matmul.label, 1},    // w_c
        {ru_gates.bias.label, 1},    // b_ru
        {c_gate.bias.label, 1},      // b_c
    };

    NodesWithType = {addV2.label,   mul1.label, mul2.label,  sub1.label,
                     concat1.label, mul.label,  split.label, concat.label};

    FusionName = "_ITEXAUGRUCell";
    pattern_ = InternalPattern(std::move(addV2));
  }
  ~AuGruFusion() {}
  std::string Name() override { return "augru"; }
};

REGISTER_FUSION(GruFusion)
REGISTER_FUSION(AuGruFusion)
}  // namespace graph
}  // namespace itex
