/* Copyright (c) 2023 Intel Corporation

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
#include "itex/core/graph/utils/symbolic_shapes.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/op_kernel.h"

namespace itex {
namespace graph {

class MHAFusionWithReshapeMatmul : public Fusion {
 public:
  MHAFusionWithReshapeMatmul() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;

    OpTypePattern query = {kAny, "query", NodeStatus::kRemain};
    OpTypePattern key = {kAny, "key", NodeStatus::kRemain};
    OpTypePattern value = {kAny, "value", NodeStatus::kRemain};
    OpTypePattern shape_0 = {kConst, "shape_0", NodeStatus::kRemain};
    OpTypePattern reshape_0 = {kReshape, "reshape_0", NodeStatus::kRemove};
    OpTypePattern perm_0 = {kConst, "perm_0", NodeStatus::kRemain};
    OpTypePattern transpose_0 = {kTranspose, "transpose_0",
                                 NodeStatus::kRemain};
    OpTypePattern shape_1 = {kConst, "shape_1", NodeStatus::kRemain};
    OpTypePattern reshape_1 = {kReshape, "reshape_1", NodeStatus::kRemove};
    OpTypePattern batch_matmul_0 = {kBatchMatMulV2, "batch_matmul_0",
                                    NodeStatus::kRemove};
    OpTypePattern shape_2 = {kConst, "shape_2", NodeStatus::kRemain};
    OpTypePattern reshape_2 = {kReshape, "reshape_2", NodeStatus::kRemove};
    OpTypePattern scale = {kConst, "scale", NodeStatus::kRemain};
    OpTypePattern mul = {kMul, "mul", NodeStatus::kRemove};
    OpTypePattern softmax = {kSoftmax, "softmax", NodeStatus::kRemove};
    OpTypePattern shape_3 = {kConst, "shape_3", NodeStatus::kRemain};
    OpTypePattern reshape_3 = {kReshape, "reshape_3", NodeStatus::kRemove};
    OpTypePattern shape_4 = {kConst, "shape_4", NodeStatus::kRemain};
    OpTypePattern reshape_4 = {kReshape, "reshape_4", NodeStatus::kRemove};
    OpTypePattern batch_matmul_1 = {kBatchMatMulV2, "batch_matmul_1",
                                    NodeStatus::kRemove};
    OpTypePattern shape_5 = {kConst, "shape_5", NodeStatus::kRemain};
    OpTypePattern reshape_5 = {kReshape, "reshape_5", NodeStatus::kRemove};
    OpTypePattern perm_1 = {kConst, "perm_1", NodeStatus::kRemain};
    OpTypePattern transpose_1 = {kTranspose, "output", NodeStatus::kReplace};

    reshape_0.AddInput(query).AddInput(shape_0);
    transpose_0.AddInput(key).AddInput(perm_0);
    reshape_1.AddInput(transpose_0).AddInput(shape_1);
    batch_matmul_0.AddInput(reshape_0).AddInput(reshape_1);
    reshape_2.AddInput(batch_matmul_0).AddInput(shape_2);
    mul.AddInput(reshape_2).AddInput(scale);
    softmax.AddInput(mul);
    reshape_3.AddInput(softmax).AddInput(shape_3);
    reshape_4.AddInput(value).AddInput(shape_4);
    batch_matmul_1.AddInput(reshape_3).AddInput(reshape_4);
    reshape_5.AddInput(batch_matmul_1).AddInput(shape_5);
    transpose_1.AddInput(reshape_5).AddInput(perm_1);

    pattern_ = InternalPattern(std::move(transpose_1));
  }

  ~MHAFusionWithReshapeMatmul() {}

  std::string Name() override { return "mha_keras_stable_diffusion"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    auto& graph_view = ctx->graph_view;
    MatchedProperties ret =
        FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);

    bool is_ok = !ret.Empty() && CheckShapes(ctx, ret);

    if (!is_ok) return ret.ToEmpty();

    return ret;
  }

  bool CheckShapes(RemapperContext* ctx,
                   const MatchedProperties& properties) const {
    int reshape_0_index = properties.map.at("reshape_0");
    NodeDef* reshape_0 = ctx->graph_view.GetNode(reshape_0_index)->node();
    std::vector<OpInfo_TensorProperties> props;
    TF_ABORT_IF_ERROR(
        ctx->graph_properties.GetInputProperties(reshape_0->name(), &props));
    auto left_shape = props[0].shape();
    if (left_shape.unknown_rank() || left_shape.dim_size() != 4) {
      return false;
    }

    const auto check_reshape = [&](const string& shape) -> bool {
      int shape_index = properties.map.at(shape);
      NodeDef* shape_node = ctx->graph_view.GetNode(shape_index)->node();
      if (shape_node->attr().at("dtype").type() != DT_INT32) {
        return false;
      }
      Tensor shape_t;
      shape_t.FromProto(shape_node->attr().at("value").tensor());
      if (shape_t.NumElements() != 3 || shape_t.flat<int32>()(0) != -1) {
        return false;
      }
      return true;
    };

    if (!(check_reshape("shape_0") && check_reshape("shape_1") &&
          check_reshape("shape_3") && check_reshape("shape_4"))) {
      return false;
    }

    const auto check_perm = [&](const string& perm,
                                absl::InlinedVector<int32, 4> axes) {
      int perm_index = properties.map.at(perm);
      NodeDef* perm_node = ctx->graph_view.GetNode(perm_index)->node();
      if (perm_node->attr().at("dtype").type() != DT_INT32) {
        return false;
      }
      Tensor perm_t;
      perm_t.FromProto(perm_node->attr().at("value").tensor());
      if (perm_t.NumElements() != 4) {
        return false;
      }
      auto perm_flat = perm_t.flat<int32>();
      if (!(perm_flat(0) == axes[0] && perm_flat(1) == axes[1] &&
            perm_flat(2) == axes[2] && perm_flat(3) == axes[3])) {
        return false;
      }
      return true;
    };

    if (!(check_perm("perm_0", {0, 2, 3, 1}) &&
          check_perm("perm_1", {0, 2, 1, 3}))) {
      return false;
    }
    return true;
  }

  Status Update(RemapperContext* ctx /** in and out **/,
                const MatchedProperties& properties) const override {
    auto* transpose_1 =
        ctx->graph_view.GetNode(properties.map.at("output"))->node();
    auto* transpose_0_view =
        ctx->graph_view.GetNode(properties.map.at("transpose_0"));
    auto* reshape_0 =
        ctx->graph_view.GetNode(properties.map.at("reshape_0"))->node();
    auto* reshape_1 =
        ctx->graph_view.GetNode(properties.map.at("reshape_1"))->node();
    auto* perm_0 = ctx->graph_view.GetNode(properties.map.at("perm_0"))->node();
    auto* reshape_4 =
        ctx->graph_view.GetNode(properties.map.at("reshape_4"))->node();

    Status status;
    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();

    NodeDef perm_node;
    string perm_node_name = perm_0->name() + "_1";
    perm_node.set_name(perm_node_name);
    perm_node.set_op(kConst);
    perm_node.set_device(transpose_1->device());
    auto* perm_attr = perm_node.mutable_attr();
    SetAttrValue(DT_INT32, &(*perm_attr)["dtype"]);
    Tensor perm_tensor(DT_INT32, {4});
    perm_tensor.flat<int32>()(0) = 0;
    perm_tensor.flat<int32>()(1) = 2;
    perm_tensor.flat<int32>()(2) = 1;
    perm_tensor.flat<int32>()(3) = 3;
    perm_tensor.AsProtoTensorContent((*perm_attr)["value"].mutable_tensor());
    mutation->AddNode(std::move(perm_node), &status);
    TF_RETURN_IF_ERROR(status);

    TensorId new_fanin_id = ParseTensorName(perm_node_name);
    mutation->AddOrUpdateRegularFanin(transpose_0_view, 1, new_fanin_id);

    NodeDef mask_node;
    string mask_node_name = transpose_1->name() + "/dummy_mask";
    mask_node.set_name(mask_node_name);
    mask_node.set_op(kConst);
    mask_node.set_device(transpose_1->device());
    auto* mask_attr = mask_node.mutable_attr();
    SetAttrValue(GetDataTypeFromAttr(*transpose_1, "T"),
                 &(*mask_attr)["dtype"]);
    Tensor mask_t(GetDataTypeFromAttr(*transpose_1, "T"));
    mask_t.AsProtoTensorContent((*mask_attr)["value"].mutable_tensor());
    mutation->AddNode(std::move(mask_node), &status);
    TF_RETURN_IF_ERROR(status);

    NodeDef fused_node;
    fused_node.set_name(transpose_1->name());
    fused_node.set_op("ScaledDotProductAttentionInference");
    fused_node.set_device(transpose_1->device());
    fused_node.add_input(reshape_0->input(0));
    fused_node.add_input(reshape_1->input(0));
    fused_node.add_input(reshape_4->input(0));
    fused_node.add_input(mask_node_name);

    AttrValue attr_dtype, attr_inf, attr_causal, attr_mask;
    attr_dtype.set_type(GetDataTypeFromAttr(*transpose_1, "T"));
    attr_inf.set_b(true);
    attr_causal.set_b(false);
    attr_mask.set_b(false);
    fused_node.mutable_attr()->insert({"T", attr_dtype});
    fused_node.mutable_attr()->insert({"use_mask", attr_mask});
    fused_node.mutable_attr()->insert({"use_causal", attr_causal});
    fused_node.mutable_attr()->insert({"is_inference", attr_inf});

    mutation->AddNode(std::move(fused_node), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }
};

class MHAPatternWithMulAndAdd : public Fusion {
 public:
  MHAPatternWithMulAndAdd() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;

    OpTypePattern query = {kAny, "query", NodeStatus::kRemain};
    OpTypePattern key = {kAny, "key", NodeStatus::kRemain};
    OpTypePattern value = {kAny, "value", NodeStatus::kRemain};
    OpTypePattern batch_matmul = {kBatchMatMulV2, "batch_matmul",
                                  NodeStatus::kRemove};
    OpTypePattern scale = {kConst, "scale", NodeStatus::kRemain};
    OpTypePattern mask = {kAny, "mask", NodeStatus::kRemain};
    OpTypePattern mul = {kMul, "mul", NodeStatus::kRemove};
    OpTypePattern add = {kAddV2, "add", NodeStatus::kRemove};
    OpTypePattern softmax = {kSoftmax, "softmax", NodeStatus::kRemove};
    OpTypePattern batch_matmul_1 = {kBatchMatMulV2, "batch_matmul_1",
                                    NodeStatus::kRemove};
    OpTypePattern perm = {kConst, "perm", NodeStatus::kRemain};
    OpTypePattern output = {kTranspose, "output", NodeStatus::kReplace};

    batch_matmul.AddInput(query).AddInput(key);
    mul.AddInput(batch_matmul).AddInput(scale);
    add.AddInput(mask).AddInput(mul);
    softmax.AddInput(add);
    batch_matmul_1.AddInput(softmax).AddInput(value);
    output.AddInput(batch_matmul_1).AddInput(perm);

    pattern_ = InternalPattern(std::move(output));
  }

  ~MHAPatternWithMulAndAdd() {}

  std::string Name() override { return "mha_pattern_with_mul_and_add"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    auto& graph_view = ctx->graph_view;

    MatchedProperties ret = FillProperties(
        &graph_view, graph_view.GetNode(node_index), pattern_, false);

    bool is_ok = !ret.Empty() && CheckShapes(ctx, ret);

    if (!is_ok) return ret.ToEmpty();

    return ret;
  }

  bool CheckShapes(RemapperContext* ctx,
                   const MatchedProperties& properties) const {
    int perm_index = properties.map.at("perm");
    NodeDef* perm_node = ctx->graph_view.GetNode(perm_index)->node();
    Tensor perm_t;
    perm_t.FromProto(perm_node->attr().at("value").tensor());
    if (perm_t.NumElements() != 4) {
      return false;
    }
    auto perm_flat = perm_t.flat<int32>();
    if (!(perm_flat(0) == 0 && perm_flat(1) == 2 && perm_flat(2) == 1 &&
          perm_flat(3) == 3)) {
      return false;
    }
    return true;
  }

  Status Update(RemapperContext* ctx /** in and out **/,
                const MatchedProperties& properties) const override {
    auto* output = ctx->graph_view.GetNode(properties.map.at("output"))->node();
    auto* batch_matmul =
        ctx->graph_view.GetNode(properties.map.at("batch_matmul"))->node();
    auto* batch_matmul_1 =
        ctx->graph_view.GetNode(properties.map.at("batch_matmul_1"))->node();
    auto* mask = ctx->graph_view.GetNode(properties.map.at("mask"))->node();

    NodeDef fused_node;
    fused_node.set_name(output->name());
    fused_node.set_op("ScaledDotProductAttentionInference");
    fused_node.set_device(output->device());
    fused_node.add_input(batch_matmul->input(0));
    fused_node.add_input(batch_matmul->input(1));
    fused_node.add_input(batch_matmul_1->input(1));
    fused_node.add_input(mask->name());

    AttrValue attr_dtype, attr_inf, attr_causal, attr_mask;
    attr_dtype.set_type(GetDataTypeFromAttr(*batch_matmul, "T"));
    attr_inf.set_b(true);
    attr_causal.set_b(false);
    attr_mask.set_b(true);
    fused_node.mutable_attr()->insert({"T", attr_dtype});
    fused_node.mutable_attr()->insert({"use_mask", attr_mask});
    fused_node.mutable_attr()->insert({"use_causal", attr_causal});
    fused_node.mutable_attr()->insert({"is_inference", attr_inf});

    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    Status status;
    mutation->AddNode(std::move(fused_node), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }
};

REGISTER_FUSION(MHAPatternWithMulAndAdd)
REGISTER_FUSION(MHAFusionWithReshapeMatmul)

}  // namespace graph
}  // namespace itex
