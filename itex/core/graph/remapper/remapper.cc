/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/graph/remapper/remapper.h"

#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/remapper/constant_names.h"
#include "itex/core/graph/remapper/fusion.h"
#include "itex/core/graph/utils/graph_common_utils.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/layout_utils.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/pattern_utils.h"
#include "itex/core/graph/utils/symbolic_shapes.h"
#include "itex/core/utils/op_kernel.h"

namespace itex {
namespace graph {

bool HasDataType(const NodeDef* node, const DataType& expected,
                 const string& type_attr) {
  DataType dtype = GetDataTypeFromAttr(*node, type_attr);
  return dtype == expected;
}

void SetFusedOpAttributes(NodeDef* fused,
                          const absl::Span<const absl::string_view> fused_ops,
                          int num_args = 1) {
  auto* attr = fused->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
}

namespace {

// Fuse l2loss + addN
struct FusedAddN {
  FusedAddN() = default;
  FusedAddN(std::vector<int> inputs_of_addN, int addN)
      : inputs_of_addN(inputs_of_addN), addN(addN) {}

  std::vector<int> inputs_of_addN;
  int addN = kMissingIndex;
};

// Bf16FusedMatmulGrad + Castfp32 pattern. will substitute with
// _ITEXFusedAccMatMulGrad.
struct Bf16ContractionGradWithCastFp32 {
  Bf16ContractionGradWithCastFp32() = default;
  Bf16ContractionGradWithCastFp32(int contraction, int bias_cast)
      : contraction(contraction), bias_cast(bias_cast) {}

  int contraction = kMissingIndex;
  int bias_cast = kMissingIndex;
  std::vector<int> bias_cast_outs;
};

// Bf16(Fused)Matmul + Castfp32 pattern. will substitute with
// _ITEX(Fused)AccMatMul.
struct Bf16ContractionWithCastFp32 {
  Bf16ContractionWithCastFp32() = default;

  int contraction = kMissingIndex;
  int cast = kMissingIndex;
};

// Comparison op followed by a cast, e.g., GreaterEqual + Cast.
struct ComparisonWithCast {
  ComparisonWithCast() = default;

  int comparison = kMissingIndex;
  int cast = kMissingIndex;
  string fused_op = "_";
};

// Random op followed by Comparison and cast.
struct RandomWithComparisonAndCast {
  RandomWithComparisonAndCast() = default;

  int comparison = kMissingIndex;
  int cast = kMissingIndex;
  int random = kMissingIndex;
  // Direction of compare
  // 0: Random comapre with X
  // 1: X compare with Random
  int direction = -1;
};

// Mul + Maximum pattern. will substitute Mul + Maximum with LeakyRelu.
struct MulWithMaximum {
  MulWithMaximum() = default;

  int input = kMissingIndex;
  int mul = kMissingIndex;
  int maximum = kMissingIndex;
  float alpha = -1;
};

// Const + Cast pattern. will substitute Const + Cast with Const.
struct ConstWithCast {
  ConstWithCast() = default;

  int constant = kMissingIndex;
  int cast = kMissingIndex;
};

// Contraction node followed by a BiasAdd.
struct ContractionWithBiasAdd {
  ContractionWithBiasAdd() = default;
  ContractionWithBiasAdd(int contraction, int bias_add, int bias_port)
      : contraction(contraction), bias_add(bias_add), bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int bias_port = kMissingIndex;
};

struct DequantizeWithShape {
  DequantizeWithShape() = default;
  DequantizeWithShape(int dequantizeIndex, int shapeIndex)
      : dequantizeIndex(dequantizeIndex), shapeIndex(shapeIndex) {}

  int dequantizeIndex = kMissingIndex;
  int shapeIndex = kMissingIndex;
};

struct DequantizeWithReshape {
  DequantizeWithReshape() = default;
  DequantizeWithReshape(int dequantizeIndex, int reshapeIndex)
      : dequantizeIndex_(dequantizeIndex), reshapeIndex_(reshapeIndex) {}

  int dequantizeIndex_ = kMissingIndex;
  int reshapeIndex_ = kMissingIndex;
};

// Contraction node followed by a BiasAddGrad.
struct ContractionWithBiasAddGrad {
  ContractionWithBiasAddGrad() = default;
  ContractionWithBiasAddGrad(int contraction, int bias_add_grad)
      : contraction(contraction), bias_add_grad(bias_add_grad) {}

  int contraction = kMissingIndex;
  int bias_add_grad = kMissingIndex;
  std::vector<int> bias_add_grad_outs;
};

// Contraction node followed by a BiasAdd and Activation.
struct ContractionWithBiasAddAndActivation {
  ContractionWithBiasAddAndActivation() = default;
  ContractionWithBiasAddAndActivation(int contraction, int bias_add,
                                      int activation, int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        activation(activation),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int activation = kMissingIndex;
  int bias_port = kMissingIndex;
};

// Contraction node followed by a BiasAdd and Add.
struct ContractionWithBiasAddAndAdd {
  ContractionWithBiasAddAndAdd() = default;
  ContractionWithBiasAddAndAdd(int contraction, int bias_add, int add,
                               int port_id, int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int bias_port = kMissingIndex;
};

// Contraction node followed by a BiasAdd, Add and Relu.
struct ContractionWithBiasAndAddActivation {
  ContractionWithBiasAndAddActivation() = default;
  ContractionWithBiasAndAddActivation(int contraction, int bias_add, int add,
                                      int port_id, int activation,
                                      int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        activation(activation),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int activation = kMissingIndex;
  int bias_port = kMissingIndex;
};

struct ContractionWithBiasAndActivationAdd {
  ContractionWithBiasAndActivationAdd() = default;
  ContractionWithBiasAndActivationAdd(int contraction, int bias_add,
                                      int activation, int add, int port_id,
                                      int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        activation(activation),
        add(add),
        port_id(port_id),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int activation = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int bias_port = kMissingIndex;
};

// BatchMatMul + Mul fusion
struct ContractionWithMul {
  ContractionWithMul() = default;
  ContractionWithMul(int contraction, int mul, int scalar)
      : contraction(contraction), mul(mul), scalar(scalar) {}

  int contraction = kMissingIndex;
  int mul = kMissingIndex;
  int scalar = kMissingIndex;
};

// FusedBatchNorm[$is_training] with fused side input and/or activation.
struct FusedBatchNormEx {
  FusedBatchNormEx() = default;

  int fused_batch_norm = kMissingIndex;
  int side_input = kMissingIndex;
  int activation = kMissingIndex;
  // Add node that will be invalidated by fusing side input and fused batch norm
  int invalidated = kMissingIndex;
};

// FusedBatchNormGrad with fused side output and/or activation.
struct FusedBatchNormGradEx {
  int fused_batch_norm_grad = kMissingIndex;
  int activation_grad = kMissingIndex;
  int side_input_grad = kMissingIndex;
  // Add node of the forward pass to access its "offset" input.
  int fwd_fused_batch_norm = kMissingIndex;
};

// Pad with `VALID` padding Conv2D/_ITEXFusedConv2D.
// Only `Pad` is supported rather than PadV2/MirrorPad.
struct PadWithContraction {
  PadWithContraction() = default;
  PadWithContraction(int pad, int contraction)
      : pad(pad), contraction(contraction) {}

  int pad = kMissingIndex;
  int contraction = kMissingIndex;
};

// The Pad gradient is Slice.
struct ConvBackpropInputWithSlice {
  ConvBackpropInputWithSlice() = default;
  ConvBackpropInputWithSlice(int slice, int contraction)
      : slice(slice), contraction(contraction) {}

  int slice = kMissingIndex;
  int contraction = kMissingIndex;
};

// TrainingOp with weight decay (Mul + AddN + TrainingOp)
struct FusedTrainingOp {
  FusedTrainingOp() = default;

  int mul = kMissingIndex;
  int mul_port = kMissingIndex;
  int mul_scalar_input = kMissingIndex;
  int addn = kMissingIndex;
  int training_op = kMissingIndex;
};

struct QuantizeV2WithQuantizedConv2D {
  QuantizeV2WithQuantizedConv2D() = default;
  QuantizeV2WithQuantizedConv2D(int quantizeV2Index, int quantizedConv2DIndex)
      : quantizeV2Index_(quantizeV2Index),
        quantizedConv2DIndex_(quantizedConv2DIndex) {}

  int quantizeV2Index_ = kMissingIndex;
  int quantizedConv2DIndex_ = kMissingIndex;
};

struct FusedBinary {
  FusedBinary() = default;
  int root_ = kMissingIndex;
  std::vector<int> fused_ops_;
  std::vector<int> input_order_;
  int num_ = kMissingIndex;
};

struct AddV2WithSoftmax {
  AddV2WithSoftmax() = default;
  AddV2WithSoftmax(int addv2Index, int softmaxIndex)
      : addv2Index_(addv2Index), softmaxIndex_(softmaxIndex) {}

  int addv2Index_ = kMissingIndex;
  int softmaxIndex_ = kMissingIndex;
};

bool IsAddWithNoBroadcast(const RemapperContext& ctx, const NodeDef& node) {
  if (!IsAdd(node)) return false;

  // Check if this is case of broadcasting - Add node supports broadcasting.
  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node.name(), &props));
  if (props.size() == 2 &&
      ShapesSymbolicallyEqual(props[0].shape(), props[1].shape())) {
    return true;
  }
  return false;
}

// Generic function to check contraction kernel.
bool IsConvOrMatMul(const NodeDef& node) {
  return IsConv3D(node) || IsConv2D(node) || IsDepthwiseConv2dNative(node) ||
         IsMatMul(node);
}

// Returns true if one input to Add is Conv2D/3D or DepthwiseConv2dNative or
// MatMul, and the other input is semantically equivalent to BiasAdd.
bool IsBiasSemanticAdd(const RemapperContext& ctx,
                       const utils::MutableNodeView& node_view,
                       int* bias_port) {
  const auto* node_def = node_view.node();
  if (!IsAdd(*node_def) || node_view.NumRegularFanins() != 2) return false;

  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node_def->name(), &props));

  if (props.size() < 2) return false;

  const auto& regular_fanin_0 = node_view.GetRegularFanin(0);
  const auto* node_view_0 = regular_fanin_0.node_view();
  const auto* node_def_0 = node_view_0->node();
  const auto& regular_fanin_1 = node_view.GetRegularFanin(1);
  const auto* node_view_1 = regular_fanin_1.node_view();
  const auto* node_def_1 = node_view_1->node();

  // Currently supported data formats are NHWC and NDHWC.
  auto is_channel_last_format = [](const NodeDef& node) -> bool {
    if (node.attr().contains("data_format")) {
      const string data_format = node.attr().at("data_format").s();
      return (data_format == "NHWC" || data_format == "NDHWC");
    }
    return true;
  };

  if (IsConvOrMatMul(*node_def_0) && is_channel_last_format(*node_def_0)) {
    *bias_port = 1;
  } else if (IsConvOrMatMul(*node_def_1) &&
             is_channel_last_format(*node_def_1)) {
    *bias_port = 0;
  } else {
    return false;
  }

  const TensorShapeProto& contraction_shape = props[1 - *bias_port].shape();
  const TensorShapeProto& bias_shape = props[*bias_port].shape();

  if (contraction_shape.unknown_rank() || bias_shape.unknown_rank() ||
      contraction_shape.dim_size() < 1 || bias_shape.dim_size() < 1 ||
      IsUnknown(contraction_shape.dim(contraction_shape.dim_size() - 1)) ||
      IsUnknown(bias_shape.dim(bias_shape.dim_size() - 1)))
    return false;

  // Helper function to check Add/AddV2 could be replaced with BiasAdd.
  const auto is_supported_shape =
      [&](const TensorShapeProto& shape,
          const TensorShapeProto& bcast_shape) -> bool {
    int conv_channel_dim;
    conv_channel_dim = shape.dim(shape.dim_size() - 1).size();

    if (shape.dim_size() == 4 && bcast_shape.dim_size() > 4) return false;
    if (shape.dim_size() == 5 && bcast_shape.dim_size() > 5) return false;

    if (shape.dim_size() < 2) return false;
    // Check that the conv node's channel dim is equal to the 1-dim add node's
    // dim
    if (conv_channel_dim != bcast_shape.dim(bcast_shape.dim_size() - 1).size())
      return false;

    // Check that add nodes dims are all 1's except the channel dim
    for (int i = 0; i < bcast_shape.dim_size() - 1; i++) {
      if (1 != bcast_shape.dim(i).size()) return false;
    }
    return true;
  };

  if (ShapesSymbolicallyEqual(contraction_shape, bias_shape) ||
      !ShapesBroadcastable(contraction_shape, bias_shape))
    return false;

  return is_supported_shape(contraction_shape, bias_shape);
}

// Returns 0: left input scalar, 1: right input scalar, -1: no scalar inputs
int GetMulScalarInputIndex(const RemapperContext& ctx,
                           const NodeDef& node_def) {
  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node_def.name(), &props));
  if (props.size() != 2) return -1;

  bool left_is_scalar = IsScalar(props[0].shape());
  bool right_is_scalar = IsScalar(props[1].shape());
  if (left_is_scalar) {
    return 0;
  } else if (right_is_scalar) {
    return 1;
  } else {
    return -1;
  }
}

int FindFusedTrainingOpInPort(const utils::MutableNodeView& mul_node_view,
                              const NodeDef& addn_node_def) {
  const auto* mul_node_def = mul_node_view.node();
  if (IsMul(*mul_node_def) && HaveSameDataType(mul_node_def, &addn_node_def)) {
    return mul_node_view.node_index();
  }
  return -1;
}

// Check if a node is a candidate to one of the patterns that require inferred
// shapes:
//   (1) TODO(itex): Splitting FusedBatchNorm into primitives.
//   (2) Fusing side input and/or activation into FusedBatchNorm.
//   (3) Conv2D -> Add or Conv2D -> BiasAdd -> Add.
//   (4) BatchMatMul + Mul
//   (5) Mul+ (AddN) + TrainingOp
[[maybe_unused]] bool RequiresInferredShapes(const RemapperContext& ctx,
                                             int node_index) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // Candidate for a FusedBatchNorm fusion.
  const auto is_batch_norm_fusion_candidate = [&]() -> bool {
    if (!IsRelu(*node_def)) return false;

    if (node_view->NumRegularFanins() < 1) return false;
    const auto& relu_fanin_0 = node_view->GetRegularFanin(0);
    const auto* relu_fanin_0_node_view = relu_fanin_0.node_view();
    const auto* relu_fanin_0_node_def = relu_fanin_0_node_view->node();

    if (IsFusedBatchNorm(*relu_fanin_0_node_def)) {
      // FusedBatchNorm + Relu.
      return true;
    } else if (IsAdd(*relu_fanin_0_node_def)) {
      // FusedBatchNorm + Add + Relu.

      if (relu_fanin_0_node_view->NumRegularFanins() < 2) return false;
      const auto& add_regular_fanin_0 =
          relu_fanin_0_node_view->GetRegularFanin(0);
      if (IsFusedBatchNorm(*add_regular_fanin_0.node_view()->node()))
        return true;
      const auto& add_regular_fanin_1 =
          relu_fanin_0_node_view->GetRegularFanin(1);
      if (IsFusedBatchNorm(*add_regular_fanin_1.node_view()->node()))
        return true;
    }

    return false;
  };

  // This function supports below patterns that require inferred
  // shapes:
  // 1. Contraction + Add.
  // 2. Contraction + Add + Activation.
  // 3. Contraction + BiasAdd/BiasSemanticAdd + Add.
  // 4. Contraction + BiasAdd/BiasSemanticAdd + Add + Activation.
  // Contraction candidate: MatMul, Conv2D, Conv3D, DepthwiseConv2dNative.
  const auto is_contraction_fusion_candidate = [&]() -> bool {
    auto is_supported_add_input = [](const auto* node_view) -> bool {
      if (IsConvOrMatMul(*node_view->node())) return true;
      // IsAdd will verify BiasSemanticAdd.
      if (IsBiasAdd(*node_view->node()) || IsAdd(*node_view->node())) {
        if (node_view->NumRegularFanins() < 2) return false;
        const auto& bias_add_fanin_0 = node_view->GetRegularFanin(0);
        const auto& bias_add_fanin_1 = node_view->GetRegularFanin(1);
        return IsConvOrMatMul(*bias_add_fanin_0.node_view()->node()) ||
               IsConvOrMatMul(*bias_add_fanin_1.node_view()->node());
      }
      return false;
    };

    auto is_supported_add = [&](const auto* node_view) -> bool {
      const auto* node_def = node_view->node();
      if (IsAdd(*node_def)) {
        if (node_view->NumRegularFanins() < 2) return false;
        const auto& add_fanin_0 = node_view->GetRegularFanin(0);
        const auto& add_fanin_1 = node_view->GetRegularFanin(1);
        return is_supported_add_input(add_fanin_0.node_view()) ||
               is_supported_add_input(add_fanin_1.node_view());
      }
      return false;
    };

    // Dealing with the Contraction + Add or Contraction + BiasAdd or
    // BiasSemanticAdd + Add patterns.
    if (is_supported_add(node_view)) {
      return true;
    }

    // Dealing with the Contraction + Add + Activation  or Contraction +
    // BiasAdd or BiasSemanticAdd + Add + Activation pattern.
    if (IsSupportedActivation(*node_view->node())) {
      for (int i = 0; i < node_view->NumRegularFanins(); i++) {
        const auto& fanin_i = node_view->GetRegularFanin(i);
        if (is_supported_add(fanin_i.node_view())) return true;
      }
    }

    return false;
  };

  const auto is_batchmatmul_mul_fusion_candidate = [&]() -> bool {
    // Candidate for BatchMatMul + Mul fusion.

    const auto* node_view = ctx.graph_view.GetNode(node_index);
    const auto* node_def = node_view->node();

    // The second node must be Mul
    if (!IsAnyMul(*node_def)) return false;

    if (node_view->NumRegularFanins() < 2) return false;
    const auto& mul_fanin_0 = node_view->GetRegularFanin(0);
    const auto& mul_fanin_1 = node_view->GetRegularFanin(1);

    // The fisrt node must be BatchMatMul
    auto is_supported_mul_input =
        [](const utils::MutableNodeView* node_view) -> bool {
      return IsAnyBatchMatMul(*node_view->node());
    };

    return is_supported_mul_input(mul_fanin_0.node_view()) ||
           is_supported_mul_input(mul_fanin_1.node_view());
  };

  const auto is_training_addn_mul_fusion_candidate = [&]() -> bool {
    // Candidate for Mul+ (AddN) + TrainingOp fusion.

    const auto* node_view = ctx.graph_view.GetNode(node_index);
    const auto* node_def = node_view->node();

    // Check training op
    int input_index = -1;
    if (IsApplyMomentum(*node_def) || IsResourceApplyMomentum(*node_def)) {
      // Input: var, accum, lr, grad, momentum
      if (node_view->NumRegularFanins() != 5) return false;
      input_index = 3;
    } else if (IsApplyAdam(*node_def) || IsResourceApplyAdam(*node_def)) {
      // Input : var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
      // grad
      if (node_view->NumRegularFanins() != 10) return false;
      input_index = 9;
    } else if (IsApplyAdamWithWeightDecay(*node_def) ||
               IsResourceApplyAdamWithWeightDecay(*node_def)) {
      // Input : var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
      // weight_decay, grad
      if (node_view->NumRegularFanins() != 11) return false;
      input_index = 10;
    } else {
      return false;
    }

    const auto* input_node_view =
        node_view->GetRegularFanin(input_index).node_view();
    const auto* input_node_def = input_node_view->node();

    if (IsAddN(*input_node_def)) {
      // Mul + AddN + Adam_op is not supported
      if (IsApplyAdam(*node_def) || IsResourceApplyAdam(*node_def) ||
          IsApplyAdamWithWeightDecay(*node_def) ||
          IsResourceApplyAdamWithWeightDecay(*node_def))
        return false;

      auto* mul_node_view = input_node_view->GetRegularFanin(0).node_view();
      int mul_node_index =
          FindFusedTrainingOpInPort(*mul_node_view, *input_node_def);
      if (mul_node_index == -1) {
        mul_node_view = input_node_view->GetRegularFanin(1).node_view();
        mul_node_index =
            FindFusedTrainingOpInPort(*mul_node_view, *input_node_def);
      }
      if (mul_node_index != -1) {
        return true;
      } else {
        return false;
      }
    } else if (IsMul(*input_node_def)) {
      return true;
    } else {
      return false;
    }
  };

  return is_batch_norm_fusion_candidate() ||
         is_contraction_fusion_candidate() ||
         is_batchmatmul_mul_fusion_candidate() ||
         is_training_addn_mul_fusion_candidate();
}

// Check whether current MatMulGrad has shared same dz with BiasAddGrad.
const auto IsLegalMatMulGrad = [](const RemapperContext& ctx, int node_index,
                                  int node_dz) -> bool {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_view == nullptr) return false;

  const auto* grad_input = node_view->GetRegularFanin(1).node_view();
  if (grad_input == nullptr) return false;

  // Input grad tensor should have index 1
  if (grad_input->node_index() != node_dz) return false;

  bool transpose_b = true;

  if (!GetNodeAttr(*node_def, "transpose_b", &transpose_b).ok()) return false;

  // Transposed input grad tensor is unsafe for BiasAddGrad fusion
  if (transpose_b) return false;

  return true;
};

bool FindFusedAddN(const RemapperContext& ctx, int node_index,
                   FusedAddN* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a AddN.
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* addN = node_view->node();
  if (!addN || !IsAddN(*addN)) return false;

  // TODO(itex): only support AddN+L2Loss fusion on GPU for now, will remove
  // this limitation once supported
  if (!NodeIsOnGpu(addN)) return false;

  int num = addN->attr().at("N").i();
  std::vector<int> inputs;
  for (int i = 0; i < num; ++i) {
    const auto* l2loss = node_view->GetRegularFanin(i).node_view();
    bool is_l2loss = l2loss->node() && (IsL2Loss(*(l2loss->node())));
    if (!is_l2loss || !HaveSameDataType(addN, l2loss->node(), "T") ||
        HasControlFaninOrFanout(*l2loss) ||
        !HasAtMostOneFanoutAtPort0(*l2loss) ||
        IsInPreserveSet(ctx, l2loss->node()))
      return false;
    inputs.push_back(l2loss->node_index());
  }
  const FusedAddN pattern{inputs, node_index};
  *matched = pattern;
  return true;
}

bool FindContractionWithBias(const RemapperContext& ctx, int node_index,
                             ContractionWithBiasAdd* matched,
                             bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  // verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  const auto* node_def = node_view->node();
  int bias_port = 1;
  if (!IsBiasAdd(*node_def) && !IsBiasSemanticAdd(ctx, *node_view, &bias_port))
    return false;

  // Input to the BiasAdd must be a Conv2D or a MatMul.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(1 - bias_port);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // verify the input node has a control fanout edge or not.
  if (HasControlFanout(*contraction_node_view)) return false;

  if (IsAccMatMul(*contraction_node_def) &&
      GetDataTypeFromAttr(*node_def, "T") == DT_FLOAT &&
      HasAtMostOneFanoutAtPort0(*contraction_node_view) &&
      !IsInPreserveSet(ctx, contraction_node_def)) {
    const ContractionWithBiasAdd pattern{contraction_node_view->node_index(),
                                         node_index, bias_port};
    if (check_device_compatible && !IsDeviceCompatible(ctx, pattern))
      return false;

    // We successfully found a {BF16MatMul+CastFp32}+Fp32BiasAdd pattern.
    *matched = pattern;

    return true;
  }
  // Conv, MatMul or DepthwiseConv2D.
  bool is_contraction = IsConvOrMatMul(*contraction_node_def);
  // TODO(itex): oneDNN does not support double dtype currently
  if (is_contraction && HasDataType(contraction_node_def, DT_DOUBLE))
    return false;

  if (!is_contraction || !HaveSameDataType(node_def, contraction_node_def) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAdd pattern{contraction_node_view->node_index(),
                                       node_index, bias_port};
  if (check_device_compatible && !IsDeviceCompatible(ctx, pattern))
    return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd pattern.
  *matched = pattern;

  return true;
}

bool FindContractionWithBiasAddGrad(const RemapperContext& ctx, int node_index,
                                    ContractionWithBiasAddGrad* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (node_view == nullptr) return false;
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Need use BiasAddGrad to find the MatMulGradFilter
  const auto* node_def = node_view->node();
  if (!IsBiasAddGrad(*node_def)) return false;

  if (!(HasDataType(node_def, DT_FLOAT) || HasDataType(node_def, DT_BFLOAT16)))
    return false;

  // Don't do FP32/BF16 fusion on CPU since it has lower perf.
  // TODO(itex): Remove this limitation once oneDNN fixes it.
  if (NodeIsOnCpu(node_def)) return false;

  // BiasAddGrad, MatMulGradFilter and MatMulGradInput use the same input.
  //
  // OP                  | Input
  // ---------------------------------------------------------------------
  // BiasAddGrad         | dz
  // MatMul(grad filter) | x and dz
  // MatMul(grad input)  | y and dz
  //
  // MatMul(forward)     | 0: x, 1; y
  //
  // Need fuse the BiasAddGrad and MatMul. OneDNN inner-product backward
  // primitive can compute gradients of weights and bias together based on
  // dz and x/y, where BiasAddGrad shares dz with MatMul. Since current
  // OneDNN inner-product backward primitive defaults the input:1 as dz,
  // BiasAddGrad will be fused with the MatMul has dz at input:1, otherwise
  // the FusedMatMulGrad kernel will need Transpose to maintain correctness.
  // Furthermore, for x:(m, k) and y:(k, n), dz shape for BiasAddGrad should
  // be (m, n). So the transpose_b of MatMul to be fused must be false.

  const auto* dz = node_view->GetRegularFanin(0).node_view();
  if (dz == nullptr) return false;
  // The node index for MatMulGradFilter if found.
  int matmul_grad_filter_idx = -1;

  // Limit this patter that dz only has 3 output, BiasAddGrad, MatMulGradFilter
  // and MatMulGradInput.
  if (dz->NumRegularFanouts() != 3) return false;

  std::vector<int> matmuls;
  for (const auto& dz_fanout_i : dz->GetRegularFanouts()) {
    for (const auto dz_fanout : dz_fanout_i) {
      if (IsMatMul(*(dz_fanout.node_view()->node()))) {
        matmuls.push_back(dz_fanout.node_view()->node_index());
      }
    }
  }

  if (matmuls.size() != 2) return false;
  if (IsLegalMatMulGrad(ctx, matmuls.at(0), dz->node_index())) {
    matmul_grad_filter_idx = matmuls.at(0);
  } else if (IsLegalMatMulGrad(ctx, matmuls.at(1), dz->node_index())) {
    matmul_grad_filter_idx = matmuls.at(1);
  }

  if (matmul_grad_filter_idx < 0) return false;

  // We successfully found a BiasAddGrad and MatMulGradFilter pattern.
  matched->contraction = matmul_grad_filter_idx;
  matched->bias_add_grad = node_view->node_index();

  for (auto const& bias_out : node_view->GetRegularFanouts()) {
    for (auto const bias_out_i : bias_out) {
      matched->bias_add_grad_outs.push_back(
          bias_out_i.node_view()->node_index());
    }
  }
  return true;
}

bool FindConvContractionWithBiasAddGrad(const RemapperContext& ctx,
                                        int node_index,
                                        ContractionWithBiasAddGrad* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (node_view == nullptr) return false;
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Need use BiasAddGrad to find the ContractionBackpropFilter.
  const auto* node_def = node_view->node();
  if (!IsBiasAddGrad(*node_def)) return false;

  if (!(HasDataType(node_def, DT_FLOAT) || HasDataType(node_def, DT_BFLOAT16)))
    return false;

  const auto* dz = node_view->GetRegularFanin(0).node_view();
  if (dz == nullptr) return false;
  int conv_grad_filter_idx = -1;

  int64_t out_port = -1;
  for (size_t i = 0; i < dz->GetRegularFanouts().size(); ++i) {
    const auto& dz_fanout_i = dz->GetRegularFanout(i);
    for (const auto dz_fanout : dz_fanout_i) {
      if (dz_fanout.node_view()->node() == node_def) {
        out_port = i;
        break;
      }
    }
  }
  if (out_port == -1) return false;

  // 1. Conv2DBackpropFilter and BiasAddGrad should share the same out port of
  // dz since dz may have multiple output.
  // 2. Make sure ConvBackpropFilter's 3rd input is fused with BiasAddGrad. A
  // typical negative example is deconv, which is implemented by
  // ConvBackpropFilter.
  for (const auto dz_fanout : dz->GetRegularFanout(out_port)) {
    if (IsConv2DBackpropFilter(*(dz_fanout.node_view()->node())) ||
        IsConv3DBackpropFilterV2(*(dz_fanout.node_view()->node()))) {
      if (dz->node() !=
          dz_fanout.node_view()->GetRegularFanin(2).node_view()->node())
        continue;
      conv_grad_filter_idx = dz_fanout.node_view()->node_index();
      break;
    }
  }

  if (conv_grad_filter_idx == -1) return false;
  // We successfully found a BiasAddGrad and ContractionBackpropFilter pattern.
  matched->contraction = conv_grad_filter_idx;
  matched->bias_add_grad = node_view->node_index();

  for (auto const& bias_out : node_view->GetRegularFanouts()) {
    for (auto const bias_out_i : bias_out) {
      matched->bias_add_grad_outs.push_back(
          bias_out_i.node_view()->node_index());
    }
  }
  return true;
}

// As AddN has multiple inputs, this function tries to find Conv2D + Bias
// pattern in specific input port.
bool FindContractionWithBiasInPort(const RemapperContext& ctx,
                                   const utils::MutableNodeView& add_node_view,
                                   const NodeDef& add_node_def, int port_id,
                                   ContractionWithBiasAdd* base) {
  // Input to AddN must match ContractionWithBiasAdd pattern.
  if (add_node_view.NumRegularFanins() < port_id + 1) return false;
  const auto& bias_add_node_view =
      add_node_view.GetRegularFanin(port_id).node_view();
  if (bias_add_node_view == nullptr) return false;
  const auto* bias_add_node_def = bias_add_node_view->node();

  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), base,
                               /*check_device_compatible=*/false))
    return false;
  if (!HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      !HaveSameDataType(&add_node_def, bias_add_node_def) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;
  return true;
}

bool FindContractionWithBiasAddAndAdd(const RemapperContext& ctx,
                                      const int node_index,
                                      ContractionWithBiasAddAndAdd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Fusion with AddN is supported only when it has two inputs.
  if (HasControlFaninOrFanout(*node_view) || node_view->NumRegularFanins() != 2)
    return false;

  // Root of the pattern must be a AddN or Add with same input shapes
  // (no broadcasting).
  const auto* node_def = node_view->node();

  if (!IsAddN(*node_def) && !IsAddWithNoBroadcast(ctx, *node_def)) return false;

  // OneDnn AddN ops only support float, float16 and bfloat16 data types on GPU.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16) &&
      !(HasDataType(node_def, DT_HALF) && NodeIsOnGpu(node_def)))
    return false;

  ContractionWithBiasAdd base;
  matched->port_id = 0;

  // Find the conv+bias pattern in specific port.
  if (!FindContractionWithBiasInPort(ctx, *node_view, *node_def,
                                     matched->port_id, &base)) {
    matched->port_id = 1;
    if (!FindContractionWithBiasInPort(ctx, *node_view, *node_def,
                                       matched->port_id, &base)) {
      return false;
    }
  }

  // We successfully found a Conv2D+BiasAdd+{AddN,Add} pattern.
  matched->contraction = base.contraction;
  matched->bias_add = base.bias_add;
  matched->bias_port = base.bias_port;
  matched->add = node_view->node_index();

  return true;
}

bool FindContractionWithBiasAndActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAddAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (!IsSupportedActivation(*node_def)) return false;

  // verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  // And input to the activation node must match ContractionWithBiasAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* bias_add_node_view = regular_fanin_0.node_view();
  const auto* bias_add_node_def = bias_add_node_view->node();

  ContractionWithBiasAdd base;
  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), &base,
                               /*check_device_compatible=*/false) ||
      !HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      (!HaveSameDataType(node_def, bias_add_node_def) &&
       !(GetDataTypeFromAttr(*node_def, "T") == DT_FLOAT &&
         IsFusedAccMatMul(*bias_add_node_def))) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;

  // TODO(itex): Public TF doesn't have MatMul + LeakyRelu fusion, remove this
  //       limitation once it's supported.
  const auto* contraction_node_view = ctx.graph_view.GetNode(base.contraction);
  const auto* contraction_def = contraction_node_view->node();

  // verify the inter node has control fanin&fanout or not.
  if (HasControlFaninOrFanout(*bias_add_node_view)) {
    return false;
  }

  // TODO(itex): oneDNN does not support double dtype currently
  if (HasDataType(contraction_def, DT_DOUBLE)) return false;
  if (IsLeakyRelu(*node_def) &&
      (IsMatMul(*contraction_def) || IsAccMatMul(*contraction_def)))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAddAndActivation pattern{
      base.contraction, base.bias_add, node_index, base.bias_port};
  if (!IsDeviceCompatible(ctx, pattern)) return false;

  // verify the input node has a control fanout edge or not.
  if (HasControlFanout(*contraction_node_view)) return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd+Activation pattern.
  *matched = pattern;

  return true;
}

bool FindContractionWithBiasAndAddActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAndAddActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  if (!IsSupportedActivation(*node_def)) return false;

  // OneDnn activation op only supports float, float16 and bfloat16 data types
  // on GPU.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16) &&
      !(HasDataType(node_def, DT_HALF) && NodeIsOnGpu(node_def)))
    return false;

  // And input to activation must match ContractionWithBiasAddAndAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* add_node_view = regular_fanin_0.node_view();
  const auto* add_node_def = add_node_view->node();

  ContractionWithBiasAddAndAdd base;
  if (!FindContractionWithBiasAddAndAdd(ctx, add_node_view->node_index(),
                                        &base) ||
      !HasAtMostOneFanoutAtPort0(*add_node_view) ||
      !HaveSameDataType(node_def, add_node_def) ||
      IsInPreserveSet(ctx, add_node_def)) {
    return false;
  }

  // TODO(itex): Public TF doesn't have MatMul + LeakyRelu fusion, remove this
  //       limitation once it's supported.
  const auto* contraction_def =
      ctx.graph_view.GetNode(base.contraction)->node();
  if (IsLeakyRelu(*node_def) &&
      (IsMatMul(*contraction_def) || IsAccMatMul(*contraction_def)))
    return false;

  // We successfully found a Conv2D+BiasAdd+AddN+activation pattern.
  const ContractionWithBiasAndAddActivation pattern{
      base.contraction, base.bias_add, base.add,
      base.port_id,     node_index,    base.bias_port};
  *matched = pattern;

  return true;
}

bool FindContractionWithBiasAndActivationInPort(
    const RemapperContext& ctx, const utils::MutableNodeView& add_node_view,
    const NodeDef& add_node_def, int port_id) {
  if (add_node_view.NumRegularFanins() < port_id + 1) return false;

  const auto& act_node_view =
      add_node_view.GetRegularFanin(port_id).node_view();
  if (act_node_view == nullptr) return false;
  const auto* act_node_def = act_node_view->node();

  if (!IsSupportedActivation(*act_node_def)) {
    return false;
  }
  if (!HasAtMostOneFanoutAtPort0(*act_node_view)) {
    return false;
  }
  return true;
}

bool FindContractionWithBiasAndActivationAdd(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAndActivationAdd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (!IsAdd(*node_def)) return false;
  // OneDnn activation op only supports float, float16 and bfloat16 data types
  // on GPU.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16) &&
      !(HasDataType(node_def, DT_HALF) && NodeIsOnGpu(node_def)))
    return false;

  ContractionWithBiasAddAndActivation base;
  if (!FindContractionWithBiasAndActivationInPort(ctx, *node_view, *node_def,
                                                  matched->port_id)) {
    matched->port_id = 1;
    if (!FindContractionWithBiasAndActivationInPort(ctx, *node_view, *node_def,
                                                    matched->port_id)) {
      return false;
    }
  }
  const auto& act_node_view =
      node_view->GetRegularFanin(matched->port_id).node_view();
  if (!FindContractionWithBiasAndActivation(ctx, act_node_view->node_index(),
                                            &base)) {
    return false;
  }

  const auto* contraction_def =
      ctx.graph_view.GetNode(base.contraction)->node();
  if (IsLeakyRelu(*node_def) &&
      (IsMatMul(*contraction_def) || IsAccMatMul(*contraction_def)))
    return false;

  // We successfully found a Conv2D+BiasAdd+AddN+activation pattern.
  const ContractionWithBiasAndActivationAdd pattern{
      base.contraction, base.bias_add,    base.activation,
      node_index,       matched->port_id, base.bias_port};
  *matched = pattern;

  return true;
}

bool FindFusedBatchNormEx(const RemapperContext& ctx, int node_index,
                          FusedBatchNormEx* matched) {
  // Root of the pattern must be a Relu.
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (!IsRelu(*node_def)) return false;

  // Returns true iff the node is a compatible FusedBatchNorm node.
  const auto valid_batch_norm =
      [&](const utils::MutableNodeView& fused_batch_norm) -> bool {
    const auto* fused_batch_norm_node_def = fused_batch_norm.node();
    if (!IsFusedBatchNorm(*fused_batch_norm_node_def)) return false;

    DataType t_dtype = GetDataTypeFromAttr(*fused_batch_norm_node_def, "T");

    // GPU supports float and bfloat16.
    if (t_dtype != DT_FLOAT && t_dtype != DT_BFLOAT16) return false;

    string data_format;
    if (!GetNodeAttr(*fused_batch_norm_node_def, kDataFormat, &data_format)
             .ok())
      return false;
    if (data_format != "NHWC" && data_format != "NCHW") return false;

    // FusedBatchNormV2 and V3 have an extra type parameter.
    if ((fused_batch_norm_node_def->op() != "FusedBatchNorm") &&
        !HasDataType(fused_batch_norm_node_def, DT_FLOAT, "U"))
      return false;

    // Check that only one node consumes the 0-th output of a FusedBatchNorm.
    if (HasControlFaninOrFanout(fused_batch_norm) ||
        !HasAtMostOneFanoutAtPort0(fused_batch_norm) ||
        IsInPreserveSet(ctx, fused_batch_norm_node_def))
      return false;

    return true;
  };

  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* relu_fanin_0_node_view = regular_fanin_0.node_view();
  const auto* relu_fanin_0_node_def = relu_fanin_0_node_view->node();

  // Input to a Relu can be a FusedBatchNorm.
  if (valid_batch_norm(*relu_fanin_0_node_view)) {
    matched->activation = node_index;
    matched->fused_batch_norm = regular_fanin_0.node_index();
    return true;
  }

  // Input to a Relu can be an Add node with FusedBatchNorm as one of the inputs
  if (IsAdd(*relu_fanin_0_node_def)) {
    // Currently no CPU implementation for "FusedBatchNorm + SideInput +
    // <Activation>""
    if (!NodeIsOnGpu(node_def)) return false;

    // Check that only Relu node consumes the output of an Add node.
    if (HasControlFaninOrFanout(*relu_fanin_0_node_view) ||
        !HasAtMostOneFanoutAtPort0(*relu_fanin_0_node_view) ||
        IsInPreserveSet(ctx, relu_fanin_0_node_def))
      return false;

    // Add node supports broadcasting, FusedBatchNormEx does not.
    std::vector<OpInfo_TensorProperties> props;
    TF_ABORT_IF_ERROR(ctx.graph_properties.GetInputProperties(
        relu_fanin_0_node_def->name(), &props));
    if (props.size() < 2 ||
        !ShapesSymbolicallyEqual(props[0].shape(), props[1].shape()))
      return false;

    if (relu_fanin_0_node_view->NumRegularFanins() < 2) return false;
    const auto& add_regular_fanin_0 =
        relu_fanin_0_node_view->GetRegularFanin(0);
    const auto& add_regular_fanin_1 =
        relu_fanin_0_node_view->GetRegularFanin(1);

    if (valid_batch_norm(*add_regular_fanin_0.node_view())) {
      matched->activation = node_index;
      matched->side_input = add_regular_fanin_1.node_index();
      matched->fused_batch_norm = add_regular_fanin_0.node_index();
      matched->invalidated = regular_fanin_0.node_index();
      return true;
    }

    if (valid_batch_norm(*add_regular_fanin_1.node_view())) {
      matched->activation = node_index;
      matched->side_input = add_regular_fanin_0.node_index();
      matched->fused_batch_norm = add_regular_fanin_1.node_index();
      matched->invalidated = regular_fanin_0.node_index();
      return true;
    }
  }

  return false;
}

bool FindDequantizeWithShape(const RemapperContext& ctx, int node_index,
                             DequantizeWithShape* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  if (!IsShape(*node_def)) return false;
  auto* dequantize_node_view = node_view->GetRegularFanin(0).node_view();
  auto* dequantize_node_def = dequantize_node_view->node();

  if (!IsDequantize(*dequantize_node_def)) return false;

  if (HasControlFaninOrFanout(*dequantize_node_view) ||
      !HasAtMostOneFanoutAtPort0(*dequantize_node_view) ||
      IsInPreserveSet(ctx, dequantize_node_def)) {
    return false;
  }

  const DequantizeWithShape pattern{dequantize_node_view->node_index(),
                                    node_view->node_index()};
  *matched = pattern;

  return true;
}

bool FindDequantizeWithReshape(const RemapperContext& ctx, int node_index,
                               DequantizeWithReshape* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // TODO(itex): only support DequantizeWithReshape fusion on GPU for now, will
  // remove this limitation once supported
  if (!NodeIsOnGpu(node_def)) return false;

  if (!IsReshape(*node_def)) return false;
  auto* dequantize_node_view = node_view->GetRegularFanin(0).node_view();
  auto* dequantize_node_def = dequantize_node_view->node();

  if (!IsDequantize(*dequantize_node_def)) return false;

  if (HasControlFaninOrFanout(*dequantize_node_view) ||
      !HasAtMostOneFanoutAtPort0(*dequantize_node_view) ||
      IsInPreserveSet(ctx, dequantize_node_def)) {
    return false;
  }

  const DequantizeWithReshape pattern{dequantize_node_view->node_index(),
                                      node_view->node_index()};
  *matched = pattern;

  return true;
}

bool FindQuantizeV2WithQuantizedConv2D(const RemapperContext& ctx,
                                       int node_index,
                                       QuantizeV2WithQuantizedConv2D* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // TODO(itex): will support more QuantizedConv2D-based ops
  if (!IsQuantizedConv2DWithPostOps(*node_def)) return false;
  auto* quantizev2_node_view = node_view->GetRegularFanin(0).node_view();
  auto* quantizev2_node_def = quantizev2_node_view->node();

  if (!IsQuantizeV2(*quantizev2_node_def)) return false;

  if (HasControlFaninOrFanout(*quantizev2_node_view) ||
      !HasAtMostOneFanoutAtPort0(*quantizev2_node_view) ||
      IsInPreserveSet(ctx, quantizev2_node_def)) {
    return false;
  }

  const QuantizeV2WithQuantizedConv2D pattern{
      quantizev2_node_view->node_index(), node_view->node_index()};
  *matched = pattern;

  return true;
}

bool FindAddV2WithSoftmax(const RemapperContext& ctx, int node_index,
                          AddV2WithSoftmax* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (HasControlFaninOrFanout(*node_view)) return false;
  if (!NodeIsOnGpu(node_def)) return false;

  if (!IsSoftmax(*node_def)) return false;
  auto* addv2_node_view = node_view->GetRegularFanin(0).node_view();
  auto* addv2_node_def = addv2_node_view->node();

  if (!IsAdd(*addv2_node_def)) return false;

  // check the shape of input nodes of AddV2
  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(addv2_node_def->name(), &props));

  const TensorShapeProto& left_shape = props[0].shape();
  const TensorShapeProto& right_shape = props[1].shape();

  bool is_non_supported_shape =
      (left_shape.dim_size() != 4) ||
      left_shape.dim_size() != right_shape.dim_size() ||
      (left_shape.dim(0).size() != right_shape.dim(0).size()) ||
      (left_shape.dim(2).size() != right_shape.dim(2).size()) ||
      (left_shape.dim(3).size() != right_shape.dim(3).size());
  if (is_non_supported_shape) {
    return false;
  }

  if (HasControlFaninOrFanout(*addv2_node_view) ||
      !HasAtMostOneFanoutAtPort0(*addv2_node_view) ||
      IsInPreserveSet(ctx, addv2_node_def)) {
    return false;
  }

  const AddV2WithSoftmax pattern{addv2_node_view->node_index(),
                                 node_view->node_index()};
  *matched = pattern;

  return true;
}

bool FindFusedBatchNormGradEx(const RemapperContext& ctx, int node_index,
                              FusedBatchNormGradEx* matched) {
  // Root of the pattern must be a FusedBatchNormGrad.
  const utils::MutableNodeView* node_view = ctx.graph_view.GetNode(node_index);

  // Returns true iff the node is a compatible FusedBatchNormGrad node.
  const auto valid_batch_norm_grad =
      [&](const utils::MutableNodeView& fused_batch_norm_grad) -> bool {
    const NodeDef* node_def = fused_batch_norm_grad.node();
    if (!IsFusedBatchNormGrad(*node_def) ||
        HasControlFaninOrFanout(fused_batch_norm_grad))
      return false;

    // We fuse FusedBatchNormGrad only for the training mode.
    bool is_training;
    if (!GetNodeAttr(*node_def, kIsTraining, &is_training).ok() || !is_training)
      return false;

    // FusedBatchNormV2 and V3 have an extra type parameter.
    if (node_def->op() != "FusedBatchNorm" &&
        !HasDataType(node_def, DT_FLOAT, "U"))
      return false;

    return true;
  };

  if (!valid_batch_norm_grad(*node_view)) return false;

  if (node_view->NumRegularFanins() < 1) return false;

  const utils::MutableFanoutView& regular_fanin_0 =
      node_view->GetRegularFanin(0);
  const utils::MutableNodeView* relugrad_node_view =
      regular_fanin_0.node_view();
  const NodeDef* relugrad_node_def = relugrad_node_view->node();
  bool is_relugrad = IsReluGrad(*relugrad_node_def);

  if (!is_relugrad || HasControlFaninOrFanout(*relugrad_node_view))
    return false;

  if (relugrad_node_view->NumRegularFanins() < 1) return false;
  // Find its corresponding forward node. We need the node to determine if the
  // type is bn+add+act or bn+act. Also, we need to access its "offset" input.
  const utils::MutableFanoutView& fanin_1 =
      relugrad_node_view->GetRegularFanin(1);
  const utils::MutableNodeView* fwd_node_view = fanin_1.node_view();
  FusedBatchNormEx fwd_matched;
  FindFusedBatchNormEx(ctx, fwd_node_view->node_index(), &fwd_matched);
  bool fwd_bn_act_used = fwd_matched.activation != kMissingIndex &&
                         fwd_matched.side_input == kMissingIndex;
  bool fwd_bn_add_act_used = fwd_matched.activation != kMissingIndex &&
                             fwd_matched.side_input != kMissingIndex;

  // Check that only 1 node consumes the output of the ReluGrad node.
  if (fwd_bn_act_used && relugrad_node_view->GetRegularFanout(0).size() == 1) {
    matched->activation_grad = regular_fanin_0.node_index();
    matched->fused_batch_norm_grad = node_index;
    matched->fwd_fused_batch_norm = fwd_matched.fused_batch_norm;
    return true;
  }

  // Check that only 2 nodes consume the output of the ReluGrad node.
  if (fwd_bn_add_act_used &&
      relugrad_node_view->GetRegularFanout(0).size() == 2) {
    // In a graph with the Add node having two BatchNorm nodes as the inputs, we
    // need to make sure only the one backward BatchNorm that correponds to the
    // to-be-fused forward BatchNorm should be fused. We use the edge for the
    // reserve space to get the directly corresponded forward BatchNorm node.
    const utils::MutableFanoutView& fwd_batch_norm_node =
        node_view->GetRegularFanin(5);
    if (fwd_matched.fused_batch_norm != fwd_batch_norm_node.node_index()) {
      return false;
    }

    const std::vector<utils::MutableFaninView>& fanouts_at_port_0 =
        relugrad_node_view->GetRegularFanouts()[0];
    const utils::MutableNodeView* fanout_0_node_view =
        ctx.graph_view.GetNode(fanouts_at_port_0[0].node_view()->GetName());
    const utils::MutableNodeView* fanout_1_node_view =
        ctx.graph_view.GetNode(fanouts_at_port_0[1].node_view()->GetName());
    const NodeDef* fanout_0_node_def = fanout_0_node_view->node();
    const NodeDef* fanout_1_node_def = fanout_1_node_view->node();
    const NodeDef* node_def = node_view->node();

    // We fuse FusedBatchNormGrad with side input on GPU.
    if (!NodeIsOnGpu(node_def)) return false;

    matched->activation_grad = regular_fanin_0.node_index();
    matched->fused_batch_norm_grad = node_index;
    matched->fwd_fused_batch_norm = fwd_matched.fused_batch_norm;

    if (fanout_0_node_def == node_def) {
      matched->side_input_grad = fanout_1_node_view->node_index();
      return true;
    }

    if (fanout_1_node_def == node_def) {
      matched->side_input_grad = fanout_0_node_view->node_index();
      return true;
    }
  }

  return false;
}

bool FindPadWithContraction(const RemapperContext& ctx, int node_index,
                            PadWithContraction* matched,
                            bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a Conv or FusedConv.
  // TODO(itex): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root node must be Conv2D/_FusedITEXConv2D.
  const auto* node_def = node_view->node();
  const bool is_ok = IsConv2D(*node_def) || node_def->op() == kFusedConv2D ||
                     IsConv3D(*node_def) || node_def->op() == kFusedConv3D;
  if (!is_ok) {
    return false;
  }

  // Input to the contraction must be Pad.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* pad_node_view = regular_fanin_0.node_view();
  const auto* pad_node_def = pad_node_view->node();

  // Only Pad is allowed, PadV2 will be prevented.
  if (pad_node_def->op() != "Pad") return false;

  // Only fuse contraction with `VALID` padding.
  // TODO(itex): Support more padding type in future.
  string padding_str;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "padding", &padding_str));
  if (padding_str != "VALID") return false;

  // Only fuse contraction with INT32 padding.
  // TODO(itex): support INT64 padding in future.
  if (!HasDataType(pad_node_def, DT_INT32, "Tpaddings")) return false;

  // If contraction has been fused, only fuse it with Pad when only has Bias.
  if (node_def->op() == kFusedConv2D || node_def->op() == kFusedConv3D) {
    int num_args;
    TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "num_args", &num_args));
    if (num_args != 1) return false;
  }

  if (!HaveSameDataType(node_def, pad_node_def) ||
      HasControlFaninOrFanout(*pad_node_view) ||
      !HasAtMostOneFanoutAtPort0(*pad_node_view) ||
      IsInPreserveSet(ctx, pad_node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const PadWithContraction pattern{pad_node_view->node_index(), node_index};
  if (check_device_compatible && !IsDeviceCompatible(ctx, pattern))
    return false;

  // We successfully found a Pad + Conv2D/_ITEXFusedConv2D pattern.
  *matched = pattern;

  return true;
}

bool FindConvBackpropInputWithSlice(const RemapperContext& ctx, int node_index,
                                    ConvBackpropInputWithSlice* matched,
                                    bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsSlice(*node_def)) return false;

  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* conv_node_view = regular_fanin_0.node_view();
  const auto* conv_node_def = conv_node_view->node();
  bool is_ok = IsConv2DBackpropInput(*conv_node_def) ||
               IsConv3DBackpropInputV2(*conv_node_def);
  is_ok = is_ok && conv_node_view->NumRegularFanouts() == 1;

  if (!is_ok) {
    return false;
  }

  // Only fuse contraction with `VALID` padding.
  // TODO(itex): Support more padding type in future.
  string padding_str;
  TF_ABORT_IF_ERROR(GetNodeAttr(*conv_node_def, "padding", &padding_str));
  if (padding_str != "VALID") return false;

  if (!HaveSameDataType(node_def, conv_node_def) ||
      HasControlFaninOrFanout(*conv_node_view) ||
      !HasAtMostOneFanoutAtPort0(*conv_node_view) ||
      IsInPreserveSet(ctx, conv_node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ConvBackpropInputWithSlice pattern{node_index,
                                           conv_node_view->node_index()};
  if (check_device_compatible && !IsDeviceCompatible(ctx, pattern))
    return false;

  *matched = pattern;

  return true;
}

// Optimize:
/*
          TrainingOp
            /    \
         AddN*   others...  ----->  FusedTrainingOp
        /   |                   /    \    \      \
      Mul  in3*                in1   in2  in3*  others
     /   \
   in1   in2
*/

// * means optional ops.
bool FindFusedTrainingOp(const RemapperContext& ctx, int node_index,
                         FusedTrainingOp* matched) {
  // Root of the pattern must be TrainingOp.
  // TODO(itex): Forward control dependencies.
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // Only GPU supports this fusion.
  // TODO(itex): Remove this limitation once it's supported.
  if (!NodeIsOnGpu(node_def)) return false;

  int input_index = -1;
  if (IsApplyMomentum(*node_def) || IsResourceApplyMomentum(*node_def)) {
    // Input: var, accum, lr, grad, momentum
    if (node_view->NumRegularFanins() != 5) return false;
    input_index = 3;
  } else if (IsApplyAdam(*node_def) || IsResourceApplyAdam(*node_def)) {
    // Input : var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
    // grad
    if (node_view->NumRegularFanins() != 10) return false;
    input_index = 9;
  } else if (IsApplyAdamWithWeightDecay(*node_def) ||
             IsResourceApplyAdamWithWeightDecay(*node_def)) {
    // Input : var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
    // weight_decay, grad
    if (node_view->NumRegularFanins() != 11) return false;
    input_index = 10;
  } else {
    return false;
  }
  // TrainingOp has control output
  if (HasControlFanin(*node_view)) return false;

  const auto* input_node_view =
      node_view->GetRegularFanin(input_index).node_view();
  const auto* input_node_def = input_node_view->node();

  if (!HasAtMostOneFanoutAtPort0(*input_node_view) ||
      !HaveSameDataType(node_def, input_node_def) ||
      IsInPreserveSet(ctx, input_node_def))
    return false;

  if (IsAddN(*input_node_def)) {
    // Mul + AddN + Adam_op is not supported
    if (IsApplyAdam(*node_def) || IsResourceApplyAdam(*node_def) ||
        IsApplyAdamWithWeightDecay(*node_def) ||
        IsResourceApplyAdamWithWeightDecay(*node_def))
      return false;
    int input_num = input_node_def->attr().at("N").i();
    if (input_num != 2) return false;

    int input_mul_port = 0;
    auto* mul_node_view = input_node_view->GetRegularFanin(0).node_view();
    int mul_node_index =
        FindFusedTrainingOpInPort(*mul_node_view, *input_node_def);
    if (mul_node_index == -1) {
      input_mul_port = 1;
      mul_node_view = input_node_view->GetRegularFanin(1).node_view();
      mul_node_index =
          FindFusedTrainingOpInPort(*mul_node_view, *input_node_def);
    }
    if (mul_node_index == -1) return false;

    const auto* mul_node_def = mul_node_view->node();
    if (!HasAtMostOneFanoutAtPort0(*mul_node_view) ||
        !HaveSameDataType(node_def, mul_node_def) ||
        IsInPreserveSet(ctx, mul_node_def))
      return false;

    // Mul has two inputs, at least one input should be scalar
    int scalar_input_index =
        GetMulScalarInputIndex(ctx, *mul_node_view->node());
    if (scalar_input_index == -1) return false;

    matched->mul = mul_node_index;
    matched->mul_port = input_mul_port;
    matched->addn = input_node_view->node_index();
    matched->training_op = node_index;

    // Handle a special case:
    // If weight and identity share the same input, there will be additional
    // tensor copy, which shouild be avoided
    //        TrainingOp
    //          /     |
    //       AddN*    |
    //      /   |     |
    //    Mul  in3*   |
    //   /   \        |
    // in1   identity |
    //          |     |
    //        ReadVar |
    //             \  |
    //              weight
    auto* weight_node_view = node_view->GetRegularFanin(0).node_view();
    for (int index = 0; index < 2; index++) {
      auto* variable_node_view =
          mul_node_view->GetRegularFanin(index).node_view();
      while (IsIdentity(*(variable_node_view->node())))
        variable_node_view = variable_node_view->GetRegularFanin(0).node_view();
      if (!IsReadVariableOp(*(variable_node_view->node()))) continue;
      if (variable_node_view->GetRegularFanin(0).node_view()->node_index() ==
          weight_node_view->node_index()) {
        matched->mul_scalar_input = 1 - index;
        break;
      }
    }
    return true;
  } else if (IsMul(*input_node_def)) {
    // Currently, we don't implement Mul + Momemtum fusion. Only Mul + AddN +
    // Momemtum is supported.
    if (IsApplyMomentum(*node_def) || IsResourceApplyMomentum(*node_def))
      return false;

    // Mul has two inputs, at least one input should be scalar
    int scalar_input_index = GetMulScalarInputIndex(ctx, *input_node_def);
    if (scalar_input_index == -1) return false;

    matched->mul = input_node_view->node_index();
    matched->training_op = node_index;
    return true;
  }
  return false;
}

// Fuse BatchMatMul and Mul into FusedBatchMatmul if the other input of
// Mul is a scalar. For example, we can optimize
/*
              Mul
             /  \
    BatchMatMul scale*  ->       FusedBatchMatmul
       /   \                     /      |       \
   input1  input2             input1  input2   scale
*/
// *) scale must be a scalar
bool FindContractionWithMul(const RemapperContext& ctx, int node_index,
                            ContractionWithMul* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsAnyMul(*node_def)) return false;

  // Mul has two inputs, one input should be scalar
  int scalar_input_index = GetMulScalarInputIndex(ctx, *node_def);
  if (scalar_input_index == -1) return false;

  auto* const_node_view =
      node_view->GetRegularFanin(scalar_input_index).node_view();
  auto* contraction_node_view =
      node_view->GetRegularFanin(1 - scalar_input_index).node_view();

  // Currently we only fuse BatchMatMul with Mul
  auto* contraction_node_def = contraction_node_view->node();
  if (!IsAnyBatchMatMul(*contraction_node_def)) return false;

  auto* const_node_def = const_node_view->node();
  if (!IsAnyConst(*const_node_def)) return false;

  bool hasValidType = false;
  hasValidType =
      (HasDataType(node_def, DT_FLOAT) || HasDataType(node_def, DT_BFLOAT16) ||
       (HasDataType(node_def, DT_HALF) && NodeIsOnGpu(node_def)));

  if (!hasValidType) return false;

  if (!HaveSameDataType(node_def, contraction_node_def) ||
      HasControlFaninOrFanout(*contraction_node_view) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  const ContractionWithMul pattern{contraction_node_view->node_index(),
                                   node_index, const_node_view->node_index()};

  *matched = pattern;

  return true;
}

bool FindBf16ContractionGradWithCastFp32(
    const RemapperContext& ctx, int node_index,
    Bf16ContractionGradWithCastFp32* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (node_view == nullptr) return false;
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsCast(*node_def)) return false;

  DataType dst_dtype = GetDataTypeFromAttr(*node_def, "DstT");
  DataType src_dtype = GetDataTypeFromAttr(*node_def, "SrcT");
  if (dst_dtype != DT_FLOAT || src_dtype != DT_BFLOAT16) return false;

  if (node_view->NumRegularFanins() != 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction->node();
  if (!IsFusedMatmulGrad(*contraction_node_def)) return false;

  const auto& contraction_fanout1 = contraction->GetRegularFanout(1);

  if (contraction_fanout1.size() > 1 ||
      IsInPreserveSet(ctx, contraction_node_def) ||
      HasControlFaninOrFanout(*contraction))
    return false;

  const auto* cast1 = contraction_fanout1[0].node_view();
  if (cast1->node_index() == node_view->node_index()) {
    matched->contraction = contraction->node_index();
    matched->bias_cast = cast1->node_index();
    for (auto const& bias_out : cast1->GetRegularFanouts()) {
      for (auto const bias_out_i : bias_out) {
        matched->bias_cast_outs.push_back(bias_out_i.node_view()->node_index());
      }
    }
    return true;
  }
  return false;
}

// Find Bf16(Fused)Matmul + CastFp32 pattern.
bool FindBf16ContractionWithCastFp32(const RemapperContext& ctx, int node_index,
                                     Bf16ContractionWithCastFp32* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  if (!IsCast(*node_def) || HasControlFaninOrFanout(*node_view)) return false;

  if (node_view->NumRegularFanins() != 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction->node();
  if (!IsMatMul(*contraction_node_def) && !IsFusedMatmul(*contraction_node_def))
    return false;

  DataType contraction_dtype = GetDataTypeFromAttr(*contraction_node_def, "T");
  DataType dst_dtype = GetDataTypeFromAttr(*node_def, "DstT");

  // Now, (Fused)Matmul + Cast fusion only support T is DT_BFLOAT16, DstT is
  // DT_FLOAT.
  if ((contraction_dtype != DT_BFLOAT16) || (dst_dtype != DT_FLOAT))
    return false;

  if (!HasAtMostOneFanoutAtPort0(*contraction) ||
      IsInPreserveSet(ctx, contraction_node_def) ||
      HasControlFaninOrFanout(*contraction))
    return false;

  if (IsMatMul(*contraction_node_def)) {
    bool is_BiasAddGrad = false;
    int dz_index;

    const auto* grad_input = contraction->GetRegularFanin(1).node_view();
    if (grad_input != nullptr) {
      if (grad_input->NumRegularFanouts() == 3) {
        for (const auto& input_fanout_i : grad_input->GetRegularFanouts()) {
          for (const auto input_fanout : input_fanout_i) {
            if (IsBiasAddGrad(*(input_fanout.node_view()->node()))) {
              is_BiasAddGrad = true;
              const auto& dz = input_fanout.node_view()->GetRegularFanin(0);
              dz_index = dz.node_view()->node_index();
            }
          }
        }
      }
    }

    if (is_BiasAddGrad) {
      if (IsLegalMatMulGrad(ctx, contraction->node_index(), dz_index))
        return false;
    }
  }

  matched->cast = node_index;
  matched->contraction = regular_fanin_0.node_index();
  return true;
}

// Comparison op + cast
bool FindComparisonWithCast(const RemapperContext& ctx, int node_index,
                            ComparisonWithCast* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  if (!IsCast(*node_def) || HasControlFaninOrFanout(*node_view)) return false;

  if (node_view->NumRegularFanins() != 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* comparison = regular_fanin_0.node_view();
  const auto* comparison_node_def = comparison->node();
  if (!IsComparison(*comparison_node_def) ||
      HasControlFaninOrFanout(*comparison))
    return false;

  DataType comparator_dtype = GetDataTypeFromAttr(*comparison_node_def, "T");
  DataType src_dtype = GetDataTypeFromAttr(*node_def, "SrcT");
  DataType dst_dtype = GetDataTypeFromAttr(*node_def, "DstT");

  if ((comparator_dtype != DT_FLOAT) && (comparator_dtype != DT_BFLOAT16) &&
      (comparator_dtype != DT_HALF))
    return false;
  if ((comparator_dtype != dst_dtype) || (src_dtype != DT_BOOL)) return false;

  // Check that only one node consumes the 0-th output of a comparison.
  if (!HasAtMostOneFanoutAtPort0(*comparison) ||
      IsInPreserveSet(ctx, comparison_node_def))
    return false;

  matched->cast = node_index;
  matched->comparison = regular_fanin_0.node_index();
  matched->fused_op =
      matched->fused_op + comparison_node_def->op() + "WithCast";

  return true;
}

// Random op + Comparison + cast
bool FindRandomWithComparisonAndCast(const RemapperContext& ctx, int node_index,
                                     RandomWithComparisonAndCast* matched) {
  ComparisonWithCast comparison_with_cast;
  if (!FindComparisonWithCast(ctx, node_index, &comparison_with_cast))
    return false;

  const auto* node_view =
      ctx.graph_view.GetNode(comparison_with_cast.comparison);

  if (node_view->NumRegularFanins() != 2) return false;
  const auto* node_def = node_view->node();
  if (!IsGreaterEqual(*node_def)) return false;

  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node_def->name(), &props));

  const auto HasRandom = [&](int direction) -> bool {
    const auto& regular_fanin = node_view->GetRegularFanin(direction);
    const auto* random = regular_fanin.node_view();
    const auto* random_node_def = random->node();
    return IsRandomUniform(*random_node_def);
  };

  matched->direction = 0;
  if (!HasRandom(matched->direction)) {
    return false;
  }
  auto compare_shape = props[1 - matched->direction].shape();
  if (Rank(compare_shape) != 0) return false;

  const auto& regular_fanin = node_view->GetRegularFanin(matched->direction);
  const auto* random = regular_fanin.node_view();
  const auto* random_node_def = random->node();

  // TODO(itex) Add CPU support
  if (NodeIsOnCpu(random_node_def)) return false;

  if (HasControlFaninOrFanout(*random)) return false;

  DataType random_dtype = GetDataTypeFromAttr(*random_node_def, "dtype");

  if ((random_dtype != DT_FLOAT) && (random_dtype != DT_BFLOAT16) &&
      (random_dtype != DT_HALF))
    return false;

  // Check that only one node consumes the 0-th output of a random.
  if (!HasAtMostOneFanoutAtPort0(*random) ||
      IsInPreserveSet(ctx, random_node_def) || HasControlFaninOrFanout(*random))
    return false;

  matched->cast = comparison_with_cast.cast;
  matched->comparison = comparison_with_cast.comparison;
  matched->random = regular_fanin.node_index();

  return true;
}

// Fuse Mul and Maximum into LeakyRelu
/*
       maximum
        /    \
       mul    |      =>   LeakyRelu
      /  \    |               |
  const*  input             input
*/
// *) const must be a scalar with float datatype. value < 1.
bool FindMulWithMaximum(const RemapperContext& ctx, int node_index,
                        MulWithMaximum* matched) {
  // Check Maximum node
  const auto* maximum_view = ctx.graph_view.GetNode(node_index);
  const auto* maximum_node = maximum_view->node();
  if (!IsMaximum(*maximum_node) || HasControlFaninOrFanout(*maximum_view) ||
      maximum_view->NumRegularFanins() != 2)
    return false;

  // Check Mul node has a scalar const fanin and the same input fanin with
  // Maximum.
  for (int attempt_mul_fanin = 0; attempt_mul_fanin <= 1; ++attempt_mul_fanin) {
    const auto* mul_view =
        maximum_view->GetRegularFanin(attempt_mul_fanin).node_view();
    const auto* mul_node = mul_view->node();
    if (!IsAnyMul(*mul_node) || mul_view->NumRegularFanins() != 2 ||
        mul_view->NumRegularFanouts() != 1 ||
        HasControlFaninOrFanout(*mul_view))
      continue;

    // Check scalar const node
    int attempt_scalar_fanin = GetMulScalarInputIndex(ctx, *mul_node);
    if (attempt_scalar_fanin == -1) continue;
    auto* const_view =
        mul_view->GetRegularFanin(attempt_scalar_fanin).node_view();
    auto* const_node = const_view->node();
    if (!IsAnyConst(*const_node) || HasControlFaninOrFanout(*const_view))
      continue;

    // Check float datatype and value < 1
    float alpha_value = -1;
    DataType const_dtype = GetDataTypeFromAttr(*const_node, "dtype");
    Tensor const_tensor;
    const_tensor.FromProto(const_node->attr().at("value").tensor());
    if (const_dtype == DT_BFLOAT16) {
      alpha_value = static_cast<float>(const_tensor.flat<Eigen::bfloat16>()(0));
    } else if (const_dtype == DT_HALF) {
      alpha_value = static_cast<float>(const_tensor.flat<Eigen::half>()(0));
    } else if (const_dtype == DT_DOUBLE) {
      alpha_value = static_cast<float>(const_tensor.flat<double>()(0));
    } else if (const_dtype == DT_FLOAT) {
      alpha_value = const_tensor.flat<float>()(0);
    } else {
      continue;
    }
    if (alpha_value > 1) continue;

    // Check same input node
    if (maximum_view->GetRegularFanin(1 - attempt_mul_fanin).node_index() !=
        mul_view->GetRegularFanin(1 - attempt_scalar_fanin).node_index())
      continue;

    if (!HasAtMostOneFanoutAtPort0(*mul_view) ||
        IsInPreserveSet(ctx, mul_view->node()))
      continue;

    matched->maximum = node_index;
    matched->mul = mul_view->node_index();
    matched->input =
        maximum_view->GetRegularFanin(1 - attempt_mul_fanin).node_index();
    matched->alpha = alpha_value;
    return true;
  }
  return false;
}

// Find Const + Cast pattern.
bool FindConstWithCast(const RemapperContext& ctx, int node_index,
                       ConstWithCast* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  if (!IsCast(*node_def) || HasControlFaninOrFanout(*node_view)) return false;

  if (node_view->NumRegularFanins() != 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* constant = regular_fanin_0.node_view();
  const auto* constant_node_def = constant->node();
  if (!IsConstant(*constant_node_def) || HasControlFaninOrFanout(*constant))
    return false;

  DataType constant_dtype = GetDataTypeFromAttr(*constant_node_def, "dtype");
  DataType src_dtype = GetDataTypeFromAttr(*node_def, "SrcT");
  DataType dst_dtype = GetDataTypeFromAttr(*node_def, "DstT");
  bool truncate = GetDataTypeFromAttr(*node_def, "Truncate");

  // Now, Const + Cast fusion only support SrcT is DT_FLOAT, DstT is DT_BFLOAT16
  // or DT_HALF, and truncate is false.
  if ((constant_dtype != DT_FLOAT) || (src_dtype == dst_dtype) ||
      (truncate == true))
    return false;

  // As Const + Cast fusion will create a new tensor, it requires the tensor
  // size is valid
  const TensorProto& raw_val = constant_node_def->attr().at("value").tensor();
  const TensorShape shape(raw_val.tensor_shape());
  const int64_t num_tensor_values = shape.num_elements();

  if (num_tensor_values <= 0) return false;

  if ((dst_dtype != DT_BFLOAT16) && (dst_dtype != DT_HALF)) return false;

  if (!HasAtMostOneFanoutAtPort0(*constant) ||
      IsInPreserveSet(ctx, constant_node_def))
    return false;

  matched->cast = node_index;
  matched->constant = regular_fanin_0.node_index();
  return true;
}

// Find sequatial binary ops.
bool FindFusedBinary(const RemapperContext& ctx, int node_index,
                     FusedBinary* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // Only check control fanin for output node is enough.
  if (HasControlFanin(*node_view)) return false;

  // Only support on GPU.
  if (NodeIsOnCpu(node_def)) return false;

  // Only support Add/Mul/Sub now because they satisfy the commutative law.
  if (!IsAdd(*node_def) && !IsMul(*node_def) && !IsSub(*node_def)) return false;

  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16) &&
      !(HasDataType(node_def, DT_HALF) && NodeIsOnGpu(node_def)))
    return false;

  // Returns true iff the node is a compatible FusedBatchNorm node.
  const auto valid_shape = [&](const utils::MutableNodeView& binary) -> bool {
    const auto* binary_def = binary.node();
    std::vector<OpInfo_TensorProperties> props;
    TF_ABORT_IF_ERROR(
        ctx.graph_properties.GetInputProperties(binary_def->name(), &props));

    if (props.size() < 2) return false;
    bool same_input =
        ShapesSymbolicallyEqual(props[0].shape(), props[1].shape());
    bool has_scalar =
        Rank(props[0].shape()) == 0 || Rank(props[1].shape()) == 0;
    if (!(same_input || has_scalar)) return false;
    return true;
  };

  if (!valid_shape(*node_view)) return false;

  // Initialize root node.
  matched->root_ = node_index;
  matched->num_ = 1;

  const int max_depth = 3;

  // Check inputs iteratively til they can't match sequatial Binary op.
  bool is_found = true;
  while (is_found) {
    is_found = false;
    node_view = ctx.graph_view.GetNode(node_index);

    ITEX_CHECK(node_view->NumRegularFanins() == 2)
        << "Incorrect inputs for BinaryOp.";

    for (int i = 0; i < node_view->NumRegularFanins(); ++i) {
      auto& regular_fanin = node_view->GetRegularFanin(i);
      const auto* input_node_view = regular_fanin.node_view();
      const auto* input_node_def = input_node_view->node();

      if (!IsAdd(*input_node_def) && !IsMul(*input_node_def) &&
          !IsSub(*input_node_def))
        continue;

      if (!HasDataType(input_node_def, DT_FLOAT) &&
          !HasDataType(input_node_def, DT_BFLOAT16) &&
          !(HasDataType(input_node_def, DT_HALF) &&
            NodeIsOnGpu(input_node_def)))
        continue;

      if (HasControlFaninOrFanout(*input_node_view) ||
          !HasAtMostOneFanoutAtPort0(*input_node_view) ||
          IsInPreserveSet(ctx, input_node_def))
        continue;

      if (!valid_shape(*input_node_view)) continue;

      is_found = true;
      node_index = regular_fanin.node_index();
      matched->fused_ops_.push_back(node_index);
      matched->input_order_.push_back(i);
      matched->num_++;

      break;
    }
    if (matched->num_ >= max_depth) break;
  }

  // Reture `true` if find more than 1 sequatial Bianry ops.
  return matched->num_ > 1;
}

void CopyFusedBatchNormAttributes(const NodeDef& fused_batch_norm,
                                  NodeDef* fused_batch_norm_ex) {
  ITEX_DCHECK(IsFusedBatchNorm(fused_batch_norm) ||
              IsFusedBatchNormGrad(fused_batch_norm))
      << "Input node must be a FusedBatchNorm";

  CopyAllAttrs(fused_batch_norm, fused_batch_norm_ex);

  // FusedBatchNorm doesn't have an extra type parameter.
  if ((fused_batch_norm.op() == "FusedBatchNorm") ||
      (fused_batch_norm.op() == "FusedBatchNormGrad")) {
    AddNodeAttr("U", DT_FLOAT, fused_batch_norm_ex);
  }
}

// Helper function to set fused op attributes with activation.
// `fused_ops` should not contain `activation`, it will add activation
// in this function.
void SetFusedOpAttributesWithActivation(
    NodeDef* fused, const NodeDef* activation,
    std::vector<absl::string_view> fused_ops, int num_args = 1) {
  // Handle special activation.
  if (activation != nullptr) {
    auto& activation_attr = activation->attr();

    if (IsLeakyRelu(*activation)) {
      AddNodeAttr("leakyrelu_alpha", activation_attr.at("alpha"), fused);
      fused_ops.push_back(activation->op());
    } else if (IsGelu(*activation)) {
      fused_ops.push_back(activation_attr.at("approximate").b()
                              ? "GeluApproximate"
                              : "GeluExact");
    } else {
      fused_ops.push_back(activation->op());
    }
  }

  SetFusedOpAttributes(fused, fused_ops, num_args);
}

// Contraction + BiasAdd.
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  ITEX_DCHECK(IsDeviceCompatible(*ctx, matched))
      << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  ITEX_VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd: "
               << " bias_add=" << bias_add.name()
               << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(bias_add.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));               // 0: input
  fused_op.add_input(contraction.input(1));               // 1: filter
  fused_op.add_input(bias_add.input(matched.bias_port));  // 2: bias

  if (IsConv2D(contraction)) {
    fused_op.set_op(kFusedConv2D);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_op.set_op(kFusedDepthwiseConv2dNative);
  } else if (IsConv3D(contraction)) {
    fused_op.set_op(kFusedConv3D);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMul);
  } else if (IsAccMatMul(contraction)) {
    fused_op.set_op(kFusedAccMatMul);
  } else {
    ITEX_CHECK(false);
  }

  CopyAllAttrs(contraction, &fused_op);
  SetFusedOpAttributes(&fused_op, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return Status::OK();
}

Status AddFusedAddN(RemapperContext* ctx, const FusedAddN& matched,
                    std::vector<bool>* invalidated_nodes,
                    std::vector<bool>* nodes_to_delete) {
  ITEX_DCHECK(IsDeviceCompatible(*ctx, matched))
      << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& addN = graph->node(matched.addN);

  ITEX_DCHECK(IsAddN(addN));

  int num = matched.inputs_of_addN.size();
  ITEX_DCHECK_GE(num, 0);
  ITEX_VLOG(2) << "Fuse " << addN.op() << " with " << num << " L2Loss"
               << " AddN=" << addN.name() << " the first L2Loss="
               << (graph->node(matched.inputs_of_addN[0])).name();

  NodeDef fused_op;
  fused_op.set_op(kFusedAddN);
  fused_op.set_name(addN.name());
  fused_op.set_device(addN.device());

  for (int i = 0; i < num; ++i) {
    const int l2loss_index = matched.inputs_of_addN[i];
    fused_op.add_input(graph->node(l2loss_index).input(0));
  }
  CopyAllAttrs(addN, &fused_op);
  AddNodeAttr("fused_ops",
              absl::Span<const absl::string_view>{"AddN", "l2loss"}, &fused_op);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.addN] = true;
  for (int i = 0; i < num; ++i) {
    (*nodes_to_delete)[matched.inputs_of_addN[i]] = true;
  }
  return Status::OK();
}

// MatMulGrad + BiasGrad or Conv2DBackpropFilter + BiassAdd.
Status AddFusedContractionGradNode(RemapperContext* ctx,
                                   const ContractionWithBiasAddGrad& matched,
                                   std::vector<bool>* invalidated_nodes,
                                   std::vector<bool>* nodes_to_delete) {
  ITEX_DCHECK(IsDeviceCompatible(*ctx, matched))
      << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);

  const NodeDef& bias_add_grad = graph->node(matched.bias_add_grad);
  ITEX_DCHECK(IsMatMul(contraction) || IsConv2DBackpropFilter(contraction) ||
              IsConv3DBackpropFilterV2(contraction))
      << "Input node must be a MatMul or ConvBackpropFilter";

  ITEX_VLOG(2) << "Fuse " << contraction.op() << " with BiasAddGrad: "
               << " bias_add_grad=" << bias_add_grad.name()
               << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(contraction.name());
  fused_op.set_device(contraction.device());
  if (IsConv2DBackpropFilter(contraction)) {
    fused_op.set_op(kConv2DBackpropFilterWithBias);
  } else if (IsConv3DBackpropFilterV2(contraction)) {
    fused_op.set_op(kConv3DBackpropFilterWithBias);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMulGrad);
  } else {
    ITEX_CHECK(false);
  }
  auto* fused_op_attr = fused_op.mutable_attr();
  auto& contraction_attr = contraction.attr();

  if (IsMatMul(contraction)) {
    if (contraction.input(0) == bias_add_grad.input(0)) {
      fused_op.add_input(contraction.input(1));  // 0: input
      (*fused_op_attr)["transpose_a"] = contraction_attr.at("transpose_b");
      (*fused_op_attr)["transpose_b"] = contraction_attr.at("transpose_a");
    } else {
      fused_op.add_input(contraction.input(0));  // 0: input
      const AttrValue ta_attr = contraction_attr.at("transpose_a");
      SetAttrValue(!ta_attr.b(), &(*fused_op_attr)["transpose_a"]);
      (*fused_op_attr)["transpose_b"] = contraction_attr.at("transpose_b");
    }
    fused_op.add_input(bias_add_grad.input(0));  // 1: dz
    (*fused_op_attr)["T"] = contraction_attr.at("T");
  } else {
    // Contraction is checked before. It must be `Conv2DBackpropFilter` or
    // `Conv3DBackpropFilter` here.
    fused_op.add_input(contraction.input(0));    // 0: input
    fused_op.add_input(contraction.input(1));    // 1: filter_size
    fused_op.add_input(bias_add_grad.input(0));  // 2: grad
    CopyAllAttrs(contraction, &fused_op);
  }

  std::vector<NodeDef> bias_add_grad_outs;
  bias_add_grad_outs.resize(matched.bias_add_grad_outs.size());
  for (size_t i = 0; i < matched.bias_add_grad_outs.size(); ++i) {
    const NodeDef& out_i = graph->node(matched.bias_add_grad_outs[i]);
    bias_add_grad_outs[i].set_name(out_i.name());
    bias_add_grad_outs[i].set_device(out_i.device());
    bias_add_grad_outs[i].set_op(out_i.op());
    for (int j = 0; j < out_i.input_size(); ++j) {
      auto out_i_input = out_i.input(j);
      if (out_i_input == bias_add_grad.name()) {
        out_i_input = contraction.name() + ":1";
      }
      bias_add_grad_outs[i].add_input(out_i_input);
    }
    CopyAllAttrs(out_i, &bias_add_grad_outs[i]);
  }

  AddNodeAttr("fused_ops", absl::Span<const absl::string_view>{"BiasAddGrad"},
              &fused_op);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  for (size_t i = 0; i < matched.bias_add_grad_outs.size(); ++i) {
    mutation->AddNode(std::move(bias_add_grad_outs[i]), &status);
  }
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add_grad] = true;
  for (size_t i = 0; i < matched.bias_add_grad_outs.size(); ++i) {
    (*invalidated_nodes)[matched.bias_add_grad_outs[i]] = true;
  }

  return Status::OK();
}

// Contraction + BiasAdd + Add.
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAddAndAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);

  // OneDnn version only support fusion for Conv and MatMul.
  ITEX_DCHECK(IsConvOrMatMul(contraction) || IsAccMatMul(contraction));

  NodeDef contraction_node;
  const NodeDef& add = graph->node(matched.add);
  contraction_node.set_name(add.name());
  contraction_node.set_device(contraction.device());
  contraction_node.add_input(
      contraction.input(0));  // 0: input(conv) / a (matmul)
  contraction_node.add_input(
      contraction.input(1));  // 1: filter(conv) / b (matmul)
  contraction_node.add_input(bias_add.input(matched.bias_port));  // 2: bias

  // Add OP has two inputs, one is conv+bias/matmul+bias pattern matched
  // previously, the other input to add is fused here.
  contraction_node.add_input(add.input(1 - matched.port_id));

  if (IsConv2D(contraction)) {
    contraction_node.set_op(kFusedConv2DWithSum);
  } else if (IsMatMul(contraction)) {
    contraction_node.set_op(kFusedMatMulWithSum);
  } else if (IsConv3D(contraction)) {
    contraction_node.set_op(kFusedConv3D);
  } else if (IsAccMatMul(contraction)) {
    contraction_node.set_op(kFusedAccMatMulWithSum);
  } else {
    ITEX_CHECK(false);
  }

  CopyAllAttrs(contraction, &contraction_node);
  SetFusedOpAttributes(&contraction_node, {"BiasAdd", "Add"}, 2);

  // TODO(itex): Support in-place optimization for Conv3D.

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(contraction_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.add] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;

  return Status::OK();
}

// Contractoin + BiasAdd + Activation.
Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAddAndActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  ITEX_DCHECK(IsDeviceCompatible(*ctx, matched))
      << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& activation = graph->node(matched.activation);

  ITEX_VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd and "
               << activation.op() << ":"
               << " activation=" << activation.name()
               << " bias_add=" << bias_add.name()
               << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(activation.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));               // 0: input
  fused_op.add_input(contraction.input(1));               // 1: filter
  fused_op.add_input(bias_add.input(matched.bias_port));  // 2: bias

  if (IsConv2D(contraction)) {
    fused_op.set_op(kFusedConv2D);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_op.set_op(kFusedDepthwiseConv2dNative);
  } else if (IsConv3D(contraction)) {
    fused_op.set_op(kFusedConv3D);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMul);
  } else if (IsAccMatMul(contraction)) {
    fused_op.set_op(kFusedAccMatMul);
  } else {
    ITEX_CHECK(false);
  }

  CopyAllAttrs(contraction, &fused_op);
  SetFusedOpAttributesWithActivation(&fused_op, &activation, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return Status::OK();
}

// Contraction + BiasAdd + Add + Activation.
Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAndAddActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  ITEX_DCHECK(IsConvOrMatMul(contraction) || IsAccMatMul(contraction));
  const NodeDef& activation = graph->node(matched.activation);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& add = graph->node(matched.add);

  ITEX_VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd and Add and "
               << activation.op() << ":"
               << " activation=" << activation.name()
               << " bias_add=" << bias_add.name() << " add=" << add.name()
               << " contraction=" << contraction.name();

  NodeDef fused_node;
  fused_node.set_name(activation.name());
  if (IsConv2D(contraction))
    fused_node.set_op(kFusedConv2DWithSum);
  else if (IsDepthwiseConv2dNative(contraction))
    fused_node.set_op(kFusedDepthwiseConv2dNative);
  else if (IsConv3D(contraction))
    fused_node.set_op(kFusedConv3D);
  else if (IsMatMul(contraction))
    fused_node.set_op(kFusedMatMulWithSum);
  else if (IsAccMatMul(contraction))
    fused_node.set_op(kFusedAccMatMulWithSum);
  else
    ITEX_CHECK(false);

  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));  // 0: input
  fused_node.add_input(contraction.input(1));  // 1: filter
  fused_node.add_input(bias_add.input(1));     // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  fused_node.add_input(add.input(1 - matched.port_id));

  CopyAllAttrs(contraction, &fused_node);
  SetFusedOpAttributesWithActivation(&fused_node, &activation,
                                     {"BiasAdd", "Add"}, 2);

  // TODO(itex): Support in-place optimization for Conv3D.

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.activation] = true;
  (*nodes_to_delete)[matched.add] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return Status::OK();
}

// Contraction + BiasAdd + Activation + Add.
Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAndActivationAdd& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  ITEX_DCHECK(IsConvOrMatMul(contraction) || IsAccMatMul(contraction));
  const NodeDef& activation = graph->node(matched.activation);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& add = graph->node(matched.add);

  NodeDef fused_node;
  fused_node.set_name(add.name());
  if (IsConv2D(contraction))
    fused_node.set_op(kFusedConv2DWithSum);
  else if (IsDepthwiseConv2dNative(contraction))
    fused_node.set_op(kFusedDepthwiseConv2dNative);
  else if (IsConv3D(contraction))
    fused_node.set_op(kFusedConv3D);
  else if (IsMatMul(contraction))
    fused_node.set_op(kFusedMatMulWithSum);
  else if (IsAccMatMul(contraction))
    fused_node.set_op(kFusedAccMatMulWithSum);
  else
    ITEX_CHECK(false);

  fused_node.set_device(add.device());
  fused_node.add_input(contraction.input(0));  // 0: input
  fused_node.add_input(contraction.input(1));  // 1: filter
  fused_node.add_input(bias_add.input(1));     // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  fused_node.add_input(add.input(1 - matched.port_id));

  CopyAllAttrs(contraction, &fused_node);
  SetFusedOpAttributesWithActivation(&fused_node, &activation,
                                     {"BiasAdd", "Add"}, 2);

  // TODO(itex): Support in-place optimization for Conv3D.
  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.add] = true;
  (*nodes_to_delete)[matched.activation] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  ITEX_VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd and Add and "
               << activation.op() << ":"
               << " activation=" << activation.name()
               << " bias_add=" << bias_add.name() << " add=" << add.name()
               << " contraction=" << contraction.name();
  return Status::OK();
}

// Contraction + Mul(scale).
// TODO(itex): Try to combine this function with Conv + BiasAdd
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithMul& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& mul = graph->node(matched.mul);
  const NodeDef& scalar = graph->node(matched.scalar);
  ITEX_VLOG(2) << "Fuse " << contraction.op() << " with Mul: "
               << " mul=" << mul.name()
               << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(mul.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));  // 0: input
  fused_op.add_input(contraction.input(1));  // 1: filter
  fused_op.add_input(scalar.name());         // 2: scale
  fused_op.set_op(kFusedBatchMatMul);

  CopyAllAttrs(contraction, &fused_op);
  SetFusedOpAttributes(&fused_op, {"Mul"}, 1);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.mul] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return Status::OK();
}

Status AddFusedBatchNormExNode(RemapperContext* ctx,
                               const FusedBatchNormEx& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  const NodeDef& activation = graph->node(matched.activation);

  ITEX_VLOG(2) << "Fuse " << activation.op() << " with FusedBatchNorm:"
               << " activation=" << activation.name() << " side_input="
               << (matched.side_input != kMissingIndex
                       ? graph->node(matched.side_input).name()
                       : "<none>")
               << " invalidated="
               << (matched.invalidated != kMissingIndex
                       ? graph->node(matched.invalidated).name()
                       : "<none>")
               << " fused_batch_norm=" << fused_batch_norm.name();

  // Replace FusedBatchNorm with _FusedBatchNormEx + <SideInput> + <Activation>.
  NodeDef fused_op;
  fused_op.set_op(kFusedBatchNormEx);
  fused_op.set_name(fused_batch_norm.name());
  fused_op.set_device(fused_batch_norm.device());

  fused_op.add_input(fused_batch_norm.input(0));  // 0: input
  fused_op.add_input(fused_batch_norm.input(1));  // 1: scale
  fused_op.add_input(fused_batch_norm.input(2));  // 2: offset
  fused_op.add_input(fused_batch_norm.input(3));  // 3: estimated_mean
  fused_op.add_input(fused_batch_norm.input(4));  // 4: estimated_var

  CopyFusedBatchNormAttributes(fused_batch_norm, &fused_op);

  auto* attrs = fused_op.mutable_attr();
  SetAttrValue(activation.op(), &(*attrs)["activation_mode"]);

  if (matched.side_input != kMissingIndex) {
    AddNodeAttr("num_side_inputs", 1, &fused_op);
    const NodeDef& side_input = graph->node(matched.side_input);
    fused_op.add_input(side_input.name());  // 5: side_input
  } else {
    AddNodeAttr("num_side_inputs", 0, &fused_op);
  }

  // Turn activation node into Identity node.
  NodeDef identity_op;
  identity_op.set_op("Identity");
  identity_op.set_name(activation.name());
  identity_op.set_device(fused_batch_norm.device());
  identity_op.add_input(fused_batch_norm.name());
  (*identity_op.mutable_attr())["T"] = attrs->at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  mutation->AddNode(std::move(identity_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm] = true;
  (*invalidated_nodes)[matched.activation] = true;
  if (matched.side_input != kMissingIndex) {
    (*nodes_to_delete)[matched.invalidated] = true;
  }

  return Status::OK();
}

Status AddFusedBatchNormGradExNode(RemapperContext* ctx,
                                   const FusedBatchNormGradEx& matched,
                                   std::vector<bool>* invalidated_nodes,
                                   std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused_batch_norm_grad =
      graph->node(matched.fused_batch_norm_grad);
  const NodeDef& activation_grad = graph->node(matched.activation_grad);
  const NodeDef& fwd_fused_batch_norm =
      graph->node(matched.fwd_fused_batch_norm);

  ITEX_VLOG(2) << "Fuse FusedBatchNormGrad with " << activation_grad.op()
               << ": "
               << " fused_batch_norm_grad=" << fused_batch_norm_grad.name()
               << " side_input="
               << (matched.side_input_grad != kMissingIndex
                       ? graph->node(matched.side_input_grad).name()
                       : "<none>")
               << " activation=" << activation_grad.name()
               << " corresponding FusedBatchNorm="
               << fwd_fused_batch_norm.name();

  NodeDef fused_op;
  fused_op.set_op(kFusedBatchNormGradEx);
  fused_op.set_name(fused_batch_norm_grad.name());
  fused_op.set_device(fused_batch_norm_grad.device());

  fused_op.add_input(activation_grad.input(0));        // 0: y_backprop
  fused_op.add_input(fused_batch_norm_grad.input(1));  // 1: x
  fused_op.add_input(fused_batch_norm_grad.input(2));  // 2: scale
  fused_op.add_input(fused_batch_norm_grad.input(3));  // 3: reserve_space_1
  fused_op.add_input(fused_batch_norm_grad.input(4));  // 4: reserve_space_2
  fused_op.add_input(fused_batch_norm_grad.input(5));  // 5: reserve_space_3
  fused_op.add_input(fwd_fused_batch_norm.input(2));   // 6: offset
  fused_op.add_input(activation_grad.input(1));        // 7: y

  CopyFusedBatchNormAttributes(fused_batch_norm_grad, &fused_op);

  auto* attrs = fused_op.mutable_attr();
  // Only support Relu mode, has check in kernel.
  SetAttrValue(activation_grad.op(), &(*attrs)["activation_mode"]);

  if (matched.side_input_grad != kMissingIndex) {
    SetAttrValue(1, &(*attrs)["num_side_inputs"]);
  } else {
    SetAttrValue(0, &(*attrs)["num_side_inputs"]);
  }

  NodeDef identity_op;
  identity_op.set_op("Identity");
  identity_op.set_name(activation_grad.name());
  identity_op.set_device(fused_batch_norm_grad.device());
  identity_op.add_input(strings::StrCat(fused_batch_norm_grad.name(), ":5"));
  (*identity_op.mutable_attr())["T"] = attrs->at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  if (matched.side_input_grad != kMissingIndex) {
    mutation->AddNode(std::move(identity_op), &status);
    TF_RETURN_IF_ERROR(status);
  }
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm_grad] = true;
  if (matched.side_input_grad != kMissingIndex) {
    (*invalidated_nodes)[matched.activation_grad] = true;
  } else {
    (*nodes_to_delete)[matched.activation_grad] = true;
  }

  return Status::OK();
}

// Pad + Contraction.
Status AddPadWithContractionNode(RemapperContext* ctx,
                                 const PadWithContraction& matched,
                                 std::vector<bool>* invalidated_nodes,
                                 std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& pad = graph->node(matched.pad);
  const NodeDef& contraction = graph->node(matched.contraction);

  NodeDef pad_with_conv;
  pad_with_conv.set_name(contraction.name());
  pad_with_conv.set_device(contraction.device());
  pad_with_conv.add_input(pad.input(0));          // 0: input
  pad_with_conv.add_input(contraction.input(1));  // 1: filter
  // Add bias input if contraction is _ITEXFusedConv2D.
  if (IsConv2D(contraction)) {
    pad_with_conv.set_op(kPadWithConv2D);
  } else if (IsConv3D(contraction)) {
    pad_with_conv.set_op(kPadWithConv3D);
  } else if (contraction.op() == kFusedConv2D) {
    pad_with_conv.set_op(kPadWithFusedConv2D);
    pad_with_conv.add_input(contraction.input(2));  // 2: bias
  } else {
    pad_with_conv.set_op(kPadWithFusedConv3D);
    pad_with_conv.add_input(contraction.input(2));  // 2: bias
  }
  pad_with_conv.add_input(pad.input(1));  // Last: pad

  CopyAllAttrs(contraction, &pad_with_conv);
  DataType paddings_type;
  TF_ABORT_IF_ERROR(GetNodeAttr(pad, "Tpaddings", &paddings_type));
  AddNodeAttr("Tpaddings", paddings_type, &pad_with_conv);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(pad_with_conv), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.contraction] = true;
  (*nodes_to_delete)[matched.pad] = true;

  return Status::OK();
}

// Simply remove Dequantize before Shape since Shape has already supported
// INT8 input.
Status AddFusedDequantizeWithShape(RemapperContext* ctx,
                                   const DequantizeWithShape& matched,
                                   std::vector<bool>* invalidated_nodes,
                                   std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();

  const NodeDef& dequantize_node_def = graph->node(matched.dequantizeIndex);
  const NodeDef& shape_node_def = graph->node(matched.shapeIndex);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();

  // Create a new Shape node with INT8 input dtype.
  NodeDef fused_op;
  fused_op.set_name(shape_node_def.name());

  // TODO(itex): Since the GPU Shape of proper TF does not support qint8, we
  // forced it to execute on the CPU. But this will cause a DeviceToHost
  // MemoryCopy. In the future, we plan to add our own Shape to ITEX-GPU to
  // bypass this problem.

  fused_op.set_device("/job:localhost/replica:0/task:0/device:CPU:0");
  fused_op.add_input(dequantize_node_def.input(0));  // change input node
  fused_op.set_op(shape_node_def.op());

  DataType dtype;
  TF_ABORT_IF_ERROR(GetNodeAttr(dequantize_node_def, "T", &dtype));
  auto* new_attr = fused_op.mutable_attr();
  SetAttrValue(dtype, &(*new_attr)["T"]);

  DataType out_type;
  TF_ABORT_IF_ERROR(GetNodeAttr(shape_node_def, "out_type", &out_type));
  SetAttrValue(out_type, &(*new_attr)["out_type"]);

  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  ITEX_VLOG(2) << "Fuse Dequantize, with Shape:"
               << " Dequantize=" << dequantize_node_def.name()
               << " Shape=" << shape_node_def.name();

  (*invalidated_nodes)[matched.shapeIndex] = true;
  (*nodes_to_delete)[matched.dequantizeIndex] = true;

  return Status::OK();
}

Status AddFusedDequantizeWithReshape(RemapperContext* ctx,
                                     const DequantizeWithReshape& matched,
                                     std::vector<bool>* invalidated_nodes,
                                     std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();

  const NodeDef& dequantize_node_def = graph->node(matched.dequantizeIndex_);
  const NodeDef& reshape_node_def = graph->node(matched.reshapeIndex_);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  // create a new node with new input dtype
  NodeDef fused_op;
  fused_op.set_name(reshape_node_def.name());
  fused_op.set_device(reshape_node_def.device());
  fused_op.add_input(dequantize_node_def.input(0));  // input node
  fused_op.add_input(dequantize_node_def.input(1));  // min range
  fused_op.add_input(dequantize_node_def.input(2));  // max range
  fused_op.add_input(reshape_node_def.input(1));     // shape tensor
  fused_op.set_op(kDequantizeReshape);

  CopyAllAttrs(dequantize_node_def, &fused_op);

  DataType out_type;
  TF_ABORT_IF_ERROR(GetNodeAttr(reshape_node_def, "Tshape", &out_type));
  AddNodeAttr("Tshape", out_type, &fused_op);
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  ITEX_VLOG(2) << "Fuse Dequantize, with Reshape:"
               << " Dequantize=" << dequantize_node_def.name()
               << " Reshape=" << reshape_node_def.name();

  (*invalidated_nodes)[matched.reshapeIndex_] = true;
  (*nodes_to_delete)[matched.dequantizeIndex_] = true;

  return Status::OK();
}

Status AddQuantizeV2WithQuantizedConv2DNode(
    RemapperContext* ctx, const QuantizeV2WithQuantizedConv2D& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();

  const NodeDef& quantizev2_node_def = graph->node(matched.quantizeV2Index_);
  const NodeDef& quantized_conv_node_def =
      graph->node(matched.quantizedConv2DIndex_);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  // create a new node
  NodeDef fused_op;
  fused_op.set_name(quantized_conv_node_def.name());
  fused_op.set_device(quantized_conv_node_def.device());
  fused_op.add_input(quantizev2_node_def.input(0));      // quantizeV2 input
  fused_op.add_input(quantized_conv_node_def.input(1));  // conv filter
  fused_op.add_input(quantized_conv_node_def.input(2));  // conv bias
  fused_op.add_input(quantizev2_node_def.input(1));  // quantizeV2  min input
  fused_op.add_input(quantizev2_node_def.input(2));  // quantizeV2 max input
  fused_op.add_input(quantized_conv_node_def.input(5));  // conv min filter
  fused_op.add_input(quantized_conv_node_def.input(6));  // conv max filter
  fused_op.add_input(
      quantized_conv_node_def.input(7));  // conv min freezed output
  fused_op.add_input(
      quantized_conv_node_def.input(8));  // conv max freezed output
  fused_op.set_op(kQuantizeV2WithQuantizedConv2D);

  // Copy attr from original nodes to fused node, and set missing attr.
  AddNodeAttr("Tinput", DT_FLOAT, &fused_op);
  CopyAllAttrs(quantized_conv_node_def, &fused_op);
  CopyAllAttrs(quantizev2_node_def, &fused_op);

  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  ITEX_VLOG(2) << "Fuse QuantizeV2, with QuantizedConv2D:"
               << " QuantizeV2=" << quantizev2_node_def.name()
               << " QuantizedConv2D=" << quantized_conv_node_def.name();

  (*invalidated_nodes)[matched.quantizedConv2DIndex_] = true;
  (*nodes_to_delete)[matched.quantizeV2Index_] = true;

  return Status::OK();
}

Status AddFusedAddV2WithSoftmaxNode(RemapperContext* ctx,
                                    const AddV2WithSoftmax& matched,
                                    std::vector<bool>* invalidated_nodes,
                                    std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();

  const NodeDef& addv2_node_def = graph->node(matched.addv2Index_);
  const NodeDef& softmax_node_def = graph->node(matched.softmaxIndex_);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  // create a new node
  NodeDef fused_op;
  fused_op.set_name(softmax_node_def.name());
  fused_op.set_device(softmax_node_def.device());
  fused_op.add_input(addv2_node_def.input(0));
  fused_op.add_input(addv2_node_def.input(1));
  fused_op.set_op(kAddV2WithSoftmax);

  CopyAllAttrs(addv2_node_def, &fused_op);

  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  ITEX_VLOG(2) << "Fuse AddV2, with Softmax: "
               << " AddV2= " << addv2_node_def.name()
               << " Softmax= " << softmax_node_def.name();

  (*invalidated_nodes)[matched.softmaxIndex_] = true;
  (*nodes_to_delete)[matched.addv2Index_] = true;

  return Status::OK();
}

Status AddConvBackpropInputWithSliceNode(
    RemapperContext* ctx, const ConvBackpropInputWithSlice& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& slice = graph->node(matched.slice);
  const NodeDef& contraction = graph->node(matched.contraction);

  NodeDef conv_backprop_input_with_slice;
  conv_backprop_input_with_slice.set_name(slice.name());
  conv_backprop_input_with_slice.set_device(contraction.device());
  conv_backprop_input_with_slice.add_input(contraction.input(0));
  conv_backprop_input_with_slice.add_input(contraction.input(1));
  conv_backprop_input_with_slice.add_input(contraction.input(2));

  conv_backprop_input_with_slice.add_input(slice.input(1));
  conv_backprop_input_with_slice.add_input(slice.input(2));

  if (IsConv2DBackpropInput(contraction)) {
    conv_backprop_input_with_slice.set_op(kConv2DBackpropInputWithSlice);
  } else if (IsConv3DBackpropInputV2(contraction)) {
    conv_backprop_input_with_slice.set_op(kConv3DBackpropInputWithSlice);
  } else {
    ITEX_CHECK(false) << "Unsupported fusion";
  }

  CopyAllAttrs(contraction, &conv_backprop_input_with_slice);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(conv_backprop_input_with_slice), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.slice] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return Status::OK();
}

// Mul + AddN + TrainingOp.
Status AddFusedTrainingNode(RemapperContext* ctx,
                            const FusedTrainingOp& matched,
                            std::vector<bool>* invalidated_nodes,
                            std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& mul = graph->node(matched.mul);
  const NodeDef& training_op = graph->node(matched.training_op);

  ITEX_VLOG(2) << "Fuse Mul, AddN with TrainingOp:"
               << " Mul=" << mul.name() << " TrainingOp=" << training_op.name();

  NodeDef fused_op;
  fused_op.set_name(training_op.name());
  fused_op.set_device(training_op.device());
  if (IsApplyAdam(training_op)) {
    fused_op.set_op(kFusedApplyAdam);
    for (int i = 0; i < 9; i++) fused_op.add_input(training_op.input(i));
  } else if (IsResourceApplyAdam(training_op)) {
    fused_op.set_op(kFusedResourceApplyAdam);
    for (int i = 0; i < 9; i++) fused_op.add_input(training_op.input(i));
  } else if (IsApplyAdamWithWeightDecay(training_op)) {
    fused_op.set_op(kFusedApplyAdamWithWeightDecay);
    for (int i = 0; i < 10; i++) fused_op.add_input(training_op.input(i));
  } else if (IsResourceApplyAdamWithWeightDecay(training_op)) {
    fused_op.set_op(kFusedResourceApplyAdamWithWeightDecay);
    for (int i = 0; i < 10; i++) fused_op.add_input(training_op.input(i));
  } else if (IsApplyMomentum(training_op)) {
    fused_op.set_op(kFusedApplyMomentum);
    fused_op.add_input(training_op.input(0));
    fused_op.add_input(training_op.input(1));
    fused_op.add_input(training_op.input(2));
    fused_op.add_input(training_op.input(4));
  } else if (IsResourceApplyMomentum(training_op)) {
    fused_op.set_op(kFusedResourceApplyMomentum);
    fused_op.add_input(training_op.input(0));
    fused_op.add_input(training_op.input(1));
    fused_op.add_input(training_op.input(2));
    fused_op.add_input(training_op.input(4));
  } else {
    ITEX_CHECK(false);
  }

  auto* attrs = fused_op.mutable_attr();
  if (matched.addn != -1) {
    if (matched.mul_scalar_input == -1) {
      fused_op.add_input(mul.input(0));
      fused_op.add_input(mul.input(1));
      SetAttrValue(2, &(*attrs)["num_mul_inputs"]);
    } else {
      fused_op.add_input(mul.input(matched.mul_scalar_input));
      SetAttrValue(1, &(*attrs)["num_mul_inputs"]);
    }
    const NodeDef& addn = graph->node(matched.addn);
    fused_op.add_input(addn.input(1 - matched.mul_port));
  } else {
    fused_op.add_input(mul.input(0));
    fused_op.add_input(mul.input(1));
    // TODO(itex): check do we need to add "num_mul_inputs" for all training
    // op fusion
  }

  if (matched.addn == kMissingIndex) {
    SetAttrValue(0, &(*attrs)["num_addn_inputs"]);
    SetAttrValue(absl::Span<const absl::string_view>{{"Mul"}},
                 &(*attrs)["fused_ops"]);

  } else {
    SetAttrValue(1, &(*attrs)["num_addn_inputs"]);
    SetAttrValue(absl::Span<const absl::string_view>{{"Mul", "AddN"}},
                 &(*attrs)["fused_ops"]);
  }
  CopyAllAttrs(training_op, &fused_op);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.training_op] = true;
  (*nodes_to_delete)[matched.mul] = true;
  if (matched.addn != -1) {
    (*nodes_to_delete)[matched.addn] = true;
  }

  return Status::OK();
}

// Bf16FusedMatMulGrad + CastFP32
Status AddFusedContractionGradWithCastNode(
    RemapperContext* ctx, const Bf16ContractionGradWithCastFp32& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  ITEX_DCHECK(IsDeviceCompatible(*ctx, matched))
      << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_cast = graph->node(matched.bias_cast);

  ITEX_VLOG(2) << "Fuse " << contraction.op() << " with Cast: "
               << " fused_matmul_grad=" << contraction.name()
               << " cast=" << bias_cast.name();

  NodeDef fused_op;
  fused_op.set_name(contraction.name());
  fused_op.set_device(contraction.device());
  if (IsFusedMatmulGrad(contraction)) {
    fused_op.set_op(kFusedAccMatMulGrad);
  } else {
    ITEX_CHECK(false);
  }
  CopyAllAttrs(contraction, &fused_op);
  fused_op.add_input(contraction.input(0));
  fused_op.add_input(contraction.input(1));

  std::vector<NodeDef> bias_cast_outs;
  bias_cast_outs.resize(matched.bias_cast_outs.size());
  for (size_t i = 0; i < matched.bias_cast_outs.size(); ++i) {
    const NodeDef& out_i = graph->node(matched.bias_cast_outs[i]);
    bias_cast_outs[i].set_name(out_i.name());
    bias_cast_outs[i].set_device(out_i.device());
    bias_cast_outs[i].set_op(out_i.op());
    for (int j = 0; j < out_i.input_size(); ++j) {
      auto out_i_input = out_i.input(j);
      if (out_i_input == bias_cast.name())
        bias_cast_outs[i].add_input(bias_cast.input(0));
      else
        bias_cast_outs[i].add_input(out_i_input);
    }
    CopyAllAttrs(out_i, &bias_cast_outs[i]);
  }

  AddNodeAttr("Tgrad", bias_cast.attr().at("DstT"), &fused_op);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  for (size_t i = 0; i < matched.bias_cast_outs.size(); ++i) {
    mutation->AddNode(std::move(bias_cast_outs[i]), &status);
  }
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_cast] = true;
  for (size_t i = 0; i < matched.bias_cast_outs.size(); ++i) {
    (*invalidated_nodes)[matched.bias_cast_outs[i]] = true;
  }

  return Status::OK();
}

// Bf16(Fused)Matmul op + castFp32
Status AddBf16ContractionWithCastFp32Node(
    RemapperContext* ctx, const Bf16ContractionWithCastFp32& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& cast = graph->node(matched.cast);

  ITEX_VLOG(2) << "Fuse " << cast.op() << " with Bf16(Fused)Matmul:"
               << " cast=" << cast.name() << " invalidated="
               << " (Fused)matmul=" << contraction.name();

  // Replace matmul and Cast with _ITEX(Fused)AccMatMul.
  NodeDef fused_op;
  if (IsMatMul(contraction)) {
    fused_op.set_op(kAccMatMul);
  } else if (IsFusedMatmulWithSum(contraction)) {
    fused_op.set_op(kFusedAccMatMulWithSum);
  } else {
    fused_op.set_op(kFusedAccMatMul);
  }
  fused_op.set_name(cast.name());
  fused_op.set_device(contraction.device());
  CopyAllAttrs(contraction, &fused_op);
  fused_op.add_input(contraction.input(0));
  fused_op.add_input(contraction.input(1));
  int num = 0;
  TryGetNodeAttr(contraction, "num_args", &num);
  for (int i = 2; i < num + 2; i++) {
    fused_op.add_input(contraction.input(i));
  }

  auto* fused_op_attr = fused_op.mutable_attr();
  auto& cast_attr = cast.attr();

  (*fused_op_attr)["Tout"] = cast_attr.at("DstT");
  if (IsMatMul(contraction)) {
    (*fused_op_attr)["Tpost"] = cast_attr.at("DstT");
  } else {
    (*fused_op_attr)["Tpost"] = cast_attr.at("SrcT");
  }
  if (IsFusedMatmulWithSum(contraction)) {
    SetAttrValue(false, &(*fused_op_attr)["inplace_sum"]);
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*invalidated_nodes)[matched.cast] = true;

  return Status::OK();
}

// Comparison op + cast
Status AddComparisonWithCastNode(RemapperContext* ctx,
                                 const ComparisonWithCast& matched,
                                 std::vector<bool>* invalidated_nodes,
                                 std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& comparison = graph->node(matched.comparison);
  const NodeDef& cast = graph->node(matched.cast);

  ITEX_VLOG(2) << "Fuse " << cast.op() << " with comparison:"
               << " cast=" << cast.name() << " invalidated="
               << " comparison=" << comparison.name();

  // Replace Comparison and Cast with ComparisonWithCast.
  NodeDef fused_op;
  fused_op.set_op(matched.fused_op);
  fused_op.set_name(cast.name());
  fused_op.set_device(comparison.device());

  fused_op.add_input(comparison.input(0));
  fused_op.add_input(comparison.input(1));
  (*fused_op.mutable_attr())["T"] = comparison.attr().at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.comparison] = true;
  (*invalidated_nodes)[matched.cast] = true;

  return Status::OK();
}

// Random op + Comparison + cast
Status AddRandomWithComparisonAndCastNode(
    RemapperContext* ctx, const RandomWithComparisonAndCast& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& random = graph->node(matched.random);
  const NodeDef& comparison = graph->node(matched.comparison);
  const NodeDef& cast = graph->node(matched.cast);

  ITEX_VLOG(2) << "Fuse " << cast.op()
               << " and " + comparison.op() + " with " + random.op() + " to "
               << kFusedRandom << ": "
               << " cast=" << cast.name() << " invalidated="
               << " comparison=" << comparison.name()
               << " random=" << random.name();

  // Replace Comparison and Cast with RandomWithComparisonAndCast.
  NodeDef fused_op;
  fused_op.set_op(kFusedRandom);
  fused_op.set_name(cast.name());
  fused_op.set_device(comparison.device());

  fused_op.add_input(random.input(0));
  fused_op.add_input(comparison.input(1 - matched.direction));
  auto* attrs = fused_op.mutable_attr();

  // Random input take shape and generate value with output data type.
  (*attrs)["T"] = random.attr().at("T");
  (*attrs)["DstT"] = comparison.attr().at("T");

  (*attrs)["seed"] = random.attr().at("seed");
  (*attrs)["seed2"] = random.attr().at("seed2");
  SetAttrValue(matched.direction, &(*attrs)["direction"]);
  SetAttrValue(
      absl::Span<const absl::string_view>{
          {random.op(), comparison.op(), cast.op()}},
      &(*attrs)["fused_ops"]);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.random] = true;
  (*nodes_to_delete)[matched.comparison] = true;
  (*invalidated_nodes)[matched.cast] = true;

  return Status::OK();
}

inline bool VerifyConstants(RemapperContext* ctx,
                            std::map<string, int>* nodes_map,
                            std::map<string, float>* values_map) {
  using utils::MutableNodeView;
  for (auto it = values_map->begin(); it != values_map->end(); ++it) {
    int node_idx = nodes_map->at(it->first);
    MutableNodeView* node_view = ctx->graph_view.GetNode(node_idx);
    NodeDef* node_def = node_view->node();
    Tensor const_tensor;
    if (node_def != nullptr && node_def->op() == "Const" &&
        const_tensor.FromProto(node_def->attr().at("value").tensor())) {
      if (const_tensor.NumElements() == 1) {
        DataType dtype = const_tensor.dtype();
        if (!(dtype == DT_FLOAT || dtype == DT_BFLOAT16 || dtype == DT_HALF))
          return false;
        // TODO(itex): A workaround for GPU with FP16 data type.
        if (dtype == DT_HALF && NodeIsOnCpu(node_def)) return false;
        auto const_value = (dtype == DT_FLOAT)
                               ? const_tensor.flat<float>()(0)
                               : const_tensor.flat<Eigen::bfloat16>()(0);
        // To compare float.
        if (std::abs(const_value - it->second) > 1e-2f) return false;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

// Gelu in python api generates a number of nodes in the graph. Depending on the
// parmeter `approximate={True/False}` different types of ops are generated. We
// distinguish them as `GeluExact` that uses Erf and `GeluApproximate` that
// uses Tanh.
bool FindMatMulBiasAddAndGelu(RemapperContext* ctx, int node_index,
                              std::map<string, int>* matched_nodes_map,
                              std::set<int>* remove_node_indices,
                              bool* is_gelu_approximate) {
  using utils::MatchingDirection;
  using utils::NodeStatus;
  // clang-format off
  utils::OpTypePattern gelu_exact_pattern =
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"AddV2", "erf_plus_one", NodeStatus::kRemove,
              {
                {"Erf", "erf", NodeStatus::kRemove,
                  {
                    {"Mul", "bias_add_times_square_root_one_half", NodeStatus::kRemove,  // NOLINT(whitespace/line_length)
                      {
                        {"BiasAdd", "bias_add", NodeStatus::kRemove},
                        {"Const", "square_root_one_half", NodeStatus::kRemain}
                      }
                    }
                  }
                },
                {"Const", "one", NodeStatus::kRemain}
              }
            },
            {"Const", "one_half", NodeStatus::kRemain}
          }
        },
        {"BiasAdd", "bias_add", NodeStatus::kRemove,
          {
            {"MatMul", "matmul", NodeStatus::kRemove},
            {"*", "bias", NodeStatus::kRemain}
          }
        }
      }
    };

  utils::OpTypePattern gelu_approximate_pattern =
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"AddV2", "tanh_plus_one", NodeStatus::kRemove,
              {
                {"Tanh", "tanh", NodeStatus::kRemove,
                  {
                    {"Mul", "matmul_plus_mul_times_square_root_two_over_pi", NodeStatus::kRemove,  // NOLINT(whitespace/line_length)
                      {
                        {"AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                          {
                            {"BiasAdd", "bias_add", NodeStatus::kRemove},
                            {"Mul", "mul", NodeStatus::kRemove,
                              {
                                {"Pow", "pow", NodeStatus::kRemove,
                                  {
                                    {"BiasAdd", "bias_add", NodeStatus::kRemove},  // NOLINT(whitespace/line_length)
                                    {"Const", "exponent", NodeStatus::kRemain}
                                  }
                                },
                                {"Const", "coeff", NodeStatus::kRemain}
                              }
                            }
                          }
                        },
                        {"Const", "square_root_two_over_pi", NodeStatus::kRemain}  // NOLINT(whitespace/line_length)
                      }
                    }
                  }
                },
                {"Const", "one", NodeStatus::kRemain}
              }
            },
            {"Const", "one_half", NodeStatus::kRemain}
          }
        },
        {"BiasAdd", "bias_add", NodeStatus::kRemove,
         {
           {"MatMul", "matmul", NodeStatus::kRemove},
           {"*", "bias", NodeStatus::kRemain}
         }
        }
      }
    };


    // Gelu approximate uses Pow(x, 3) which is optimized by arithmetic
    // optimizer as Mul(x, Square(x)) with an arifact of control dependency.
    // So we try to match pattern at second pass of remapper which reccieves
    // _FusedMatMul(MatMul + BiasAdd) with control dependency removed. This
    // is enabled only on CPU.
    utils::OpTypePattern gelu_approximate_pattern_on_cpu =
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"AddV2", "tanh_plus_one", NodeStatus::kRemove,
              {
                {"Tanh", "tanh", NodeStatus::kRemove,
                  {
                    {"Mul", "matmul_plus_mul_times_square_root_two_over_pi", NodeStatus::kRemove,  // NOLINT(whitespace/line_length)
                      {
                        {"AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                          {
                            {"BiasAdd", "bias_add", NodeStatus::kRemove},
                            {"Mul", "empirical_const_times_matmul", NodeStatus::kRemove,  // NOLINT(whitespace/line_length)
                              {
                                {"Mul", "mul", NodeStatus::kRemove,
                                 {
                                   {"Square", "square", NodeStatus::kRemove,
                                    {
                                      {"BiasAdd", "bias_add", NodeStatus::kRemove}  // NOLINT(whitespace/line_length)
                                    }
                                   },
                                   {"BiasAdd", "bias_add", NodeStatus::kRemove}
                                 }
                                },
                                {"Const", "empirical_const", NodeStatus::kRemain}  // NOLINT(whitespace/line_length)
                              }
                            }
                          }
                        },
                        {"Const", "square_root_two_over_pi", NodeStatus::kRemain}  // NOLINT(whitespace/line_length)
                      }
                    }
                  }
                },
                {"Const", "one", NodeStatus::kRemain}
              }
            },
            {"Const", "one_half", NodeStatus::kRemain}
          }
        },
        {"BiasAdd", "bias_add", NodeStatus::kRemove,
         {
           {"MatMul", "matmul", NodeStatus::kRemove},
           {"*", "bias", NodeStatus::kRemain}
         }
        }
      }
    };
  // clang-format on
  bool found_gelu_exact = false;
  bool found_gelu_approximate = false;
  bool found_gelu_approximate_on_cpu = false;
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  // Find GeluExact
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_gelu_exact =
      graph_matcher.GetMatchedNodes(gelu_exact_pattern, ctx->nodes_to_preserve,
                                    ctx->graph_view.GetNode(node_index),
                                    matched_nodes_map, remove_node_indices);
  // Find GeluApproximate
  if (!found_gelu_exact) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_gelu_approximate = graph_matcher.GetMatchedNodes(
        gelu_approximate_pattern, ctx->nodes_to_preserve,
        ctx->graph_view.GetNode(node_index), matched_nodes_map,
        remove_node_indices);
  }

  if (!found_gelu_exact && !found_gelu_approximate) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_gelu_approximate_on_cpu = graph_matcher.GetMatchedNodes(
        gelu_approximate_pattern_on_cpu, ctx->nodes_to_preserve,
        ctx->graph_view.GetNode(node_index), matched_nodes_map,
        remove_node_indices);
  }

  // Pattern matcher does subgraph matching based on op types only. The matcher
  // also does a sanity check on nodes tagged as `kRemove`, i.e., they do not
  // have any consumer outside the matched nodes. In order to replace the
  // subgraph, we need additional checks, for example, if the key ops have been
  // placed on CPU, desired data type, const has desired value etc. For the
  // following fusion: MatMul + BiasAdd + Gelu (disintegrated into smaller
  // ops), we check if (i) MatMul op is CpuCompatible, (ii) const nodes have
  // desired values.
  if (found_gelu_exact) {
    std::map<string, float> values_map = {
        {"square_root_one_half", 0.707106}, {"one", 1.0}, {"one_half", 0.5}};
    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else if (found_gelu_approximate) {
    std::map<string, float> values_map = {{"square_root_two_over_pi", 0.797884},
                                          {"one", 1.0},
                                          {"one_half", 0.5},
                                          {"exponent", 3}};
    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else if (found_gelu_approximate_on_cpu) {
    std::map<string, float> values_map = {{"square_root_two_over_pi", 0.797884},
                                          {"one", 1.0},
                                          {"one_half", 0.5},
                                          {"empirical_const", 0.044715}};
    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else {
    return false;
  }
  *is_gelu_approximate =
      (found_gelu_approximate || found_gelu_approximate_on_cpu) ? true : false;
  return (found_gelu_exact || found_gelu_approximate ||
          found_gelu_approximate_on_cpu);
}

Status AddFusedMatMulBiasAddAndGelu(RemapperContext* ctx,
                                    std::map<string, int>* matched_nodes_map,
                                    std::set<int>* remove_node_indices,
                                    std::vector<bool>* invalidated_nodes,
                                    std::vector<bool>* nodes_to_delete,
                                    bool is_gelu_approximate) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("output"))->node();
  auto* matmul_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("matmul"))->node();

  NodeDef fused_node;
  // Fused node should have the name of terminal node of the fusion.
  fused_node.set_name(output_node->name());
  fused_node.set_op(kFusedMatMul);
  fused_node.set_device(matmul_node->device());
  fused_node.add_input(matmul_node->input(0));
  fused_node.add_input(matmul_node->input(1));
  auto* bias_add_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("bias_add"))->node();
  fused_node.add_input(bias_add_node->input(1));

  CopyAllAttrs(*matmul_node, &fused_node);
  if (is_gelu_approximate)
    SetFusedOpAttributes(&fused_node, {"BiasAdd", "GeluApproximate"});
  else
    SetFusedOpAttributes(&fused_node, {"BiasAdd", "GeluExact"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map->at("output")] = true;

  for (const auto& node_idx : *remove_node_indices) {
    if (node_idx < nodes_to_delete->size()) (*nodes_to_delete)[node_idx] = true;
  }

  return Status::OK();
}

// Add Mul + Maximum fusion.
Status AddMulWithMaximumNode(RemapperContext* ctx,
                             const MulWithMaximum& matched,
                             std::vector<bool>* invalidated_nodes,
                             std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& maximum = graph->node(matched.maximum);
  const NodeDef& input = graph->node(matched.input);

  DataType dst_dtype = GetDataTypeFromAttr(maximum, "T");

  // Add LeakyRelu op.
  NodeDef leakyrelu_op;
  leakyrelu_op.set_op(kLeakyRelu);
  leakyrelu_op.set_name(maximum.name());
  leakyrelu_op.set_device(maximum.device());
  leakyrelu_op.add_input(input.name());

  auto* attr = leakyrelu_op.mutable_attr();
  SetAttrValue(matched.alpha, &(*attr)["alpha"]);
  SetAttrValue(dst_dtype, &(*attr)["T"]);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(leakyrelu_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.mul] = true;
  (*invalidated_nodes)[matched.maximum] = true;
  return Status::OK();
}

// Add Const + cast fusion.
Status AddConstWithCastNode(RemapperContext* ctx, const ConstWithCast& matched,
                            std::vector<bool>* invalidated_nodes,
                            std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& constant = graph->node(matched.constant);
  const NodeDef& cast = graph->node(matched.cast);

  // Replace Const and Cast with Const.
  TF_RETURN_IF_ERROR(CheckAttrExists(constant, "value"));

  DataType dst_dtype = GetDataTypeFromAttr(cast, "DstT");

  const TensorProto& raw_val = constant.attr().at("value").tensor();
  Tensor value = Tensor(raw_val.dtype(), raw_val.tensor_shape());
  value.FromProto(raw_val);

  const Eigen::ThreadPoolDevice d =
      OpKernelContext::eigen_cpu_device_singleton();
  Tensor cast_value = Tensor(dst_dtype, raw_val.tensor_shape());
  if (dst_dtype == DT_BFLOAT16) {
    cast_value.flat<Eigen::bfloat16>().device(d) =
        value.flat<float>().template cast<Eigen::bfloat16>();
  } else if (dst_dtype == DT_HALF) {
    cast_value.flat<Eigen::half>().device(d) =
        value.flat<float>().template cast<Eigen::half>();
  } else {
    return errors::InvalidArgument(
        "Const + Cast fusion only support Const(fp32) + "
        "Cast(float->bfloat16/half) pattern.");
  }

  // Add new Const op.
  NodeDef new_const_op;
  new_const_op.set_op("Const");
  new_const_op.set_name(cast.name());
  new_const_op.set_device(constant.device());

  AttrValue attr_type;
  attr_type.set_type(dst_dtype);
  AttrValue attr_tensor;
  TensorProto* t = attr_tensor.mutable_tensor();
  cast_value.AsProtoTensorContent(t);
  new_const_op.mutable_attr()->insert({"dtype", attr_type});
  new_const_op.mutable_attr()->insert({"value", attr_tensor});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(new_const_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.constant] = true;
  (*invalidated_nodes)[matched.cast] = true;
  return Status::OK();
}

// Add sequatial Binary ops fusion.
Status AddFusedBinaryNode(RemapperContext* ctx, const FusedBinary& matched,
                          std::vector<bool>* invalidated_nodes,
                          std::vector<bool>* nodes_to_delete) {
  int node_index = matched.root_;
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& root_def = graph->node(node_index);

  ITEX_VLOG(2) << "Fuse " << root_def.op() << " with " << matched.num_ - 1
               << " sequatial Binary ops.";

  // Add new op.
  NodeDef new_node_def;
  new_node_def.set_op(kFusedBinary);
  new_node_def.set_name(root_def.name());
  new_node_def.set_device(root_def.device());

  // Add inputs to sequatial Binary fusion. Each Binary op only has 1 Binary
  // precursor. The precursor order is recorded in `input_order_`
  NodeDef* node_def = (ctx->graph_view.GetNode(node_index))->node();
  std::vector<string> fused_ops = {root_def.op()};
  ITEX_CHECK(matched.fused_ops_.size() == matched.input_order_.size());
  for (size_t i = 0; i < matched.fused_ops_.size(); ++i) {
    new_node_def.add_input(node_def->input(1 - matched.input_order_[i]));
    node_def = (ctx->graph_view.GetNode(matched.fused_ops_[i]))->node();
    fused_ops.push_back(node_def->op());
  }
  // Add the last 2 inputs.
  new_node_def.add_input(node_def->input(1));
  new_node_def.add_input(node_def->input(0));
  // Set attrs. There will be `N + 1` inputs with `N` Binary op fusion.
  CopyAllAttrs(root_def, &new_node_def);
  AddNodeAttr("num_args", matched.num_ + 1, &new_node_def);
  AddNodeAttr("fused_ops", fused_ops, &new_node_def);
  AddNodeAttr("input_order", matched.input_order_, &new_node_def);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(new_node_def), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.root_] = true;
  for (int index : matched.fused_ops_) {
    (*nodes_to_delete)[index] = true;
  }
  return Status::OK();
}

}  // namespace

// `is_full` is true by default. When oneDNN Graph is enabled, we want to set it
// as false. When is_full == false, we only fuse Const + Cast / instancenorm/
// layernorm.
// `level` means the order of current remapper pass. Simple fusions without any
// variant  will be checked under level 0 only.
Status RunRemapper(const char* device_name, const GrapplerItem& item,
                   const GraphDef& graph_def, GraphDef* optimized_graph,
                   bool is_full, int level) {
  const int default_level = 0;
  Status status;
  GraphDef multable_graph_def = graph_def;
  RemapperContext ctx(item, &multable_graph_def, &status, level);
  // TODO(itex): Currently some fusions will be disabled when LayoutOPT is off,
  //       remove this dependency once all plain fusions are supported.
  bool is_layout_opt = GetOptimizerConfigFlags().enable_layout_opt;

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  const int num_nodes = multable_graph_def.node_size();
  // Skip nodes that were invalidated by a remapper, e.g. do not process BiasAdd
  // and Activation nodes that were fused into a Conv2D node.
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);

  ITEX_VLOG(1) << "RemapperPass: Start to fuse nodes with LayoutOPT("
               << (is_layout_opt ? "ON" : "OFF") << ").";

  // _Fused{...} kernels do not have registered gradient function, so we must
  // not perform rewrite if the graph will be differentiated later.
  // bool allow_non_differentiable_rewrites =
  //     item.optimization_options().allow_non_differentiable_rewrites;

  // Maybe exist multiple patterns mapping to one key, so we need to sort it.
  // Currently we just based on the node number, which means, the more nodes,
  // the higher priority.
  FusionMgr::GetInstance().Sort();

  // Infer statically first and only once.
  ctx.GetGraphProperties();

  for (int i = num_nodes - 1; i >= 0; --i) {
    // Check if node was invalidated or deleted by one of the previous remaps.
    if (invalidated_nodes[i] || nodes_to_delete[i]) {
      continue;
    }

    // Don't fuse fetch node when layout is ON because layout won't rewrite it.
    if (IsInPreserveSet(ctx, (ctx.graph_view.GetNode(i))->node()) &&
        is_layout_opt) {
      NodeDef* node_def = (ctx.graph_view.GetNode(i))->node();
      ITEX_VLOG(3) << "The node is in preserve set " << node_def->op() << ":"
                   << node_def->name();
      continue;
    }

    // Check if node can run on current optimizer device.
    if (!NodeIsOnDevice(device_name, ctx.graph_view.GetNode(i)->node())) {
      NodeDef* node_def = (ctx.graph_view.GetNode(i))->node();
      ITEX_VLOG(3) << "The node " << node_def->op() << ":" << node_def->name()
                   << "is not at " << device_name;
      continue;
    }

    if (is_full) {
      // Remap Conv2D+BiasAdd+Activation+Add into the _ITEXFusedConv2D.
      ContractionWithBiasAndActivationAdd contract_with_bias_and_activation_add;
      if (FindContractionWithBiasAndActivationAdd(
              ctx, i, &contract_with_bias_and_activation_add)) {
        TF_ABORT_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_activation_add,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Conv2D+BiasAdd+Add+Activation into the _ITEXFusedConv2D.
      ContractionWithBiasAndAddActivation contract_with_bias_and_add_activation;
      if (FindContractionWithBiasAndAddActivation(
              ctx, i, &contract_with_bias_and_add_activation)) {
        TF_ABORT_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Conv2D+BiasAdd+Add into the _ITEXFusedConv2D.
      ContractionWithBiasAddAndAdd contract_with_bias_and_add;
      if (FindContractionWithBiasAddAndAdd(ctx, i,
                                           &contract_with_bias_and_add)) {
        TF_ABORT_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap MatMul + BiasAdd + gelu-subgraph
      std::map<string, int> matched_nodes_map;
      std::set<int> remove_node_indices;
      bool is_gelu_approximate = false;
      if (FindMatMulBiasAddAndGelu(&ctx, i, &matched_nodes_map,
                                   &remove_node_indices,
                                   &is_gelu_approximate)) {
        TF_ABORT_IF_ERROR(AddFusedMatMulBiasAddAndGelu(
            &ctx, &matched_nodes_map, &remove_node_indices, &invalidated_nodes,
            &nodes_to_delete, is_gelu_approximate));
        continue;
      }
    }
    // The entry of the fusion pass. It will iterate all fusion registered.
    TF_ABORT_IF_ERROR(LaunchPatternMatcher(&ctx, i, &invalidated_nodes,
                                           &nodes_to_delete, is_full));

    if (is_full) {
      // Remap {Conv2D,DepthwiseConv2D,Conv3D,MatMul}+BiasAdd into the
      // _ITEXFused{Conv2D,DepthwiseConv2dNative,Conv3D,MatMul}
      ContractionWithBiasAdd contract_with_bias;
      if (FindContractionWithBias(ctx, i, &contract_with_bias)) {
        TF_ABORT_IF_ERROR(AddFusedContractionNode(
            &ctx, contract_with_bias, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap MatMul+BiasAddGrad into the _fusedMatMulGrad
      ContractionWithBiasAddGrad contract_with_bias_grad;
      if (FindContractionWithBiasAddGrad(ctx, i, &contract_with_bias_grad)) {
        TF_ABORT_IF_ERROR(
            AddFusedContractionGradNode(&ctx, contract_with_bias_grad,
                                        &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap {Conv2DBackpropFilter,Conv3DBackpropFilter}+BiasAddGrad into
      // FusedContractionBackpropFiler.
      ContractionWithBiasAddGrad conv_contract_with_bias_grad;
      if (FindConvContractionWithBiasAddGrad(ctx, i,
                                             &conv_contract_with_bias_grad)) {
        TF_ABORT_IF_ERROR(
            AddFusedContractionGradNode(&ctx, conv_contract_with_bias_grad,
                                        &invalidated_nodes, &nodes_to_delete));
        continue;
      }
      // Remap {Conv2D,Conv3D,MatMul}+BiasAdd+Activation into
      // _ITEXFused{Conv2D,Conv3D,MatMul}.
      ContractionWithBiasAddAndActivation contract_with_bias_and_activation;
      if (FindContractionWithBiasAndActivation(
              ctx, i, &contract_with_bias_and_activation)) {
        TF_ABORT_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap FusedBatchNorm+<SideInput>+<Activation> into the
      // _FusedBatchNormEx.
      FusedBatchNormEx fused_batch_norm_ex;
      if (FindFusedBatchNormEx(ctx, i, &fused_batch_norm_ex)) {
        TF_ABORT_IF_ERROR(AddFusedBatchNormExNode(
            &ctx, fused_batch_norm_ex, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      FusedBatchNormGradEx fused_batch_norm_grad_ex;
      if (FindFusedBatchNormGradEx(ctx, i, &fused_batch_norm_grad_ex)) {
        TF_ABORT_IF_ERROR(
            AddFusedBatchNormGradExNode(&ctx, fused_batch_norm_grad_ex,
                                        &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Pad+{Conv2D, _ITEXFusedConv2D} into the _FusedPadConv2D.
      PadWithContraction pad_with_contract;
      if (FindPadWithContraction(ctx, i, &pad_with_contract)) {
        TF_ABORT_IF_ERROR(AddPadWithContractionNode(
            &ctx, pad_with_contract, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      ConvBackpropInputWithSlice conv_with_slice;
      if (FindConvBackpropInputWithSlice(ctx, i, &conv_with_slice)) {
        TF_ABORT_IF_ERROR(AddConvBackpropInputWithSliceNode(
            &ctx, conv_with_slice, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Mul + AddN + TrainingOp into the _FusedTrainingOp.
      FusedTrainingOp fused_training_op;
      if (level == default_level &&
          FindFusedTrainingOp(ctx, i, &fused_training_op)) {
        TF_ABORT_IF_ERROR(AddFusedTrainingNode(
            &ctx, fused_training_op, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap BatchMatMul+Mul into the _FusedBatchMatMul.
      ContractionWithMul contract_with_mul;
      if (FindContractionWithMul(ctx, i, &contract_with_mul)) {
        TF_ABORT_IF_ERROR(AddFusedContractionNode(
            &ctx, contract_with_mul, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // delete dequantize node if it finds dequantize_with_shape pattern
      DequantizeWithShape dequantize_with_shape;
      if (level == default_level &&
          FindDequantizeWithShape(ctx, i, &dequantize_with_shape)) {
        TF_ABORT_IF_ERROR(AddFusedDequantizeWithShape(
            &ctx, dequantize_with_shape, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // delete dequantize node if it finds dequantize_with_reshape pattern
      DequantizeWithReshape dequantize_with_reshape;
      if (is_layout_opt && level == default_level &&
          FindDequantizeWithReshape(ctx, i, &dequantize_with_reshape)) {
        TF_ABORT_IF_ERROR(AddFusedDequantizeWithReshape(
            &ctx, dequantize_with_reshape, &invalidated_nodes,
            &nodes_to_delete));
        continue;
      }

      // Remap QuantizeV2+QuantizedConv2D into the
      // _ITEXQuantizeV2WithQuantizedConv2D
      QuantizeV2WithQuantizedConv2D quantizev2_with_quantizedconv;
      if (is_layout_opt && FindQuantizeV2WithQuantizedConv2D(
                               ctx, i, &quantizev2_with_quantizedconv)) {
        TF_ABORT_IF_ERROR(AddQuantizeV2WithQuantizedConv2DNode(
            &ctx, quantizev2_with_quantizedconv, &invalidated_nodes,
            &nodes_to_delete));
        continue;
      }

      // Remap L2loss+AddN into the _FusedAddN
      FusedAddN fused_addn;
      if (level == default_level && FindFusedAddN(ctx, i, &fused_addn)) {
        TF_ABORT_IF_ERROR(AddFusedAddN(&ctx, fused_addn, &invalidated_nodes,
                                       &nodes_to_delete));
        continue;
      }

      AddV2WithSoftmax fused_addv2_with_softmax;
      if (level == default_level &&
          FindAddV2WithSoftmax(ctx, i, &fused_addv2_with_softmax)) {
        TF_ABORT_IF_ERROR(
            AddFusedAddV2WithSoftmaxNode(&ctx, fused_addv2_with_softmax,
                                         &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Bf16(Fused)Matmul+CastFp32 into the _ITEX(Fused)AccMatMul.
      Bf16ContractionWithCastFp32 contraction_with_cast;
      if (FindBf16ContractionWithCastFp32(ctx, i, &contraction_with_cast)) {
        TF_ABORT_IF_ERROR(AddBf16ContractionWithCastFp32Node(
            &ctx, contraction_with_cast, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Random Comparison+Cast into the RandomWithComparisonAndCast.
      RandomWithComparisonAndCast random_with_compare_and_cast;
      if (level == default_level &&
          FindRandomWithComparisonAndCast(ctx, i,
                                          &random_with_compare_and_cast)) {
        TF_ABORT_IF_ERROR(AddRandomWithComparisonAndCastNode(
            &ctx, random_with_compare_and_cast, &invalidated_nodes,
            &nodes_to_delete));
        continue;
      }

      // Remap Bf16FusedMatmulGrad+CastFp32 into the _ITEXFusedAccMatMulGrad.
      Bf16ContractionGradWithCastFp32 contraction_grad_with_cast;
      if (FindBf16ContractionGradWithCastFp32(ctx, i,
                                              &contraction_grad_with_cast)) {
        TF_ABORT_IF_ERROR(AddFusedContractionGradWithCastNode(
            &ctx, contraction_grad_with_cast, &invalidated_nodes,
            &nodes_to_delete));
        continue;
      }

      // Remap Comparison+Cast into the ComparisonWithCast.
      ComparisonWithCast comparison_with_cast;
      if (level == default_level &&
          FindComparisonWithCast(ctx, i, &comparison_with_cast)) {
        TF_ABORT_IF_ERROR(AddComparisonWithCastNode(
            &ctx, comparison_with_cast, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Mul+Max into the LeakyRelu.
      MulWithMaximum mul_with_maximum;
      if (level == default_level &&
          FindMulWithMaximum(ctx, i, &mul_with_maximum)) {
        TF_ABORT_IF_ERROR(AddMulWithMaximumNode(
            &ctx, mul_with_maximum, &invalidated_nodes, &nodes_to_delete));
        continue;
      }
    }

    // Remap Const+Cast into the Const. this fusion aims to reduce the number
    // of Cast which were produced by auto mixed precision.
    ConstWithCast const_with_cast;
    if (FindConstWithCast(ctx, i, &const_with_cast)) {
      TF_ABORT_IF_ERROR(AddConstWithCastNode(
          &ctx, const_with_cast, &invalidated_nodes, &nodes_to_delete));
      continue;
    }

    // Remap sequatial Binary ops into the _ITEXFusedBinary op.
    // Disable it in 1st remapper since it may break other high priority
    // fusions.
    FusedBinary seq_binary;
    if (level != default_level && FindFusedBinary(ctx, i, &seq_binary)) {
      TF_ABORT_IF_ERROR(AddFusedBinaryNode(&ctx, seq_binary, &invalidated_nodes,
                                           &nodes_to_delete));
    }
  }

  // Remove invalidated nodes.
  utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(ctx.graph_view.GetNode(i));
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());

  *optimized_graph = std::move(multable_graph_def);
  return Status::OK();
}

}  // namespace graph
}  // namespace itex
