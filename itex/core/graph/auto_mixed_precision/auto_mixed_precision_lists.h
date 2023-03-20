/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_LISTS_H_
#define ITEX_CORE_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_LISTS_H_

#include <string>

#include "itex/core/graph/config_util.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/gtl/flatset.h"
#include "itex/core/utils/protobuf/config.pb.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/str_util.h"

namespace itex {
namespace graph {

// Represents the four lists of ops: the allow list, infer list, deny list, and
// clear list. These lists determine which ops are converted to fp16/bf16
// (referred to as 'f16' for short) and which ops stay as fp32.
class AutoMixedPrecisionLists {
 public:
  virtual ~AutoMixedPrecisionLists() {}

  // Returns the set of ops that are considered numerically-safe (for execution
  // in f16), performance-critical, and can run in f16. These ops are always
  // converted to f16.
  virtual gtl::FlatSet<string> AllowList() = 0;
  // Returns the set of ops that can run in f16 and are considered numerically-
  // safe (for execution in f16), but which may be made unsafe by an upstream
  // denylist op.
  virtual gtl::FlatSet<string> InferList() = 0;
  // Returns the set of ops that are considered numerically-dangerous (i.e.,
  // unsafe for execution in f16) and whose effects may also be observed in
  // downstream nodes (e.g. for f16, in Exp -> Add, the Add is unsafe due to
  // the Exp).
  virtual gtl::FlatSet<string> DenyList() = 0;
  // Returns the set of ops that do not have numerically-significant effects
  // (i.e., they are always considered safe for execution in f16 precision), and
  // can run in f16.
  virtual gtl::FlatSet<string> ClearList() = 0;

 protected:
  // Adds or removes ops from list if certain environmental variables are set.

  static void UpdateList(const string& list_name, gtl::FlatSet<string>* list) {
    ITEX_CHECK(list_name == "ALLOWLIST" ||
               list_name == "INFERLIST" ||  // Crash OK.
               list_name == "DENYLIST" || list_name == "CLEARLIST");
    string add_env_var = "ITEX_AUTO_MIXED_PRECISION_" + list_name + "_ADD";
    string remove_env_var =
        "ITEX_AUTO_MIXED_PRECISION_" + list_name + "_REMOVE";
    string to_add, to_remove;

    auto cfg_ = itex::itex_get_config();

#define LIST_IS_NOT_NULL(list) \
  cfg_.graph_options().auto_mixed_precision_options().list()

    if (list_name == "ALLOWLIST") {
      to_add = LIST_IS_NOT_NULL(allowlist_add);
      to_remove = LIST_IS_NOT_NULL(allowlist_remove);
      if (to_add.empty()) {
        ITEX_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
      }
      if (to_remove.empty()) {
        ITEX_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
      }
    }
    if (list_name == "INFERLIST") {
      to_add = LIST_IS_NOT_NULL(inferlist_add);
      to_remove = LIST_IS_NOT_NULL(inferlist_remove);
      if (to_add.empty()) {
        ITEX_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
      }
      if (to_remove.empty()) {
        ITEX_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
      }
    }
    if (list_name == "CLEARLIST") {
      to_add = LIST_IS_NOT_NULL(clearlist_add);
      to_remove = LIST_IS_NOT_NULL(clearlist_remove);
      if (to_add.empty()) {
        ITEX_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
      }
      if (to_remove.empty()) {
        ITEX_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
      }
    }
    if (list_name == "DENYLIST") {
      to_add = LIST_IS_NOT_NULL(denylist_add);
      to_remove = LIST_IS_NOT_NULL(denylist_remove);
      if (to_add.empty()) {
        ITEX_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
      }
      if (to_remove.empty()) {
        ITEX_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
      }
    }

#undef LIST_IS_NOT_NULL

    for (const auto& x : str_util::Split(to_add, ",")) {
      list->insert(x);
    }
    for (const auto& x : str_util::Split(to_remove, ",")) {
      list->erase(x);
    }
  }

  // Subclasses should include these on the ClearList.
  static void AddTensorListOps(gtl::FlatSet<string>* list) {
    // Note: if a data structure op (such as TensorListPopBack) is added here,
    // IsTensorListReaderOp or IsTensorListWriterOp may need to be modified
    // LINT.IfChange
    constexpr const char* tensor_list_ops[] = {
        "TensorListConcat",
        "TensorListConcatLists",
        "TensorListConcatV2",
        "TensorListFromTensor",
        "TensorListGather",
        "TensorListGetItem",
        "TensorListPopBack",
        "TensorListPushBack",
        "TensorListPushBackBatch",
        "TensorListScatter",
        "TensorListScatterIntoExistingList",
        "TensorListScatterV2",
        "TensorListSetItem",
        "TensorListSplit",
        "TensorListStack"};
    // LINT.ThenChange(//tensorflow/core/grappler/optimizers/auto_mixed_precision.cc)
    for (auto op : tensor_list_ops) {
      list->insert(op);
    }
  }
  // TODO(Guangyong): Add the training ops to related list.
  // such as _FusedApplyAdam or _FusedApplyMomentum.
  // The default Allow list of FP16 and BF16.
  gtl::FlatSet<string> allow_list_ops = gtl::FlatSet<string>{
      "BatchMatMul",
      "BatchMatMulV2",
      "Conv2D",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "Conv3D",
      "Conv3DBackpropFilter",
      "Conv3DBackpropFilterV2",
      "Conv3DBackpropInput",
      "Conv3DBackpropInputV2",
      "DepthwiseConv2dNative",
      "DepthwiseConv2dNativeBackpropFilter",
      "DepthwiseConv2dNativeBackpropInput",
      "MatMul",
      // TODO(hfang): The following ops is from Intel-TF DIEN ops.
      // Should be remove in future.
      "MklAUGRU",
      "MklGRU",
      // Below ops are fusion ops.
      "_ITEXConv2DBackpropFilterWithBias",
      "_ITEXConv2DBackpropInputWithSlice",
      "_ITEXConv3DBackpropFilterWithBias",
      "_ITEXConv3DBackpropInputV2WithSlice",
      "_ITEXForwardAUGRU",
      "_ITEXForwardGRU",
      "_ITEXFusedBatchMatMulV2",
      "_ITEXFusedConv2D",
      "_ITEXFusedConv2DWithSum",
      "_ITEXFusedConv3D",
      "_ITEXFusedDepthwiseConv2dNative",
      "_ITEXFusedMatMul",
      "_ITEXFusedMatMulGrad",
      "_ITEXFusedMatMulWithSum",
      "_ITEXPadWithConv2D",
      "_ITEXPadWithConv3D",
      "_ITEXPadWithFusedConv2D",
      "_ITEXPadWithFusedConv3D",
  };

  // The default Infer list of FP16 and BF16.
  gtl::FlatSet<string> infer_list_ops = gtl::FlatSet<string>{
      "Add",
      "AddN",
      "AddV2",
      "AvgPool",
      "AvgPool3D",
      "AvgPool3DGrad",
      "AvgPoolGrad",
      "BiasAdd",
      "BiasAddGrad",
      "BiasAddV1",
      "Elu",
      "EluGrad",
      "Erf",
      "Erfc",
      "FloorDiv",
      "FusedBatchNormV2",
      "FusedBatchNormGradV2",
      "FusedBatchNormV3",
      "FusedBatchNormGradV3",
      "Gelu",
      "GeluGrad",
      "Inv",
      "ITEXGelu",
      "ITEXGeluGrad"
      "ITEXLayerNorm",
      "LayerNorm",
      "LeakyRelu",
      "LeakyReluGrad",
      "Log",
      "Log1p",
      "LogSoftmax",
      "Mul",
      "Prod",
      "RealDiv",
      "Reciprocal",
      "Selu",
      "SeluGrad",
      "Sigmoid",
      "SigmoidGrad",
      "Softmax",
      "Softplus",
      "SoftplusGrad",
      "Softsign",
      "SoftsignGrad",
      "Sqrt",
      "Sub",
      "SwishGrad",
      "Tanh",
      "TanhGrad",
      "_FusedBatchNormEx",
      "_ITEXFusedBatchNormGradEx",
      "_ITEXFusedBinary",
      "_ITEXFusedInstanceNorm",
      "_ITEXInstanceNorm",
      "_ITEXMish",
      "_ITEXSwish",
  };

  // The default Deny list of FP16 and BF16.
  gtl::FlatSet<string> deny_list_ops = gtl::FlatSet<string>{
      "Exp",
      "Expm1",
      "L2Loss",
      "Mean",
      "Pow",
      "SaveV2",
      "SoftmaxCrossEntropyWithLogits",
      "SparseSoftmaxCrossEntropyWithLogits",
      "Sum",
  };

  // The default Clear list of FP16 and BF16.
  gtl::FlatSet<string> clear_list_ops = gtl::FlatSet<string>{
      "Abs",
      "ArgMax",
      "ArgMin",
      "BatchToSpace",
      "BatchToSpaceND",
      "BroadcastTo",
      "Ceil",
      "CheckNumerics",
      "ClipByValue",
      "Concat",
      "ConcatV2",
      "DepthToSpace",
      "DynamicPartition",
      "DynamicStitch",
      "EnsureShape",
      "Enter",
      "Equal",
      "Exit",
      "ExpandDims",
      "Fill",
      "Floor",
      "Gather",
      "GatherNd",
      "GatherV2",
      "Greater",
      "GreaterEqual",
      "Identity",
      "IdentityN",
      "IsFinite",
      "IsInf",
      "IsNan",
      "Less",
      "LessEqual",
      "Max",
      "MaxPool",
      "MaxPool3D",
      "MaxPool3DGrad",
      "MaxPool3DGradGrad",
      "MaxPoolGrad",
      "MaxPoolGradGrad",
      "MaxPoolGradGradV2",
      "MaxPoolGradV2",
      "MaxPoolV2",
      "Maximum",
      "Merge",
      "Min",
      "Minimum",
      "MirrorPad",
      "MirrorPadGrad",
      "Neg",
      "NextIteration",
      "NotEqual",
      "OneHot",
      "OnesLike",
      "Pack",
      "Pad",
      "PadV2",
      "PreventGradient",
      "Rank",
      "Relu",
      "Relu6",
      "Relu6Grad",
      "ReluGrad",
      "Reshape",
      "ResizeNearestNeighbor",
      "ResizeNearestNeighborGrad",
      "Reverse",
      "ReverseSequence",
      "ReverseV2",
      "Round",
      "Select",
      "SelectV2",
      "Shape",
      "ShapeN",
      "Sign",
      "Size",
      "Slice",
      "Snapshot",
      "SpaceToBatch",
      "SpaceToBatchND",
      "SpaceToDepth",
      "Split",
      "SplitV",
      "Squeeze",
      "StopGradient",
      "StridedSlice",
      "StridedSliceGrad",
      "Switch",
      "Tile",
      "TopK",
      "TopKV2",
      "Transpose",
      "Unpack",
      "Where",
      "ZerosLike",
  };
};

class AutoMixedPrecisionListsGPU : public AutoMixedPrecisionLists {
 public:
  AutoMixedPrecisionListsGPU() {}

  gtl::FlatSet<string> AllowList() override {
    // Add ops supported only by GPU devices.
    auto add_list_ops =
        gtl::FlatSet<string>{"Einsum", "_ITEXFusedAddV2WithSoftmax"};
    for (auto op : add_list_ops) {
      allow_list_ops.insert(op);
    }
    // Add some ops used in P0 models into allow list
    for (auto op : special_list_ops) {
      allow_list_ops.insert(op);
    }

    UpdateList("ALLOWLIST", &allow_list_ops);
    return allow_list_ops;
  }

  gtl::FlatSet<string> InferList() override {
    UpdateList("INFERLIST", &infer_list_ops);
    return infer_list_ops;
  }

  gtl::FlatSet<string> DenyList() override {
    auto add_list_ops = gtl::FlatSet<string>{
        "_FusedAddN",
    };
    for (auto op : add_list_ops) {
      deny_list_ops.insert(op);
    }

    UpdateList("DENYLIST", &deny_list_ops);
    return deny_list_ops;
  }

  gtl::FlatSet<string> ClearList() override {
    AddTensorListOps(&clear_list_ops);
    UpdateList("CLEARLIST", &clear_list_ops);

    for (auto op : special_list_ops) {
      clear_list_ops.erase(op);
    }

    return clear_list_ops;
  }

 private:
  // The special list for GPU FP16 only
  gtl::FlatSet<string> special_list_ops =
      gtl::FlatSet<string>{"Tile", "ConcatV2", "Reshape", "Transpose",
                           "Pack", "Unpack",   "Squeeze", "Slice"};
};

class AutoMixedPrecisionListsCPU : public AutoMixedPrecisionLists {
 public:
  AutoMixedPrecisionListsCPU() {}

  gtl::FlatSet<string> AllowList() override {
    UpdateList("ALLOWLIST", &allow_list_ops);
    return allow_list_ops;
  }

  gtl::FlatSet<string> InferList() override {
    UpdateList("INFERLIST", &infer_list_ops);
    return infer_list_ops;
  }

  gtl::FlatSet<string> DenyList() override {
    UpdateList("DENYLIST", &deny_list_ops);
    return deny_list_ops;
  }

  gtl::FlatSet<string> ClearList() override {
    AddTensorListOps(&clear_list_ops);
    UpdateList("CLEARLIST", &clear_list_ops);
    return clear_list_ops;
  }
};

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_LISTS_H_
