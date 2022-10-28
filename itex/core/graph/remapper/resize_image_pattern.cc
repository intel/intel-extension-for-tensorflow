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

#include <algorithm>

#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/remapper/constant_names.h"
#include "itex/core/graph/remapper/fusion.h"
#include "itex/core/graph/remapper/remapper.h"
#include "itex/core/graph/utils/pattern_utils.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/op_kernel.h"
namespace itex {
namespace graph {

// For the UpSampling3D, the Keras will use Concat + Split to
// implement forward. The backward comes from the auto gradient, which
// will be composed by SplitV + AddN + Concat or Slice + AddN +
// Concat. For the 3 types of pattern,

class ResizeNearestNeighborFusion : public Fusion {
 public:
  ResizeNearestNeighborFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;

    OpTypePattern dim1 = {kConst, "dim1", NodeStatus::kRemain};
    OpTypePattern value1 = {kAny, "input", NodeStatus::kRemain};
    OpTypePattern split1 = {kSplit, "split1", NodeStatus::kRemove};
    OpTypePattern concat1 = {kConcatV2, "concat1", NodeStatus::kRemove};

    OpTypePattern dim2 = {kConst, "dim2", NodeStatus::kRemain};
    OpTypePattern split2 = {kSplit, "split2", NodeStatus::kRemove};
    OpTypePattern concat2 = {kConcatV2, "concat2", NodeStatus::kRemove};

    OpTypePattern dim3 = {kConst, "dim3", NodeStatus::kRemain};
    OpTypePattern split3 = {kSplit, "split3", NodeStatus::kRemove};
    OpTypePattern concat3 = {kConcatV2, "concat3", NodeStatus::kReplace};

    split1.AddInput(dim1).AddInput(value1);
    concat1.AddNSameInput(split1).AddInput(dim1);

    split2.AddInput(dim2).AddInput(concat1);
    concat2.AddNSameInput(split2).AddInput(dim2);

    split3.AddInput(dim3).AddInput(concat2);
    concat3.AddNSameInput(split3).AddInput(dim3);

    pattern_ = InternalPattern(std::move(concat3));
  }

  ~ResizeNearestNeighborFusion() {}

  std::string Name() override { return "resize-nearest-neighbor"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    // TODO(itex): Currently this fusion will be disabled when LayoutOPT is off,
    //       remove this dependency once resize_3d plain fusion are supported.
    bool is_layout_opt = GetOptimizerConfigFlags().enable_layout_opt;
    if (!is_layout_opt) return ret.ToEmpty();

    auto& graph_view = ctx->graph_view;

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);

    if (ret.Empty()) return ret;

#define GET_AXIS(name, dim) GetAxis(ret.GetNode(&graph_view, name), dim)
#define RETURN_IF_ERROR(...)                                     \
  do {                                                           \
    ::itex::Status _status(__VA_ARGS__);                         \
    if (ITEX_PREDICT_FALSE(!_status.ok())) return ret.ToEmpty(); \
  } while (0)

    int dim1, dim2, dim3;
    RETURN_IF_ERROR(GET_AXIS("dim1", &dim1));
    RETURN_IF_ERROR(GET_AXIS("dim2", &dim2));
    RETURN_IF_ERROR(GET_AXIS("dim3", &dim3));
#undef RETURN_IF_ERROR
#undef GET_AXIS

    // Check whether is NDHWC format.
    if (dim1 == 1 && dim2 == 2 && dim3 == 3) {
      return ret;
    } else {
      return ret.ToEmpty();
    }
  }

  int GetSizeFactor(const utils::MutableNodeView* concat) const {
    return concat->NumRegularFanins() - 1;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* split1 = properties.GetNode(&graph_view, "split1");
    const NodeDef* concat1 = properties.GetNode(&graph_view, "concat1");
    const NodeDef* split2 = properties.GetNode(&graph_view, "split2");
    const NodeDef* concat2 = properties.GetNode(&graph_view, "concat2");
    const NodeDef* split3 = properties.GetNode(&graph_view, "split3");
    const NodeDef* concat3 = properties.GetNode(&graph_view, "concat3");

#define GET_SIZE_FACTOR(name) \
  GetSizeFactor(graph_view.GetNode(properties.map.at(name)))

    int size1 = GET_SIZE_FACTOR("concat1");
    int size2 = GET_SIZE_FACTOR("concat2");
    int size3 = GET_SIZE_FACTOR("concat3");
#undef GET_SIZE_FACTOR

    auto name =
        strings::StrCat(absl::StripSuffix(split1->name(), "/split"), "/sizes");
    auto device = properties.GetNode(&graph_view, "dim1")->device();

    NodeDef sizes_const_op;
    sizes_const_op.set_op("Const");
    sizes_const_op.set_name(name.data());
    sizes_const_op.set_device(device);

    AttrValue dtype;
    dtype.set_type(DT_INT32);
    AttrValue value;
    TensorProto* value_proto = value.mutable_tensor();
    Tensor value_tensor = Tensor(DT_INT32, TensorShape({3}));
    int32* value_tensor_ptr = static_cast<int32*>(value_tensor.data());
    value_tensor_ptr[0] = size1;
    value_tensor_ptr[1] = size2;
    value_tensor_ptr[2] = size3;
    value_tensor.AsProtoTensorContent(value_proto);
    sizes_const_op.mutable_attr()->insert({"dtype", dtype});
    sizes_const_op.mutable_attr()->insert({"value", value});

    NodeDef fused_op;
    fused_op.set_name(concat3->name());
    fused_op.set_op(kResizeNearestNeighbor);
    fused_op.set_device(concat3->device());
    fused_op.add_input(split1->input(1));
    fused_op.add_input(name);

    auto* attr = fused_op.mutable_attr();
    (*attr)["T"] = split1->attr().at("T");
    SetAttrValue(false, &(*attr)["align_corners"]);
    SetAttrValue(true, &(*attr)["half_pixel_centers"]);

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(sizes_const_op), &status);
    TF_RETURN_IF_ERROR(status);
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }

 private:
  Status GetAxis(const NodeDef* dim_node_def, int* axis) const {
    const TensorProto& raw = dim_node_def->attr().at("value").tensor();
    Tensor value = Tensor(raw.dtype(), raw.tensor_shape());
    value.FromProto(raw);

    switch (raw.dtype()) {
      case DT_INT32:
        *axis = value.scalar<int>()();
        break;
      case DT_INT64:  // Should no dim can't repr by int32.
        *axis = static_cast<int>(value.scalar<int64>()());
        break;
      default:
        return errors::InvalidArgument(
            "The dim const only supports int32 or int64");
    }

    return Status::OK();
  }
};

class ResizeNearestNeighborGradFusion : public Fusion {
 public:
  ResizeNearestNeighborGradFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;

    OpTypePattern value1 = {kAny, "value1", NodeStatus::kRemain};
    OpTypePattern size_splits1 = {kConst, "size_splits1", NodeStatus::kRemain};
    OpTypePattern split_dim1 = {kConst, "split_dim1", NodeStatus::kRemain};
    OpTypePattern split1 = {kSplitV, "split1", NodeStatus::kRemove};
    OpTypePattern addn1 = {kAddN, "addn1", NodeStatus::kRemove};
    OpTypePattern concat_dim1 = {kConst, "concat_dim1", NodeStatus::kRemain};
    OpTypePattern concat1 = {kConcatV2, "concat1", NodeStatus::kRemove};

    OpTypePattern size_splits2 = {kConst, "size_splits2", NodeStatus::kRemain};
    OpTypePattern split_dim2 = {kConst, "split_dim2", NodeStatus::kRemain};
    OpTypePattern split2 = {kSplitV, "split2", NodeStatus::kRemove};
    OpTypePattern addn2 = {kAddN, "addn2", NodeStatus::kRemove};
    OpTypePattern concat_dim2 = {kConst, "concat_dim2", NodeStatus::kRemain};
    OpTypePattern concat2 = {kConcatV2, "concat2", NodeStatus::kRemove};

    OpTypePattern size_splits3 = {kConst, "size_splits3", NodeStatus::kRemain};
    OpTypePattern split_dim3 = {kConst, "split_dim3", NodeStatus::kRemain};
    OpTypePattern split3 = {kSplitV, "split3", NodeStatus::kRemove};
    OpTypePattern addn3 = {kAddN, "addn3", NodeStatus::kRemove};
    OpTypePattern concat_dim3 = {kConst, "concat_dim3", NodeStatus::kRemain};
    OpTypePattern concat3 = {kConcatV2, "concat3", NodeStatus::kReplace};

    split1.AddInput(value1).AddInput(size_splits1).AddInput(split_dim1);
    addn1.AddNSameInput(split1);
    concat1.AddNDifferentInput(addn1).AddInput(concat_dim1);

    split2.AddInput(concat1).AddInput(size_splits2).AddInput(split_dim2);
    addn2.AddNSameInput(split2);
    concat2.AddNDifferentInput(addn2).AddInput(concat_dim2);

    split3.AddInput(concat2).AddInput(size_splits3).AddInput(split_dim3);
    addn3.AddNSameInput(split3);
    concat3.AddNDifferentInput(addn3).AddInput(concat_dim3);

    pattern_ = InternalPattern(std::move(concat3));
  }

  ~ResizeNearestNeighborGradFusion() {}

  std::string Name() override { return "resize-nearest-neighbor-grad"; }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;

    // TODO(itex): Currently this fusion will be disabled when LayoutOPT is off,
    //       remove this dependency once resize_3d plain fusion are supported.
    bool is_layout_opt = GetOptimizerConfigFlags().enable_layout_opt;
    if (!is_layout_opt) return ret.ToEmpty();

    auto& graph_view = ctx->graph_view;

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);

    return ret;
  }

  int GetSizeFactor(const utils::MutableNodeView* concat) const {
    return concat->NumRegularFanins() - 1;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    const NodeDef* split1 = properties.GetNode(&graph_view, "split1");
    const NodeDef* concat3 = properties.GetNode(&graph_view, "concat3");

#define GET_SIZE_FACTOR(name) \
  GetSizeFactor(graph_view.GetNode(properties.map.at(name)))

    int size1 = GET_SIZE_FACTOR("concat3");
    int size2 = GET_SIZE_FACTOR("concat2");
    int size3 = GET_SIZE_FACTOR("concat1");
#undef GET_SIZE_FACTOR

    auto name =
        strings::StrCat(absl::StripSuffix(split1->name(), "/split"), "/sizes");
    auto device = properties.GetNode(&graph_view, "split_dim1")->device();

    NodeDef sizes_const_op;
    sizes_const_op.set_op("Const");
    sizes_const_op.set_name(name.data());
    sizes_const_op.set_device(device);

    AttrValue dtype;
    dtype.set_type(DT_INT32);
    AttrValue value;
    TensorProto* value_proto = value.mutable_tensor();
    Tensor value_tensor = Tensor(DT_INT32, TensorShape({3}));
    int32* value_tensor_ptr = static_cast<int32*>(value_tensor.data());
    value_tensor_ptr[0] = size1;
    value_tensor_ptr[1] = size2;
    value_tensor_ptr[2] = size3;
    value_tensor.AsProtoTensorContent(value_proto);
    sizes_const_op.mutable_attr()->insert({"dtype", dtype});
    sizes_const_op.mutable_attr()->insert({"value", value});

    NodeDef fused_op;
    fused_op.set_name(concat3->name());
    fused_op.set_op(kResizeNearestNeighborGrad);
    fused_op.set_device(concat3->device());
    fused_op.add_input(split1->input(0));
    fused_op.add_input(name);

    auto* attr = fused_op.mutable_attr();
    (*attr)["T"] = split1->attr().at("T");
    SetAttrValue(false, &(*attr)["align_corners"]);
    SetAttrValue(true, &(*attr)["half_pixel_centers"]);

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(sizes_const_op), &status);
    TF_RETURN_IF_ERROR(status);
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    return Status::OK();
  }
};

static void GetTensorFromConst(const NodeDef* node, Tensor* tensor) {
  tensor->FromProto(node->attr().at("value").tensor());
}

class ResizeNearestNeighborGradFusionV2 : public Fusion {
 public:
  ResizeNearestNeighborGradFusionV2() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;

    OpTypePattern addn = {kAddN, "addn", NodeStatus::kRemove};
    OpTypePattern concat_dim = {kConst, "concat_dim", NodeStatus::kRemain};
    OpTypePattern concat = {kConcatV2, "concat", NodeStatus::kReplace};

    concat.AddNDifferentInput(addn).AddInput(concat_dim);

    pattern_ = InternalPattern(std::move(concat));
  }

  ~ResizeNearestNeighborGradFusionV2() {}

  std::string Name() override { return "resize-nearest-neighbor-grad-v2"; }

  Tensor GetConcatDim(utils::MutableGraphView* graph_view,
                      const MatchedProperties& properties) const {
    Tensor tensor;
    auto* concat_view = graph_view->GetNode(properties.map.at("concat"));
    const NodeDef* concat_dim =
        concat_view->GetRegularFanin(concat_view->NumRegularFanins() - 1)
            .node_view()
            ->node();
    GetTensorFromConst(concat_dim, &tensor);
    return tensor;
  }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;

    // TODO(itex): Currently this fusion will be disabled when LayoutOPT is off,
    //       remove this dependency once resize_3d plain fusion are supported.
    bool is_layout_opt = GetOptimizerConfigFlags().enable_layout_opt;
    if (!is_layout_opt) return ret.ToEmpty();

    auto& graph_view = ctx->graph_view;

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);
    if (ret.Empty()) {
      return ret;
    }

    Tensor dim_tensor = GetConcatDim(&graph_view, ret);
    if (dim_tensor.NumElements() != 1) {
      return ret.ToEmpty();
    }
    int dim = dim_tensor.scalar<int>()();

    // We need to check all Slice to AddN
    int num_inputs_to_addn = 0;
    int is_ok = true;
    std::vector<int> slice_dims;
    std::vector<int> slices;
    for (auto const& item : ret.map) {
      auto label = item.first;
      if (absl::StartsWith(label, "addn")) {
        auto index = item.second;
        auto* addn = graph_view.GetNode(index);
        if (num_inputs_to_addn == 0) {
          num_inputs_to_addn = addn->NumRegularFanins();
        }

        if (addn->NumRegularFanins() != num_inputs_to_addn) {
          is_ok = false;
          break;
        }

        for (int i = 0; i < addn->NumRegularFanins(); i++) {
          auto& input = addn->GetRegularFanin(i);
          auto& op = input.node_view()->node()->op();
          if (op != kSlice) {
            is_ok = false;
            break;
          }

          // Check Slice's input is 5D
          auto slice = input.node_view();
          auto begin = slice->GetRegularFanin(1).node_view()->node();
          auto size = slice->GetRegularFanin(2).node_view()->node();
          if (begin->op() == kSlice && size->op() == kSlice) {
            is_ok = false;
            break;
          }
          Tensor begin_tensor;
          GetTensorFromConst(begin, &begin_tensor);
          Tensor size_tensor;
          GetTensorFromConst(size, &size_tensor);
          if (size_tensor.vec<int>()(dim) != 1) {
            is_ok = false;
            break;
          }
          slices.push_back(slice->node_index());
          slice_dims.push_back(begin_tensor.vec<int>()(dim));
        }

        if (!is_ok) {
          break;
        }
      }
    }

    if (!is_ok) {
      return ret.ToEmpty();
    }

    std::vector<int> tmp = std::vector<int>(slice_dims);
    std::sort(tmp.begin(), tmp.end(),
              [](const int& lhs, const int& rhs) -> bool { return lhs < rhs; });
    for (int i = 0; i < tmp.size(); i++) {
      if (tmp[i] != i) {
        return ret.ToEmpty();
      }
    }

    for (auto const& slice_index : slices) {
      ret.deleted.insert(slice_index);
    }
    for (int i = 0; i < slice_dims.size(); i++) {
      std::string name = "slice" + std::to_string(i);
      ret.map.insert(std::pair<std::string, int>(name, slices[i]));
    }

    return ret;
  }

  int GetSizeFactor(const MatchedProperties& properties) const {
    int count = 0;
    for (auto const& item : properties.map) {
      auto label = item.first;
      if (absl::StartsWith(label, "addn")) {
        count++;
      }
    }

    return count;
  }

  const NodeDef* GetSliceInput(utils::MutableGraphView* graph_view,
                               const MatchedProperties& properties) const {
    for (auto const& item : properties.map) {
      auto label = item.first;
      if (absl::StartsWith(label, "slice")) {
        return graph_view->GetNode(item.second)->node();
      }
    }
    ITEX_VLOG(FATAL) << "Has not found the node stars with slice";
    return nullptr;
  }

  const NodeDef* GetSliceSize(utils::MutableGraphView* graph_view,
                              const MatchedProperties& properties) const {
    for (auto const& item : properties.map) {
      auto label = item.first;
      if (absl::StartsWith(label, "slice")) {
        auto* slice_view = graph_view->GetNode(item.second);
        auto* size_view = slice_view->GetRegularFanin(2).node_view();
        return size_view->node();
      }
    }
    ITEX_VLOG(FATAL) << "Has not found the node stars with slice";
    return nullptr;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;
    auto* concat_view = graph_view.GetNode(properties.map.at("concat"));
    int size = concat_view->NumRegularFanins() - 1;
    auto* concat = concat_view->node();
    auto* concat_dim =
        concat_view->GetRegularFanin(concat_view->NumRegularFanins() - 1)
            .node_view()
            ->node();
    auto name = strings::StrCat(concat->name(), "/sizes");
    auto device = concat_dim->device();
    int concat_dim_value;
    Tensor concat_dim_tensor;
    GetTensorFromConst(concat_dim, &concat_dim_tensor);
    concat_dim_value = concat_dim_tensor.scalar<int>()();

    const NodeDef* slice = GetSliceInput(&graph_view, properties);
    const NodeDef* slice_size = GetSliceSize(&graph_view, properties);
    Tensor size_tensor;
    GetTensorFromConst(slice_size, &size_tensor);

    NodeDef sizes_const_op;
    sizes_const_op.set_op("Const");
    sizes_const_op.set_name(name.data());
    sizes_const_op.set_device(device);

    AttrValue dtype;
    dtype.set_type(DT_INT32);
    AttrValue value;
    TensorProto* value_proto = value.mutable_tensor();
    Tensor value_tensor = Tensor(DT_INT32, TensorShape({3}));
    int32* value_tensor_ptr = static_cast<int32*>(value_tensor.data());
    value_tensor_ptr[0] = size_tensor.vec<int32>()(1);
    value_tensor_ptr[1] = size_tensor.vec<int32>()(2);
    value_tensor_ptr[2] = size_tensor.vec<int32>()(3);
    value_tensor_ptr[concat_dim_value - 1] = size;

    value_tensor.AsProtoTensorContent(value_proto);
    sizes_const_op.mutable_attr()->insert({"dtype", dtype});
    sizes_const_op.mutable_attr()->insert({"value", value});

    NodeDef fused_op;
    fused_op.set_name(concat->name());
    fused_op.set_op(kResizeNearestNeighborGrad);
    fused_op.set_device(concat->device());
    fused_op.add_input(slice->input(0));
    fused_op.add_input(name);

    auto* attr = fused_op.mutable_attr();
    (*attr)["T"] = concat->attr().at("T");
    SetAttrValue(false, &(*attr)["align_corners"]);
    SetAttrValue(true, &(*attr)["half_pixel_centers"]);

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(sizes_const_op), &status);
    TF_RETURN_IF_ERROR(status);
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    return Status::OK();
  }
};

class SqueezeResizeNearestNeighborGradFusion : public Fusion {
 public:
  SqueezeResizeNearestNeighborGradFusion() : Fusion() {
    using utils::NodeStatus;
    using utils::OpTypePattern;

    OpTypePattern size1 = {kConst, "size1", NodeStatus::kRemain};
    OpTypePattern value1 = {kAny, "input1", NodeStatus::kRemain};
    OpTypePattern resize1 = {kResizeNearestNeighborGrad, "resize1",
                             NodeStatus::kRemove};

    OpTypePattern size2 = {kConst, "size2", NodeStatus::kRemain};
    OpTypePattern resize2 = {kResizeNearestNeighborGrad, "resize2",
                             NodeStatus::kReplace};

    resize1.AddInput(value1).AddInput(size1);
    resize2.AddInput(resize1).AddInput(size2);

    pattern_ = InternalPattern(std::move(resize2));
  }

  ~SqueezeResizeNearestNeighborGradFusion() {}

  std::string Name() override { return "squeeze-resize-nearest-neighbor-grad"; }

  Tensor GetTensorFromConst(const NodeDef* node) const {
    Tensor tensor;
    tensor.FromProto(node->attr().at("value").tensor());
    return tensor;
  }

  Tensor GetTensorFromConst(utils::MutableGraphView* graph_view,
                            const MatchedProperties& properties,
                            const std::string name) const {
    auto* node = properties.GetNode(graph_view, name.c_str());
    return GetTensorFromConst(node);
  }

  MatchedProperties Check(RemapperContext* ctx,
                          const int node_index) const override {
    MatchedProperties ret;
    auto& graph_view = ctx->graph_view;

    ret = FillProperties(&graph_view, graph_view.GetNode(node_index), pattern_);

    if (ret.Empty()) return ret;

    // Check whether there's continuous 3 resize.
    auto* resize1_view = graph_view.GetNode(ret.map.at("resize1"));
    auto* resize0_view = resize1_view->GetRegularFanin(0).node_view();
    auto exist_resize0 =
        resize0_view->node()->op() == kResizeNearestNeighborGrad;
    if (exist_resize0) {
      auto* input_view = resize0_view->GetRegularFanin(0).node_view();
      auto* size0_view = resize0_view->GetRegularFanin(1).node_view();
      ret.deleted.insert(resize0_view->node_index());
      ret.deleted.insert(size0_view->node_index());
      ret.deleted.insert(
          resize1_view->node_index());  // Need to change to kRemove.

      ret.map.insert(
          std::pair<std::string, int>({"size0", size0_view->node_index()}));
      ret.map.insert(
          std::pair<std::string, int>({"resize0", resize0_view->node_index()}));
      ret.map.insert(
          std::pair<std::string, int>({"input0", input_view->node_index()}));
    }

    return ret;
  }

  Status Update(RemapperContext* ctx,
                const MatchedProperties& properties) const override {
    auto& graph_view = ctx->graph_view;

    bool exist_resize0 = properties.map.count("size0") != 0;
    auto size2 = properties.GetNode(&graph_view, "size2");
    auto resize2 = properties.GetNode(&graph_view, "resize2");
    std::string input;
    if (exist_resize0) {
      input = properties.GetNode(&graph_view, "input0")->name();
    } else {
      input = properties.GetNode(&graph_view, "input1")->name();
    }

    NodeDef fused_op;
    fused_op.set_name(resize2->name());
    fused_op.set_op(kResizeNearestNeighborGrad);
    fused_op.set_device(resize2->device());
    fused_op.add_input(input);
    fused_op.add_input(size2->name());

    auto* attr = fused_op.mutable_attr();
    (*attr)["T"] = resize2->attr().at("T");
    SetAttrValue(false, &(*attr)["align_corners"]);
    SetAttrValue(true, &(*attr)["half_pixel_centers"]);

    Status status;
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    return Status::OK();
  }
};

REGISTER_FUSION(ResizeNearestNeighborFusion)
REGISTER_FUSION(ResizeNearestNeighborGradFusion)
REGISTER_FUSION(ResizeNearestNeighborGradFusionV2)
REGISTER_FUSION(SqueezeResizeNearestNeighborGradFusion)
}  // namespace graph
}  // namespace itex
