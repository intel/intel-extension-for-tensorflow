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

#include "itex/core/graph/memory_opt_pass/weight_prepack.h"

#include <string>
#include <unordered_map>

#include "dnnl.hpp"  // NOLINT(build/include_subdir)
#include "google/protobuf/text_format.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/attr_value_util.h"
#include "itex/core/utils/node_def_util.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/types.h"

namespace itex {

using dnnl::memory;
using Eigen::bfloat16;
using Eigen::half;

namespace graph {

namespace {

const WeightPrePackInfo* GetWeightPrePackInfo(string op_name) {
  static const std::unordered_map<string, WeightPrePackInfo>
      weight_prepack_map = {
          {"_ITEXMatMul", {1, OneDnnTensorFormat::FORMAT_NC}},
          {"_ITEXFusedMatMul", {1, OneDnnTensorFormat::FORMAT_NC}}};

  auto iter = weight_prepack_map.find(op_name);

  if (iter == weight_prepack_map.end()) return nullptr;

  return &(iter->second);
}

}  // namespace

bool IsAttrExpected(const NodeDef* node_def, StringPiece attr_name,
                    bool expected_value) {
  bool attr_value = false;

  if (!HasNodeAttr(*node_def, attr_name)) return false;

  ITEX_CHECK_OK(GetNodeAttr(*node_def, attr_name, &attr_value));

  return attr_value == expected_value;
}

bool IsLegalComputeNode(const MutableNodeView* node_view) {
  auto* node_def = node_view->node();

  // TODO(itex): support GPU/XPU Device in future.
  // Disable weight cache opt on GPU/XPU
  if (!NodeIsOnCpu(node_def)) return false;

  if (!IsAttrExpected(node_def, "is_filter_const", true)) return false;

  // TODO(itex): support filter transpose in future.
  if (!IsAttrExpected(node_def, "transpose_b", false)) return false;

  return true;
}

bool IsLegalFilterNode(const MutableNodeView* node_view) {
  const auto* node_def = node_view->node();

  // TODO(itex): support other constant filters in future.
  if (!IsConstant(*node_def)) return false;

  // TODO(itex): support filter shared by multiple computation nodes in future.
  if (node_view->GetRegularFanout(0).size() != 1) return false;

  return true;
}

template <typename T>
memory::desc GetDstFormatTag(const dnnl::engine& engine,
                             const TensorShape& wei_shape) {
  const int m = 1024;
  const int k = wei_shape.dim_size(0);
  const int n = wei_shape.dim_size(1);

  TensorShape src_shape = {m, k};
  TensorShape dst_shape = {m, n};

  auto src_dims = TFShapeToOneDnnDims(src_shape);
  auto wei_dims = TFShapeToOneDnnDims(wei_shape);
  auto dst_dims = TFShapeToOneDnnDims(dst_shape);

  auto src_strides = CalculateTFStrides(src_dims);
  auto dst_strides = CalculateTFStrides(dst_dims);

  memory::desc src_md = memory::desc(src_dims, OneDnnType<T>(), src_strides);
  memory::desc wei_md =
      memory::desc(wei_dims, OneDnnType<T>(), memory::format_tag::any);
  memory::desc dst_md = memory::desc(dst_dims, OneDnnType<T>(), dst_strides);

  dnnl::primitive_desc matmul_pd =
      dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md);

  return matmul_pd.weights_desc();
}

void UpdateConstantNodeValue(const Tensor& src, NodeDef* node_def,
                             string attr_name = "value") {
  AttrValue attr_value;
  TensorProto* node_val = attr_value.mutable_tensor();
  src.AsProtoTensorContent(node_val);
  (*node_def->mutable_attr())[attr_name].mutable_tensor()->Swap(node_val);
}

void UpdateMetaData(NodeDef* node_def, OneDnnShape* prepack_shape) {
  Tensor meta_tensor(
      DT_UINT8,
      {static_cast<int64_t>(prepack_shape->GetSerializeBufferSize())});
  prepack_shape->SerializeOneDnnShape(
      meta_tensor.flat<uint8>().data(),
      meta_tensor.flat<uint8>().size() * sizeof(uint8));

  // Update meta attr
  UpdateConstantNodeValue(meta_tensor, node_def, "meta");
}

bool SetDefaultMetaData(NodeDef* node_def) {
  // Pre-pack has been done if the tensor has non-default `meta` info.
  if (node_def->attr().contains("meta")) {
    Tensor meta;
    OneDnnShape meta_shape;

    ITEX_CHECK_OK(GetTensorFromConstant(node_def, &meta, "meta"));
    meta_shape.DeSerializeOneDnnShape(
        meta.flat<uint8>().data(), meta.flat<uint8>().size() * sizeof(uint8));

    if (meta_shape.IsOneDnnTensor()) return false;
  }

  OneDnnShape default_shape;
  default_shape.SetOneDnnTensor(false);
  UpdateMetaData(node_def, &default_shape);

  return true;
}

template <typename T>
void ProcessFilterImpl(NodeDef* node_def, NodeDef* filter_node_def) {
  // TODO(itex): support GPU/XPU Device in future.
  dnnl::engine engine = GetCPUDnnlEngine();

  const WeightPrePackInfo* rinfo = GetWeightPrePackInfo(node_def->op());

  ITEX_CHECK_NE(rinfo, nullptr);

  Tensor filter;
  ITEX_CHECK_OK(GetTensorFromConstant(filter_node_def, &filter));

  TensorShape filter_shape = filter.shape();
  memory::dims filter_dims = TFShapeToOneDnnDims(filter_shape);

  memory::desc filter_md = CreatePlainMemDescWithFormatTag<T>(filter_dims);
  memory filter_mem =
      CreateDnnlMemory(filter_md, engine, GetTensorBuffer<T>(&filter));

  memory::desc filter_md_prefer = GetDstFormatTag<T>(engine, filter_shape);

  int64_t reorder_size = filter_md_prefer.get_size() / sizeof(T);
  Tensor reorder_tensor(DataTypeToEnum<T>::v(), {reorder_size});
  memory reorder_mem = CreateDnnlMemory(filter_md_prefer, engine,
                                        GetTensorBuffer<T>(&reorder_tensor));

  // Convert weights from plain to block
  dnnl::stream onednn_stream = dnnl::stream(engine);
  ReorderMemoryInternal(&filter_mem, &reorder_mem, onednn_stream);

  // Update filter node
  UpdateConstantNodeValue(reorder_tensor, filter_node_def);

  // Update meta data
  OneDnnShape prepack_shape;
  prepack_shape.SetOneDnnTensor(true);
  prepack_shape.SetOneDnnLayout(filter_md_prefer);
  prepack_shape.SetTfDataFormat(rinfo->onednn_format);
  UpdateMetaData(node_def, &prepack_shape);
}

#define MATCH_AND_EXECUTE(TYPE)                                               \
  case TYPE:                                                                  \
    ProcessFilterImpl<EnumToDataType<TYPE>::Type>(node_def, filter_node_def); \
    return;

// Currently, Weight Pre-Pack only supports fp32, bf16 and fp16.
void ProcessFilter(NodeDef* node_def, NodeDef* filter_node_def) {
  DataType dtype = GetDataTypeFromAttr(*filter_node_def, "dtype");

  switch (dtype) {
    MATCH_AND_EXECUTE(DT_FLOAT);
    MATCH_AND_EXECUTE(DT_BFLOAT16);
    MATCH_AND_EXECUTE(DT_HALF);
    default:
      ITEX_VLOG(2) << "Unsupported data type: " << DataTypeString(dtype);
  }
}

#undef MATCH_AND_EXECUTE

void WeightPrePack(const MutableNodeView* node_view) {
  auto* node_def = node_view->node();

  const WeightPrePackInfo* rinfo = GetWeightPrePackInfo(node_def->op());

  if (rinfo == nullptr) return;

  // Explicitly set default meta data if this attribute exists.
  // Skip this node if it's already pre-packed.
  if (!SetDefaultMetaData(node_def)) return;

  if (!IsLegalComputeNode(node_view)) return;

  auto* filter_node_view =
      node_view->GetRegularFanin(rinfo->filter_index).node_view();

  if (!IsLegalFilterNode(filter_node_view)) return;

  auto* filter_node_def = filter_node_view->node();

  ProcessFilter(node_def, filter_node_def);
}

}  // namespace graph
}  // namespace itex
