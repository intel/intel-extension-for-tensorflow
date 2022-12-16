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

#ifndef ITEX_CORE_GRAPH_UTILS_NODE_TYPE_ATTR_MAP_H_
#define ITEX_CORE_GRAPH_UTILS_NODE_TYPE_ATTR_MAP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "itex/core/utils/function.h"
#include "itex/core/utils/status.h"
#include "protos/graph.pb.h"

namespace itex {
namespace graph {

// Instances of this class represent unique type attribute identifiers within a
// node. It handles regular type attributes, list type attributes (where
// type_index is set to the index in the type list), and fixed types.
struct TypeAttrId {
  static constexpr int kSingleType = -1;

  explicit TypeAttrId(const string& _attr_name, int _type_index = kSingleType)
      : attr_name(_attr_name),
        type_index(_type_index),
        fixed_type(DT_INVALID) {}

  explicit TypeAttrId(DataType _fixed_type)
      : attr_name(), type_index(kSingleType), fixed_type(_fixed_type) {}

  bool operator==(const TypeAttrId& other) const {
    return attr_name == other.attr_name && type_index == other.type_index &&
           fixed_type == other.fixed_type;
  }

  bool operator<(const TypeAttrId& other) const {
    return std::make_tuple(attr_name, type_index, fixed_type) <
           std::make_tuple(other.attr_name, other.type_index, other.fixed_type);
  }

  template <typename H>
  friend H AbslHashValue(H h, const TypeAttrId& ta) {
    return H::combine(std::move(h), ta.attr_name, ta.type_index, ta.fixed_type);
  }

  string DebugString() const {
    if (!attr_name.empty()) {
      if (type_index == kSingleType) {
        return attr_name;
      } else {
        return strings::StrCat(attr_name, "[", type_index, "]");
      }
    } else {
      return DataTypeString(fixed_type);
    }
  }

  string attr_name;
  // If attr_name is a list(type), this is the index into the list. Otherwise
  // this is kSingleType.
  int type_index;
  DataType fixed_type;
};

// Returns the data type of the given type attribute, or DT_INVALID if the type
// attribute is invalid.
DataType GetDataType(const NodeDef& node, const TypeAttrId& type_attr);

std::vector<std::pair<int, int>> ArgDefIndexes(const NodeDef& node, int arg_idx,
                                               const OpDef::ArgDef& arg_def);
// Returns a pair (arg_index, type_index) for each input to the node, where
// arg_index is the index of the input_arg in op_def and type_index is the index
// of the type in type_list_attr (only defined for list arguments).
std::vector<std::pair<int, int>> InputPortArgDefIndexes(const NodeDef& node,
                                                        const OpDef& op_def);
// Returns a pair (arg_index, type_index) for each output to the node, where
// arg_index is the index of the output_arg in op_def and type_index is the
// index of the type in type_list_attr (only defined for list arguments).
std::vector<std::pair<int, int>> OutputPortArgDefIndexes(const NodeDef& node,
                                                         const OpDef& op_def);
TypeAttrId GetTypeAttrId(const OpDef::ArgDef& arg_def, int arg_type_index);
std::vector<int> NonControlInputs(const NodeDef& node);

// A utility class to lookup node type attributes and type attribute <->
// input/output port mappings.
class NodeTypeAttrMap {
 public:
  NodeTypeAttrMap() {}

  explicit NodeTypeAttrMap(const GraphDef& graph);

  Status Init(const GraphDef& graph);

  Status Clear() {
    graph_ = nullptr;
    type2io_.clear();
    io2type_.clear();
    return Status::OK();
  }

  bool is_initialized() const;

  absl::flat_hash_set<TypeAttrId> GetTypeAttrs(const NodeDef& node) const;
  const absl::flat_hash_set<int>& GetInputPorts(
      const NodeDef& node, const TypeAttrId& type_attr) const;
  const absl::flat_hash_set<int>& GetOutputPorts(
      const NodeDef& node, const TypeAttrId& type_attr) const;
  TypeAttrId GetInputTypeAttr(const NodeDef& node, int port) const;
  TypeAttrId GetOutputTypeAttr(const NodeDef& node, int port) const;
  int GetInputSize(const NodeDef& node) const;
  int GetOutputSize(const NodeDef& node) const;

 private:
  Status AddNode(const NodeDef& node);

  // WARN: `graph_` must outlive this object (node pointers must remain valid).
  const GraphDef* graph_ = nullptr;  // do not own
  std::unique_ptr<FunctionLibraryDefinition> function_library_;

  typedef absl::flat_hash_set<int> IntSet;
  // Maps a type attr id -> (input port set, output port set)
  typedef absl::flat_hash_map<TypeAttrId, std::pair<IntSet, IntSet>> Type2IOMap;
  // Maps a node -> type attr mapping
  absl::flat_hash_map<const NodeDef*, Type2IOMap> type2io_;
  // Maps a port -> type attr id
  typedef std::vector<TypeAttrId> TypeAttrIdVec;
  // Maps a node -> (input port mapping, output port mapping)
  absl::flat_hash_map<const NodeDef*, std::pair<TypeAttrIdVec, TypeAttrIdVec>>
      io2type_;
};

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_UTILS_NODE_TYPE_ATTR_MAP_H_
