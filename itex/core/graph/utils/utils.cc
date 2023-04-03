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

#include "itex/core/graph/utils/utils.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/utils/device_name_utils.h"
#include "itex/core/utils/mutex.h"
#include "itex/core/utils/node_def_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/strcat.h"
#include "itex/core/utils/types.h"
#include "protos/attr_value.pb.h"
#include "protos/node_def.pb.h"

namespace itex {
namespace graph {

namespace {

using strings::StrCat;

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

// Is 'node' an operator that consumes only the shape of its input, not the
// data itself?
// TODO(ezhulenev): move to op_types.h. Requires to break circular dependency.
// TODO(ezhulenev): what about Identity passing tensor to Shape consumer?
bool IsShapeConsumer(const NodeDef& node) {
  const string& op = node.op();
  return op == "Shape" || op == "ShapeN" || op == "Rank" || op == "Size";
}

string MakeUniqueFilename(string name, const string& suffix = ".pbtxt") {
  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (uint64 i = 0; i < name.size(); ++i) {
    char ch = name[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?' ||
        ch == '\\') {
      name[i] = '_';
    }
  }

  int count;
  {
    mutex_lock lock(&instance.counts_mutex);
    count = instance.counts[name]++;
  }

  string filename = name;
  if (count > 0) {
    absl::StrAppend(&filename, "_", count);
  }
  absl::StrAppend(&filename, suffix);
  return filename;
}

Status WriteTextProtoToUniqueFile(const itex::protobuf::Message& proto,
                                  std::ofstream* output) {
  string s;
  if (!::google::protobuf::TextFormat::PrintToString(proto, &s)) {
    return errors::FailedPrecondition("Unable to convert proto to text.");
  }

  output->write(s.c_str(), s.length());
  if (!output->good()) return errors::Internal("Unable to dump graph to file.");

  output->close();
  if (!output->good()) return errors::Internal("Unable to close dump file.");

  return Status::OK();
}

Status WriteBinaryProtoToUniqueFile(const itex::protobuf::Message& proto,
                                    std::ofstream* output) {
  if (!proto.SerializeToOstream(output)) {
    return errors::Internal("Unable to dump graph to file.");
  }

  output->close();
  if (!output->good()) return errors::Internal("Unable to close dump file.");

  return Status::OK();
}

Status CreateWritableFile(const string& dirname, const string& name,
                          const string& suffix, string* filepath,
                          std::ofstream* output) {
  string dir;

  if (!dirname.empty()) {
    dir = dirname;
  } else {
    const char* prefix = getenv("ITEX_DUMP_GRAPH_PREFIX");
    if (prefix != nullptr) dir = prefix;
  }

  if (dir.empty()) {
    ITEX_LOG(WARNING)
        << "Failed to dump " << name << " because dump location is not "
        << " specified through either ITEX_DUMP_GRAPH_PREFIX environment "
        << "variable or function argument.";
    return errors::InvalidArgument("ITEX_DUMP_GRAPH_PREFIX not specified");
  }

  // TODO(itex): Integrate proper's FileSystem to create file.
  // TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir));
  // *filepath = io::JoinPath(dir, MakeUniqueFilename(name, suffix));
  *filepath = StrCat(dir, "/", MakeUniqueFilename(name, suffix));
  output->open(filepath->c_str(), std::ios::out);

  if (!output->is_open())
    return errors::Internal("Unable to create dump file under directory '",
                            *filepath, "'.");

  return Status::OK();
}

}  // namespace

namespace internal {
// Specialized template class method GetNodeDefFromGraph.
template <>
NodeDef* NodeMapInternal<GraphDef, NodeDef>::GetNodeDefFromGraph(
    GraphDef* graph, int64 i) const {
  return graph->mutable_node(i);
}

template <>
const NodeDef*
NodeMapInternal<const GraphDef, const NodeDef>::GetNodeDefFromGraph(
    const GraphDef* graph, int64 i) const {
  return &graph->node(i);
}
}  // namespace internal

string DumpGraphDefToFile(const string& name, GraphDef const& graph_def,
                          const string& dirname, bool is_output_binary) {
  string filepath;
  std::ofstream output;

  string ext = is_output_binary ? ".pb" : "*.pbtxt";
  Status status = CreateWritableFile(dirname, name, ext, &filepath, &output);

  if (!status.ok()) {
    return StrCat("(failed to create writable file: ", status.ToString(), ")");
  }

  if (is_output_binary) {
    status = WriteBinaryProtoToUniqueFile(graph_def, &output);
  } else {
    status = WriteTextProtoToUniqueFile(graph_def, &output);
  }

  if (!status.ok()) {
    return StrCat("(failed to dump Graph to '", filepath,
                  "': ", status.ToString(), ")");
  }
  ITEX_LOG(INFO) << "Dumped Graph to " << filepath;
  return filepath;
}

string TensorIdToString(const TensorId& tensor_id) {
  return tensor_id.index() == 0 ? string(tensor_id.node())
                                : tensor_id.ToString();
}

string SafeTensorIdToString(const SafeTensorId& tensor_id) {
  return tensor_id.index() == 0 ? tensor_id.node() : tensor_id.ToString();
}

bool IsSameInput(const string& name1, const string& name2) {
  if (name1 == name2) return true;
  TensorId tensor1 = ParseTensorName(name1);
  TensorId tensor2 = ParseTensorName(name2);
  return tensor1 == tensor2;
}

bool IsControlInput(const string& name) {
  return !name.empty() && name[0] == '^';
}

bool IsControlInput(const TensorId& tensor_id) { return tensor_id.index() < 0; }

string AddPrefixToNodeName(const string& name, const string& prefix,
                           const string& delimiter) {
  if (!name.empty()) {
    if (name[0] == '^') {
      return strings::StrCat("^", prefix, delimiter, name.substr(1));
    }
  }
  return strings::StrCat(prefix, delimiter, name);
}

string AddPrefixToNodeName(const string& name, const string& prefix) {
  return AddPrefixToNodeName(name, prefix, "/");
}

string AsControlDependency(const NodeDef& node) {
  return strings::StrCat("^", node.name());
}

string AsControlDependency(const string& node_name) {
  ITEX_CHECK(!node_name.empty());
  return (!node_name.empty() && node_name[0] == '^')
             ? node_name
             : strings::StrCat("^", node_name);
}

bool NodeIsOnDevice(const char* device_name, const NodeDef* node) {
  return !node->device().empty() &&
         absl::StrContains(node->device(), device_name);
}

bool NodeIsOnCpu(const NodeDef* node) {
  string task, device;
  return DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) &&
         absl::StartsWith(GetDeviceBackendName(device.c_str()), DEVICE_CPU);
}

bool NodeIsOnGpu(const NodeDef* node) {
  string task, device;
  return DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) &&
         absl::StartsWith(GetDeviceBackendName(device.c_str()), DEVICE_GPU);
}

bool HasControlInputs(const NodeDef& node) {
  const int num_inputs = node.input_size();
  if (num_inputs > 0 && IsControlInput(node.input(num_inputs - 1))) {
    return true;
  }
  return false;
}

bool HasRegularInputs(const NodeDef& node) {
  const int num_inputs = node.input_size();
  if (num_inputs > 0 && !IsControlInput(node.input(0))) {
    return true;
  }
  return false;
}

int NumNonControlInputs(const NodeDef& node) {
  int num_inputs = 0;
  for (; num_inputs < node.input_size(); ++num_inputs) {
    const string& input = node.input(num_inputs);
    if (IsControlInput(input)) {
      return num_inputs;
    }
  }
  return num_inputs;
}

int NumControlInputs(const NodeDef& node) {
  int num_inputs = 0;
  for (; num_inputs < node.input_size(); ++num_inputs) {
    const string& input = node.input(node.input_size() - num_inputs - 1);
    if (!IsControlInput(input)) {
      return num_inputs;
    }
  }
  return num_inputs;
}

bool HasRegularOutputs(const NodeDef& node, const NodeMap& node_map) {
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (const string& node_as_input : output->input()) {
      if (IsControlInput(node_as_input)) break;

      TensorId tensor = ParseTensorName(node_as_input);
      if (tensor.node() == node.name()) {
        return true;
      }
    }
  }
  return false;
}

bool HasControlOutputs(const NodeDef& node, const NodeMap& node_map) {
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (int idx = output->input_size() - 1; idx >= 0; --idx) {
      const string& node_as_input = output->input(idx);
      if (!IsControlInput(node_as_input)) break;

      TensorId tensor = ParseTensorName(node_as_input);
      if (tensor.node() == node.name()) {
        return true;
      }
    }
  }
  return false;
}

int NumControlOutputs(const NodeDef& node, const NodeMap& node_map) {
  int num_outputs = 0;
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (int idx = output->input_size() - 1; idx >= 0; --idx) {
      const string& node_as_input = output->input(idx);
      if (!IsControlInput(node_as_input)) break;

      TensorId tensor = ParseTensorName(node_as_input);
      if (tensor.node() == node.name()) {
        ++num_outputs;
      }
    }
  }
  return num_outputs;
}

int NumNonControlOutputs(const NodeDef& node, const NodeMap& node_map) {
  int num_outputs = 0;
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (const string& node_as_input : output->input()) {
      if (IsControlInput(node_as_input)) {
        break;
      }
      if (node_as_input == node.name()) {
        ++num_outputs;
      } else {
        const TensorId tensor = ParseTensorName(node_as_input);
        if (tensor.node() == node.name()) {
          ++num_outputs;
        }
      }
    }
  }
  return num_outputs;
}

int NumNonControlDataOutputs(const NodeDef& node, const NodeMap& node_map) {
  int num_data_outputs = 0;
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    if (IsShapeConsumer(*output)) continue;

    for (int i = 0; i < output->input_size(); ++i) {
      const string& input = output->input(i);
      if (!IsControlInput(input) && NodeName(input) == node.name()) {
        ++num_data_outputs;
        break;
      }
    }
  }
  return num_data_outputs;
}

// Returns the data type in attribute `attr_name` of `node`. If that attribute
// doesn't exist, returns DT_INVALID.
DataType GetDataTypeFromAttr(const NodeDef& node, const string& type_attr) {
  if (!node.attr().count(type_attr)) {
    return DT_INVALID;
  }
  const auto& attr = node.attr().at(type_attr);
  if (attr.value_case() != AttrValue::kType) {
    return DT_INVALID;
  }
  return attr.type();
}

NodeDef* GetTailOfChain(const NodeDef& source, const NodeMap& node_map,
                        bool follow_control_input,
                        const std::function<bool(const NodeDef&)>& pred_fn) {
  const NodeDef* current = &source;
  const NodeDef* next = current;
  while (next == &source || (next != nullptr && pred_fn(*next))) {
    current = next;
    if (current->input_size() == 0 ||
        (!follow_control_input && IsControlInput(current->input(0)))) {
      break;
    }
    next = node_map.GetNode(current->input(0));
    if (next == nullptr) {
      ITEX_LOG(ERROR) << "Node not found: " << current->input(0);
    }
  }
  return const_cast<NodeDef*>(current);
}

// Every permutation is a product of one or more cycles. Iterate over the cycles
// in the permutation, and convert each of those into a product of
// transpositions (swaps): https://en.wikipedia.org/wiki/Cyclic_permutation
void PermuteNodesInPlace(GraphDef* graph, std::vector<int>* permutation,
                         bool invert_permutation) {
  ITEX_CHECK_EQ(graph->node_size(), permutation->size());
  std::vector<int> inv_perm(permutation->size(), 0);
  if (invert_permutation) {
    for (size_t n = 0; n < permutation->size(); ++n) {
      inv_perm[(*permutation)[n]] = n;
    }
    permutation->swap(inv_perm);
  }
  for (int n = 0, end = permutation->size(); n + 1 < end; ++n) {
    while (n != (*permutation)[n]) {
      std::size_t r = (*permutation)[n];
      graph->mutable_node()->SwapElements(n, r);
      std::swap((*permutation)[n], (*permutation)[r]);
    }
  }
}

void DedupControlInputs(NodeDef* node) {
  absl::flat_hash_set<string> inputs;
  int pos = 0;
  while (pos < node->input_size()) {
    const string& input = node->input(pos);
    if (!inputs.insert(NodeName(input)).second && IsControlInput(input)) {
      node->mutable_input()->SwapElements(pos, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
    } else {
      ++pos;
    }
  }
}

namespace {

template <typename UniqueContainer>
void EraseNodesFromGraphImpl(const UniqueContainer& nodes_to_delete,
                             GraphDef* graph) {
  static_assert(std::is_same<typename UniqueContainer::value_type, int>::value,
                "Need to pass container of ints");

  int last = graph->node_size() - 1;
  for (auto it = nodes_to_delete.rbegin(); it != nodes_to_delete.rend(); ++it) {
    const int index = *it;
    graph->mutable_node()->SwapElements(index, last);
    last--;
  }
  graph->mutable_node()->DeleteSubrange(last + 1, nodes_to_delete.size());
}

template <typename T>
inline void STLSortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

}  // namespace

void EraseNodesFromGraph(const std::set<int>& nodes_to_delete,
                         GraphDef* graph) {
  EraseNodesFromGraphImpl(nodes_to_delete, graph);
}

void EraseNodesFromGraph(std::vector<int>&& nodes_to_delete, GraphDef* graph) {
  STLSortAndRemoveDuplicates(&nodes_to_delete);
  EraseNodesFromGraphImpl(nodes_to_delete, graph);
}

void EraseNodesFromGraph(const std::set<string>& nodes_to_delete,
                         GraphDef* graph) {
  std::vector<int> nodes_idx_to_delete;
  nodes_idx_to_delete.reserve(nodes_to_delete.size());
  for (int i = 0; i < graph->node_size(); ++i) {
    if (nodes_to_delete.count(graph->node(i).name()))
      nodes_idx_to_delete.push_back(i);
  }
  EraseNodesFromGraphImpl(nodes_idx_to_delete, graph);
}

#undef HANDLE_CASE

Status CheckAttrExists(const NodeDef& node, const string& key) {
  if (!HasNodeAttr(node, key)) {
    return errors::InvalidArgument("Node '", node.name(), "' lacks '", key,
                                   "' attr: ", node.ShortDebugString());
  }
  return Status::OK();
}

Status CheckAttrsExist(const NodeDef& node, absl::Span<const string> keys) {
  for (const string& key : keys) {
    TF_RETURN_IF_ERROR(CheckAttrExists(node, key));
  }
  return Status::OK();
}

Status IsKernelRegisteredForNode(const NodeDef& node) {
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(node.device(), &parsed_name)) {
    return errors::InvalidArgument("Could not parse device name: ",
                                   node.device());
  }

  Status status;
  status = FindKernelDef(DeviceType(parsed_name.type), node, nullptr, nullptr);
  return status;
}

namespace {
void RemoveAttributes(const std::vector<absl::string_view>& to_remove,
                      NodeDef* node) {
  if (to_remove.size() == static_cast<size_t>(node->attr_size())) {
    node->clear_attr();
  } else {
    for (const auto& key : to_remove) {
      node->mutable_attr()->erase(string(key));
    }
  }
}
}  // namespace

int EraseRegularNodeAttributes(NodeDef* node) {
  std::vector<absl::string_view> to_remove;
  for (const auto& attr : node->attr()) {
    if (!attr.first.empty() && (attr.first)[0] != '_') {
      to_remove.push_back(attr.first);
    }
  }
  RemoveAttributes(to_remove, node);
  return to_remove.size();
}

int EraseNodeOutputAttributes(NodeDef* node) {
  std::vector<absl::string_view> to_remove;
  for (const auto& attr : node->attr()) {
    const string& attr_name = attr.first;
    if (attr_name == "_xla_inferred_shapes" ||
        absl::StartsWith(attr_name, "_output_")) {
      to_remove.push_back(attr_name);
    }
  }
  RemoveAttributes(to_remove, node);
  return to_remove.size();
}

}  // end namespace graph
}  // end namespace itex
