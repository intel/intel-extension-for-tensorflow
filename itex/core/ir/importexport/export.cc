/* Copyright (c) 2023 Intel Corporation

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/ir/importexport/export.h"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "itex/core/ir/dialect.h"
#include "itex/core/ir/importexport/convert_attributes.h"
#include "itex/core/ir/importexport/convert_tensor.h"
#include "itex/core/ir/importexport/convert_types.h"
#include "itex/core/ir/importexport/functiondef_export.h"
#include "itex/core/ir/ops.h"
#include "itex/core/ir/types/dialect.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/statusor.h"
#include "itex/core/utils/types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"              // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/IR/FunctionInterfaces.h"      // from @llvm-project
#include "mlir/IR/Location.h"                // from @llvm-project
#include "mlir/IR/OpDefinition.h"            // from @llvm-project
#include "mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/IR/OperationSupport.h"        // from @llvm-project
#include "mlir/IR/Types.h"                   // from @llvm-project
#include "mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"               // from @llvm-project
#include "protos/attr_value.pb.h"
#include "protos/full_type.pb.h"
#include "protos/function.pb.h"
#include "protos/graph.pb.h"
#include "protos/node_def.pb.h"
#include "protos/op_def.pb.h"
#include "protos/resource_handle.pb.h"
#include "protos/tensor_shape.pb.h"
#include "protos/types.pb.h"
#include "protos/versions.pb.h"

#define DEBUG_TYPE "graphdef-to-mlir"

using itex::DataType;
using itex::FunctionDef;
using itex::GetValueNameFn;
using itex::GradientDef;
using itex::GraphDef;
using itex::NodeDef;
using itex::OpDef;
using itex::Status;
using itex::StatusOr;
using itex::VersionDef;
using itex::errors::InvalidArgument;

namespace mlir {
namespace tfg {

StatusOr<GraphOp> ValidateModuleForExport(ModuleOp module) {
  GraphOp graph_op;
  for (Operation& op : *module.getBody()) {
    if (isa<GraphFuncOp>(op)) continue;
    if (auto new_graph_op = dyn_cast<GraphOp>(op)) {
      if (graph_op) {
        return InvalidArgument(
            "Can't export module with two different tfg.graph");
      }
      graph_op = new_graph_op;
      continue;
    }
    return InvalidArgument(
        "Can't export module with other ops than tfg.graph or tfg.func, has: ",
        op.getName().getStringRef().str());
  }
  return graph_op;
}

void ExportVersionAttr(VersionAttr attr, VersionDef* version) {
  version->set_producer(attr.getProducer());
  version->set_min_consumer(attr.getMinConsumer());
  for (int32_t bad_consumer : attr.getBadConsumers())
    version->add_bad_consumers(bad_consumer);
}

void ExtractExperimentalDebugInfoFromLocation(
    Location inst_loc, NodeDef::ExperimentalDebugInfo* debug_info) {
  auto add_name_loc = [&](mlir::NameLoc name_loc) {
    StringRef node, func;
    std::tie(node, func) = name_loc.getName().strref().split('@');
    debug_info->add_original_node_names(node.str());
    if (!func.empty()) debug_info->add_original_func_names(func.str());
  };
  if (auto fused = inst_loc.dyn_cast<mlir::FusedLoc>()) {
    for (Location loc : fused.getLocations())
      if (auto name_loc = loc.dyn_cast<mlir::NameLoc>()) add_name_loc(name_loc);
    return;
  }
  if (auto name_loc = inst_loc.dyn_cast<mlir::NameLoc>())
    add_name_loc(name_loc);
}

namespace {

constexpr StringRef kNameAttr = TFGraphDialect::getNameAttrKey();
constexpr StringRef kDeviceAttr = TFGraphDialect::getDeviceAttrKey();
constexpr StringRef kFullTypeAttr = TFGraphDialect::getFullTypeAttrKey();

// Convert an MLIR operation to a NodeDef. The `control_ty` is the instance of
// the `ControlType` to compare against and detect a control dependency case.
Status ConvertOperationToNodeImpl(Operation& op, NodeDef* node,  // NOLINT
                                  GetValueNameFn get_value_name) {
  auto nameAttr = op.getAttrOfType<StringAttr>(kNameAttr);
  if (nameAttr) node->set_name(nameAttr.getValue().str());
  auto deviceAttr = op.getAttrOfType<StringAttr>(kDeviceAttr);
  if (deviceAttr) node->set_device(deviceAttr.getValue().str());
  if (auto fulltype_attr =
          op.getAttrOfType<itex_type::FullTypeAttr>(kFullTypeAttr)) {
    TF_ASSIGN_OR_RETURN(*node->mutable_experimental_type(),
                        ConvertAttribute(fulltype_attr));
  }
  std::string name;
  for (Value operand : op.getOperands()) {
    TF_RETURN_IF_ERROR(get_value_name(operand, name));
    node->add_input(name);
  }
  StringRef op_name = op.getName().stripDialect();
  if (op_name == "LegacyCall") {
    auto callee = op.getAttrOfType<FuncAttr>("callee");
    if (!callee)
      return InvalidArgument("Missing callee attribute on LegacyCall");
    StringRef callee_name = callee.getName().getRootReference().getValue();
    node->set_op({callee_name.data(), callee_name.size()});
    TF_RETURN_IF_ERROR(ConvertAttributes(
        callee.getAttrs().getValue(), {kNameAttr, kDeviceAttr, kFullTypeAttr},
        /*remove_ref_type=*/false, node->mutable_attr()));
    auto optional_device =
        op.getAttrDictionary().getNamed("_mlir_assigned_device");
    if (optional_device.hasValue()) {
      NamedAttrList assigned_device;
      assigned_device.push_back(*optional_device);
      TF_RETURN_IF_ERROR(ConvertAttributes(assigned_device, {},
                                           /*remove_ref_type=*/false,
                                           node->mutable_attr()));
    }
  } else {
    node->set_op({op_name.data(), op_name.size()});
    TF_RETURN_IF_ERROR(ConvertAttributes(
        op.getAttrs(), {kNameAttr, kDeviceAttr, kFullTypeAttr},
        /*remove_ref_type=*/false, node->mutable_attr()));
  }
  // Eliminate empty "_mlir_assigned_device" from the export. This is just
  // more friendly to the serialization.
  {
    auto it = node->mutable_attr()->find("_mlir_assigned_device");
    if (it != node->mutable_attr()->end() && it->second.s().empty())
      node->mutable_attr()->erase("_mlir_assigned_device");
  }

  // Export the location as debug info on the nodes.
  ExtractExperimentalDebugInfoFromLocation(
      op.getLoc(), node->mutable_experimental_debug_info());
  if (node->experimental_debug_info().original_node_names().empty())
    node->clear_experimental_debug_info();

  return Status::OK();
}

// Convert the handle_data_arr to the `handle_data` field of the provided arg.
// Each entry of the array is itself an array with two entries: a Type and a
// ShapeAttr.
static Status ConvertHandleDataImpl(ArrayAttr handle_data_arr,
                                    OpDef::ArgDef* arg) {
  if (!handle_data_arr) return {};
  for (auto handle_data_attr : handle_data_arr.getAsRange<TypeAttr>()) {
    TensorType handle_type = handle_data_attr.getValue().dyn_cast<TensorType>();
    if (!handle_type) {
      return InvalidArgument("Expected an array of tensor types, but got ",
                             debugString(handle_data_arr));
    }
    auto* handle_data = arg->add_handle_data();
    if (handle_type.hasRank()) {
      ConvertToTensorShapeProto(handle_type.getShape(),
                                handle_data->mutable_shape());
    } else {
      handle_data->mutable_shape()->set_unknown_rank(true);
    }
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(handle_type.getElementType(), &dtype));
    handle_data->set_dtype(dtype);
  }
  return {};
}
}  // namespace

// Compute the name to use in GraphDef for a given Value (either the result of
// an operation or a block operand if a function argument) and store the result
// in the provided name string. The `control_ty` is the instance of the
// `ControlType` to compare against and detect a control dependency case.
Status GetValueName(Value operand, std::string& name,  // NOLINT
                    Type control_ty) {
  OpResult op_result = operand.dyn_cast<OpResult>();
  if (!op_result) {
    BlockArgument block_operand = operand.dyn_cast<BlockArgument>();
    bool is_control = (block_operand.getType() == control_ty);
    int arg_num = block_operand.getArgNumber();
    name.clear();
    // Function arguments are coming as pair: the even are the actual tensors
    // while the odd position are the associated control input.
    if (is_control) name = "^";
    DictionaryAttr arg_attrs = function_interface_impl::getArgAttrDict(
        block_operand.getParentBlock()->getParentOp(), arg_num - is_control);
    if (!arg_attrs)
      return InvalidArgument("Missing attribute for argument #", arg_num);
    StringAttr arg_name = arg_attrs.getAs<StringAttr>("tfg.name");
    if (!arg_name)
      return InvalidArgument(
          "Can't export graph with missing op-name for function parameter #",
          arg_num);
    absl::StrAppend(&name, arg_name.getValue().str());
    return {};
  }
  Operation* producer = op_result.getDefiningOp();
  auto nameAttr = producer->getAttrOfType<StringAttr>(kNameAttr);
  if (!nameAttr)
    return InvalidArgument("Can't export graph with missing op-name");

  name.clear();
  if (op_result.getType() == control_ty) name = "^";
  absl::StrAppend(&name, nameAttr.getValue().str());
  if (op_result.getType() != control_ty && op_result.getResultNumber())
    absl::StrAppend(&name, ":", op_result.getResultNumber());
  return {};
}
}  // namespace tfg
}  // namespace mlir

namespace itex {

Status ConvertHandleData(mlir::ArrayAttr handle_data_arr, OpDef::ArgDef* arg) {
  return mlir::tfg::ConvertHandleDataImpl(handle_data_arr, arg);
}

// Status ExportMlirToGraphdef(mlir::ModuleOp module, GraphDef *output_graph) {
//   return mlir::tfg::ExportMlirToGraphdefImpl(module, output_graph);
// }

Status ConvertOperationToNode(mlir::Operation& op, NodeDef* node,  // NOLINT
                              GetValueNameFn get_value_name) {
  return mlir::tfg::ConvertOperationToNodeImpl(op, node, get_value_name);
}
Status ConvertOperationToNode(mlir::Operation& op, NodeDef* node) {  // NOLINT
  auto control_ty = mlir::tfg::ControlType::get(op.getContext());
  return mlir::tfg::ConvertOperationToNodeImpl(
      op, node, [&](mlir::Value operand, std::string& output_name) {
        return mlir::tfg::GetValueName(operand, output_name, control_ty);
      });
}

}  //  namespace itex
