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

#ifndef ITEX_CORE_IR_IMPORTEXPORT_EXPORT_H_
#define ITEX_CORE_IR_IMPORTEXPORT_EXPORT_H_

#include <memory>
#include <string>

#include "itex/core/ir/ops.h"
#include "itex/core/utils/function.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"   // from @llvm-project
#include "mlir/IR/Types.h"       // from @llvm-project
#include "mlir/IR/Value.h"       // from @llvm-project
#include "protos/graph.pb.h"
#include "protos/op_def.pb.h"
#include "protos/versions.pb.h"

namespace mlir {
namespace tfg {

// Compute the name to use in GraphDef for a given Value (either the result of
// an operation or a block operand if a function argument) and store the result
// in the provided name string. The `control_ty` is the instance of the
// `ControlType` to compare against and detect a control dependency case.
itex::Status GetValueName(Value operand, std::string& name,  // NOLINT
                          Type control_ty);

// Returns a validated graph to export. A TFG module is valid for export if it
// contains at most one graph operation and any number of graph functions.
// Otherwise, returns an error.
itex::StatusOr<GraphOp> ValidateModuleForExport(ModuleOp module);  // NOLINT

// Exports a GraphFunc operation as a new entry in the function library,
// overwriting any existing functions.
itex::Status ExportFunction(GraphFuncOp func_op,
                            itex::FunctionLibraryDefinition& flib);  // NOLINT

// Converts a version attribute to VersionDef.
void ExportVersionAttr(VersionAttr attr, itex::VersionDef* version);  // NOLINT

// Converts a location to the debug information for the node def, if we find
// supported location, that is a top-level NameLoc or any NameLoc nested inside
// a FusedLoc. Other kind of location are ignored. If a NameLoc is of the form
// "name@func" we parse it and import the two appropriately.
void ExtractExperimentalDebugInfoFromLocation(
    Location inst_loc, itex::NodeDef::ExperimentalDebugInfo* debug_info);

}  // namespace tfg
}  // namespace mlir

namespace itex {

// Given an MLIR module, returns a `output_graph` GraphDef. The module must
// contain at most a single Graph operation and zero or more TFFunc operations.
Status ExportMlirToGraphdef(mlir::ModuleOp module, GraphDef* output_graph);

// Callback type for `ConvertOperationToNode`.
using GetValueNameFn = llvm::function_ref<Status(mlir::Value /*operand*/,
                                                 std::string& /*output_name*/)>;

// Converts an Operation to a NodeDef. The provided `get_value_name` callback
// computes the name to use in GraphDef for a given Value (either the result of
// an operation or a block operand if a function argument) and stores the result
// in the provided `output_name` string.
Status ConvertOperationToNode(mlir::Operation& op, NodeDef* node,  // NOLINT
                              GetValueNameFn get_value_name);
Status ConvertOperationToNode(mlir::Operation& op, NodeDef* node);  // NOLINT

// Convert the handle_data_arr to the `handle_data` field of the provided arg.
// Each entry of the array is itself an array with two entries: a Type and a
// ShapeAttr.
Status ConvertHandleData(mlir::ArrayAttr handle_data_arr, OpDef::ArgDef* arg);

}  // namespace itex

#endif  // ITEX_CORE_IR_IMPORTEXPORT_EXPORT_H_
