/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_
#define ITEX_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"       // from @llvm-project
#include "mlir/Support/LLVM.h"   // from @llvm-project
// #include "tensorflow/core/framework/function.h"
#include "itex/core/ir/dialect.h"
#include "itex/core/ir/ops.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/statusor.h"
#include "protos/graph.pb.h"
#include "protos/node_def.pb.h"

namespace mlir {
namespace tfg {

// Get the name of a value as if it were an edge in a graph.
itex::StatusOr<std::string> GetValueName(Value value, TFGraphDialect* dialect);

// Convert a TFG graph directly to GraphDef. Graph functions in the module are
// added to the GraphDef's function library.
itex::Status ConvertToGraphDef(ModuleOp module, itex::GraphDef* graph);

// Convert a single TFG op to NodeDef. This utliity function requires a callback
// `get_value_name` that returns the edge name of the given operand.
itex::Status ConvertToNodeDef(
    Operation* op, itex::NodeDef* node, TFGraphDialect* dialect,
    function_ref<itex::StatusOr<std::string>(Value)> get_value_name);

// // Convert a single TFG function to a FunctionDef and add it to the function
// // library. If a function with the same name already exists, replace it.
// itex::Status ConvertToFunctionDef(
//     GraphFuncOp func, itex::FunctionLibraryDefinition &library);

}  // namespace tfg
}  // namespace mlir

#endif  // ITEX_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_
