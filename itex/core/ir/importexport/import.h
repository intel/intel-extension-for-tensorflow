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

#ifndef ITEX_CORE_IR_IMPORTEXPORT_IMPORT_H_
#define ITEX_CORE_IR_IMPORTEXPORT_IMPORT_H_

#include <string>

#include "absl/strings/string_view.h"
#include "itex/core/ir/dialect.h"
#include "itex/core/utils/node_def_util.h"
#include "itex/core/utils/statusor.h"
#include "itex/core/utils/types.h"
#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "protos/graph.pb.h"
#include "protos/versions.pb.h"

namespace mlir {
namespace tfg {

// Constructs the MLIR VersionAttr for the provided GraphDef.
VersionAttr ConvertVersionAttr(MLIRContext* context,
                               const itex::VersionDef& version);

// Returns true if the function is a generic function. I.e. it contains
// placeholder attributes.
bool IsGenericFunction(const itex::FunctionDef& fdef);
/*
// Converts a Graph and function libs to a MLIR module containing the graph and
// expressed in TFG dialect.
itex::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportGraphAndFunctionsToMlir(
    MLIRContext* context, const itex::Graph& graph,
    const itex::GraphDebugInfo& debug_info,
    const itex::FunctionLibraryDefinition& flib_def);

// Converts a GraphDef to a MLIR module containing the graph and expressed in
// TFG dialect.
itex::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportGraphDefToMlir(
    MLIRContext* context, const itex::GraphDebugInfo& debug_info,
    const itex::GraphDef& graphdef);
*/
// Converts an array of "handle_data" (a DType and a Shape) to an MLIR array
// attribute. Each entry will be itself an ArrayAttribute containing a TypeAttr
// and a ShapeAttr
itex::StatusOr<ArrayAttr> ConvertHandleData(
    Builder builder, const itex::protobuf::RepeatedPtrField<
                         itex::ResourceHandleProto_DtypeAndShape>& handle_data);

}  // namespace tfg
}  // namespace mlir

#endif  // ITEX_CORE_IR_IMPORTEXPORT_IMPORT_H_
