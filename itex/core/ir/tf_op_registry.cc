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

#include "itex/core/ir/tf_op_registry.h"

#include "itex/core/ir/ops.h"
#include "protos/op_def.pb.h"

using itex::FunctionLibraryDefinition;
using itex::GraphDef;
using itex::OpDef;
using itex::Status;

namespace mlir {
namespace tfg {
TensorFlowOpRegistryInterface::TensorFlowOpRegistryInterface(Dialect* dialect)
    : TensorFlowOpRegistryInterface(dialect, GraphDef()) {}

// Returns true if the op is stateful.
static bool IsStatefulImpl(const FunctionLibraryDefinition& func,
                           StringRef op_name) {
  OpDef op_def;
  Status status = func.LookUpOpDef(op_name.str(), &op_def);
  // If an op definition was not found, conservatively assume stateful.
  if (!status.ok()) return true;
  return op_def.is_stateful();
}

bool TensorFlowOpRegistryInterface::isStateful(Operation* op) const {
  // Handle TFG internal ops.
  if (isa<ReturnOp, YieldOp, ConditionOp>(op)) return false;
  if (auto func = dyn_cast<GraphFuncOp>(op)) return func.is_stateful();
  // Handle TFG region ops.
  // TODO(jeffniu): Region ops should be marked with a trait.
  StringRef op_name = op->getName().stripDialect();
  if (op->getNumRegions() && op_name.endswith("Region"))
    op_name = op_name.drop_back(/*len("Region")=*/6);
  return IsStatefulImpl(func_, op_name);
}
}  // namespace tfg
}  // namespace mlir
