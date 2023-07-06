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

#ifndef ITEX_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_IMPORT_H_
#define ITEX_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_IMPORT_H_

#include "itex/core/ir/ops.h"
#include "itex/core/utils/status.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "protos/function.pb.h"

namespace mlir {
namespace tfg {

// Import the FunctionDef `func` as a TFG generic function (see GraphFuncOp
// documentation). The function will be inserted using the provided `builder`.
itex::Status ConvertGenericFunction(GraphFuncOp func_op,
                                    const itex::FunctionDef& func,
                                    OpBuilder& builder);  // NOLINT

}  // namespace tfg
}  // namespace mlir

#endif  // ITEX_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_IMPORT_H_
