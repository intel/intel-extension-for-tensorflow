/* Copyright (c) 2023 Intel Corporation

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

#include "xpuautoshard/common/mlir/ops.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"                // from @llvm-project
#include "mlir/IR/Builders.h"                // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/IR/Dialect.h"                 // from @llvm-project
#include "mlir/IR/DialectImplementation.h"   // from @llvm-project
#include "mlir/IR/FunctionImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"             // from @llvm-project
#include "mlir/IR/OperationSupport.h"        // from @llvm-project
#include "mlir/IR/TypeRange.h"               // from @llvm-project
#include "mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/Support/LogicalResult.h"      // from @llvm-project
#include "xpuautoshard/common/mlir/dialect.h"

namespace mlir {
namespace hs {

/// Get Sharding Property
::mlir::hs::ShardingPropertyAttr getShardingProperty(Operation* op) {
  return {};
}

}  // namespace hs
}  // namespace mlir

#define GET_OP_CLASSES
#include "xpuautoshard/common/mlir/ops.cpp.inc"
