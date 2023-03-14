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

#include "xpuautoshard/common/mlir/dialect.h"

#include "llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/IR/AsmState.h"               // from @llvm-project
#include "mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/Support/LogicalResult.h"     // from @llvm-project
#include "xpuautoshard/common/mlir/attributes.h"
#include "xpuautoshard/common/mlir/ops.h"

// Generated definitions.
#include "xpuautoshard/common/mlir/dialect.cpp.inc"

namespace llvm {

::llvm::hash_code hash_value(const float& v) {
  return ::llvm::hash_value(static_cast<int64_t>(1.0f * v));
}

::llvm::hash_code hash_value(const as::ShardingPropertyRef& prop) {
  return ::llvm::hash_value(prop ? (int64_t)prop.get() : (int64_t)0);
}

::llvm::hash_code hash_value(const as::Device& device) {
  return ::llvm::hash_combine(device.getId(), device.getName(),
                              device.getScore());
}

::llvm::hash_code hash_value(const as::DeviceInfo& info) {
  return ::llvm::hash_combine_range(info.getDevices().begin(),
                                    info.getDevices().end());
}
}  // namespace llvm

#define GET_ATTRDEF_CLASSES
#include "xpuautoshard/common/mlir/attributes.cpp.inc"

#define DEBUG_TYPE "hs_dialect"
namespace mlir {
namespace hs {
//===----------------------------------------------------------------------===//
// HSDialect dialect.
//===----------------------------------------------------------------------===//
// Dialect construction: there is one instance per context and it registers its
// operations, attributes, types, and interfaces here.
void HSDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xpuautoshard/common/mlir/attributes.cpp.inc"  //NOLINT
      >();

  addOperations<
#define GET_OP_LIST
#include "xpuautoshard/common/mlir/ops.cpp.inc"
      >();

  // Support unknown operations.
  allowsUnknownOperations();

  LLVM_DEBUG(llvm::dbgs() << "HSDialect registered";);
}

bool ShardingPropertyAttr::isInitialized() const {
  return getImpl()->prop && getImpl()->prop->isInitialized();
}

as::ShardingPropertyRef ShardingPropertyAttr::getShardingProperty() const {
  return getImpl()->prop;
}

const as::DeviceInfo& DeviceInfoAttr::getDeviceInfo() const {
  return getImpl()->info;
}

}  // namespace hs

}  // namespace mlir
