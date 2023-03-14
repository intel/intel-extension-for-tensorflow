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

#ifndef ITEX_CORE_EXPERIMENTAL_XPUAUTOSHARD_SRC_XPUAUTOSHARD_COMMON_MLIR_ATTRIBUTES_H_
#define ITEX_CORE_EXPERIMENTAL_XPUAUTOSHARD_SRC_XPUAUTOSHARD_COMMON_MLIR_ATTRIBUTES_H_

#include "mlir/IR/Attributes.h"                     // from @llvm-project
#include "mlir/IR/Builders.h"                       // from @llvm-project
#include "mlir/IR/BuiltinOps.h"                     // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                   // from @llvm-project
#include "mlir/IR/Dialect.h"                        // from @llvm-project
#include "mlir/IR/Matchers.h"                       // from @llvm-project
#include "mlir/IR/OpImplementation.h"               // from @llvm-project
#include "mlir/IR/PatternMatch.h"                   // from @llvm-project
#include "mlir/IR/RegionKindInterface.h"            // from @llvm-project
#include "mlir/IR/TypeUtilities.h"                  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"   // from @llvm-project
#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/sharding_property.h"

// Get the C++ declaration for all the attrs defined in ODS for the dialect.

#define GET_ATTRDEF_CLASSES
#include "xpuautoshard/common/mlir/attributes.h.inc"

#endif  // ITEX_CORE_EXPERIMENTAL_XPUAUTOSHARD_SRC_XPUAUTOSHARD_COMMON_MLIR_ATTRIBUTES_H_
