/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_COMPILER_MLIR_XLA_HLO_MODULE_IMPORTER_H_
#define ITEX_CORE_COMPILER_MLIR_XLA_HLO_MODULE_IMPORTER_H_

#include <unordered_map>

#include "itex/core/compiler/mlir/tensorflow/utils/error_util.h"
#include "itex/core/compiler/xla/status.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/IR/MLIRContext.h"           // from @llvm-project
#include "protos/xla_data.pb.h"

namespace itex_xla {
class HloModule;
class HloModuleProto;
class HloComputation;
class HloInstruction;
class Shape;

// Importer that takes an HloModule and imports it as an MLIR module in the XLA
// dialect. HloModuleImporter does not take ownership.
class HloModuleImporter {
 public:
  explicit HloModuleImporter(mlir::ModuleOp module,
                             bool import_all_computation = false);

  // Import the HloModule into the MLIR Module.
  Status Import(const itex_xla::HloModule& module);

  // Import the HloModuleProto into the MLIR Module.
  Status Import(const itex_xla::HloModuleProto& module);

 private:
  bool import_all_computation_;
  mlir::ModuleOp module_;
  mlir::Builder builder_;

  // Map for tracking which MLIR function map to which HLO Computation. This
  // tracks functions as they are imported and provides a quick lookup for
  // functions invoked by control flow related operations (e.g. while, call).
  std::unordered_map<const itex_xla::HloComputation*, mlir::func::FuncOp>
      function_map_;
};

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_MLIR_XLA_HLO_MODULE_IMPORTER_H_
