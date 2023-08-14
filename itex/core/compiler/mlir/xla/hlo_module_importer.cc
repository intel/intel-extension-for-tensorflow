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

#include "itex/core/compiler/mlir/xla/hlo_module_importer.h"

#include "itex/core/compiler/mlir/xla/hlo_function_importer.h"
#include "itex/core/compiler/xla/service/hlo_computation.h"
#include "itex/core/compiler/xla/service/hlo_instruction.h"
#include "itex/core/compiler/xla/service/hlo_module.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/Location.h"              // from @llvm-project
#include "mlir/IR/OperationSupport.h"      // from @llvm-project
#include "mlir/IR/Types.h"                 // from @llvm-project
#include "protos/xla.pb.h"

namespace itex_xla {

HloModuleImporter::HloModuleImporter(mlir::ModuleOp module,
                                     bool import_all_computation)
    : import_all_computation_(import_all_computation),
      module_(module),
      builder_(module.getContext()) {
  module.getContext()->loadDialect<mlir::arith::ArithDialect>();
  module.getContext()->loadDialect<mlir::func::FuncDialect>();
  module.getContext()->loadDialect<mlir::mhlo::MhloDialect>();
}

Status HloModuleImporter::Import(const itex_xla::HloModule& module) {
  module_.setName(module.name());
  if (!import_all_computation_)
    // Only import the entry computation, any reachable one will be imported
    // unless turned into a region operation.
    return HloFunctionImporter::ImportAsFunc(
        *module.entry_computation(), module_, &function_map_, &builder_);

  for (const auto* computation : module.computations())
    TF_RETURN_IF_ERROR(HloFunctionImporter::ImportAsFunc(
        *computation, module_, &function_map_, &builder_));

  return Status::OK();
}

Status HloModuleImporter::Import(const itex_xla::HloModuleProto& module_proto) {
  itex_xla::DebugOptions debug_options;
  TF_ASSIGN_OR_RETURN(auto module_config,
                      itex_xla::HloModule::CreateModuleConfigFromProto(
                          module_proto, debug_options));
  TF_ASSIGN_OR_RETURN(auto module, itex_xla::HloModule::CreateFromProto(
                                       module_proto, module_config));

  return Import(*module);
}

}  // namespace itex_xla