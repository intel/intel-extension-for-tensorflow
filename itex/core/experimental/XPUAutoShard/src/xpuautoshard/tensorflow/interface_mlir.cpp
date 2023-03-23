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

#include "xpuautoshard/tensorflow/interface_mlir.h"

#include "itex/core/ir/ops.h"
#include "itex/core/utils/status.h"
#include "mlir/IR/MLIRContext.h"    // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/mlir/passes/auto_sharding_pass.h"
#include "xpuautoshard/tensorflow/passes/hs_to_tfg.h"
#include "xpuautoshard/tensorflow/passes/tfg_to_hs.h"
#include "xpuautoshard/tensorflow/passes/type_inference.h"

using itex::Status;
using mlir::MLIRContext;
using mlir::OpPassManager;
using mlir::PassManager;

namespace as {
namespace tensorflow {
namespace {
void AddPreprocessTFGPasses(PassManager* pm, GraphProperties* graph_prop) {
  // TODO(itex): Add passes to preprocess TFG IR with inlining
  // and shape propagation etc.
  OpPassManager& nestedModulePM = pm->nest<mlir::tfg::GraphOp>();
  nestedModulePM.addPass(
      std::make_unique<mlir::tfg::TypeInference>(graph_prop));
}
}  // anonymous namespace

void auto_sharding_pass_mlir(mlir::MLIRContext* context, mlir::ModuleOp* module,
                             const ShardingConfig& sharding_config,
                             const DeviceInfo& device_info,
                             GraphProperties* graph_prop) {
  PassManager pm(context);
  // Step 1: Preprocess TFG IR with inlining and type propagation etc.
  AddPreprocessTFGPasses(&pm, graph_prop);
  // Step 2: Convert TFG IR to HS IR
  // The conversion assumes that the mlir::tfg::GraphOp as root, inlined without
  // nested calls to mlir::tfg::GraphFuncOp, and with shape info propagated.
  // The inlining assumption fits most DL model design.
  pm.addPass(std::make_unique<mlir::hs::TFGtoHSConversion>(device_info,
                                                           sharding_config));
  // Step 3: Auto sharding with HS IR
  pm.addPass(std::make_unique<mlir::hs::AutoShardingPass>(sharding_config));
  // Step 4: Convert HS IR back to TFG IR
  pm.addPass(std::make_unique<mlir::hs::HStoTFGConversion>(sharding_config));
  // Run passes
  if (failed(pm.run(*module))) {
    llvm::errs() << "Failed to run sharding passes \n";
  }
}

}  // namespace tensorflow
}  // namespace as
