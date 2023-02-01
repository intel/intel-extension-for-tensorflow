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

#include "itex/core/graph/tfg_optimizer_hook/tfg_optimizer_hook.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "itex/core/graph/tfg_optimizer_hook/tfg_passes_builder.h"
#include "itex/core/graph/utils/grappler_item.h"
#include "itex/core/ir/dialect.h"
#include "itex/core/ir/importexport/graphdef_export.h"
#include "itex/core/ir/importexport/graphdef_import.h"
#include "itex/core/ir/ops.h"
#include "itex/core/ir/tf_op_registry.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/errors.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"                // from @llvm-project
#include "mlir/IR/Dialect.h"                   // from @llvm-project
#include "mlir/IR/MLIRContext.h"               // from @llvm-project
#include "mlir/Pass/PassManager.h"             // from @llvm-project
#include "mlir/Pass/PassRegistry.h"            // from @llvm-project
#include "mlir/Support/FileUtilities.h"        // from @llvm-project
#include "mlir/Support/LogicalResult.h"        // from @llvm-project
#include "mlir/Transforms/LocationSnapshot.h"  // from @llvm-project
#include "protos/graph_debug_info.pb.h"
#include "protos/versions.pb.h"

using itex::GraphDef;
using itex::Status;
using itex::errors::InvalidArgument;

namespace mlir {
class PassManager;

namespace tfg {
// A function that builds the TFG pass pipeline.
using TFGPassPipelineBuilder = std::function<void(PassManager& pm)>;  // NOLINT

// The implementation of the TFG optimizer. It holds the MLIR context and the
// pass manager.
class Impl {
 public:
  // Builds the pass pipeline. The context is initialized with threading
  // disabled. If the user specifies to run the optimizer with more than zero
  // threads, a threadpool is initialized and passed to the MLIR context.
  explicit Impl(TFGPassPipelineBuilder builder, unsigned num_tfg_threads)
      : ctx_(MLIRContext::Threading::DISABLED), mgr_(&ctx_) {
    DialectRegistry registry;
    // Register the TF op registry interface so that passes can query it.
    registry.addExtension(+[](MLIRContext* ctx, TFGraphDialect* dialect) {
      dialect->addInterfaces<TensorFlowOpRegistryInterface>();
    });
    ctx_.appendDialectRegistry(registry);
    builder(mgr_);
    if (num_tfg_threads) {
      llvm::ThreadPoolStrategy strategy;
      strategy.ThreadsRequested = num_tfg_threads;
      threadpool_ = std::make_unique<llvm::ThreadPool>(strategy);
      ctx_.setThreadPool(*threadpool_);
    }
  }

  // Runs the pass manager.
  LogicalResult RunPipeline(ModuleOp module) { return mgr_.run(module); }

  // Get the context.
  MLIRContext* GetContext() { return &ctx_; }

  // Convert the pass pipeline to a textual string.
  std::string GetPipelineString() {
    std::string pipeline;
    llvm::raw_string_ostream os(pipeline);
    mgr_.printAsTextualPipeline(os);
    return os.str();
  }

 private:
  // An optional threadpool for running MLIR with threading. Use an external
  // threadpool so the number of threads can be controlled.
  std::unique_ptr<llvm::ThreadPool> threadpool_;
  // The MLIR context.
  MLIRContext ctx_;
  // The pass manager containing the loaded TFG pass pipeline.
  PassManager mgr_;
};

void DumpToFileInDirOrStdout(const std::string& file_name,
                             mlir::Operation* op) {
  ITEX_LOG(INFO) << "Dump tfg module: " << file_name;
  std::string file_path = "./" + file_name;
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(llvm::SmallString<32>(file_path), &error);
  if (!outputFile) {
    ITEX_LOG(ERROR) << "Error: " << error << std::endl
                    << "Failed to open file: " << file_path;
    return;
  }

  op->print(outputFile->os(), mlir::OpPrintingFlags().useLocalScope());
  outputFile->keep();
}

Status RunAutoShard(const GraphDef& graph_def, GraphDef* optimized_graph,
                    itex::graph::GraphProperties* graph_prop) {
  TFGPassPipelineBuilder builder = DefaultGrapplerPipeline;
  unsigned num_tfg_threads = 0;
  std::unique_ptr<Impl> impl =
      std::make_unique<Impl>(std::move(builder), num_tfg_threads);
  ITEX_VLOG(5) << "TFG Before Graph: \n" << graph_def.DebugString();

  // TODO(itex): load dialect if needed.
  // impl->GetContext()->getOrLoadDialect<mlir::hs::HSDialect>();
  // Import the GraphDef to TFG.
  itex::GraphDebugInfo debug_info;
  auto error_or_module =
      mlir::tfg::ImportGraphDef(impl->GetContext(), debug_info, graph_def);
  if (!error_or_module.ok()) {
    auto status = error_or_module.status();
    itex::errors::AppendToMessage(
        &status, "when importing GraphDef to MLIR module in GrapplerHook");
    ITEX_VLOG(4) << "GraphDef import error: " << status.ToString();
    return status;
  }

  auto module_ref = std::move(error_or_module.ValueOrDie());

  // Run the pipeline on the graph.
  if (failed(impl->RunPipeline(*module_ref)))
    return InvalidArgument("MLIR Graph Optimizer failed");

  // TODO(itex): Add autoshard integration when it is ready.

  // Export the TFG module to GraphDef.
  GraphDef graphdef;
  *graphdef.mutable_library() = graph_def.library();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      mlir::tfg::ConvertToGraphDef(*module_ref, &graphdef),
      "when exporting MLIR module to GraphDef in GrapplerHook");
  *optimized_graph = std::move(graphdef);

  return Status::OK();
}
}  // end namespace tfg
}  // end namespace mlir
