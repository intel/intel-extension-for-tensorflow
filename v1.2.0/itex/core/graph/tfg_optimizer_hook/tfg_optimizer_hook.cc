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
#include <unordered_set>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "itex/core/graph/config_util.h"
#include "itex/core/graph/tfg_optimizer_hook/tfg_passes_builder.h"
#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/grappler_item.h"
#include "itex/core/graph/utils/utils.h"
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
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/tensorflow/interface_mlir.h"

using itex::GraphDef;
using itex::Status;
using itex::errors::InvalidArgument;

namespace itex {
namespace graph {

struct AutoShardContext {
  explicit AutoShardContext(const GrapplerItem& item, GraphDef* g_def,
                            Status* status)
      : nodes_to_preserve(item.NodesToPreserve()),
        graph_view(g_def, status),
        graph_properties(item),
        inferred_graph_properties(false) {}

  std::unordered_set<string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
  GraphProperties graph_properties;
  bool inferred_graph_properties;

  GraphProperties& GetGraphProperties() {
    if (!inferred_graph_properties) {
      Status s = graph_properties.InferStatically(
          /*assume_valid_feeds=*/true,
          /*aggressive_shape_inference=*/false,
          /*include_input_tensor_values=*/true,
          /*include_output_tensor_values=*/true);

      // TODO(itex) Is there any case that InferStatically will return an
      // unsuccessful state?
      TF_ABORT_IF_ERROR(s);
      inferred_graph_properties = true;
    }
    return graph_properties;
  }
};
}  // namespace graph
}  // namespace itex

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

Status RunAutoShard(const itex::graph::GrapplerItem& item,
                    const GraphDef& graph_def, GraphDef* optimized_graph,
                    bool have_matmul_or_conv) {
  // Use shape inference feature.
  Status status;
  GraphDef multable_graph_def = graph_def;
  auto ctx = itex::graph::AutoShardContext(item, &multable_graph_def, &status);
  // Infer statically first and only once.
  ctx.GetGraphProperties();

  // It is assumed here that the GraphDef containing MatMul or Conv OP is
  // the main part of the model, which can be converted to MLIR normally,
  // and then AutoShard can be performed.
  // On the contrary, the GraphDef that does not contain MatMul or Conv OP
  // is not what we need to pay attention to, and it may not be converted into
  // MLIR normally, and then AutoShard cannot be performed, so it will return
  // directly.
  if (have_matmul_or_conv == false) {
    *optimized_graph = multable_graph_def;
    return Status::OK();
  }

  TFGPassPipelineBuilder builder = DefaultGrapplerPipeline;
  unsigned num_tfg_threads = 0;
  std::unique_ptr<Impl> impl =
      std::make_unique<Impl>(std::move(builder), num_tfg_threads);
  ITEX_VLOG(5) << "TFG Before Graph: \n" << multable_graph_def.DebugString();

  impl->GetContext()->getOrLoadDialect<mlir::hs::HSDialect>();
  // Import the GraphDef to TFG.
  itex::GraphDebugInfo debug_info;
  auto error_or_module = mlir::tfg::ImportGraphDef(
      impl->GetContext(), debug_info, multable_graph_def);
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
  auto module = module_ref.get();
  {
    itex::int64 itex_num_cpus = 0;
    itex::int64 itex_num_gpus = 0;
    itex::int64 itex_cpu_bs = -1;
    itex::int64 itex_gpu_bs = -1;
    itex::int64 itex_cpu_steps = 1;
    itex::int64 itex_gpu_steps = 1;

    auto configs = itex::itex_get_config().graph_options().sharding_config();
    if (configs.auto_mode())
      ITEX_LOG(WARNING) << "auto_mode is not supported in sharding_config";

    if (configs.devices().size() == 0) {
      ITEX_CHECK_OK(itex::ReadInt64FromEnvVar("ITEX_SHARDING_CPU_DEVICE_NUM", 0,
                                              &itex_num_cpus));
      ITEX_CHECK_OK(itex::ReadInt64FromEnvVar("ITEX_SHARDING_GPU_DEVICE_NUM", 0,
                                              &itex_num_gpus));
      ITEX_CHECK_OK(
          itex::ReadInt64FromEnvVar("ITEX_SHARDING_CPU_BS", -1, &itex_cpu_bs));
      ITEX_CHECK_OK(
          itex::ReadInt64FromEnvVar("ITEX_SHARDING_GPU_BS", -1, &itex_gpu_bs));
      ITEX_CHECK_OK(itex::ReadInt64FromEnvVar("ITEX_SHARDING_CPU_STAGE_NUM", 1,
                                              &itex_cpu_steps));
      ITEX_CHECK_OK(itex::ReadInt64FromEnvVar("ITEX_SHARDING_GPU_STAGE_NUM", 1,
                                              &itex_gpu_steps));
    }

    for (auto cfg : configs.devices()) {
      if (absl::AsciiStrToLower(cfg.device_type().c_str()) == "gpu") {
        itex_num_gpus = cfg.device_num();
        itex_gpu_bs = cfg.batch_size();
        itex_gpu_steps = cfg.stage_num();
      } else if (absl::AsciiStrToLower(cfg.device_type().c_str()) == "cpu") {
        itex_num_cpus = cfg.device_num();
        itex_cpu_bs = cfg.batch_size();
        itex_cpu_steps = cfg.stage_num();
      } else {
        ITEX_LOG(WARNING) << "Only CPU and GPU is supported in ShardingConfig";
      }
    }

    ITEX_VLOG(1) << "AutoShard pass, itex_num_cpus: " << itex_num_cpus;
    ITEX_VLOG(1) << "AutoShard pass, itex_num_gpus: " << itex_num_gpus;
    ITEX_VLOG(1) << "AutoShard pass, itex_cpu_bs: " << itex_cpu_bs;
    ITEX_VLOG(1) << "AutoShard pass, itex_gpu_bs: " << itex_gpu_bs;
    ITEX_VLOG(1) << "AutoShard pass, itex_cpu_steps: " << itex_cpu_steps;
    ITEX_VLOG(1) << "AutoShard pass, itex_gpu_steps: " << itex_gpu_steps;

    float gpu_score = itex_gpu_bs * itex_gpu_steps;
    float cpu_score = itex_cpu_bs * itex_cpu_steps;

    as::DeviceInfo device_info(/*add_cpu_host=*/false);
    for (int i = 0; i < itex_num_gpus; i++) {
      as::Device gpu(i + 1, "XPU:" + std::to_string(i), gpu_score);
      gpu.setNumStages(itex_gpu_steps);
      device_info.addDevice(gpu);
    }
    for (int i = 0; i < itex_num_cpus; i++) {
      as::Device cpu(i + itex_num_gpus + 1, "CPU:" + std::to_string(i),
                     cpu_score);
      cpu.setNumStages(itex_cpu_steps);
      device_info.addDevice(cpu);
    }

    bool model_prune = false;
    ITEX_CHECK_OK(itex::ReadBoolFromEnvVar("MODEL_PRUNE", false, &model_prune));
    as::ShardingConfig config;
    config.setStrategyKind(as::StrategyKind::HEURISTIC);
    config.setUseMultiStageJoin(true);
    config.getHeuristicsConfig().setMultiStageEnabled((itex_gpu_steps != 1) ||
                                                      (itex_cpu_steps != 1));
    config.setNeedDeadNodePrune(model_prune);
    as::tensorflow::auto_sharding_pass_mlir(impl->GetContext(), &module, config,
                                            device_info,
                                            &(ctx.graph_properties));
    ITEX_LOG(INFO) << "Run AutoShard pass successfully";
  }

  // Export the TFG module to GraphDef.
  GraphDef graphdef;
  *graphdef.mutable_library() = multable_graph_def.library();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      mlir::tfg::ConvertToGraphDef(*module_ref, &graphdef),
      "when exporting MLIR module to GraphDef in GrapplerHook");
  *optimized_graph = std::move(graphdef);

  return Status::OK();
}
}  // end namespace tfg
}  // end namespace mlir
