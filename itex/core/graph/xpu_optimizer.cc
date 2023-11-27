/* Copyright (c) 2021-2022 Intel Corporation
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

#include "itex/core/graph/xpu_optimizer.h"

#include <string>

#include "itex/core/graph/auto_mixed_precision/auto_mixed_precision.h"
#include "itex/core/graph/generic_layout_optimizer/generic_layout_optimizer.h"
#include "itex/core/graph/memory_opt_pass/memory_opt_pass.h"
#include "itex/core/graph/native_layout/native_layout.h"
#ifdef ITEX_ONEDNN_GRAPH
#include "itex/core/graph/onednn_graph/onednn_graph.h"
#endif  // ITEX_ONEDNN_GRAPH
#include "itex/core/graph/onednn_layout/onednn_layout.h"
#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/remapper/remapper.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "tensorflow/c/experimental/grappler/grappler.h"

#if 0
#include "itex/core/graph/tfg_optimizer_hook/tfg_optimizer_hook.h"
#endif

namespace itex {
namespace graph {

void* Optimizer_CPU_Create() {
  auto* optimizer = new Optimizer;
  optimizer->device_name = DEVICE_CPU;
  return reinterpret_cast<void*>(optimizer);
}

void* Optimizer_GPU_Create() {
  auto* optimizer = new Optimizer;
  optimizer->device_name = DEVICE_GPU;
  return reinterpret_cast<void*>(optimizer);
}

void* Optimizer_XPU_Create() {
  auto* optimizer = new Optimizer;
  optimizer->device_name = DEVICE_XPU;
  return reinterpret_cast<void*>(optimizer);
}

void Optimizer_Destroy(void* optimizer) {
  if (optimizer) delete reinterpret_cast<Optimizer*>(optimizer);
}

void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf,
                        const TF_GrapplerItem* tf_item,
                        TF_Buffer* optimized_graph_buf, TF_Status* tf_status) {
  Status status;
  std::string dev_name = (static_cast<Optimizer*>(optimizer))->device_name;
#ifdef INTEL_CPU_ONLY
  if (dev_name.find("CPU") == std::string::npos) return;
#else
  if (dev_name.find("CPU") != std::string::npos) return;
#endif  // INTEL_CPU_ONLY
  OptimizerContext opt_ctx((static_cast<Optimizer*>(optimizer))->device_name);
  ITEX_VLOG(2) << "Optimizer_Optimize  device is " << dev_name;

  // Used for calculating time consumption of ITEX graph optimization.
  std::chrono::steady_clock::time_point start, end;
  if (IsVerboseEnabled()) {
    start = std::chrono::steady_clock::now();
  }

  // Get GrapplerItem.
  GrapplerItem item(tf_item);

  // Deserialize graph_buf into GraphDef
  GraphDef graph_def;
  SET_STATUS_IF_ERROR(tf_status, BufferToMessage(graph_buf, graph_def));
  GraphDef optimized_graph_def = graph_def;
  auto config = GetOptimizerConfigFlags();

  opt_ctx.is_compute_intensive = HaveComputeIntensiveNode(graph_def);
  opt_ctx.is_quantization_graph = HaveQuantizeDequantizeNode(graph_def);
#ifndef INTEL_CPU_ONLY
  // Compute-extensive check is not required on GPU except AutoShard.
  opt_ctx.enable_complete_opt = true;
#else
  opt_ctx.enable_complete_opt =
      (opt_ctx.is_compute_intensive || config.enable_test_mode);
#endif  // INTEL_CPU_ONLY

#ifndef INTEL_CPU_ONLY
  // TF runs plugin optimizer twice, we only run AutoShard in 1st pass.
  bool sharded = false;
  for (int i = 0; i < graph_def.node_size(); i++) {
    const auto& node = graph_def.node(i);
    if (node.name().find("XPURemapper-Replica") != std::string::npos) {
      sharded = true;
      break;
    }
    if (node.name().find("AutoShard") != std::string::npos) {
      sharded = true;
      break;
    }
  }

  if (!sharded) {
    // It is assumed here that the GraphDef containing MatMul or Conv OP is
    // the main part of the model, which can be converted to MLIR normally,
    // and then AutoShard can be performed.
    // On the contrary, the GraphDef that does not contain MatMul or Conv OP
    // is not what we need to pay attention to, and it may not be converted into
    // MLIR normally, and then AutoShard cannot be performed, so it will return
    // directly.
    if (config.enable_sharding && opt_ctx.is_compute_intensive) {
      optimized_graph_def.Swap(&graph_def);

      if (ITEX_VLOG_IS_ON(4)) {
        DumpGraphDefToFile("itex_optimizer_before_sharding", graph_def, "./");
      }

#if 0
      SET_STATUS_IF_ERROR(tf_status,
                          mlir::tfg::RunAutoShard(&opt_ctx, item, graph_def,
                                                  &optimized_graph_def));
#endif
      if (ITEX_VLOG_IS_ON(4)) {
        DumpGraphDefToFile("itex_optimizer_after_sharding", optimized_graph_def,
                           "./");
      }
    }
  }
#endif  // INTEL_CPU_ONLY

  // The optimization pass order with or without oneDNN Graph are different.
  // With oneDNN Graph:     partial_remapper -> auto_mixed_precision
  //                                         -> onednn_graph -> full_remapper
  // Without oneDNN Graph:  full_remapper -> auto_mixed_precision
  bool onednn_graph_optimize =
      config.enable_onednn_graph &&
      (opt_ctx.is_quantization_graph || config.enable_onednn_graph_all_type);

  optimized_graph_def.Swap(&graph_def);
  GenericLayoutOptimizer generic_layout_opt;
  SET_STATUS_IF_ERROR(tf_status,
                      generic_layout_opt.Optimize(&opt_ctx, item, graph_def,
                                                  &optimized_graph_def));

  if (config.enable_remapper && opt_ctx.enable_complete_opt) {
    if (onednn_graph_optimize) {
      // We don't want full scope remapper here if oneDNN graph is enabled.
      optimized_graph_def.Swap(&graph_def);
      SET_STATUS_IF_ERROR(tf_status, RunRemapper(&opt_ctx, item, graph_def,
                                                 &optimized_graph_def, false));
    } else {
      // Run remapper twice for full scope fusions if oneDNN graph is disabled.
      for (int i = 0; i < config.remapper_run_pass; ++i) {
        optimized_graph_def.Swap(&graph_def);
        SET_STATUS_IF_ERROR(tf_status, RunRemapper(&opt_ctx, item, graph_def,
                                                   &optimized_graph_def, true,
                                                   RemapperLevel(i)));
      }
    }
  }

  if (config.enable_auto_mixed_precision && opt_ctx.enable_complete_opt) {
    optimized_graph_def.Swap(&graph_def);
    SET_STATUS_IF_ERROR(
        tf_status,
        RunAutoMixedPrecision(&opt_ctx, item, graph_def, &optimized_graph_def));
    // Because after running auto_mixed_precision, it will insert Cast op
    // before Const op. So run remapper Const + Cast fusion will remove
    // these overhead.
    // We don't want ITEX remapper pass change graph before LLGA pass
    if (config.enable_remapper && !onednn_graph_optimize) {
      optimized_graph_def.Swap(&graph_def);
      SET_STATUS_IF_ERROR(tf_status, RunRemapper(&opt_ctx, item, graph_def,
                                                 &optimized_graph_def));
    }
  }

#ifdef ITEX_ONEDNN_GRAPH
  if (onednn_graph_optimize && opt_ctx.enable_complete_opt) {
    optimized_graph_def.Swap(&graph_def);
    SET_STATUS_IF_ERROR(tf_status,
                        RunOneDnnGraph(item, graph_def, &optimized_graph_def));

    // Run the full scope remapper here since only got partial remapper before
    // if oneDNN graph is enabled.
    if (config.enable_remapper) {
      for (int i = 0; i < config.remapper_run_pass; ++i) {
        optimized_graph_def.Swap(&graph_def);
        SET_STATUS_IF_ERROR(tf_status, RunRemapper(&opt_ctx, item, graph_def,
                                                   &optimized_graph_def, true,
                                                   RemapperLevel(i)));
      }
    }
  }
#endif  // ITEX_ONEDNN_GRAPH

  if (config.enable_layout_opt && opt_ctx.enable_complete_opt) {
    optimized_graph_def.Swap(&graph_def);
    SET_STATUS_IF_ERROR(tf_status, RunOneDnnLayout(&opt_ctx, item, graph_def,
                                                   &optimized_graph_def));
  }

  // Put post Native Format rewrite pass for better co-working with oneDNN
  // layout.
  optimized_graph_def.Swap(&graph_def);
  SET_STATUS_IF_ERROR(tf_status, RunNativeLayout(&opt_ctx, item, graph_def,
                                                 &optimized_graph_def));

  // Memory Optimization
  optimized_graph_def.Swap(&graph_def);
  SET_STATUS_IF_ERROR(tf_status, RunMemoryOptPass(&opt_ctx, item, graph_def,
                                                  &optimized_graph_def));

  if (IsVerboseEnabled()) {
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;
    ITEX_VLOG(0) << "Time for graph optimize costs " << duration.count()
                 << " sec\n";
  }

  if (ITEX_VLOG_IS_ON(4)) {
    DumpGraphDefToFile("itex_optimizer", optimized_graph_def, "./");
  }

  // Serialize output GraphDef into optimized_graph_buf.
  SET_STATUS_IF_ERROR(
      tf_status, MessageToBuffer(optimized_graph_def, optimized_graph_buf));

  TF_StatusFromStatus(status, tf_status);
}

}  // namespace graph
}  // namespace itex
