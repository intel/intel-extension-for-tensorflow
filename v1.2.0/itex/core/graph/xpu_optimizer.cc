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
#include "itex/core/graph/onednn_graph/onednn_graph.h"
#include "itex/core/graph/onednn_layout/onednn_layout.h"
#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/remapper/remapper.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "tensorflow/c/experimental/grappler/grappler.h"

#ifndef INTEL_CPU_ONLY
#include "itex/core/graph/tfg_optimizer_hook/tfg_optimizer_hook.h"
#endif  // INTEL_CPU_ONLY

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
  const char* device_name = (static_cast<Optimizer*>(optimizer))->device_name;

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

  bool have_matmul_or_conv = false;
  for (auto node : graph_def.node()) {
    if (node.op().find("MatMul") != std::string::npos ||
        node.op().find("Conv") != std::string::npos) {
      have_matmul_or_conv = true;
    }
  }

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
    if (config.enable_sharding) {
      optimized_graph_def.Swap(&graph_def);

      if (ITEX_VLOG_IS_ON(4)) {
        DumpGraphDefToFile("itex_optimizer_before_sharding", graph_def, "./");
      }
      SET_STATUS_IF_ERROR(tf_status, mlir::tfg::RunAutoShard(
                                         item, graph_def, &optimized_graph_def,
                                         have_matmul_or_conv));
      if (ITEX_VLOG_IS_ON(4)) {
        DumpGraphDefToFile("itex_optimizer_after_sharding", optimized_graph_def,
                           "./");
      }
    }
  }
#endif  // INTEL_CPU_ONLY

  optimized_graph_def.Swap(&graph_def);
  GenericLayoutOptimizer generic_layout_opt;
  SET_STATUS_IF_ERROR(tf_status,
                      generic_layout_opt.Optimize(device_name, item, graph_def,
                                                  &optimized_graph_def));

  if (config.enable_remapper) {
    // We don't want full scope remapper here if oneDNN graph is enabled.
    for (int i = 0; i < config.remapper_run_pass; ++i) {
      optimized_graph_def.Swap(&graph_def);
      SET_STATUS_IF_ERROR(
          tf_status,
          RunRemapper(device_name, item, graph_def, &optimized_graph_def,
                      !config.enable_onednn_graph, i));
    }
  }

  if (config.enable_auto_mixed_precision) {
    optimized_graph_def.Swap(&graph_def);
    SET_STATUS_IF_ERROR(tf_status,
                        RunAutoMixedPrecision(device_name, item, graph_def,
                                              &optimized_graph_def));
    // Because after running auto_mixed_precision, it will insert Cast op
    // before Const op. So run remapper Const + Cast fusion will remove
    // these overhead.
    // We don't want ITEX remapper pass change graph before LLGA pass
    if (config.enable_remapper) {
      optimized_graph_def.Swap(&graph_def);
      SET_STATUS_IF_ERROR(tf_status, RunRemapper(device_name, item, graph_def,
                                                 &optimized_graph_def,
                                                 !config.enable_onednn_graph));
    }
  }

  if (config.enable_onednn_graph) {
    optimized_graph_def.Swap(&graph_def);
    SET_STATUS_IF_ERROR(tf_status,
                        RunOneDnnGraph(item, graph_def, &optimized_graph_def));

    // Run the full scope remapper here since only got partial remapper before
    // if oneDNN graph is enabled.
    if (config.enable_remapper) {
      for (int i = 0; i < config.remapper_run_pass; ++i) {
        optimized_graph_def.Swap(&graph_def);
        SET_STATUS_IF_ERROR(tf_status,
                            RunRemapper(device_name, item, graph_def,
                                        &optimized_graph_def, true, i));
      }
    }
  }

  if (config.enable_layout_opt) {
    optimized_graph_def.Swap(&graph_def);
    SET_STATUS_IF_ERROR(tf_status, RunOneDnnLayout(device_name, item, graph_def,
                                                   &optimized_graph_def));
  }

  // Put post Native Format rewrite pass for better co-working with oneDNN
  // layout.
  optimized_graph_def.Swap(&graph_def);
  SET_STATUS_IF_ERROR(tf_status, RunNativeLayout(device_name, item, graph_def,
                                                 &optimized_graph_def));

  // Memory Optimization
  optimized_graph_def.Swap(&graph_def);
  SET_STATUS_IF_ERROR(tf_status, RunMemoryOptPass(device_name, item, graph_def,
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
