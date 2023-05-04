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

#ifndef ITEX_CORE_GRAPH_OPTIMIZER_CONFIG_H_
#define ITEX_CORE_GRAPH_OPTIMIZER_CONFIG_H_
#include <memory>

#include "tensorflow/c/experimental/grappler/grappler.h"

constexpr static bool enable_itex_sharding = false;
#ifndef INTEL_CPU_ONLY
constexpr static bool enable_itex_onednn_graph = false;
#else
constexpr static bool enable_itex_onednn_graph = true;
#endif  // INTEL_CPU_ONLY
constexpr static bool enable_itex_onednn_graph_all_type = false;
constexpr static bool enable_itex_onednn_graph_compiler_backend = false;
constexpr static bool enable_itex_onednn_graph_dnnl_backend = true;
constexpr static bool enable_itex_tf_constant_folding = true;
constexpr static bool enable_itex_optimize_aggressive = false;
constexpr static bool enable_itex_remapper = true;
constexpr static bool enable_itex_auto_mixed_precision = false;
constexpr static bool enable_itex_layout_opt = true;
constexpr static bool enable_itex_test_mode = false;
constexpr static int32_t remapper_run_pass = 2;

typedef struct _OptimizerConfigFlags {
  bool enable_sharding;
  bool enable_onednn_graph;
  bool enable_onednn_graph_all_type;
  bool enable_onednn_graph_compiler_backend;
  bool enable_onednn_graph_dnnl_backend;
  bool enable_tf_constant_folding;
  bool enable_optimize_aggressive;
  bool enable_remapper;
  bool enable_auto_mixed_precision;
  // TODO(itex): To integrate DOC & GraphOptions
  bool enable_layout_opt;
  bool enable_test_mode;
  int32_t remapper_run_pass;
} OptimizerConfigFlags;

OptimizerConfigFlags GetOptimizerConfigFlags();

#endif  // ITEX_CORE_GRAPH_OPTIMIZER_CONFIG_H_
