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

#include "itex/core/graph/optimizer_config.h"

#include <string>
#include <utility>

#include "itex/core/graph/config_util.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/hw_info.h"
#include "itex/core/utils/protobuf/config.pb.h"
#include "itex/core/utils/status.h"

namespace {
// Helper function to set optimizer environment variable flags. The 'new_name'
// is for compatibility with the `old_name`.
void HelperSetEnvOptimzerConfig(std::string new_name, std::string old_name,
                                bool default_flag, bool* flag) {
  // The `old_name` environment variable is vaild.
  if (std::getenv(old_name.c_str()) && !std::getenv(new_name.c_str())) {
    ITEX_CHECK_OK(itex::ReadBoolFromEnvVar(old_name, default_flag, flag));
    ITEX_LOG(WARNING) << old_name << " will be deprecated, please use "
                      << new_name << " instead.";
  } else {
    ITEX_CHECK_OK(itex::ReadBoolFromEnvVar(new_name, default_flag, flag));
  }
}

}  // namespace

void SetOptimizerConfigFlags(OptimizerConfigFlags* opt_config_flags) {
  bool onednn_graph_flag;
  bool onednn_graph_all_type_flag;
  bool onednn_graph_compiler_backend_flag;
  bool onednn_graph_dnnl_backend_flag;
  bool tf_constant_folding_flag;
  bool remapper_flag;
  bool auto_mixed_precision_flag;
  bool layout_opt_flag;

  auto cfg_ = itex::itex_get_config();
#define USER_IS_ON(CFG) cfg_.graph_options().CFG() == itex::Toggle::ON
#define USER_IS_OFF(CFG) cfg_.graph_options().CFG() == itex::Toggle::OFF
#define USER_IS_SET(CFG) cfg_.graph_options().CFG()

  if (USER_IS_SET(onednn_graph)) {
    onednn_graph_flag = false;
    if (USER_IS_ON(onednn_graph)) {
      onednn_graph_flag = true;
    }
  } else {
    HelperSetEnvOptimzerConfig("ITEX_ONEDNN_GRAPH", "ITEX_ENABLE_ONEDNN_GRAPH",
                               enable_itex_onednn_graph, &onednn_graph_flag);
  }

  HelperSetEnvOptimzerConfig(
      "_ITEX_ONEDNN_GRAPH_ALL_TYPE", "_ITEX_ENABLE_ONEDNN_GRAPH_ALL_TYPE",
      enable_itex_onednn_graph_all_type, &onednn_graph_all_type_flag);

  HelperSetEnvOptimzerConfig("_ITEX_ONEDNN_GRAPH_COMPILER_BACKEND",
                             "_ITEX_ENABLE_ONEDNN_GRAPH_COMPILER_BACKEND",
                             enable_itex_onednn_graph_compiler_backend,
                             &onednn_graph_compiler_backend_flag);

  HelperSetEnvOptimzerConfig("_ITEX_ONEDNN_GRAPH_DNNL_BACKEND",
                             "_ITEX_ENABLE_ONEDNN_GRAPH_DNNL_BACKEND",
                             enable_itex_onednn_graph_dnnl_backend,
                             &onednn_graph_dnnl_backend_flag);

  if (USER_IS_SET(remapper)) {
    remapper_flag = true;
    if (USER_IS_OFF(remapper)) {
      remapper_flag = false;
    }
  } else {
    HelperSetEnvOptimzerConfig("ITEX_REMAPPER", "ITEX_ENABLE_REMAPPER",
                               enable_itex_remapper, &remapper_flag);
  }

  if (USER_IS_SET(layout_opt)) {
    layout_opt_flag = true;
    if (USER_IS_OFF(layout_opt)) {
      layout_opt_flag = false;
    }
  } else {
    bool default_value = enable_itex_layout_opt;
#ifdef INTEL_CPU_ONLY
    // Set CPU default format is plain.
    default_value = false;
#else
    // GPU set XeHPC default format is plain.
    default_value = !IsXeHPC();
#endif
    if (std::getenv("ITEX_ONEDNN_LAYOUT_OPT")) {
      HelperSetEnvOptimzerConfig("ITEX_LAYOUT_OPT", "ITEX_ONEDNN_LAYOUT_OPT",
                                 default_value, &layout_opt_flag);
    } else {
      HelperSetEnvOptimzerConfig("ITEX_LAYOUT_OPT",
                                 "ITEX_ENABLE_ONEDNN_LAYOUT_OPT", default_value,
                                 &layout_opt_flag);
    }
  }

  ITEX_CHECK_OK(itex::ReadBoolFromEnvVar("ITEX_TF_CONSTANT_FOLDING",
                                         enable_itex_tf_constant_folding,
                                         &tf_constant_folding_flag));

  if (USER_IS_SET(auto_mixed_precision)) {
    auto_mixed_precision_flag = false;
    if (USER_IS_ON(auto_mixed_precision)) {
      auto_mixed_precision_flag = true;
    }
  } else {
    ITEX_CHECK_OK(itex::ReadBoolFromEnvVar("ITEX_AUTO_MIXED_PRECISION",
                                           enable_itex_auto_mixed_precision,
                                           &auto_mixed_precision_flag));
  }

#undef USER_IS_ON
#undef USER_IS_OFF
#undef USER_IS_SET

  // Set OptimizerConfigFlags.
  opt_config_flags->enable_onednn_graph = onednn_graph_flag;
  opt_config_flags->enable_onednn_graph_all_type = onednn_graph_all_type_flag;
  opt_config_flags->enable_onednn_graph_compiler_backend =
      onednn_graph_compiler_backend_flag;
  opt_config_flags->enable_onednn_graph_dnnl_backend =
      onednn_graph_dnnl_backend_flag;
  opt_config_flags->enable_tf_constant_folding = tf_constant_folding_flag;
  opt_config_flags->enable_remapper = remapper_flag;
  opt_config_flags->enable_auto_mixed_precision = auto_mixed_precision_flag;
  opt_config_flags->enable_layout_opt = layout_opt_flag;
  opt_config_flags->remapper_run_pass = remapper_run_pass;
}

OptimizerConfigFlags GetOptimizerConfigFlags() {
  OptimizerConfigFlags opt_config_flags;
  SetOptimizerConfigFlags(&opt_config_flags);
  return opt_config_flags;
}
