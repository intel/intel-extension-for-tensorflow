/* Copyright (c) 2022 Intel Corporation

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

#include "itex/core/graph/config_util.h"

#include <cstring>

namespace itex {
namespace {
ConfigProto& Configs() {
  static ConfigProto config;
  return config;
}
}  // namespace

void itex_set_config(const ConfigProto& config) { Configs() = config; }

ConfigProto itex_get_config() { return Configs(); }

bool isxehpc_value;
ConfigProto itex_get_isxehpc() {
  ConfigProto isxehpc_proto;
  GraphOptions* isxehpc_graph = isxehpc_proto.mutable_graph_options();
  isxehpc_graph->set_device_isxehpc(isxehpc_value);
  return isxehpc_proto;
}

}  // namespace itex
