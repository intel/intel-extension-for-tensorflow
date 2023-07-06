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

#pragma once

#include <memory>

#include "itex/core/graph/utils/graph_properties.h"
#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xpuautoshard/common/config.h"
#include "xpuautoshard/common/device_info.h"

using itex::graph::GraphProperties;

namespace as {
namespace tensorflow {

void auto_sharding_pass_mlir(mlir::MLIRContext* context, mlir::ModuleOp* module,
                             const ShardingConfig& sharding_config,
                             const DeviceInfo& device_info,
                             GraphProperties* graph_properties);

}
}  // namespace as
