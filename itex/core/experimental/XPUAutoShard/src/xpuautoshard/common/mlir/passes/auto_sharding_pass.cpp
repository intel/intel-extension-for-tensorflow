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

#include "xpuautoshard/common/mlir/passes/auto_sharding_pass.h"

#include "itex/core/utils/logging.h"
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/mlir/passes/pass_utils.h"
#include "xpuautoshard/tensorflow/macro.h"

namespace mlir {
namespace hs {

using as::ShardingConfig;

AutoShardingPass::~AutoShardingPass() = default;

AutoShardingPass::AutoShardingPass(const ShardingConfig& sharding_config)
    : sharding_config_(sharding_config) {}

void AutoShardingPass::handleRootOp(Operation* root_op) {
  auto device_info_attr =
      root_op->getAttrOfType<DeviceInfoAttr>(HSDialect::getDeviceInfoAttrKey());
  auto device_info = device_info_attr.getDeviceInfo();
  auto&& graph_handle = mlirGraphToGraphHandle(root_op);
  auto&& hsp_tuner =
      createHspTunerMlir(graph_handle, device_info, sharding_config_);
  setAnnotationToGraph(graph_handle, hsp_tuner->tune(graph_handle));
}

void AutoShardingPass::runOnOperation() {
  auto this_op = getOperation();
  if (auto graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(*this_op)) {
    if (checkGraphShapeInferenceException(&graph_op) == true) {
      ITEX_LOG(WARNING)
          << "Graph has shape inference exception, skip auto_sharding_pass.";
      return;
    }
  }
  if (isRootOp(this_op)) {
    handleRootOp(this_op);
    assert(allHspInitialized(this_op) &&
           "Expect all sharding properties are initialized after auto-sharding "
           "pass");
    return;
  }
  // TODO(itex): handle further nested sub-graphs
  for (Region& region : this_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (auto graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(op)) {
          if (checkGraphShapeInferenceException(&graph_op) == true) {
            ITEX_LOG(WARNING) << "Graph has shape inference exception, skip "
                                 "auto_sharding_pass.";
            return;
          }
        }
        if (isRootOp(&op)) {
          handleRootOp(&op);
          bool all_inited = allHspInitialized(&op);
          if (!all_inited) {
            llvm::outs() << op << "\n";
          }
          TF_ASSERT(all_inited,
                    "Expect all sharding properties are initialized after "
                    "auto-sharding pass");
        }
      }
    }
  }
}

bool AutoShardingPass::allHspInitialized(Operation* root_op) {
  for (Region& region : root_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        for (auto named_attr : op.getAttrs()) {
          if (auto hsp =
                  named_attr.getValue().dyn_cast<ShardingPropertyAttr>()) {
            if (!hsp.isInitialized()) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

}  // namespace hs
}  // namespace mlir
