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

#include "xpuautoshard/common/mlir/passes/cpu_host_initializer.h"

#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/mlir/passes/pass_utils.h"
#include "xpuautoshard/common/sharding_property.h"

namespace mlir {
namespace hs {

using ::as::ShardingProperty;
using ::as::ShardingPropertyRef;

namespace {
ShardingPropertyRef createShardingPropertyCpuHostForType(Type type) {
  auto cpu = as::Device::getCpuHost(1.0f, "CPU:0");
  cpu.setNumStages(1);
  as::DeviceInfo device_info(/*add_cpu_host=*/false);
  device_info.addDevice(cpu);
  auto rank_and_shapes = rankAndShapes(type);
  return as::makeRef<ShardingProperty>(
      device_info, getElementTypeFromMlirType(type), rank_and_shapes.first,
      rank_and_shapes.second);
}
}  // namespace

void CpuHostInitializer::handleShardOp(ShardOp* shard_op) {
  auto& result_hsps = annot_->getResultHsps(shard_op->getOperation());
  if (!result_hsps[0]->isInitialized()) {
    result_hsps[0] =
        createShardingPropertyCpuHostForType(shard_op->getResult().getType());
    changed_ = true;
  }
}

void CpuHostInitializer::handleUnshardOp(UnshardOp* unshard_op) {
  // do nothing
}

void CpuHostInitializer::handleReshardOp(ReshardOp* reshard_op) {
  // do not expect reshard op
  assert(false);
}

void CpuHostInitializer::handleFrameworkOp(Operation* fwk_op) {
  auto& result_hsps = annot_->getResultHsps(fwk_op);
  for (size_t i = 0; i < result_hsps.size(); i++) {
    if (!result_hsps[i]->isInitialized()) {
      result_hsps[i] =
          createShardingPropertyCpuHostForType(fwk_op->getResult(i).getType());
      changed_ = true;
    }
  }
}

bool CpuHostInitializer::initSome(Operation* root_op) {
  changed_ = false;
  for (Region& region : root_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (auto shard_op = llvm::dyn_cast<ShardOp>(op)) {
          handleShardOp(&shard_op);
        } else if (auto unshard_op = llvm::dyn_cast<UnshardOp>(op)) {
          handleUnshardOp(&unshard_op);
        } else if (auto reshard_op = llvm::dyn_cast<ReshardOp>(op)) {
          handleReshardOp(&reshard_op);
        } else if (isFrameworkOp(&op)) {
          handleFrameworkOp(&op);
        }
      }
    }
  }
  return changed_;
}

}  // namespace hs
}  // namespace mlir
