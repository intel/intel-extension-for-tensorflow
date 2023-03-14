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

#include "xpuautoshard/common/mlir/passes/hsp_propagator.h"

#include <queue>
#include <unordered_map>

#include "xpuautoshard/common/hsp_inference/hsp_inference.h"
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/mlir/passes/pass_utils.h"
#include "xpuautoshard/common/sharding_property.h"

namespace mlir {
namespace hs {

using as::ShardingProperty;

void HspPropagator::propagate(Operation* root_op,
                              HspOpPropagator* op_propagator) {
  bool changed;
  do {
    changed = false;
    changed |= forwardPropagate(root_op, op_propagator);
    changed |= backwardPropagate(root_op, op_propagator);
  } while (changed);
}

bool HspPropagator::forwardPropagate(Operation* root_op,
                                     HspOpPropagator* op_propagator) {
  bool changed = false;
  std::queue<Operation*> ready_queue;
  for (Region& region : root_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (llvm::dyn_cast<ShardOp>(op)) {
          ready_queue.push(&op);
        }
      }
    }
  }
  std::unordered_map<Operation*, size_t> num_operands_visited;
  while (!ready_queue.empty()) {
    Operation* op = ready_queue.front();
    ready_queue.pop();
    if (llvm::dyn_cast<UnshardOp>(op)) {
      continue;
    }
    changed |= op_propagator->forward(op);
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        if (num_operands_visited.find(user) == num_operands_visited.end()) {
          num_operands_visited[user] = 0;
        }
        if (++num_operands_visited[user] == user->getNumOperands()) {
          ready_queue.push(user);
        }
      }
    }
  }
  return changed;
}

bool HspPropagator::backwardPropagate(Operation* root_op,
                                      HspOpPropagator* op_propagator) {
  bool changed = false;
  std::queue<Operation*> ready_queue;
  for (Region& region : root_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (llvm::dyn_cast<UnshardOp>(op)) {
          ready_queue.push(&op);
        }
      }
    }
  }
  std::unordered_map<Operation*, size_t> num_sharded_results_visited;
  while (!ready_queue.empty()) {
    Operation* op = ready_queue.front();
    ready_queue.pop();
    if (llvm::dyn_cast<ShardOp>(op)) {
      continue;
    }
    changed |= op_propagator->backward(op);
    for (auto operand : op->getOperands()) {
      auto defining_op = operand.getDefiningOp();
      if (num_sharded_results_visited.find(defining_op) ==
          num_sharded_results_visited.end()) {
        num_sharded_results_visited[defining_op] = 0;
      }
      if (++num_sharded_results_visited[defining_op] ==
          numShardedResults(defining_op)) {
        ready_queue.push(defining_op);
      }
    }
  }
  return changed;
}

}  // namespace hs
}  // namespace mlir
