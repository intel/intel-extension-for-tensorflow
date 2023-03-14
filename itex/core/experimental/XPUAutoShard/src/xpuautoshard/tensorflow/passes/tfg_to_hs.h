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

#include "itex/core/ir/ops.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/mlir/passes/pass_utils.h"

namespace mlir {
namespace hs {
class TFGtoHSConversion
    : public PassWrapper<TFGtoHSConversion, OperationPass<> > {
 public:
  TFGtoHSConversion(const as::DeviceInfo& device_info,
                    const as::ShardingConfig& sharding_config);
  TFGtoHSConversion(const TFGtoHSConversion& pass) = default;
  ~TFGtoHSConversion() override;
  void runOnOperation() override;

 protected:
  as::DeviceInfo device_info_;
  as::ShardingConfig sharding_config_;

 private:
  void DeadNodePrune(mlir::tfg::GraphOp* graph_op);
  void handleResourceOp(Operation* op);
  void handleGenericOp(Operation* op);
  void handleGraphOp(mlir::tfg::GraphOp* graph_op);
  // void mergeReadResourceOP(mlir::tfg::GraphOp* graph_op);
  bool cancelShardUnshardPair(mlir::Block* block);
  void addUnshardToGraphOutputs(mlir::Block* block);
};
}  // namespace hs
}  // namespace mlir
