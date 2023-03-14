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

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "xpuautoshard/common/config.h"
#include "xpuautoshard/common/device_info.h"

namespace mlir {
namespace hs {
class AutoShardingPass
    : public PassWrapper<AutoShardingPass, OperationPass<> > {
 public:
  explicit AutoShardingPass(const as::ShardingConfig& sharding_config);
  AutoShardingPass(const AutoShardingPass& pass) = default;
  ~AutoShardingPass() override;
  void runOnOperation() override;

 protected:
  as::ShardingConfig sharding_config_;

 private:
  void handleRootOp(Operation* root_op);
  bool allHspInitialized(Operation* root_op);
};
}  // namespace hs
}  // namespace mlir
