/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This files implements a pass that partially bufferized IR.

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include "itex/core/compiler/mlir/hlo/transforms/itex_gpu_passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"

namespace mlir {

#define GEN_PASS_DEF_REPLACEALLOCWITHARGPASS
#include "transforms/itex_gpu_passes.h.inc"

using ::mlir::func::FuncOp;

namespace {
class ReplaceAllocWithArgPass
    : public impl::ReplaceAllocWithArgPassBase<ReplaceAllocWithArgPass> {
  void runOnOperation() override;
};
}  // namespace

void ReplaceAllocWithArgPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  IRRewriter rewriter(funcOp.getContext());
  BitVector resultsToErase(funcOp.getNumResults());
  Operation* terminator = funcOp.getBody().back().getTerminator();
  for (OpOperand& result : terminator->getOpOperands()) {
    Operation* allocOp = result.get().getDefiningOp();
    if (!allocOp || !isa<memref::AllocOp>(allocOp)) {
      // Do nothing if return value is not from allocOp.
      return;
    }
    resultsToErase.set(result.getOperandNumber());
    auto attrs = funcOp.getResultAttrDict(result.getOperandNumber());
    funcOp.insertArgument(funcOp.getNumArguments(), result.get().getType(),
                          attrs, result.get().getLoc());
    rewriter.replaceOp(allocOp, funcOp.getArguments().back());
  }
  funcOp.eraseResults(resultsToErase);
  terminator->eraseOperands(resultsToErase);
}

std::unique_ptr<OperationPass<FuncOp>> createReplaceAllocWithArgPass() {
  return std::make_unique<ReplaceAllocWithArgPass>();
}

}  // namespace mlir
