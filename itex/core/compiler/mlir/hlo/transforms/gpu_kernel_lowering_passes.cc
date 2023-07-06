/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "itex/core/compiler/mlir/hlo/transforms/itex_gpu_passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

#define GEN_PASS_DEF_GPUKERNELTOLLVMPASS
#include "transforms/itex_gpu_passes.h.inc"

namespace {

/// A pass that does the final lowering to LLVM. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class GpuKernelToLLVMPass
    : public impl::GpuKernelToLLVMPassBase<GpuKernelToLLVMPass> {
  void runOnOperation() override;
};

}  // namespace

static void populateCommonPatterns(LLVMTypeConverter& converter,
                                   RewritePatternSet& patterns) {
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);
  populateMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateComplexToLLVMConversionPatterns(converter, patterns);
}

void GpuKernelToLLVMPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions llvmOpts(&getContext(), DataLayout(getOperation()));
  LLVMTypeConverter converter(&getContext(), llvmOpts);
  populateCommonPatterns(converter, patterns);
  populateGpuToLLVMConversionPatterns(converter, patterns);
  ConversionTarget target(getContext());
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGpuKernelToLlvmPass() {
  return std::make_unique<GpuKernelToLLVMPass>();
}

}  // namespace mlir
