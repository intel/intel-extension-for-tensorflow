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

#ifndef ITEX_CORE_COMPILER_MLIR_HLO_TRANSFORMS_ITEX_GPU_PASSES_H_
#define ITEX_CORE_COMPILER_MLIR_HLO_TRANSFORMS_ITEX_GPU_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
class PassManager;
namespace gpu {
class GPUModuleOp;
}  // namespace gpu
namespace func {
class FuncOp;
}

#define GEN_PASS_DECL
#include "transforms/itex_gpu_passes.h.inc"

// Create a pass which lowers a subset of lmhlo.fusion ops to gpu.launch_func
// plus a gpu.module containing the kernel.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createGpuFusionRewritePass();

// Returns array of bool attributes. The value of each element specifies whether
// the corresponding operand is written. This attribute is attached to
// 'gpu.launc_func' ops during the fusion rewrite pass above.
ArrayAttr getWrittenOperandsAttribute(Operation* op);

/// Pass that transforms gpu modules in standard dialect to LLVM.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGpuKernelToLlvmPass();

std::unique_ptr<OperationPass<func::FuncOp>> createReplaceAllocWithArgPass();

#define GEN_PASS_REGISTRATION
#include "transforms/itex_gpu_passes.h.inc"  //NOLINT

}  // namespace mlir

#endif  // ITEX_CORE_COMPILER_MLIR_HLO_TRANSFORMS_ITEX_GPU_PASSES_H_
