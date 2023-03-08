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

#ifndef ITEX_CORE_COMPILER_MLIR_HLO_TRANSFORMS_GPU_PASSES
#define ITEX_CORE_COMPILER_MLIR_HLO_TRANSFORMS_GPU_PASSES

include "mlir/Pass/PassBase.td"

def GpuFusionRewritePass : Pass<"gpu-fusion-rewrite", "ModuleOp"> {
  let summary = "Pass to rewrite lmhlo.fusion ops to gpu.launch_func.";
  let constructor = "createGpuFusionRewritePass()";
}

def GpuKernelToLLVMPass : Pass<"gpu-kernel-to-llvm", "gpu::GPUModuleOp"> {
  let summary = "Pass to transform a gpu module to llvm.";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let constructor = "createGpuKernelToLlvmPass()";
}

def ReplaceAllocWithArgPass : Pass<"replace-alloc-with-arg", "mlir::func::FuncOp"> {
  let summary = "Hoist memref allocations to function arguments.";
  let description = [{
    Replaces memref.alloc uses with a new argument of the parent function.
  }];
  let constructor = "createReplaceAllocWithArgPass()";
}

#endif // ITEX_CORE_COMPILER_MLIR_HLO_TRANSFORMS_GPU_PASSES
