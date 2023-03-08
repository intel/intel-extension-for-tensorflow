/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/service/gpu/spir_compiler.h"

#include <stdlib.h>

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "itex/core/compiler/xla/service/algebraic_simplifier.h"
#include "itex/core/compiler/xla/service/call_inliner.h"
#include "itex/core/compiler/xla/service/dump.h"
#include "itex/core/compiler/xla/service/gpu/cusolver_rewriter.h"
#include "itex/core/compiler/xla/service/gpu/gpu_conv_padding_legalization.h"
#include "itex/core/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "itex/core/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "itex/core/compiler/xla/service/gpu/ir_emission_utils.h"
#include "itex/core/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "itex/core/compiler/xla/service/gpu/mkl.h"
#include "itex/core/compiler/xla/service/gpu/onednn_fused_conv_rewriter.h"
#include "itex/core/compiler/xla/service/gpu/target_constants.h"
#include "itex/core/compiler/xla/service/gpu/triangular_solve_rewriter.h"
#include "itex/core/compiler/xla/service/hlo_constant_folding.h"
#include "itex/core/compiler/xla/service/hlo_cse.h"
#include "itex/core/compiler/xla/service/hlo_opcode.h"
#include "itex/core/compiler/xla/service/hlo_pass_fix.h"
#include "itex/core/compiler/xla/service/hlo_pass_pipeline.h"
#include "itex/core/compiler/xla/service/hlo_verifier.h"
#include "itex/core/compiler/xla/service/llvm_ir/llvm_util.h"
#include "itex/core/compiler/xla/service/tuple_simplifier.h"
#include "itex/core/compiler/xla/status_macros.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_platform_id.h"
#include "itex/core/compiler/xla/types.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/path.h"
#include "itex/core/utils/status.h"

namespace itex_xla {
namespace gpu {

Status SPIRCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module) {
  // Convert convolutions into CustomCalls to cudnn, then canonicalize them
  // (GpuConvPaddingLegalization). Also expand cuSolver calls.
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);
  pipeline.AddPass<GpusolverRewriter>();
  pipeline.AddPass<GpuConvRewriter>();
  pipeline.AddPass<CudnnFusedConvRewriter>();
  pipeline.AddPass<GpuConvPaddingLegalization>();
  // pipeline.AddPass<CudnnPadForConvolutions>(
  //    stream_exec->GetDeviceDescription().cuda_compute_capability());
  // pipeline.AddPass<CudnnVectorizeConvolutions>(
  //    stream_exec->GetDeviceDescription().cuda_compute_capability());
  // The conv padding/vectorization passes which we need to get rid of.  They
  // also leave behind unnecessary tuple/get-tuple-element pairs that
  // TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  AlgebraicSimplifierOptions algsimp_options;
  algsimp_options.set_enable_conv_operand_swap(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(algsimp_options);

  // CudnnSimplifyPadding gets rid of some padding introduced by
  // CudnnPadForConvolutions and used by CudnnVectorizeConvolutions.  The
  // pattern-matches in this pass need to be run after inlining and simplifying
  // tuples from CudnnVectorizeConvolutions.  We also need to run algsimp to
  // e.g. clean up unnecessary nop `convert`s.
  // pipeline.AddPass<CudnnSimplifyPadding>();

  // tf2xla bridge, DepthwiseConvolutionConverter, GpuConvRewriter, and
  // CudnnSimplifyPadding introduce reshapes and transposes that can be
  // eliminated using AlgebraicSimplifier  We run algsimp to a fixed point.
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(algsimp_options);

  // GpuConvRewriter, GpuConvPaddingLegalization and
  // CudnnConvPadForTensorCores may add instructions which can be simplified
  // by constant folding.
  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return Status::OK();
}

Status SPIRCompiler::OptimizeHloPostLayoutAssignment(HloModule* hlo_module) {
  HloPassPipeline pre_pipeline("spir post-layout_assignment part 1");

  // // This needs to run before GemmRewriter, which is part of
  // // OptimizeHloPostLayoutAssignment().
  // if
  // (stream_exec->GetDeviceDescription().cuda_compute_capability().IsAtLeast(
  //         se::CudaComputeCapability::AMPERE)) {
  //   pre_pipeline.AddPass<CublasPadForGemms>(PrimitiveType::BF16,
  //                                           /*pad_to_multiple_of=*/8);
  // }
  // if
  // (stream_exec->GetDeviceDescription().cuda_compute_capability().IsAtLeast(
  //         se::CudaComputeCapability::VOLTA)) {
  //   // Pad gemms over S8 to multiples of 4 so cuBLAS can run them.
  //   pre_pipeline.AddPass<CublasPadForGemms>(PrimitiveType::S8,
  //                                           /*pad_to_multiple_of=*/4);

  //   // Pad the dimensions of matrices in dot operations to multiples of 8.
  //   pre_pipeline.AddPass<CublasPadForGemms>(PrimitiveType::F16,
  //                                           /*pad_to_multiple_of=*/8);
  // }
  // Padding a gemm operand that's a constant results in pad(constant).  Run
  // constant-folding to simplify this into a new constant.
  pre_pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pre_pipeline.Run(hlo_module).status());

  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(hlo_module));

  HloPassPipeline post_pipeline("spir post-layout_assignment part 2");

  // Find the fastest algorithm for GEMMs. Skip on Ampere and later as the
  // algorithm goes unused.
  // if
  // (!stream_exec->GetDeviceDescription().cuda_compute_capability().IsAtLeast(
  //         se::CudaComputeCapability::AMPERE)) {
  //   post_pipeline.AddPass<GemmAlgorithmPicker>(stream_exec,
  //   device_allocator);
  // }

  if (!IsBefEnabled(hlo_module->config())) {
    // Transform TriangularSolve ops into custom-calls, so we can add temp
    // memory. XLIR allocates temp memory, and so the custom-call implementation
    // for TriangularSolve is not needed.
    post_pipeline.AddPass<TriangularSolveRewriter>();
  }

  TF_RETURN_IF_ERROR(post_pipeline.Run(hlo_module).status());

  return Status::OK();
}

namespace {
absl::optional<bool> CanShareBufferHint(const HloInstruction* user,
                                        const HloInstruction* operand,
                                        const ShapeIndex& user_index) {
  switch (user->opcode()) {
    case HloOpcode::kAllReduce:
      // NCCL all-reduce can be performed in-place.
      return user->operand_count() == 1 ||
             (user_index.size() == 1 &&
              user->operand(user_index[0]) == operand);
    case HloOpcode::kCustomCall:
      // Share the bias buffer with the parent instruction.
      if (user->custom_call_target() == kGemmCallTarget) {
        return user->operand_count() == 3 && user->operand(2) == operand;
      }
      // The operand of cholesky can be shared with the first output.
      if (user->custom_call_target() == kCusolverCholeskyCallTarget) {
        return user_index.size() == 1 && user_index[0] == 0;
      }
      return false;
    default:
      return absl::nullopt;
  }
}

// Try to load textual LLVM IR from files defined in the FLAGS. If
// successful, return the llvm::Module, otherwise return nullptr.
std::unique_ptr<llvm::Module> MaybeLoadLLVMFromFile(const HloModule* module,
                                                    llvm::Module* llvm_module) {
  // If the xla_gpu_llvm_ir_file option is set, be explicit if a file is used
  // and warn when a file is not used to ease catching typo in filename.
  if (module == nullptr) {
    return nullptr;
  }

  std::string prefix = itex_xla::FilenameFor(*module, "", "");
  auto xla_gpu_llvm_ir_file =
      module->config().debug_options().xla_gpu_llvm_ir_file();
  auto matched_filename = absl::c_find_if(
      xla_gpu_llvm_ir_file, [prefix](const std::string& full_filename) {
        // To ease comparing many LLVM versions, accept different suffixes then
        // the original filename.
        return absl::StartsWith(itex::io::Basename(full_filename), prefix);
      });
  if (!xla_gpu_llvm_ir_file.empty() &&
      matched_filename == std::end(xla_gpu_llvm_ir_file)) {
    ITEX_VLOG(0) << "RunBackend() - For module with prefix '" << prefix
                 << "', we did not found a LLVM file to load.";
  }

  if (matched_filename != std::end(xla_gpu_llvm_ir_file)) {
    ITEX_VLOG(0) << "RunBackend() - Will load LLVM from file: "
                 << *matched_filename;
    llvm::LLVMContext& context = llvm_module->getContext();
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> loaded_module =
        llvm::parseIRFile(*matched_filename, err, context);

    if (!loaded_module) {
      err.print("ERR", llvm::errs());
      ITEX_LOG(FATAL)
          << "Failed to load an LLVM file. It is probably invalid LLVM.";
    }
    // Overwrite the dumped not optimized LLVM to show which one will be used.
    llvm_ir::DumpIrIfEnabled(*module, *loaded_module, /*optimized=*/false);
    return loaded_module;
  }
  return nullptr;
}

}  // namespace

SPIRCompiler::SPIRCompiler()
    : GpuCompiler(stream_executor::gpu::kSyclPlatformId, spir::TargetTriple(),
                  spir::DataLayout()) {}

HloDataflowAnalysis::CanShareBuffer SPIRCompiler::GetCanShareBuffer() {
  return &CanShareBufferHint;
}

StatusOr<std::pair<std::string, std::vector<uint8_t>>>
SPIRCompiler::CompileTargetBinary(const HloModuleConfig& module_config,
                                  llvm::Module* llvm_module,
                                  const HloModule* debug_module) {
  std::string libdevice_dir;
  ITEX_VLOG(2) << "Libdevice dir = " << libdevice_dir << "\n";
  std::unique_ptr<llvm::Module> loaded_module =
      MaybeLoadLLVMFromFile(debug_module, llvm_module);
  llvm::Module* selected_module = nullptr;
  if (loaded_module) {
    selected_module = loaded_module.get();
  } else {
    selected_module = llvm_module;
  }

  std::string spir;
  if (debug_module) {
    XLA_SCOPED_LOGGING_TIMER("CompileTargetBinary - CompileToSpir");
    TF_ASSIGN_OR_RETURN(
        spir,
        spir::CompileToSpir(selected_module, module_config, libdevice_dir));
  }
  DumpStringToFileInDirOrStdout("module.spv", spir,
                                module_config.debug_options());

  std::vector<uint8_t> spir_bin(spir.begin(), spir.end());
  return std::pair<std::string, std::vector<uint8_t>>("", std::move(spir_bin));
}

/*static*/ SPIRCompiler* SPIRCompiler::CreateSPIRCompiler() {
  static auto compiler = absl::make_unique<SPIRCompiler>();
  return compiler.get();
}

}  // namespace gpu
}  // namespace itex_xla
