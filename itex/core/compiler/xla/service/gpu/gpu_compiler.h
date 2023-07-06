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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "itex/core/compiler/xla/service/gpu/gpu_executable.h"
#include "itex/core/compiler/xla/service/hlo_pass_pipeline.h"
#include "itex/core/compiler/xla/service/llvm_compiler.h"
#include "itex/core/compiler/xla/service/service_executable_run_options.h"
#include "itex/core/compiler/xla/status_macros.h"
#include "itex/core/compiler/xla/types.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/statusor.h"

namespace itex_xla {
namespace gpu {
// The GPU compiler generates efficient GPU executables.
class GpuCompiler : public LLVMCompiler {
 public:
  GpuCompiler(se::Platform::Id platform_id, const char* target_triple,
              const char* data_layout);
  ~GpuCompiler() override {}

  using LLVMCompiler::Compile;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  // StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
  //     const HloModule* hlo_module) override;

  // virtual GpuVersion GetGpuVersion(se::StreamExecutor* stream_exec) = 0;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<AotCompilationResult>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override {
    ITEX_LOG(FATAL) << "LoadAotCompilationResult is not supported";
  }

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     AotCompilationOptions const& options) override {
    ITEX_LOG(FATAL) << "CompileAheadOfTime is not supported";
  }

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileToTargetBinary(
      const HloModuleConfig& module_config,
      std::unique_ptr<llvm::Module> llvm_module, const HloModule* debug_module);

  se::Platform::Id PlatformId() const override { return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

 protected:
  virtual Status OptimizeHloPostLayoutAssignment(HloModule* hlo_module);

 private:
  Status OptimizeHloModule(HloModule* hlo_module);

  virtual Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module) = 0;

  virtual HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() {
    return
        [](const HloInstruction*, const HloInstruction*,
           const ShapeIndex&) -> absl::optional<bool> { return absl::nullopt; };
  }

  // TODO(timshen): Replace `debug_module` with some portable debug information
  // that accommodates both HLO and MLIR.
  virtual StatusOr<std::pair<std::string, std::vector<uint8_t>>>
  CompileTargetBinary(const HloModuleConfig& module_config,
                      llvm::Module* llvm_module,
                      const HloModule* debug_module) = 0;

  Status PrepareHloModuleForIrEmitting(HloModule* hlo_module);

  // virtual StatusOr<std::vector<uint8_t>> LinkModules(
  //     se::StreamExecutor* stream_exec,
  //     std::vector<std::vector<uint8_t>> modules) {
  //   return Unimplemented("LinkModules is not implemented.");
  // }

  // Optional HloProto, stashed for dumping snapshots.
  std::unique_ptr<HloProto> hlo_proto_;

  se::Platform::Id platform_id_;

  // The triple that represents our target.
  const char* target_triple_;

  // The data layout of the emitted module.
  const char* data_layout_;

  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64_t pointer_size_;

  GpuCompiler(const GpuCompiler&) = delete;
  GpuCompiler& operator=(const GpuCompiler&) = delete;
};

}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
