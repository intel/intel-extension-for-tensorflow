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

// LLVM-based compiler backend.
#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "itex/core/compiler/xla/service/hlo_module_config.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/compiler/xla/types.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace itex_xla {
namespace gpu {

namespace spir {
StatusOr<std::string> CompileToSpir(llvm::Module* module,
                                    const HloModuleConfig& hlo_module_config,
                                    const std::string& libdevice_dir_path);
}  // namespace spir

}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
