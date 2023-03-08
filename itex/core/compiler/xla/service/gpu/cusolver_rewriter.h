/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_CUSOLVER_REWRITER_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_CUSOLVER_REWRITER_H_

#include "itex/core/compiler/xla/service/hlo_computation.h"
#include "itex/core/compiler/xla/service/hlo_module.h"
#include "itex/core/compiler/xla/service/hlo_pass_interface.h"
#include "itex/core/compiler/xla/stream_executor/device_memory_allocator.h"

namespace itex_xla {
namespace gpu {

// Rewrites Cholesky calls into CustomCall HLOs that call into cuSolver.
class GpusolverRewriter : public HloModulePass {
 public:
  GpusolverRewriter();
  absl::string_view name() const override { return "gpusolver-rewriter"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation* computation);
};

}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_CUSOLVER_REWRITER_H_
