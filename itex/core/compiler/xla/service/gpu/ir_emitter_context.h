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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_

#include <string>
#include <utility>
#include <vector>

#include "itex/core/compiler/xla/service/buffer_assignment.h"
#include "itex/core/compiler/xla/service/gpu/buffer_allocations.h"
#include "itex/core/compiler/xla/service/gpu/gpu_executable.h"
#include "itex/core/compiler/xla/service/gpu/launch_dimensions.h"
#include "itex/core/compiler/xla/service/hlo_execution_profile.h"
#include "itex/core/compiler/xla/service/name_uniquer.h"
#include "lhlo/IR/lhlo_ops.h"
#include "lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "llvm/IR/Module.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"           // from @llvm-project

namespace itex_xla {
namespace gpu {

// IrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class IrEmitterContext {
 public:
  IrEmitterContext(const HloModule* hlo_module,
                   const BufferAssignment* buffer_assignment,
                   std::string platform_name, GpuDeviceInfo gpu_device_info,
                   mlir::MLIRContext* mlir_context, llvm::Module* llvm_module,
                   bool mlir_pass_success = false)
      : hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        platform_name_(std::move(platform_name)),
        gpu_device_info_(gpu_device_info),
        mlir_context_(mlir_context),
        llvm_module_(llvm_module),
        mlir_pass_success_(mlir_pass_success) {}
  // Disallow copy and assign.
  IrEmitterContext(const IrEmitterContext&) = delete;
  IrEmitterContext& operator=(const IrEmitterContext&) = delete;

  // Simple accessors.
  const HloModule& hlo_module() const { return *hlo_module_; }
  const BufferAssignment& buffer_assignment() const {
    return *buffer_assignment_;
  }
  absl::string_view platform_name() const { return platform_name_; }
  GpuDeviceInfo gpu_device_info() const { return gpu_device_info_; }
  mlir::MLIRContext* mlir_context() { return mlir_context_; }
  llvm::Module* llvm_module() { return llvm_module_; }
  NameUniquer* name_uniquer() { return &name_uniquer_; }
  bool mlir_pass_success() const { return mlir_pass_success_; }

  std::vector<GpuExecutable::ConstantInfo>& constants() { return constants_; }

  absl::Span<const BufferAllocation> allocations() const {
    if (buffer_assignment_) {
      return buffer_assignment_->Allocations();
    }
    return allocations_;
  }

  void set_allocations(absl::Span<const BufferAllocation> allocations) {
    ITEX_CHECK_EQ(nullptr, buffer_assignment_);
    allocations_ = allocations;
  }

 private:
  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;
  absl::Span<const BufferAllocation> allocations_;
  std::string platform_name_;
  GpuDeviceInfo gpu_device_info_;
  mlir::MLIRContext* mlir_context_;
  llvm::Module* llvm_module_;
  NameUniquer name_uniquer_;
  std::vector<GpuExecutable::ConstantInfo> constants_;
  bool mlir_pass_success_;
};

}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
