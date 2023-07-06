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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_TARGET_CONSTANTS_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_TARGET_CONSTANTS_H_

namespace itex_xla {
namespace gpu {

namespace spir {
// The triple that represents our target on LLVM AMDGPU backend.
inline const char* TargetTriple() {
  static constexpr char kTargetTriple[] = "spir64-unknown-unknown";
  return kTargetTriple;
}

// The data layout of the emitted module.
inline const char* DataLayout() {
  // Specifies the address space as global address space
  // A1: Specifies the address space of objects created by ‘alloca’.
  // P1: Specifies the address space that corresponds to program memory.
  // G1: Specifies the address space of global variables.
  static constexpr char kDataLayout[] =
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:"
      "32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:"
      "128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:"
      "1024";
  return kDataLayout;
}

}  // namespace spir
}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_TARGET_CONSTANTS_H_
