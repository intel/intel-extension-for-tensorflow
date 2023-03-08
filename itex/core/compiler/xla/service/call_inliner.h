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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_CALL_INLINER_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_CALL_INLINER_H_

#include <deque>

#include "absl/container/flat_hash_map.h"
#include "itex/core/compiler/xla/service/hlo_pass_interface.h"
#include "itex/core/compiler/xla/statusor.h"

namespace itex_xla {

// For every kCall operation in the main computation, we inline the body of the
// called function, and proceed recursively.
class CallInliner : public HloModulePass {
 public:
  using InlinedInstructionMap =
      absl::flat_hash_map<HloInstruction*, HloInstruction*>;

  // Inlines one call instruction.  Returns a mapping from the original
  // instructions to their inlined versions.
  static StatusOr<InlinedInstructionMap> Inline(HloInstruction* call);

  // If single_call_site is true, only functions with a single call site will be
  // inlined.
  // If update_domain is true, the exit domains could be updated for calls which
  // are being inlined if necessary.
  explicit CallInliner(bool single_call_site = false,
                       bool update_domain = false)
      : single_call_site_(single_call_site), update_domain_(update_domain) {}
  ~CallInliner() override = default;
  absl::string_view name() const override { return "CallInliner"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool single_call_site_;
  bool update_domain_;
};

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_CALL_INLINER_H_
