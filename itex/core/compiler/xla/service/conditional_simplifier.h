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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_CONDITIONAL_SIMPLIFIER_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_CONDITIONAL_SIMPLIFIER_H_

#include "absl/strings/string_view.h"
#include "itex/core/compiler/xla/service/hlo_module.h"
#include "itex/core/compiler/xla/service/hlo_pass_interface.h"
#include "itex/core/compiler/xla/statusor.h"

namespace itex_xla {

// HLO pass that removes kConditional with a constant predicate, replacing them
// with their true or false computation as appropriate.
class ConditionalSimplifier : public HloModulePass {
 public:
  absl::string_view name() const override { return "simplify-conditional"; }
  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> TryRemoveConditional(HloInstruction* conditional);
};

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_CONDITIONAL_SIMPLIFIER_H_
