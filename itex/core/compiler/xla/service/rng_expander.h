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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_RNG_EXPANDER_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_RNG_EXPANDER_H_

#include <tuple>

#include "itex/core/compiler/xla/service/op_expander_pass.h"

namespace itex_xla {

class RngExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "rng-expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(HloInstruction* rng) override;

 private:
  // Cache RNG computations based on the distribution, output shape and shapes
  // of the first and second operand.
  absl::flat_hash_map<std::tuple<RandomDistribution, Shape, Shape, Shape>,
                      HloComputation*>
      expanded_rng_instructions_;
};

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_RNG_EXPANDER_H_
