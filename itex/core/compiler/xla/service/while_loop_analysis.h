/* Copyright (c) 2023 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_
#include <vector>

#include "absl/types/optional.h"
#include "itex/core/compiler/xla/service/hlo_instruction.h"

namespace itex_xla {

// Returns the precise trip count of the loop if it's statically known,
// nullopt otherwise.
//
// max_brute_force_iters limits the number of steps that are evaluated while
// trying to brute force a loop trip count. trip counts larger than
// max_brute_force_iters may be returned if we can pattern-match the loop
// condition.
absl::optional<int64_t> ComputeWhileLoopTripCount(
    HloInstruction* while_op, int64_t max_brute_force_iters = 128);

// Returns an upper bound on the trip count of the loop if it's statically
// known, nullopt otherwise.
absl::optional<int64_t> ComputeWhileLoopTripCountUpperBound(
    HloInstruction* while_op);

// The below function identifies a subset of all possible auxiliary
// induction variables (AIV). Specifically, candidates are gtes, e.g.,
// gte(param0, N)
std::vector<const HloInstruction*> GetAuxiliaryLoopInductionVars(
    const HloInstruction* while_op);
// Returns the tuple index of the loop induction variable if there is such an
// induction variable detected. Otherwise returns nullopt.
absl::optional<int64_t> GetLoopInductionVarTupleIdx(
    const HloInstruction* while_op);
}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_
