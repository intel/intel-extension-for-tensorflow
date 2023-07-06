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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_TRANSPOSE_FOLDING_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_TRANSPOSE_FOLDING_H_

#include <vector>

#include "itex/core/compiler/xla/service/hlo_module.h"
#include "itex/core/compiler/xla/service/hlo_pass_interface.h"

namespace itex_xla {

// HLO pass that folds transpose operators into Dot operators, where the Dot
// operator is implemented by a GEMM kernel that can transpose its inputs.
class TransposeFolding : public HloModulePass {
 public:
  using OperandIndices = std::vector<int64_t>;

  // Returns the set of foldable operands for a given HLO and some candidate
  // operands.
  using FoldableOperands = std::function<OperandIndices(const HloInstruction&,
                                                        const OperandIndices&)>;
  using TransposableGemmOperandsFn = FoldableOperands;
  using TransposableConvOperandsFn = FoldableOperands;

  // Helper function to explicitly not fold transposes.
  static OperandIndices NeverFoldTranspose(const HloInstruction&,
                                           const OperandIndices&) {
    return {};
  }

  // Helper function to always fold transposes.
  static OperandIndices AlwaysFoldTranspose(const HloInstruction&,
                                            const OperandIndices& ids) {
    return ids;
  }

  // transposable_gemm_operands returns the set of operands it wants to fold if
  // the instruction argument is implemented as a GEMM kernel that supports
  // transposing its arguments.
  //
  // transposable_conv_operands returns the set of operands it wants to fold if
  // the instruction argument is implemented as a convolution that supports
  // transposing its arguments.
  explicit TransposeFolding(
      TransposableGemmOperandsFn transposable_gemm_operands =
          AlwaysFoldTranspose,
      TransposableConvOperandsFn transposable_conv_operands =
          AlwaysFoldTranspose);
  absl::string_view name() const override { return "transpose-folding"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  TransposableGemmOperandsFn transposable_gemm_operands_;
  TransposableConvOperandsFn transposable_conv_operands_;
};

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_TRANSPOSE_FOLDING_H_
