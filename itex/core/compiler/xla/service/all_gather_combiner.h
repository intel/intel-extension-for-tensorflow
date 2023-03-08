/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_ALL_GATHER_COMBINER_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_ALL_GATHER_COMBINER_H_

#include "absl/strings/string_view.h"
#include "itex/core/compiler/xla/array2d.h"
#include "itex/core/compiler/xla/service/hlo_module.h"
#include "itex/core/compiler/xla/service/hlo_pass_interface.h"
#include "itex/core/compiler/xla/statusor.h"
#include "protos/xla_data.pb.h"

namespace itex_xla {

// Combines small non-dependent AllGather ops into larger combined
// AllGather ops. A typical AllGather implementation has a minimum
// latency-induced time for a AllGather op so a single combined op can be
// more efficient than many small ones.
class AllGatherCombiner : public HloModulePass {
 public:
  AllGatherCombiner(int64_t combine_threshold_in_bytes,
                    int64_t combine_threshold_count);

  absl::string_view name() const override { return "all-gather-combiner"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Combine all gather ops up to this threshold.
  int64_t combine_threshold_in_bytes_;

  // Combine all gather ops up to this threshold (number of operands).
  int64_t combine_threshold_count_;
};

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_ALL_GATHER_COMBINER_H_
