/* Copyright (c) 2023 Intel Corporation

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

#pragma once

#include <vector>

#include "xpuautoshard/common/config.h"
#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/mlir/passes/hsp_initializer.h"
#include "xpuautoshard/common/mlir/passes/mlir_hsp_annotator.h"
#include "xpuautoshard/common/ref_base.h"

namespace mlir {
namespace hs {

class HeuristicsInitializer : public HspInitializer {
 public:
  HeuristicsInitializer(const as::DeviceInfo& device_info,
                        const as::HeuristicsConfig& heuristics_config,
                        MLIRAnnotationRef annot)
      : device_info_(device_info),
        heuristics_config_(heuristics_config),
        annot_(annot) {}

  bool initSome(Operation* root_op) override;

 private:
  as::DeviceInfo device_info_;
  as::HeuristicsConfig heuristics_config_;
  MLIRAnnotationRef annot_;

  bool tryBatchSplitValue(Value value, const std::vector<float>& ratios);
  /**
   * @brief Get the Split Ratios by devices in `device_info_` according to
   * the DAG given by `root_op`.
   *
   * @param root_op
   * @return std::vector<float>
   */
  std::vector<float> getSplitRatios(Operation* root_op);
  bool tryBatchSplit(Operation* root_op);
  bool tryPropSingleSplitOnlyInputs(Operation* root_op);
  bool tryInitReshapeMatmul(Operation* root_op);
  bool trySingleSplitOnlyForShardOp(Operation* root_op,
                                    bool only_const_flag = false);
  bool tryInitReshape(Operation* root_op);
};

using HeuristicsInitializerRef = as::Ref<HeuristicsInitializer>;

}  // namespace hs
}  // namespace mlir
