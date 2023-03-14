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

#include "xpuautoshard/common/config.h"
#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/hsp_tuner.h"

namespace mlir {
namespace hs {

class MLIRHspTuner : public as::HspTuner {
 public:
  MLIRHspTuner(as::GraphRef graph, const as::DeviceInfo& device_info,
               const as::ShardingConfig& sharding_config)
      : graph_(graph),
        device_info_(device_info),
        sharding_config_(sharding_config) {}

  as::HspCostEvaluatorRef getCostModel() override {
    return as::makeRef<as::DummyHspCostEvaluator, as::HspCostEvaluator>();
  }

  as::TuningStateRef nextState() override { return nullptr; }

  void updateScore(const as::ScoreRef& score,
                   as::TuningStateRef tuning_state = nullptr) override {}

  as::HspAnnotatorRef createAnnotator(
      as::TuningStateRef tuning_state = nullptr) override;

  bool stopCriterionMet() override { return true; }

 private:
  as::GraphRef graph_;
  as::DeviceInfo device_info_;
  as::ShardingConfig sharding_config_;
};

}  // namespace hs
}  // namespace mlir
