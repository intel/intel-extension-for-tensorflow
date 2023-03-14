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
#include <memory>

#include "xpuautoshard/common/hsp_annotator.h"
#include "xpuautoshard/common/hsp_cost_evaluator.h"
#include "xpuautoshard/common/ref_base.h"

namespace as {

/**
 * @brief The state of tuning usually contains the parameters for the tuning
 * algorithm.
 *
 */
class TuningState {
 public:
  virtual ~TuningState() = default;
};

using TuningStateRef = std::shared_ptr<TuningState>;

/**
 * @brief The tuner for identifying the optimal HSP annotation solution for
 * framework graphs given device info and sharding configuration.
 *
 */
class HspTuner {
 public:
  virtual ~HspTuner() = default;

  /**
   * @brief Tune the best HSP annotation given a `graph`.
   *
   * @param graph
   * @return HspAnnotationRef
   */
  virtual HspAnnotationRef tune(GraphRef graph);

  /**
   * @brief Get the Cost Model for evaluating the performance of
   * an HSP annotation solution.
   *
   * @return CostModelRef
   */
  virtual HspCostEvaluatorRef getCostModel() = 0;

  /**
   * @brief Identify the next tuning state to tune. The current tuning
   * state of the tuner is updated with this next tuning state accordingly.
   *
   * @return TuningStateRef The next tuning state.
   */
  virtual TuningStateRef nextState() = 0;

  /**
   * @brief Update the score corresponding to the given `tuning_state`. If the
   * `tuning_state` is nullptr, the current tuning state is implied.
   *
   * @param score
   * @param tuning_state
   */
  virtual void updateScore(const ScoreRef& score,
                           TuningStateRef tuning_state = nullptr) = 0;

  /**
   * @brief Create an HSPAnnotator object given the `tuning_state`. If the
   * `tuning_state` is nullptr, the current tuning state is implied.
   *
   * @param tuning_state
   * @return HspAnnotatorRef
   */
  virtual HspAnnotatorRef createAnnotator(
      TuningStateRef tuning_state = nullptr) = 0;

  /**
   * @brief Check if the stop criterion is met to stop the tuning loop.
   *
   * @return true
   * @return false
   */
  virtual bool stopCriterionMet() = 0;
};

using HspTunerRef = Ref<HspTuner>;

}  // namespace as
