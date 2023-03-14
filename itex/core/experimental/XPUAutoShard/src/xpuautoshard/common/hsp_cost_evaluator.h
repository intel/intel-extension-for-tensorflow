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

#include <limits>

#include "xpuautoshard/common/graph.h"
#include "xpuautoshard/common/hsp_annotator.h"
#include "xpuautoshard/common/ref_base.h"

namespace as {

/**
 * @brief The score evaluated by the cost model. The higher, the better.
 *
 */
class Score {
 public:
  virtual ~Score() = default;
  virtual bool operator==(const Score& rhs) = 0;
  virtual bool operator<(const Score& rhs) = 0;
};

using ScoreRef = Ref<Score>;

/**
 * @brief A cost model used to evaluate the goodness (e.g. performance) of
 * the framework graph annotated with HSPs.
 *
 */
class HspCostEvaluator {
 public:
  virtual ~HspCostEvaluator() = default;
  /**
   * @brief The lowest score as measured by this cost model.
   *
   * @return ScoreRef
   */
  virtual ScoreRef lowestScore() = 0;

  /**
   * @brief Evaluate the score of the graph given an HSP annotation. If the HSP
   * annotation is null, the default HSP info on the graph will be used.
   *
   * @param graph
   * @param annotation
   * @return ScoreRef
   */
  virtual ScoreRef evaluate(GraphRef graph,
                            HspAnnotationRef annotation = nullptr) = 0;
};

using HspCostEvaluatorRef = Ref<HspCostEvaluator>;

/**
 * @brief A dummy cost model that always returns the minimal score
 *
 */
class DummyHspCostEvaluator : public HspCostEvaluator {
 private:
  class FloatScore : public Score {
   public:
    explicit FloatScore(float score = std::numeric_limits<float>::lowest())
        : score_(score) {}

    bool operator==(const Score& rhs) override {
      const auto& float_rhs = dynamic_cast<const FloatScore&>(rhs);
      return score_ == float_rhs.score_;
    }

    bool operator<(const Score& rhs) override {
      const auto& float_rhs = dynamic_cast<const FloatScore&>(rhs);
      return score_ < float_rhs.score_;
    }

   private:
    float score_;
  };

 public:
  DummyHspCostEvaluator() {}

  ScoreRef lowestScore() override { return makeRef<FloatScore, Score>(); }

  ScoreRef evaluate(GraphRef graph,
                    HspAnnotationRef annotation = nullptr) override {
    return makeRef<FloatScore, Score>(0.0f);
  }
};

};  // namespace as
