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

#include "xpuautoshard/common/graph.h"

namespace as {

/**
 * @brief An abstraction of a compute workload. The implementation can
 * characterize the workload at a specific abstraction level. Below is a
 * candidate list of characteristics from more abstract to more concrete:
 * 1. A total amount of arithmetic calculation, memory load and store.
 * 2. A sequence of arithmetic calculation, memory load and store operations.
 * 3. Materialized code that can run on a specific HW.
 *
 */
class ComputeCharacteristics {
 public:
  virtual ~ComputeCharacteristics() = default;
};

using ComputeCharacteristicsRef = Ref<ComputeCharacteristics>;

/**
 * @brief Returned when a characterizer cannot decide the compute
 * characteristics.
 *
 */
class UnknownComputeCharacteristics : public ComputeCharacteristics {
 public:
  /**
   * @brief A static helper function for constructing an unknown compute
   * characteristics
   *
   * @return Ref<UnknownComputeCharacteristics>
   */
  static Ref<UnknownComputeCharacteristics> get() {
    static auto ref = makeRef<UnknownComputeCharacteristics>();
    return ref;
  }
};

using UnknownComputeCharacteristicsRef = Ref<UnknownComputeCharacteristics>;

/**
 * @brief Characterize the computation of a DAG or an op
 *
 */
class ComputeCharacterizer {
 public:
  virtual ~ComputeCharacterizer() = default;
  virtual ComputeCharacteristicsRef characterize(GraphRef graph) = 0;
  virtual ComputeCharacteristicsRef characterize(OpDescRef op_desc) = 0;
};

using ComputeCharacterizerRef = Ref<ComputeCharacterizer>;

/**
 * @brief A cost model for evaluating the cost of a characterized computation
 *
 */
class CostModel {
 public:
  /**
   * @brief Time cost with metrics defined by the cost model
   *
   */
  using TimeCost = float;

  virtual ~CostModel() = default;
  /**
   * @brief Create a Compute Characterizer object corresponding to this cost
   * model. The computer characterizer is used to characterize the computation
   * for this cost model to evaluate the cost.
   *
   * @return ComputeCharacterizerRef
   */
  virtual ComputeCharacterizerRef createComputeCharacterizer() = 0;

  /**
   * @brief Evaluate the time cost of a compute.
   *
   * @param comp_ch
   * @return TimeCost
   */
  virtual TimeCost evaluateTime(ComputeCharacteristicsRef comp_ch) = 0;

  // TODO(itex): add evaluatePower and evaluateMemUsage
};

using CostModelRef = Ref<CostModel>;

}  // namespace as
