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

#include "xpuautoshard/common/cost_model.h"
#include "xpuautoshard/common/device_info.h"

namespace as {

/**
 * @brief Analytically model a workload w.r.t. the amount of arithmetic and
 * memory operations in the workload computation.
 *
 */
class QuantitativeComputeCharacteristics : public ComputeCharacteristics {
 public:
  /**
   * @brief Construct a default quantitative compute characteristics object with
   * zero statistics.
   *
   */
  QuantitativeComputeCharacteristics()
      : num_float_ops_(0),
        num_bfloat16_ops_(0),
        num_float16_ops_(0),
        num_int8_ops_(0),
        memory_load_bytes_(0),
        memory_store_bytes_(0) {}

  /**
   * @brief Sum up the compute characteristics in `rhs` to this.
   *
   * @param rhs
   * @return QuantitativeComputeCharacteristics&
   */
  QuantitativeComputeCharacteristics& operator+=(
      const QuantitativeComputeCharacteristics& rhs);

  /**
   * @brief Get the number of float ops
   *
   * @return float
   */
  float getNumFloatOps() const { return num_float_ops_; }

  void setNumFloatOps(float ops) { num_float_ops_ = ops; }

  /**
   * @brief Get the number of bfloat16 ops
   *
   * @return float
   */
  float getNumBfloat16Ops() const { return num_bfloat16_ops_; }

  void setNumBfloat16Ops(float ops) { num_bfloat16_ops_ = ops; }

  /**
   * @brief Get the number of float16 ops
   *
   * @return float
   */
  float getNumFloat16Ops() const { return num_float16_ops_; }

  void setNumFloat16Ops(float ops) { num_float16_ops_ = ops; }

  /**
   * @brief Get the number of int8 ops
   *
   * @return float
   */
  float getNumInt8Ops() const { return num_int8_ops_; }

  void setNumInt8Ops(float ops) { num_int8_ops_ = ops; }

  /**
   * @brief Get the memory load in bytes
   *
   * @return float
   */
  float getMemoryLoadBytes() const { return memory_load_bytes_; }

  void setMemoryLoadBytes(float bytes) { memory_load_bytes_ = bytes; }

  /**
   * @brief Get the memory writes in bytes
   *
   * @return float
   */
  float getMemoryStoreBytes() const { return memory_store_bytes_; }

  void setMemoryStoreBytes(float bytes) { memory_store_bytes_ = bytes; }

 private:
  float num_float_ops_;
  float num_bfloat16_ops_;
  float num_float16_ops_;
  float num_int8_ops_;
  float memory_load_bytes_;
  float memory_store_bytes_;
};

using QuantitativeComputeCharacteristicsRef =
    Ref<QuantitativeComputeCharacteristics>;

/**
 * @brief Analytically model a workload w.r.t. its qualitative characteristics
 * like compute data types, training or inference etc.
 *
 */
class QualitativeComputeCharacteristics : public ComputeCharacteristics {
 public:
  QualitativeComputeCharacteristics(DataType dtype = DataType::FLOAT32,
                                    bool maybe_training = false);

  /**
   * @brief Implicitly instantiate a qualitative characteristics from a
   * quantitative one.
   *
   * @param quan_comp_ch
   */
  QualitativeComputeCharacteristics(
      const QuantitativeComputeCharacteristics& quan_comp_ch);

  QualitativeComputeCharacteristics& operator+=(
      const QualitativeComputeCharacteristics& rhs);

  /**
   * @brief The workload may be for training
   *
   * @return true
   * @return false
   */
  bool maybeTraining() const { return maybe_training_; }

  /**
   * @brief The workload has bfloat16 multiply-add compute
   *
   * @return true
   * @return false
   */
  bool hasBfloat16Compute() const { return has_bfloat16_; }

  /**
   * @brief The workload has float16 multiply-add compute
   *
   * @return true
   * @return false
   */
  bool hasFloat16Compute() const { return has_float16_; }

  /**
   * @brief The workload has int8 multiply-add compute
   *
   * @return true
   * @return false
   */
  bool hasInt8Compute() const { return has_int8_; }

 private:
  bool has_bfloat16_;
  bool has_float16_;
  bool has_int8_;
  bool maybe_training_;
};

using QualitativeComputeCharacteristicsRef =
    Ref<QualitativeComputeCharacteristics>;

/**
 * @brief Builder for analytic compute characteristics of a workload w.r.t. the
 * amount of arithmetic and memory operations in the workload computation, or
 * its qualitative characteristics if quantitative approach is not feasible
 * (e.g., shapes unknown)
 *
 */
class AnalyticComputeCharacterizer : public ComputeCharacterizer {
 public:
  ComputeCharacteristicsRef characterize(GraphRef graph) override;
  ComputeCharacteristicsRef characterize(OpDescRef op_desc) override;

 private:
  /**
   * @brief Characterize conv forward
   *
   * @param op_desc The op descriptor for conv forward
   * @return ComputeCharacteristicsRef
   */
  ComputeCharacteristicsRef characterizeConvForward(OpDescRef op_desc);

  /**
   * @brief Characterize conv backard by data
   *
   * @param op_desc The op descriptor for conv backward by data
   * @return ComputeCharacteristicsRef
   */
  ComputeCharacteristicsRef characterizeConvBackwardData(OpDescRef op_desc);

  /**
   * @brief Characterize conv backward by weight
   *
   * @param op_desc The op descriptor for conv backward by weight
   * @return ComputeCharacteristicsRef
   */
  ComputeCharacteristicsRef characterizeConvBackwardWeight(OpDescRef op_desc);

  /**
   * @brief Characterize matmul
   *
   * @param op_desc The op descriptor for matmul
   * @return ComputeCharacteristicsRef
   */
  ComputeCharacteristicsRef characterizeMatMul(OpDescRef op_desc);

  /**
   * @brief
   *
   * @param op_desc The op descriptor for a memory-bound op
   * @return ComputeCharacteristicsRef
   */
  ComputeCharacteristicsRef characterizeMemoryOp(OpDescRef op_desc);

  /**
   * @brief Add memory ops to the `analytic_ch`
   *
   * @param analytic_ch
   * @param dtype Data type of the memory ops
   * @param ops Number of memory ops
   * @param is_load Whether it is a memory load
   */
  void addMemoryOps(QuantitativeComputeCharacteristicsRef analytic_ch,
                    DataType dtype, float ops, bool is_load);

  /**
   * @brief Add multiply-add compute ops to the `analytic_ch`. Multiply and add
   * are counted separately.
   *
   * @param analytic_ch
   * @param dtype Data type of the compute ops
   * @param ops Number of compute ops
   */
  void addMultiplyAddComputeOps(
      QuantitativeComputeCharacteristicsRef analytic_ch, DataType dtype,
      float ops);

  float getConvOps(int64_t bs, int64_t ic, int64_t oc,
                   const std::vector<int64_t>& kernel_dims,
                   const std::vector<int64_t>& out_dims);
};

using AnalyticComputeCharacterizerRef = Ref<AnalyticComputeCharacterizer>;

/**
 * @brief Evaluate the cost of a workload with the analytic model
 *
 */
class AnalyticCostModel : public CostModel {
 public:
  explicit AnalyticCostModel(const DeviceComputeCapability& device_cap)
      : device_cap_(device_cap) {}

  ComputeCharacterizerRef createComputeCharacterizer() override;
  TimeCost evaluateTime(ComputeCharacteristicsRef comp_ch) override;

 private:
  DeviceComputeCapability device_cap_;

  /**
   * @brief Evaluate time cost qualitatively.
   *
   * @return TimeCost
   */
  TimeCost evaluateTimeQualitative(
      QualitativeComputeCharacteristicsRef comp_ch);

  /**
   * @brief Evaluate the time for fp32 multiply-add ops
   *
   * @param comp_ch
   * @return TimeCost
   */
  TimeCost evaluateTimeFloat(QuantitativeComputeCharacteristicsRef comp_ch);

  /**
   * @brief Evaluate the time for bf16 multiply-add ops
   *
   * @param comp_ch
   * @return TimeCost
   */
  TimeCost evaluateTimeBfloat16(QuantitativeComputeCharacteristicsRef comp_ch);

  /**
   * @brief Evaluate the time for fp16 multiply-add ops
   *
   * @param comp_ch
   * @return TimeCost
   */
  TimeCost evaluateTimeFloat16(QuantitativeComputeCharacteristicsRef comp_ch);

  /**
   * @brief Evaluate the time for int8 multiply-add ops
   *
   * @param comp_ch
   * @return TimeCost
   */
  TimeCost evaluateTimeInt8(QuantitativeComputeCharacteristicsRef comp_ch);

  /**
   * @brief Evaluate the time for memory load and store
   *
   * @param comp_ch
   * @return TimeCost
   */
  TimeCost evaluateTimeMemory(QuantitativeComputeCharacteristicsRef comp_ch);

  // TODO(itex): model cache access
};

using AnalyticCostModelRef = Ref<AnalyticCostModel>;

}  // namespace as
