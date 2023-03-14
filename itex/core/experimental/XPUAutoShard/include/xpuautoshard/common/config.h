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
#include "xpuautoshard/common/ref_base.h"
namespace as {

// TODO(itex): Refactor the data structure to support additional
// params for each strategy.
enum StrategyKind {
  CPU_HOST,
  HEURISTIC,
  LEARNED,
};

struct HeuristicsConfig {
  HeuristicsConfig() : multi_stage_enabled_(false), batch_grain_size_(1) {}

  bool isMultiStageEnabled() const { return multi_stage_enabled_; }
  void setMultiStageEnabled(bool enable) { multi_stage_enabled_ = enable; }
  int64_t getBatchGrainSize() const { return batch_grain_size_; }
  void setBatchGrainSize(int64_t grain_size) { batch_grain_size_ = grain_size; }

 private:
  bool multi_stage_enabled_;
  int64_t batch_grain_size_;
};

struct ShardingConfig {
  ShardingConfig()
      : strategy_kind_(StrategyKind::CPU_HOST),
        use_nccl_comm_backend_(false),
        use_multi_stage_join_(true),
        need_dead_node_prune_(true) {}

  HeuristicsConfig& getHeuristicsConfig() { return heuristics_config_; }
  const HeuristicsConfig& getHeuristicsConfig() const {
    return heuristics_config_;
  }

  void setStrategyKind(StrategyKind strategy_kind) {
    strategy_kind_ = strategy_kind;
  }

  StrategyKind getStrategyKind() const { return strategy_kind_; }

  void setUseNcclCommBackend(bool use_nccl_comm_backend) {
    use_nccl_comm_backend_ = use_nccl_comm_backend;
  }

  /**
   * @brief Whether we use NCCL pr oneCCL communication backend for all-reduce
   * etc.
   *
   * @return true Use NCCL comm backend
   * @return false Use oneCCL comm backend
   */
  bool isUseNcclCommBackend() const { return use_nccl_comm_backend_; }

  void setUseMultiStageJoin(bool use_multi_stage_join) {
    use_multi_stage_join_ = use_multi_stage_join;
  }

  bool isUseMultiStageJoin() const { return use_multi_stage_join_; }

  void setNeedDeadNodePrune(bool need_dead_node_prune) {
    need_dead_node_prune_ = need_dead_node_prune;
  }

  /**
   * @brief Whether we prune dead nodes that cannot reach `Preserve Nodes`
   * before auto-sharding.
   *
   * @return true need Prune dead nodes
   * @return false need Prune dead nodes
   */
  bool isNeedDeadNodePrune() const { return need_dead_node_prune_; }

 private:
  HeuristicsConfig heuristics_config_;
  StrategyKind strategy_kind_;
  bool use_nccl_comm_backend_;
  bool use_multi_stage_join_;
  bool need_dead_node_prune_;
};

}  // namespace as
