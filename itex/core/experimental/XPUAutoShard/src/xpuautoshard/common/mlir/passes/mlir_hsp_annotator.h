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

#include <unordered_map>
#include <utility>

#include "xpuautoshard/common/config.h"
#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/hsp_annotator.h"
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/mlir/passes/mlir_graph.h"
#include "xpuautoshard/common/sharding_property.h"

namespace mlir {
namespace hs {

using as::ShardingPropertyRefVec;

struct MLIRAnnotation : public as::HspAnnotation {
  /**
   * @brief Get the Result Hsps object of an op
   *
   * @param op
   * @return ShardingPropertyRefVec&
   */
  ShardingPropertyRefVec& getResultHsps(Operation* op) {
    return annot_map_[op];
  }

  /**
   * @brief Add the result hsps of an op to the annotation
   *
   * @param op
   * @param hsps_pair
   */
  void insert(Operation* op, as::ShardingPropertyRefVec&& result_hsps) {
    annot_map_.insert({op, std::move(result_hsps)});
  }

  std::pair<ShardingPropertyRefVec, ShardingPropertyRefVec>
  getShardingPropertiesForOp(Operation* op);

  as::ShardingPropertyRef getShardingPropertyForValue(Value value);

 private:
  std::unordered_map<Operation*, ShardingPropertyRefVec> annot_map_;
};
using MLIRAnnotationRef = as::Ref<MLIRAnnotation>;

class MLIRHspAnnotator : public as::HspAnnotator {
 public:
  MLIRHspAnnotator(as::GraphRef graph, const as::DeviceInfo& device_info,
                   const as::ShardingConfig& sharding_config)
      : mlir_graph_(as::downcastRef<MLIRGraph>(graph)),
        device_info_(device_info),
        sharding_config_(sharding_config) {}

  as::HspAnnotationRef annotate(as::GraphRef graph) override;

 private:
  MLIRGraphRef mlir_graph_;
  as::DeviceInfo device_info_;
  as::ShardingConfig sharding_config_;
};

}  // namespace hs
}  // namespace mlir
