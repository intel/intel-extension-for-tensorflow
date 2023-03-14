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

#include "xpuautoshard/common/mlir/passes/mlir_hsp_annotator.h"

#include <memory>
#include <utility>
#include <vector>

#include "xpuautoshard/common/hsp_inference/hsp_inference.h"
#include "xpuautoshard/common/hsp_inference/hsp_inference_factory.h"
#include "xpuautoshard/common/mlir/passes/cpu_host_initializer.h"
#include "xpuautoshard/common/mlir/passes/heuristics_initializer.h"
#include "xpuautoshard/common/mlir/passes/hsp_initializer.h"
#include "xpuautoshard/common/mlir/passes/hsp_propagator.h"
#include "xpuautoshard/common/mlir/passes/learned_initializer.h"
#include "xpuautoshard/common/mlir/passes/pass_utils.h"

namespace mlir {
namespace hs {

using as::DeviceInfo;
using as::GraphRef;
using as::HspAnnotationRef;
using as::HspInference;
using as::HspInferenceFactory;
using as::HspInferenceRef;
using as::makeRef;
using as::ShardingProperty;
using as::ShardingPropertyRef;
using as::StrategyKind;

namespace {
/**
 * @brief The HSP op propagator that populates the annotation with a given op
 *
 */
struct GetHspOpPropagator : public HspOpPropagator {
  explicit GetHspOpPropagator(MLIRAnnotationRef mlir_annot)
      : mlir_annot_(mlir_annot) {}

  bool forward(Operation* op) override {
    ShardingPropertyRefVec hsps;
    for (size_t i = 0; i < op->getNumResults(); i++) {
      hsps.push_back(getShardingPropertyForValue(op->getResult(i)));
    }
    mlir_annot_->insert(op, std::move(hsps));
    return false;  // only need one pass, so returning false to avoid pass the
                   // second time.
  }

  bool backward(Operation* op) override { return false; }

 private:
  MLIRAnnotationRef mlir_annot_;
};

class HspInferenceOpPropagator : public HspOpPropagator {
 public:
  explicit HspInferenceOpPropagator(MLIRAnnotationRef mlir_annot)
      : mlir_annot_(mlir_annot) {}

  bool forward(Operation* op) override {
    bool changed = false;
    if (llvm::dyn_cast<ShardOp>(op) || llvm::dyn_cast<ReshardOp>(op) ||
        llvm::dyn_cast<UnshardOp>(op)) {
      return changed;
    }
    auto&& inference = getHspInference(op);
    ShardingPropertyRefVec input_hsps;
    ShardingPropertyRefVec output_hsps;
    std::tie(input_hsps, output_hsps) =
        mlir_annot_->getShardingPropertiesForOp(op);
    changed |= inference.infer(&input_hsps, &output_hsps);
    return changed;
  }
  bool backward(Operation* op) override {
    // XXX: Anything special for backward?
    return forward(op);
  }

 private:
  /**
   * @brief Get the hsp inference object for the given op
   *
   * @param op
   * @return as::HspInference&
   */
  HspInference& getHspInference(Operation* op) {
    if (hsp_inf_cache_.find(op) == hsp_inf_cache_.end()) {
      auto&& hsp_inference =
          HspInferenceFactory::get(op->getName().getStringRef().str())
              .create(mlirOpToOpDesc(op));
      hsp_inf_cache_.insert({op, hsp_inference});
    }
    return *hsp_inf_cache_[op];
  }

  std::unordered_map<Operation*, HspInferenceRef> hsp_inf_cache_;
  MLIRAnnotationRef mlir_annot_;
};

}  // anonymous namespace

HspAnnotationRef MLIRHspAnnotator::annotate(GraphRef graph) {
  HspPropagator propagator;
  auto&& mlir_annot = makeRef<MLIRAnnotation>();
  GetHspOpPropagator get_hsp_op_propagator(mlir_annot);
  propagator.propagate(mlir_graph_->getRoot(), &get_hsp_op_propagator);

  std::shared_ptr<HspInitializer> initializer;
  switch (sharding_config_.getStrategyKind()) {
    case StrategyKind::CPU_HOST:
      initializer =
          std::make_shared<CpuHostInitializer>(device_info_, mlir_annot);
      break;
    case StrategyKind::HEURISTIC:
      initializer = std::make_shared<HeuristicsInitializer>(
          device_info_, sharding_config_.getHeuristicsConfig(), mlir_annot);
      break;
    case StrategyKind::LEARNED:
      initializer =
          std::make_shared<LearnedInitializer>(device_info_, mlir_annot);
      break;
    default:
      assert(false && "unreachable");
  }
  HspInferenceOpPropagator inference_op_propagator(mlir_annot);
  do {
    propagator.propagate(mlir_graph_->getRoot(), &inference_op_propagator);
  } while (initializer->initSome(mlir_graph_->getRoot()));

  return mlir_annot;
}

std::pair<ShardingPropertyRefVec, ShardingPropertyRefVec>
MLIRAnnotation::getShardingPropertiesForOp(Operation* op) {
  std::vector<ShardingPropertyRef> input_hsps;
  for (auto&& operand : op->getOperands()) {
    input_hsps.push_back(getShardingPropertyForValue(operand));
  }
  return std::make_pair(input_hsps, annot_map_[op]);
}

ShardingPropertyRef MLIRAnnotation::getShardingPropertyForValue(Value value) {
  auto defining_op = value.getDefiningOp();
  if (annot_map_.find(defining_op) != annot_map_.end()) {
    for (unsigned int i = 0; i < defining_op->getNumResults(); i++) {
      auto op_result = defining_op->getResult(i);
      if (op_result == value) {
        return annot_map_[defining_op][i];
      }
    }
  }
  assert(false &&
         "Should not reach here, value should be in the results of the "
         "defining op");
  return as::makeRef<as::ShardingProperty>(
      as::DeviceInfo(/*add_cpu_host=*/false));
}

}  // namespace hs
}  // namespace mlir
