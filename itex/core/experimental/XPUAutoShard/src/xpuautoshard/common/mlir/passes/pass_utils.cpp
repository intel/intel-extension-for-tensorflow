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

#include "xpuautoshard/common/mlir/passes/pass_utils.h"

#include <utility>

#include "xpuautoshard/common/mlir/passes/hsp_propagator.h"
#include "xpuautoshard/common/mlir/passes/mlir_graph.h"
#include "xpuautoshard/common/mlir/passes/mlir_hsp_annotator.h"
#include "xpuautoshard/common/mlir/passes/mlir_hsp_tuner.h"

namespace mlir {
namespace hs {

using as::Graph;
using as::GraphRef;
using as::HspAnnotationRef;
using as::HspTuner;
using as::HspTunerRef;
using as::makeRef;
using as::OpDescRef;
using as::ShardingConfig;

OpDescRef mlirOpToOpDesc(Operation* op) {
  return makeRef<MLIROpDesc, OpDesc>(op);
}

as::GraphRef mlirGraphToGraphHandle(Operation* root_op) {
  return makeRef<MLIRGraph, Graph>(root_op);
}

HspTunerRef createHspTunerMlir(GraphRef graph, const DeviceInfo& device_info,
                               const ShardingConfig& sharding_config) {
  return makeRef<MLIRHspTuner, HspTuner>(graph, device_info, sharding_config);
}

void setAnnotationToGraph(as::GraphRef graph, as::HspAnnotationRef annot) {
  /**
   * @brief The HSP op propagator that sets the HSP strictly
   * according to the given annotation.
   *
   */
  struct SetHspOpPropagator : public HspOpPropagator {
    explicit SetHspOpPropagator(MLIRAnnotationRef mlir_annot)
        : mlir_annot_(mlir_annot) {}

    bool forward(Operation* op) override {
      auto&& hsps = mlir_annot_->getResultHsps(op);
      assert(hsps.size() == op->getNumResults() &&
             "Expect each result of op has an HSP");
      for (size_t i = 0; i < hsps.size(); i++) {
        setShardingPropertyForValue(op->getResult(i), hsps[i]);
      }
      return false;  // only need one pass, so returning false to avoid pass the
                     // second time.
    }

    bool backward(Operation* op) override { return false; }

   private:
    MLIRAnnotationRef mlir_annot_;
  };
  MLIRGraphRef mlir_graph = as::downcastRef<MLIRGraph, Graph>(graph);
  assert(mlir_graph && "Expect graph is of MLIRGraph type");
  MLIRAnnotationRef mlir_annot =
      as::downcastRef<MLIRAnnotation, as::HspAnnotation>(annot);
  assert(mlir_annot && "Expect annotation is of MLIRAnnotation type");
  Operation* root_op = mlir_graph->getRoot();
  HspPropagator propagator;
  SetHspOpPropagator set_hsp_op_propagator(mlir_annot);
  propagator.propagate(root_op, &set_hsp_op_propagator);
}

}  // namespace hs
}  // namespace mlir
