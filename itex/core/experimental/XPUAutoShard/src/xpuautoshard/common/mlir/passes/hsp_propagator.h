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
#include <utility>

#include "mlir/IR/OpDefinition.h"
#include "xpuautoshard/common/device_info.h"

namespace mlir {
namespace hs {

class HspOpPropagator {
 public:
  virtual ~HspOpPropagator() = default;
  virtual bool forward(Operation* op) = 0;
  virtual bool backward(Operation* op) = 0;
};

class HspPropagator {
 public:
  HspPropagator() {}
  /**
   * @brief Propagate HSP values through the graphs under `root_op` assuming the
   * graphs are DAG.
   *
   * @param root_op
   */
  void propagate(Operation* root_op, HspOpPropagator* op_propagator);

 protected:
  bool forwardPropagate(Operation* root_op, HspOpPropagator* op_propagator);
  bool backwardPropagate(Operation* root_op, HspOpPropagator* op_propagator);
};

}  // namespace hs
}  // namespace mlir
