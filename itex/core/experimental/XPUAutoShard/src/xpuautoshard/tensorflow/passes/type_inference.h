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

#include <unordered_set>

#include "itex/core/graph/utils/graph_properties.h"
#include "itex/core/ir/ops.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

using itex::graph::GraphProperties;

namespace mlir {
namespace tfg {

class TypeInference
    : public PassWrapper<TypeInference, OperationPass<GraphOp> > {
 public:
  explicit TypeInference(GraphProperties* properties)
      : properties_(properties) {}
  ~TypeInference() override;
  void runOnOperation() override;

 protected:
  bool forwardOp(Operation* op);
  void eliminateConstValueImplicitInfer();

 private:
  GraphProperties* properties_;
};

}  // namespace tfg
}  // namespace mlir
