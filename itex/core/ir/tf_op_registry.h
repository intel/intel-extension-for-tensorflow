/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_IR_TF_OP_REGISTRY_H_
#define ITEX_CORE_IR_TF_OP_REGISTRY_H_

#include "itex/core/ir/interfaces.h"
#include "itex/core/utils/function.h"
#include "protos/graph.pb.h"

namespace mlir {
namespace tfg {
class TensorFlowOpRegistryInterface : public TensorFlowRegistryInterfaceBase {
 public:
  // Create the interface model with a provided registry.
  TensorFlowOpRegistryInterface(Dialect* dialect, itex::GraphDef g_def)
      : TensorFlowRegistryInterfaceBase(dialect), func_(g_def) {}
  // Create the interface model with the global registry.
  explicit TensorFlowOpRegistryInterface(Dialect* dialect);

  // Returns true if the operation is stateful.
  bool isStateful(Operation* op) const override;

 private:
  itex::FunctionLibraryDefinition func_;
};
}  // namespace tfg
}  // namespace mlir

#endif  // ITEX_CORE_IR_TF_OP_REGISTRY_H_
