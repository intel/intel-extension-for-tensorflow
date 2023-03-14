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

#include "mlir/IR/OpDefinition.h"
#include "xpuautoshard/common/ref_base.h"

namespace mlir {
namespace hs {

class HspInitializer {
 public:
  virtual ~HspInitializer() = default;
  /**
   * @brief Initialize some hsps in the DAG specified with `root_op`. The
   * algorithm should make sure to initialize at least one uninitialized hsp.
   *
   * @param root_op
   * @return true Initialized at least one HSP.
   * @return false Nothing to work on. All HSPs are initialized.
   */
  virtual bool initSome(Operation* root_op) = 0;
};

using HspInitializerRef = as::Ref<HspInitializer>;

}  // namespace hs
}  // namespace mlir
