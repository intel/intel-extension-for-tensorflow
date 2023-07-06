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

#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/mlir/passes/hsp_initializer.h"
#include "xpuautoshard/common/mlir/passes/mlir_hsp_annotator.h"
#include "xpuautoshard/common/ref_base.h"

namespace mlir {
namespace hs {

class LearnedInitializer : public HspInitializer {
 public:
  LearnedInitializer(const as::DeviceInfo& device_info, MLIRAnnotationRef annot)
      : device_info_(device_info), annot_(annot) {}

  bool initSome(Operation* root_op) override {
    // TODO(itex)
    return false;
  }

 private:
  as::DeviceInfo device_info_;
  MLIRAnnotationRef annot_;
};

using LearnedInitializerRef = as::Ref<LearnedInitializer>;

}  // namespace hs
}  // namespace mlir
