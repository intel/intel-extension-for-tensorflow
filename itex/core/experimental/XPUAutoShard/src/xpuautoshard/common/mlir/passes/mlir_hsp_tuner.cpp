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

#include "xpuautoshard/common/mlir/passes/mlir_hsp_tuner.h"

#include "xpuautoshard/common/mlir/passes/mlir_hsp_annotator.h"

namespace mlir {
namespace hs {

using as::HspAnnotator;
using as::HspAnnotatorRef;
using as::makeRef;
using as::TuningStateRef;
using ::mlir::hs::MLIRHspAnnotator;

HspAnnotatorRef MLIRHspTuner::createAnnotator(TuningStateRef tuning_state) {
  return makeRef<MLIRHspAnnotator, HspAnnotator>(graph_, device_info_,
                                                 sharding_config_);
}

}  // namespace hs
}  // namespace mlir
