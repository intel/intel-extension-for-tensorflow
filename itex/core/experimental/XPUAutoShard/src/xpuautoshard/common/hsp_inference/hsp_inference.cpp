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

#include "xpuautoshard/common/hsp_inference/hsp_inference.h"

#include <assert.h>

#include <algorithm>
#include <sstream>

#include "xpuautoshard/common/hsp_exception.h"

namespace as {

bool HspInference::infer(ShardingPropertyRefVec* input_props,
                         ShardingPropertyRefVec* output_props) {
  bool changed = doInfer(*input_props, *output_props);
  changed |= alignSingleSplitStageOffsets(input_props, output_props);
  return changed;
}

bool HspInference::alignSingleSplitStageOffsets(
    ShardingPropertyRefVec* input_props, ShardingPropertyRefVec* output_props) {
  size_t max_num_logical_shards = 0;
  std::vector<int64_t> stage_offsets;
  auto init_stage_offsets = [&](as::ShardingPropertyRef prop) {
    if (!prop->isInitialized()) {
      return;
    }
    size_t num_logical_shards = prop->getNumLogicalShards();
    if (num_logical_shards > max_num_logical_shards) {
      max_num_logical_shards = num_logical_shards;
      stage_offsets = prop->getStageOffsets();
    }
  };
  for (auto& prop : *input_props) {
    init_stage_offsets(prop);
  }
  for (auto& prop : *output_props) {
    init_stage_offsets(prop);
  }
  bool changed = false;
  if (stage_offsets.empty()) {
    return changed;
  }
  auto update_stage_offsets_for_single_split =
      [&](as::ShardingPropertyRef prop) {
        if (prop->isSplitSingleOnly()) {
          changed |= prop->splitSingleOnly(stage_offsets);
        }
      };
  for (auto& prop : *input_props) {
    update_stage_offsets_for_single_split(prop);
  }
  for (auto& prop : *output_props) {
    update_stage_offsets_for_single_split(prop);
  }
  return changed;
}

}  // namespace as
