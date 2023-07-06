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

#include <memory>
#include <unordered_map>
#include <vector>

#include "xpuautoshard/common/hsp_inference/dim_group.h"
#include "xpuautoshard/common/op_desc.h"
#include "xpuautoshard/common/ref_base.h"
#include "xpuautoshard/common/sharding_property.h"

namespace as {

class HspInference {
 public:
  virtual ~HspInference() = default;
  virtual bool infer(ShardingPropertyRefVec* input_props,
                     ShardingPropertyRefVec* output_props);

 protected:
  explicit HspInference(OpDescRef op_desc) : op_desc_(op_desc) {}
  OpDescRef op_desc_;

  virtual bool doInfer(const ShardingPropertyRefVec& input_props,
                       const ShardingPropertyRefVec& output_props) = 0;

 private:
  bool alignSingleSplitStageOffsets(ShardingPropertyRefVec* input_props,
                                    ShardingPropertyRefVec* output_props);
};

using HspInferenceRef = Ref<HspInference>;

}  // namespace as
