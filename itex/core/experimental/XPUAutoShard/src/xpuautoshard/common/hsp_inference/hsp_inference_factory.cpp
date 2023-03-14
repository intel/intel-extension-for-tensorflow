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

#include "xpuautoshard/common/hsp_inference/hsp_inference_factory.h"

#include "xpuautoshard/common/hsp_inference/hsp_inference_utils.h"

namespace as {

std::unordered_map<std::string, HspInferenceFactoryRef>
    HspInferenceFactory::factory_map_;

HspInferenceFactory& HspInferenceFactory::get(const std::string& op_name) {
  class DefaultHspInference : public HspInference {
   public:
    explicit DefaultHspInference(OpDescRef op_desc)
        : as::HspInference(op_desc) {}
    bool doInfer(const as::ShardingPropertyRefVec& input_props,
                 const as::ShardingPropertyRefVec& output_props) override {
      return utils::inferDefault(input_props, output_props);
    }
  };
  class DefaultHspInferenceFactory : public HspInferenceFactory {
   public:
    HspInferenceRef create(OpDescRef op_desc) override {
      return makeRef<DefaultHspInference>(op_desc);
    }
  };
  if (factory_map_.find(op_name) != factory_map_.end()) {
    return *factory_map_[op_name];
  } else {
    static DefaultHspInferenceFactory default_factory;
    return default_factory;
  }
}

void HspInferenceFactory::registerFactory(const std::string& op_name,
                                          HspInferenceFactoryRef factory) {
  factory_map_.insert({op_name, factory});
}

}  // namespace as
