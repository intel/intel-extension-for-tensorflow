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

#include "xpuautoshard/common/sharding_property_grouper.h"

#include <unordered_map>

#include "xpuautoshard/common/hsp_exception.h"

namespace as {

ShardingPropertyGrouper::ShardingPropertyGrouper(
    const ShardingPropertyRefVec& props) {
  if (props.size() == 0) {
    return;
  }
  // make sure the props have same device set
  // TODO(itex): can we allow same device set but different order of ids?
  auto&& first_prop = props[0];
  for (size_t i = 1; i < props.size(); i++) {
    if (first_prop->getDeviceIds() != props[i]->getDeviceIds() ||
        first_prop->getNumLogicalShards() != props[i]->getNumLogicalShards()) {
      throw HspGroupingConflictException(props);
    }
  }
  for (size_t prop_num = 0; prop_num < props.size(); prop_num++) {
    auto shard_descs = props[prop_num]->getShardDescriptors();
    for (size_t shard_num = 0; shard_num < shard_descs.size(); shard_num++) {
      if (prop_num == 0) {
        shard_desc_groups_.emplace_back(
            ShardDescGroup(props[prop_num]->getDeviceIdPerShardNum(shard_num)));
      }
      shard_desc_groups_[shard_num].emplace_back(shard_descs[shard_num]);
    }
  }
}

}  // namespace as
