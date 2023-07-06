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

#include <vector>

#include "xpuautoshard/common/sharding_property.h"
namespace as {

/**
 * @brief A group of ShardDescs corresponding to a group of sharded tensors
 * which are sharded along the same dim and along the same order.
 *
 */
class ShardDescGroup : public std::vector<ShardDesc> {
 public:
  explicit ShardDescGroup(DeviceId device_id)
      : std::vector<ShardDesc>(), device_id_(device_id) {}

  /**
   * @brief Get the device id of this group. If the device id
   * of different ShardDesc is different, the device id of
   * ShardDesc having split is preferred. Split ShardDescs should
   * have the same device id.
   *
   * @return int64_t
   */
  DeviceId getDeviceId() const { return device_id_; }

 private:
  DeviceId device_id_;
};

/**
 * @brief A helper class to produce the ShardDescGroups for sharding
 * properties. The properties should be compatible with each other.
 * TODO: define the meaning of "compatible".
 *
 */
class ShardingPropertyGrouper {
 public:
  /**
   * @brief Construct the grouper object given sharding properties
   *
   * @param props
   */
  explicit ShardingPropertyGrouper(const ShardingPropertyRefVec& props);

  /**
   * @brief Get the ShardDescGroups. The number of the groups equals to
   * the maximum number of shards of the sharding properties.
   *
   * @return const std::vector<ShardDescGroup>&
   */
  const std::vector<ShardDescGroup>& getShardDescriptorGroups() const {
    return shard_desc_groups_;
  }

 private:
  std::vector<ShardDescGroup> shard_desc_groups_;
};

}  // namespace as
