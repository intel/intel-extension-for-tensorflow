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

#include "xpuautoshard/common/device_info.h"

namespace as {

const Device DeviceInfo::INVALID_DEVICE;

DeviceInfo::DeviceInfo(bool add_cpu_host, float host_score,
                       const std::string& host_name) {
  if (add_cpu_host) {
    Device cpu_host = Device::getCpuHost(host_score, host_name);
    addDevice(cpu_host);
  }
}

bool DeviceInfo::operator==(const DeviceInfo& rhs) const {
  return device_map_ == rhs.device_map_;
}

bool DeviceInfo::operator!=(const DeviceInfo& rhs) const {
  return !(*this == rhs);
}

bool DeviceInfo::addDevice(const Device& device) {
  if (device_map_.find(device.getId()) != device_map_.end()) {
    return false;
  }
  device_map_.insert(DeviceMap::value_type(device.getId(), device));
  return true;
}

const Device& DeviceInfo::getDevice(DeviceId id) const {
  auto iter = device_map_.find(id);
  if (iter != device_map_.end()) {
    return iter->second;
  } else {
    return INVALID_DEVICE;
  }
}

bool Device::operator==(const Device& rhs) const {
  return getId() == rhs.getId() && getName() == rhs.getName() &&
         getScore() == rhs.getScore();
}

bool Device::operator!=(const Device& rhs) const { return !(*this == rhs); }

}  // namespace as
