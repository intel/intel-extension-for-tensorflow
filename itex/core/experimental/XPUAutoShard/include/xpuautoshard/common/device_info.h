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

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace as {

using DeviceId = size_t;

/**
 * @brief Device compute capability
 * TODO(itex): add memory capacity and latency. add cache capacity, bandwidth
 * and latency.
 *
 */
class DeviceComputeCapability {
 public:
  DeviceComputeCapability(float float_OPS = 0, float bfloat16_OPS = 0,
                          float float16_OPS = 0, float int8_OPS = 0,
                          float mem_bandwidth_triad = 0)
      : float_OPS_(float_OPS),
        bfloat16_OPS_(bfloat16_OPS),
        float16_OPS_(float16_OPS),
        int8_OPS_(int8_OPS),
        mem_bandwidth_triad_(mem_bandwidth_triad) {}

  /**
   * @brief Throughput: fp32 multiply-add ops per second
   *
   * @return float
   */
  float getFloatOPS() const { return float_OPS_; }

  /**
   * @brief Throughput: bf16 multiply-add ops per second
   *
   * @return float
   */
  float getBfloat16OPS() const { return bfloat16_OPS_; }

  /**
   * @brief Throughput: fp16 multiply-add ops per second
   *
   * @return float
   */
  float getFloat16OPS() const { return float16_OPS_; }

  /**
   * @brief Throughput: int8 multiply-add ops per second
   *
   * @return float
   */
  float getInt8OPS() const { return int8_OPS_; }

  /**
   * @brief Stream triad memory bandwidth
   *
   * @return float
   */
  float getMemBandwidthTriad() const { return mem_bandwidth_triad_; }

 private:
  float float_OPS_;
  float bfloat16_OPS_;
  float float16_OPS_;
  float int8_OPS_;
  float mem_bandwidth_triad_;
};

class Device {
 public:
  static constexpr DeviceId CPU_HOST_ID = 0;
  static constexpr DeviceId INVALID_DEVICE_ID = -1;
  static constexpr const char* CPU_HOST_NAME = "__cpu_host__";
  static constexpr const char* DEVICE_UNKNOWN = "__unknown__";

  static Device getCpuHost(float score = 1.0f,
                           const std::string& host_name = "") {
    return {CPU_HOST_ID, host_name.empty() ? CPU_HOST_NAME : host_name, score};
  }

  Device() : Device(INVALID_DEVICE_ID) {}

  Device(DeviceId id, const std::string& name = DEVICE_UNKNOWN,
         float score = -1.0f)
      : id_(id), name_(name), score_(score), num_stages_(0) {}

  bool operator==(const Device& rhs) const;
  bool operator!=(const Device& rhs) const;

  DeviceId getId() const { return id_; }
  void setId(DeviceId id) { id_ = id; }

  std::string getName() const { return name_; }
  void setName(const std::string& name) { name_ = name; }

  bool isCpuHost() const { return id_ == CPU_HOST_ID; }

  /**
   * @brief A normalized score from (0, 1] to rank the overall device
   * compute capability.
   *
   * @return float
   */
  float getScore() const { return score_; }

  void setScore(float score) { score_ = score; }

  void setComputeCapability(const DeviceComputeCapability& device_comp_cap) {
    device_comp_cap_ = device_comp_cap;
  }

  const DeviceComputeCapability& getComputeCapability() const {
    return device_comp_cap_;
  }

  /**
   * @brief Recommended number of stages for this device
   *
   * @return size_t >0 number of stages or 0 if no recommendation.
   */
  size_t getNumStages() const { return num_stages_; }

  void setNumStages(size_t num_stages) { num_stages_ = num_stages; }

 private:
  DeviceId id_;
  std::string name_;
  float score_;
  // TODO(itex): hard-code score temporarily, to be replaced with cost model.
  size_t num_stages_;
  DeviceComputeCapability device_comp_cap_;
};

class DeviceInfo {
 public:
  using DeviceMap = std::map<DeviceId, Device>;

  static const Device INVALID_DEVICE;

  struct DeviceIterConst : public DeviceMap::const_iterator {
    explicit DeviceIterConst(const DeviceMap::const_iterator& iter)
        : DeviceMap::const_iterator(iter) {}
    const Device& operator*() const {
      return DeviceMap::const_iterator::operator*().second;
    }
  };

  struct DeviceRangeConst {
    explicit DeviceRangeConst(const DeviceMap* map) : map_(map) {}

    DeviceIterConst begin() const { return DeviceIterConst(map_->begin()); }
    DeviceIterConst end() const { return DeviceIterConst(map_->end()); }

   private:
    const DeviceMap* map_;
  };

  DeviceInfo(bool add_cpu_host = false, float host_score = 1.0f,
             const std::string& host_name = "");

  bool operator==(const DeviceInfo& rhs) const;
  bool operator!=(const DeviceInfo& rhs) const;

  size_t getNumDevices() const { return device_map_.size(); }
  DeviceRangeConst getDevices() const { return DeviceRangeConst(&device_map_); }

  std::vector<DeviceId> getDeviceIds() const {
    std::vector<DeviceId> deviceIds;
    auto range = getDevices();
    std::transform(range.begin(), range.end(), std::back_inserter(deviceIds),
                   [](const Device& device) { return device.getId(); });
    return deviceIds;
  }

  /**
   * @brief Add a new device to the device info struct.
   *
   * @param device The device to add, its id not exists in the device info.
   * TODO(itex): assign manage device id from device info instead. This
   * simplifies the implementation of container, i.e., no need to maintain a
   * map, a vector is good enough.
   * @return true Device added successfully
   * @return false Device adding fails
   */
  bool addDevice(const Device& device);
  const Device& getDevice(DeviceId id) const;

 private:
  DeviceMap device_map_;
  // TODO(itex): add inter-connectivity
};

}  // namespace as
