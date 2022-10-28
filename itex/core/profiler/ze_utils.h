/* Copyright (c) 2021-2022 Intel Corporation

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

//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef ITEX_CORE_PROFILER_ZE_UTILS_H_
#define ITEX_CORE_PROFILER_ZE_UTILS_H_

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <cstring>
#include <string>
#include <vector>

#include "itex/core/profiler/pti_assert.h"
#include "itex/core/profiler/utils.h"

namespace utils {
namespace ze {

inline std::vector<ze_driver_handle_t> GetDriverList() {
  ze_result_t status = ZE_RESULT_SUCCESS;

  uint32_t driver_count = 0;
  status = zeDriverGet(&driver_count, nullptr);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  if (driver_count == 0) {
    return std::vector<ze_driver_handle_t>();
  }

  std::vector<ze_driver_handle_t> driver_list(driver_count);
  status = zeDriverGet(&driver_count, driver_list.data());
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  return driver_list;
}

inline std::vector<ze_device_handle_t> GetDeviceList(
    ze_driver_handle_t driver) {
  PTI_ASSERT(driver != nullptr);
  ze_result_t status = ZE_RESULT_SUCCESS;

  uint32_t device_count = 0;
  status = zeDeviceGet(driver, &device_count, nullptr);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  if (device_count == 0) {
    return std::vector<ze_device_handle_t>();
  }

  std::vector<ze_device_handle_t> device_list(device_count);
  status = zeDeviceGet(driver, &device_count, device_list.data());
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  return device_list;
}

inline std::vector<ze_device_handle_t> GetSubDeviceList(
    ze_device_handle_t device) {
  PTI_ASSERT(device != nullptr);
  ze_result_t status = ZE_RESULT_SUCCESS;

  uint32_t sub_device_count = 0;
  status = zeDeviceGetSubDevices(device, &sub_device_count, nullptr);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  if (sub_device_count == 0) {
    return std::vector<ze_device_handle_t>();
  }

  std::vector<ze_device_handle_t> sub_device_list(sub_device_count);
  status =
      zeDeviceGetSubDevices(device, &sub_device_count, sub_device_list.data());
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  return sub_device_list;
}

inline ze_driver_handle_t GetGpuDriver() {
  std::vector<ze_driver_handle_t> driver_list;

  for (auto driver : GetDriverList()) {
    for (auto device : GetDeviceList(driver)) {
      ze_device_properties_t props{
          ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
      };
      ze_result_t status = zeDeviceGetProperties(device, &props);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      if (props.type == ZE_DEVICE_TYPE_GPU) {
        driver_list.push_back(driver);
      }
    }
  }

  if (driver_list.empty()) {
    return nullptr;
  }

  std::string value = utils::GetEnv("PTI_DEVICE_ID");
  uint32_t device_id = value.empty() ? 0 : std::stoul(value);
  PTI_ASSERT(device_id >= 0 && device_id < driver_list.size());
  return driver_list[device_id];
}

inline ze_device_handle_t GetGpuDevice() {
  std::vector<ze_device_handle_t> device_list;

  for (auto driver : GetDriverList()) {
    for (auto device : GetDeviceList(driver)) {
      ze_device_properties_t props{
          ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
      };
      ze_result_t status = zeDeviceGetProperties(device, &props);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      if (props.type == ZE_DEVICE_TYPE_GPU) {
        device_list.push_back(device);
      }
    }
  }

  if (device_list.empty()) {
    return nullptr;
  }

  std::string value = utils::GetEnv("PTI_DEVICE_ID");
  uint32_t device_id = value.empty() ? 0 : std::stoul(value);
  PTI_ASSERT(device_id >= 0 && device_id < device_list.size());

  std::vector<ze_device_handle_t> sub_device_list =
      GetSubDeviceList(device_list[device_id]);
  if (sub_device_list.empty()) {
    return device_list[device_id];
  }

  value = utils::GetEnv("PTI_SUB_DEVICE_ID");
  uint32_t sub_device_id = value.empty() ? 0 : std::stoul(value);
  PTI_ASSERT(sub_device_id >= 0 && sub_device_id < sub_device_list.size());
  return sub_device_list[sub_device_id];
}

inline ze_context_handle_t GetContext(ze_driver_handle_t driver) {
  PTI_ASSERT(driver != nullptr);

  ze_result_t status = ZE_RESULT_SUCCESS;
  ze_context_handle_t context = nullptr;
  ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};

  status = zeContextCreate(driver, &context_desc, &context);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  return context;
}

inline std::string GetDeviceName(ze_device_handle_t device) {
  PTI_ASSERT(device != nullptr);
  ze_result_t status = ZE_RESULT_SUCCESS;
  ze_device_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  status = zeDeviceGetProperties(device, &props);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  return props.name;
}

inline int GetMetricId(zet_metric_group_handle_t group, std::string name) {
  PTI_ASSERT(group != nullptr);

  ze_result_t status = ZE_RESULT_SUCCESS;
  uint32_t metric_count = 0;
  status = zetMetricGet(group, &metric_count, nullptr);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  if (metric_count == 0) {
    return -1;
  }

  std::vector<zet_metric_handle_t> metric_list(metric_count, nullptr);
  status = zetMetricGet(group, &metric_count, metric_list.data());
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  int target = -1;
  for (uint32_t i = 0; i < metric_count; ++i) {
    zet_metric_properties_t metric_props{};
    status = zetMetricGetProperties(metric_list[i], &metric_props);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    if (name == metric_props.name) {
      target = i;
      break;
    }
  }

  return target;
}

inline zet_metric_group_handle_t FindMetricGroup(
    ze_device_handle_t device, std::string name,
    zet_metric_group_sampling_type_flag_t type) {
  PTI_ASSERT(device != nullptr);

  ze_result_t status = ZE_RESULT_SUCCESS;
  uint32_t group_count = 0;
  status = zetMetricGroupGet(device, &group_count, nullptr);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  if (group_count == 0) {
    return nullptr;
  }

  std::vector<zet_metric_group_handle_t> group_list(group_count, nullptr);
  status = zetMetricGroupGet(device, &group_count, group_list.data());
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  zet_metric_group_handle_t target = nullptr;
  for (uint32_t i = 0; i < group_count; ++i) {
    zet_metric_group_properties_t group_props{};
    group_props.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
    status = zetMetricGroupGetProperties(group_list[i], &group_props);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    if (name == group_props.name && (group_props.samplingType & type)) {
      target = group_list[i];
      break;
    }
  }

  return target;
}

inline size_t GetKernelMaxSubgroupSize(ze_kernel_handle_t kernel) {
  PTI_ASSERT(kernel != nullptr);
  ze_kernel_properties_t props{};
  ze_result_t status = zeKernelGetProperties(kernel, &props);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  return props.maxSubgroupSize;
}

inline std::string GetKernelName(ze_kernel_handle_t kernel) {
  PTI_ASSERT(kernel != nullptr);

  size_t size = 0;
  ze_result_t status = zeKernelGetName(kernel, &size, nullptr);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  PTI_ASSERT(size > 0);

  std::vector<char> name(size);
  status = zeKernelGetName(kernel, &size, name.data());
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  PTI_ASSERT(name[size - 1] == '\0');
  return std::string(name.begin(), name.end() - 1);
}

inline uint64_t GetDeviceTimestamp(ze_device_handle_t device) {
  PTI_ASSERT(device != nullptr);
  uint64_t host_timestamp = 0, device_timestamp = 0;
  ze_result_t status =
      zeDeviceGetGlobalTimestamps(device, &host_timestamp, &device_timestamp);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  return device_timestamp;
}

inline uint64_t GetDeviceTimerFrequency(ze_device_handle_t device) {
  PTI_ASSERT(device != nullptr);
  ze_device_properties_t props{
      ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2,
  };
  ze_result_t status = zeDeviceGetProperties(device, &props);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  return props.timerResolution;
}

inline uint64_t GetDeviceTimestampMask(ze_device_handle_t device) {
  PTI_ASSERT(device != nullptr);
  ze_device_properties_t props{
      ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2,
  };
  ze_result_t status = zeDeviceGetProperties(device, &props);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  return (1ull << props.kernelTimestampValidBits) - 1ull;
}

inline ze_api_version_t GetDriverVersion(ze_driver_handle_t driver) {
  PTI_ASSERT(driver != nullptr);

  ze_api_version_t version = ZE_API_VERSION_FORCE_UINT32;
  ze_result_t status = zeDriverGetApiVersion(driver, &version);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  return version;
}

inline ze_api_version_t GetVersion() {
  auto driver_list = GetDriverList();
  if (driver_list.empty()) {
    return ZE_API_VERSION_FORCE_UINT32;
  }
  return GetDriverVersion(driver_list.front());
}

}  // namespace ze
}  // namespace utils

#endif  // ITEX_CORE_PROFILER_ZE_UTILS_H_
