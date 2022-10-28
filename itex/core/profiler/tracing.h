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

#ifndef ITEX_CORE_PROFILER_TRACING_H_
#define ITEX_CORE_PROFILER_TRACING_H_

#include <string>
#include <vector>

static const char* GetResultString(unsigned result) {
  switch (result) {
    case ZE_RESULT_SUCCESS:
      return "ZE_RESULT_SUCCESS";
    case ZE_RESULT_NOT_READY:
      return "ZE_RESULT_NOT_READY";
    case ZE_RESULT_ERROR_DEVICE_LOST:
      return "ZE_RESULT_ERROR_DEVICE_LOST";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
      return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
      return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
      return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
      return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
      return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
    case ZE_RESULT_ERROR_NOT_AVAILABLE:
      return "ZE_RESULT_ERROR_NOT_AVAILABLE";
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
      return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
    case ZE_RESULT_ERROR_UNINITIALIZED:
      return "ZE_RESULT_ERROR_UNINITIALIZED";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
      return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
      return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
      return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
      return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
      return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
      return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
    case ZE_RESULT_ERROR_INVALID_SIZE:
      return "ZE_RESULT_ERROR_INVALID_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
      return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
      return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
      return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
      return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
      return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
      return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
      return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
      return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
      return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
      return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
      return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
      return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
      return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
      return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
    case ZE_RESULT_ERROR_UNKNOWN:
      return "ZE_RESULT_ERROR_UNKNOWN";
    case ZE_RESULT_FORCE_UINT32:
      return "ZE_RESULT_FORCE_UINT32";
    default:
      break;
  }
  return "UNKNOWN";
}

static const char* GetStructureTypeString(unsigned structure_type) {
  switch (structure_type) {
    case ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES";
    case ZE_STRUCTURE_TYPE_CONTEXT_DESC:
      return "ZE_STRUCTURE_TYPE_CONTEXT_DESC";
    case ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC:
      return "ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC";
    case ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC:
      return "ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC";
    case ZE_STRUCTURE_TYPE_EVENT_POOL_DESC:
      return "ZE_STRUCTURE_TYPE_EVENT_POOL_DESC";
    case ZE_STRUCTURE_TYPE_EVENT_DESC:
      return "ZE_STRUCTURE_TYPE_EVENT_DESC";
    case ZE_STRUCTURE_TYPE_FENCE_DESC:
      return "ZE_STRUCTURE_TYPE_FENCE_DESC";
    case ZE_STRUCTURE_TYPE_IMAGE_DESC:
      return "ZE_STRUCTURE_TYPE_IMAGE_DESC";
    case ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC:
      return "ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC";
    case ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC:
      return "ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC";
    case ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES";
    case ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC";
    case ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD";
    case ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD";
    case ZE_STRUCTURE_TYPE_MODULE_DESC:
      return "ZE_STRUCTURE_TYPE_MODULE_DESC";
    case ZE_STRUCTURE_TYPE_MODULE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_MODULE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_KERNEL_DESC:
      return "ZE_STRUCTURE_TYPE_KERNEL_DESC";
    case ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES";
    case ZE_STRUCTURE_TYPE_SAMPLER_DESC:
      return "ZE_STRUCTURE_TYPE_SAMPLER_DESC";
    case ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC:
      return "ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC";
    case ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC";
    case ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC";
    case ZE_STRUCTURE_TYPE_FORCE_UINT32:
      return "ZE_STRUCTURE_TYPE_FORCE_UINT32";
    default:
      break;
  }
  return "UNKNOWN";
}

static void zeInitOnEnter(ze_init_params_t* params, ze_result_t result,
                          void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeInitOnExit(ze_init_params_t* params, ze_result_t result,
                         void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeInit", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeInit", start_time,
                         end_time);
  }
}

static void zeDriverGetOnEnter(ze_driver_get_params_t* params,
                               ze_result_t result, void* global_user_data,
                               void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDriverGetOnExit(ze_driver_get_params_t* params,
                              ze_result_t result, void* global_user_data,
                              void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDriverGet", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeDriverGet",
                         start_time, end_time);
  }
}

static void zeDriverGetApiVersionOnEnter(
    ze_driver_get_api_version_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDriverGetApiVersionOnExit(
    ze_driver_get_api_version_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDriverGetApiVersion", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDriverGetApiVersion", start_time, end_time);
  }
}

static void zeDriverGetPropertiesOnEnter(
    ze_driver_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDriverGetPropertiesOnExit(
    ze_driver_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDriverGetProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDriverGetProperties", start_time, end_time);
  }
}

static void zeDriverGetIpcPropertiesOnEnter(
    ze_driver_get_ipc_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDriverGetIpcPropertiesOnExit(
    ze_driver_get_ipc_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDriverGetIpcProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDriverGetIpcProperties", start_time, end_time);
  }
}

static void zeDriverGetExtensionPropertiesOnEnter(
    ze_driver_get_extension_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDriverGetExtensionPropertiesOnExit(
    ze_driver_get_extension_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDriverGetExtensionProperties", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDriverGetExtensionProperties", start_time,
                         end_time);
  }
}

static void zeDeviceGetOnEnter(ze_device_get_params_t* params,
                               ze_result_t result, void* global_user_data,
                               void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetOnExit(ze_device_get_params_t* params,
                              ze_result_t result, void* global_user_data,
                              void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGet", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeDeviceGet",
                         start_time, end_time);
  }
}

static void zeDeviceGetSubDevicesOnEnter(
    ze_device_get_sub_devices_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetSubDevicesOnExit(
    ze_device_get_sub_devices_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetSubDevices", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetSubDevices", start_time, end_time);
  }
}

static void zeDeviceGetPropertiesOnEnter(
    ze_device_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetPropertiesOnExit(
    ze_device_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetProperties", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetProperties", start_time, end_time);
  }
}

static void zeDeviceGetComputePropertiesOnEnter(
    ze_device_get_compute_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetComputePropertiesOnExit(
    ze_device_get_compute_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetComputeProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetComputeProperties", start_time, end_time);
  }
}

static void zeDeviceGetModulePropertiesOnEnter(
    ze_device_get_module_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetModulePropertiesOnExit(
    ze_device_get_module_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetModuleProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetModuleProperties", start_time, end_time);
  }
}

static void zeDeviceGetCommandQueueGroupPropertiesOnEnter(
    ze_device_get_command_queue_group_properties_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetCommandQueueGroupPropertiesOnExit(
    ze_device_get_command_queue_group_properties_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetCommandQueueGroupProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetCommandQueueGroupProperties", start_time,
                         end_time);
  }
}

static void zeDeviceGetMemoryPropertiesOnEnter(
    ze_device_get_memory_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetMemoryPropertiesOnExit(
    ze_device_get_memory_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetMemoryProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetMemoryProperties", start_time, end_time);
  }
}

static void zeDeviceGetMemoryAccessPropertiesOnEnter(
    ze_device_get_memory_access_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetMemoryAccessPropertiesOnExit(
    ze_device_get_memory_access_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetMemoryAccessProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetMemoryAccessProperties", start_time,
                         end_time);
  }
}

static void zeDeviceGetCachePropertiesOnEnter(
    ze_device_get_cache_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetCachePropertiesOnExit(
    ze_device_get_cache_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetCacheProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetCacheProperties", start_time, end_time);
  }
}

static void zeDeviceGetImagePropertiesOnEnter(
    ze_device_get_image_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetImagePropertiesOnExit(
    ze_device_get_image_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetImageProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetImageProperties", start_time, end_time);
  }
}

static void zeDeviceGetExternalMemoryPropertiesOnEnter(
    ze_device_get_external_memory_properties_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetExternalMemoryPropertiesOnExit(
    ze_device_get_external_memory_properties_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetExternalMemoryProperties", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetExternalMemoryProperties", start_time,
                         end_time);
  }
}

static void zeDeviceGetP2PPropertiesOnEnter(
    ze_device_get_p2_p_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetP2PPropertiesOnExit(
    ze_device_get_p2_p_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetP2PProperties", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceGetP2PProperties", start_time, end_time);
  }
}

static void zeDeviceCanAccessPeerOnEnter(
    ze_device_can_access_peer_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceCanAccessPeerOnExit(
    ze_device_can_access_peer_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceCanAccessPeer", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeDeviceCanAccessPeer", start_time, end_time);
  }
}

static void zeDeviceGetStatusOnEnter(ze_device_get_status_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeDeviceGetStatusOnExit(ze_device_get_status_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeDeviceGetStatus", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeDeviceGetStatus",
                         start_time, end_time);
  }
}

static void zeContextCreateOnEnter(ze_context_create_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeContextCreateOnExit(ze_context_create_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeContextCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeContextCreate",
                         start_time, end_time);
  }
}

static void zeContextDestroyOnEnter(ze_context_destroy_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeContextDestroyOnExit(ze_context_destroy_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeContextDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeContextDestroy",
                         start_time, end_time);
  }
}

static void zeContextGetStatusOnEnter(ze_context_get_status_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeContextGetStatusOnExit(ze_context_get_status_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeContextGetStatus", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeContextGetStatus",
                         start_time, end_time);
  }
}

static void zeContextSystemBarrierOnEnter(
    ze_context_system_barrier_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeContextSystemBarrierOnExit(
    ze_context_system_barrier_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeContextSystemBarrier", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeContextSystemBarrier", start_time, end_time);
  }
}

static void zeContextMakeMemoryResidentOnEnter(
    ze_context_make_memory_resident_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeContextMakeMemoryResidentOnExit(
    ze_context_make_memory_resident_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeContextMakeMemoryResident", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeContextMakeMemoryResident", start_time, end_time);
  }
}

static void zeContextEvictMemoryOnEnter(
    ze_context_evict_memory_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeContextEvictMemoryOnExit(ze_context_evict_memory_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeContextEvictMemory", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeContextEvictMemory",
                         start_time, end_time);
  }
}

static void zeContextMakeImageResidentOnEnter(
    ze_context_make_image_resident_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeContextMakeImageResidentOnExit(
    ze_context_make_image_resident_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeContextMakeImageResident", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeContextMakeImageResident", start_time, end_time);
  }
}

static void zeContextEvictImageOnEnter(ze_context_evict_image_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeContextEvictImageOnExit(ze_context_evict_image_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeContextEvictImage", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeContextEvictImage",
                         start_time, end_time);
  }
}

static void zeCommandQueueCreateOnEnter(
    ze_command_queue_create_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandQueueCreateOnExit(ze_command_queue_create_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandQueueCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeCommandQueueCreate",
                         start_time, end_time);
  }
}

static void zeCommandQueueDestroyOnEnter(
    ze_command_queue_destroy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandQueueDestroyOnExit(
    ze_command_queue_destroy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandQueueDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandQueueDestroy", start_time, end_time);
  }
}

static void zeCommandQueueExecuteCommandListsOnEnter(
    ze_command_queue_execute_command_lists_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandQueueExecuteCommandListsOnExit(
    ze_command_queue_execute_command_lists_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandQueueExecuteCommandLists", time);
  if (collector->callback_ != nullptr) {
    uint32_t command_list_count = *(params->pnumCommandLists);
    ze_command_list_handle_t* command_lists = *(params->pphCommandLists);
    std::string kernel_call_id;
    if (command_lists != nullptr) {
      for (uint32_t i = 0; i < command_list_count; ++i) {
        std::vector<uint64_t> kernel_id_list =
            collector->correlator_->GetKernelId(command_lists[i]);
        std::vector<uint64_t> call_id_list =
            collector->correlator_->GetCallId(command_lists[i]);
        PTI_ASSERT(kernel_id_list.size() == call_id_list.size());
        for (size_t j = 0; j < kernel_id_list.size(); ++j) {
          kernel_call_id += std::to_string(kernel_id_list[j]) + "." +
                            std::to_string(call_id_list[j]) + ",";
        }
      }
    }

    if (!kernel_call_id.empty()) {
      kernel_call_id = kernel_call_id.substr(0, kernel_call_id.size() - 1);
    } else {
      kernel_call_id = "0";
    }
    collector->callback_(collector->callback_data_, kernel_call_id,
                         "zeCommandQueueExecuteCommandLists", start_time,
                         end_time);
  }
}

static void zeCommandQueueSynchronizeOnEnter(
    ze_command_queue_synchronize_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandQueueSynchronizeOnExit(
    ze_command_queue_synchronize_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandQueueSynchronize", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandQueueSynchronize", start_time, end_time);
  }
}

static void zeCommandListCreateOnEnter(ze_command_list_create_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListCreateOnExit(ze_command_list_create_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeCommandListCreate",
                         start_time, end_time);
  }
}

static void zeCommandListCreateImmediateOnEnter(
    ze_command_list_create_immediate_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListCreateImmediateOnExit(
    ze_command_list_create_immediate_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListCreateImmediate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListCreateImmediate", start_time, end_time);
  }
}

static void zeCommandListDestroyOnEnter(
    ze_command_list_destroy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListDestroyOnExit(ze_command_list_destroy_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeCommandListDestroy",
                         start_time, end_time);
  }
}

static void zeCommandListCloseOnEnter(ze_command_list_close_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListCloseOnExit(ze_command_list_close_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListClose", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeCommandListClose",
                         start_time, end_time);
  }
}

static void zeCommandListResetOnEnter(ze_command_list_reset_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListResetOnExit(ze_command_list_reset_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListReset", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeCommandListReset",
                         start_time, end_time);
  }
}

static void zeCommandListAppendWriteGlobalTimestampOnEnter(
    ze_command_list_append_write_global_timestamp_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendWriteGlobalTimestampOnExit(
    ze_command_list_append_write_global_timestamp_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendWriteGlobalTimestamp", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListAppendWriteGlobalTimestamp", start_time,
                         end_time);
  }
}

static void zeCommandListAppendBarrierOnEnter(
    ze_command_list_append_barrier_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendBarrierOnExit(
    ze_command_list_append_barrier_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendBarrier", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendBarrier", start_time, end_time);
  }
}

static void zeCommandListAppendMemoryRangesBarrierOnEnter(
    ze_command_list_append_memory_ranges_barrier_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendMemoryRangesBarrierOnExit(
    ze_command_list_append_memory_ranges_barrier_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendMemoryRangesBarrier", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendMemoryRangesBarrier", start_time,
                         end_time);
  }
}

static void zeCommandListAppendMemoryCopyOnEnter(
    ze_command_list_append_memory_copy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendMemoryCopyOnExit(
    ze_command_list_append_memory_copy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendMemoryCopy", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendMemoryCopy", start_time, end_time);
  }
}

static void zeCommandListAppendMemoryFillOnEnter(
    ze_command_list_append_memory_fill_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendMemoryFillOnExit(
    ze_command_list_append_memory_fill_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendMemoryFill", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendMemoryFill", start_time, end_time);
  }
}

static void zeCommandListAppendMemoryCopyRegionOnEnter(
    ze_command_list_append_memory_copy_region_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendMemoryCopyRegionOnExit(
    ze_command_list_append_memory_copy_region_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendMemoryCopyRegion", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendMemoryCopyRegion", start_time,
                         end_time);
  }
}

static void zeCommandListAppendMemoryCopyFromContextOnEnter(
    ze_command_list_append_memory_copy_from_context_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendMemoryCopyFromContextOnExit(
    ze_command_list_append_memory_copy_from_context_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendMemoryCopyFromContext", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendMemoryCopyFromContext", start_time,
                         end_time);
  }
}

static void zeCommandListAppendImageCopyOnEnter(
    ze_command_list_append_image_copy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendImageCopyOnExit(
    ze_command_list_append_image_copy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendImageCopy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendImageCopy", start_time, end_time);
  }
}

static void zeCommandListAppendImageCopyRegionOnEnter(
    ze_command_list_append_image_copy_region_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendImageCopyRegionOnExit(
    ze_command_list_append_image_copy_region_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendImageCopyRegion", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendImageCopyRegion", start_time,
                         end_time);
  }
}

static void zeCommandListAppendImageCopyToMemoryOnEnter(
    ze_command_list_append_image_copy_to_memory_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendImageCopyToMemoryOnExit(
    ze_command_list_append_image_copy_to_memory_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendImageCopyToMemory", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendImageCopyToMemory", start_time,
                         end_time);
  }
}

static void zeCommandListAppendImageCopyFromMemoryOnEnter(
    ze_command_list_append_image_copy_from_memory_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendImageCopyFromMemoryOnExit(
    ze_command_list_append_image_copy_from_memory_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendImageCopyFromMemory", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendImageCopyFromMemory", start_time,
                         end_time);
  }
}

static void zeCommandListAppendMemoryPrefetchOnEnter(
    ze_command_list_append_memory_prefetch_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendMemoryPrefetchOnExit(
    ze_command_list_append_memory_prefetch_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendMemoryPrefetch", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListAppendMemoryPrefetch", start_time,
                         end_time);
  }
}

static void zeCommandListAppendMemAdviseOnEnter(
    ze_command_list_append_mem_advise_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendMemAdviseOnExit(
    ze_command_list_append_mem_advise_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendMemAdvise", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListAppendMemAdvise", start_time, end_time);
  }
}

static void zeCommandListAppendSignalEventOnEnter(
    ze_command_list_append_signal_event_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendSignalEventOnExit(
    ze_command_list_append_signal_event_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendSignalEvent", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListAppendSignalEvent", start_time,
                         end_time);
  }
}

static void zeCommandListAppendWaitOnEventsOnEnter(
    ze_command_list_append_wait_on_events_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendWaitOnEventsOnExit(
    ze_command_list_append_wait_on_events_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendWaitOnEvents", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListAppendWaitOnEvents", start_time,
                         end_time);
  }
}

static void zeCommandListAppendEventResetOnEnter(
    ze_command_list_append_event_reset_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendEventResetOnExit(
    ze_command_list_append_event_reset_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendEventReset", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListAppendEventReset", start_time, end_time);
  }
}

static void zeCommandListAppendQueryKernelTimestampsOnEnter(
    ze_command_list_append_query_kernel_timestamps_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendQueryKernelTimestampsOnExit(
    ze_command_list_append_query_kernel_timestamps_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendQueryKernelTimestamps", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListAppendQueryKernelTimestamps", start_time,
                         end_time);
  }
}

static void zeCommandListAppendLaunchKernelOnEnter(
    ze_command_list_append_launch_kernel_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendLaunchKernelOnExit(
    ze_command_list_append_launch_kernel_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendLaunchKernel", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendLaunchKernel", start_time,
                         end_time);
  }
}

static void zeCommandListAppendLaunchCooperativeKernelOnEnter(
    ze_command_list_append_launch_cooperative_kernel_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendLaunchCooperativeKernelOnExit(
    ze_command_list_append_launch_cooperative_kernel_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendLaunchCooperativeKernel",
                             time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendLaunchCooperativeKernel",
                         start_time, end_time);
  }
}

static void zeCommandListAppendLaunchKernelIndirectOnEnter(
    ze_command_list_append_launch_kernel_indirect_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendLaunchKernelIndirectOnExit(
    ze_command_list_append_launch_kernel_indirect_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendLaunchKernelIndirect", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_,
                         std::to_string(collector->correlator_->GetKernelId()),
                         "zeCommandListAppendLaunchKernelIndirect", start_time,
                         end_time);
  }
}

static void zeCommandListAppendLaunchMultipleKernelsIndirectOnEnter(
    ze_command_list_append_launch_multiple_kernels_indirect_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeCommandListAppendLaunchMultipleKernelsIndirectOnExit(
    ze_command_list_append_launch_multiple_kernels_indirect_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeCommandListAppendLaunchMultipleKernelsIndirect",
                             time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeCommandListAppendLaunchMultipleKernelsIndirect",
                         start_time, end_time);
  }
}

static void zeFenceCreateOnEnter(ze_fence_create_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeFenceCreateOnExit(ze_fence_create_params_t* params,
                                ze_result_t result, void* global_user_data,
                                void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeFenceCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeFenceCreate",
                         start_time, end_time);
  }
}

static void zeFenceDestroyOnEnter(ze_fence_destroy_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeFenceDestroyOnExit(ze_fence_destroy_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeFenceDestroy", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeFenceDestroy",
                         start_time, end_time);
  }
}

static void zeFenceHostSynchronizeOnEnter(
    ze_fence_host_synchronize_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeFenceHostSynchronizeOnExit(
    ze_fence_host_synchronize_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeFenceHostSynchronize", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeFenceHostSynchronize", start_time, end_time);
  }
}

static void zeFenceQueryStatusOnEnter(ze_fence_query_status_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeFenceQueryStatusOnExit(ze_fence_query_status_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeFenceQueryStatus", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeFenceQueryStatus",
                         start_time, end_time);
  }
}

static void zeFenceResetOnEnter(ze_fence_reset_params_t* params,
                                ze_result_t result, void* global_user_data,
                                void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeFenceResetOnExit(ze_fence_reset_params_t* params,
                               ze_result_t result, void* global_user_data,
                               void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeFenceReset", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeFenceReset",
                         start_time, end_time);
  }
}

static void zeEventPoolCreateOnEnter(ze_event_pool_create_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventPoolCreateOnExit(ze_event_pool_create_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventPoolCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeEventPoolCreate",
                         start_time, end_time);
  }
}

static void zeEventPoolDestroyOnEnter(ze_event_pool_destroy_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventPoolDestroyOnExit(ze_event_pool_destroy_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventPoolDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeEventPoolDestroy",
                         start_time, end_time);
  }
}

static void zeEventPoolGetIpcHandleOnEnter(
    ze_event_pool_get_ipc_handle_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventPoolGetIpcHandleOnExit(
    ze_event_pool_get_ipc_handle_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventPoolGetIpcHandle", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeEventPoolGetIpcHandle", start_time, end_time);
  }
}

static void zeEventPoolOpenIpcHandleOnEnter(
    ze_event_pool_open_ipc_handle_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventPoolOpenIpcHandleOnExit(
    ze_event_pool_open_ipc_handle_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventPoolOpenIpcHandle", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeEventPoolOpenIpcHandle", start_time, end_time);
  }
}

static void zeEventPoolCloseIpcHandleOnEnter(
    ze_event_pool_close_ipc_handle_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventPoolCloseIpcHandleOnExit(
    ze_event_pool_close_ipc_handle_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventPoolCloseIpcHandle", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeEventPoolCloseIpcHandle", start_time, end_time);
  }
}

static void zeEventCreateOnEnter(ze_event_create_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventCreateOnExit(ze_event_create_params_t* params,
                                ze_result_t result, void* global_user_data,
                                void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeEventCreate",
                         start_time, end_time);
  }
}

static void zeEventDestroyOnEnter(ze_event_destroy_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventDestroyOnExit(ze_event_destroy_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeEventDestroy",
                         start_time, end_time);
  }
}

static void zeEventHostSignalOnEnter(ze_event_host_signal_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventHostSignalOnExit(ze_event_host_signal_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventHostSignal", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeEventHostSignal",
                         start_time, end_time);
  }
}

static void zeEventHostSynchronizeOnEnter(
    ze_event_host_synchronize_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventHostSynchronizeOnExit(
    ze_event_host_synchronize_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventHostSynchronize", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeEventHostSynchronize", start_time, end_time);
  }
}

static void zeEventQueryStatusOnEnter(ze_event_query_status_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventQueryStatusOnExit(ze_event_query_status_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventQueryStatus", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeEventQueryStatus",
                         start_time, end_time);
  }
}

static void zeEventHostResetOnEnter(ze_event_host_reset_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventHostResetOnExit(ze_event_host_reset_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventHostReset", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeEventHostReset",
                         start_time, end_time);
  }
}

static void zeEventQueryKernelTimestampOnEnter(
    ze_event_query_kernel_timestamp_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeEventQueryKernelTimestampOnExit(
    ze_event_query_kernel_timestamp_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeEventQueryKernelTimestamp", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeEventQueryKernelTimestamp", start_time, end_time);
  }
}

static void zeImageGetPropertiesOnEnter(
    ze_image_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeImageGetPropertiesOnExit(ze_image_get_properties_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeImageGetProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeImageGetProperties",
                         start_time, end_time);
  }
}

static void zeImageCreateOnEnter(ze_image_create_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeImageCreateOnExit(ze_image_create_params_t* params,
                                ze_result_t result, void* global_user_data,
                                void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeImageCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeImageCreate",
                         start_time, end_time);
  }
}

static void zeImageDestroyOnEnter(ze_image_destroy_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeImageDestroyOnExit(ze_image_destroy_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeImageDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeImageDestroy",
                         start_time, end_time);
  }
}

static void zeModuleCreateOnEnter(ze_module_create_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleCreateOnExit(ze_module_create_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeModuleCreate",
                         start_time, end_time);
  }
}

static void zeModuleDestroyOnEnter(ze_module_destroy_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleDestroyOnExit(ze_module_destroy_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleDestroy", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeModuleDestroy",
                         start_time, end_time);
  }
}

static void zeModuleDynamicLinkOnEnter(ze_module_dynamic_link_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleDynamicLinkOnExit(ze_module_dynamic_link_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleDynamicLink", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeModuleDynamicLink",
                         start_time, end_time);
  }
}

static void zeModuleGetNativeBinaryOnEnter(
    ze_module_get_native_binary_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleGetNativeBinaryOnExit(
    ze_module_get_native_binary_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleGetNativeBinary", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeModuleGetNativeBinary", start_time, end_time);
  }
}

static void zeModuleGetGlobalPointerOnEnter(
    ze_module_get_global_pointer_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleGetGlobalPointerOnExit(
    ze_module_get_global_pointer_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleGetGlobalPointer", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeModuleGetGlobalPointer", start_time, end_time);
  }
}

static void zeModuleGetKernelNamesOnEnter(
    ze_module_get_kernel_names_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleGetKernelNamesOnExit(
    ze_module_get_kernel_names_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleGetKernelNames", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeModuleGetKernelNames", start_time, end_time);
  }
}

static void zeModuleGetPropertiesOnEnter(
    ze_module_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleGetPropertiesOnExit(
    ze_module_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleGetProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeModuleGetProperties", start_time, end_time);
  }
}

static void zeModuleGetFunctionPointerOnEnter(
    ze_module_get_function_pointer_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleGetFunctionPointerOnExit(
    ze_module_get_function_pointer_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleGetFunctionPointer", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeModuleGetFunctionPointer", start_time, end_time);
  }
}

static void zeModuleBuildLogDestroyOnEnter(
    ze_module_build_log_destroy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleBuildLogDestroyOnExit(
    ze_module_build_log_destroy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleBuildLogDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeModuleBuildLogDestroy", start_time, end_time);
  }
}

static void zeModuleBuildLogGetStringOnEnter(
    ze_module_build_log_get_string_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeModuleBuildLogGetStringOnExit(
    ze_module_build_log_get_string_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeModuleBuildLogGetString", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeModuleBuildLogGetString", start_time, end_time);
  }
}

static void zeKernelCreateOnEnter(ze_kernel_create_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelCreateOnExit(ze_kernel_create_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeKernelCreate",
                         start_time, end_time);
  }
}

static void zeKernelDestroyOnEnter(ze_kernel_destroy_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelDestroyOnExit(ze_kernel_destroy_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeKernelDestroy",
                         start_time, end_time);
  }
}

static void zeKernelSetCacheConfigOnEnter(
    ze_kernel_set_cache_config_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelSetCacheConfigOnExit(
    ze_kernel_set_cache_config_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelSetCacheConfig", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeKernelSetCacheConfig", start_time, end_time);
  }
}

static void zeKernelSetGroupSizeOnEnter(
    ze_kernel_set_group_size_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelSetGroupSizeOnExit(
    ze_kernel_set_group_size_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelSetGroupSize", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeKernelSetGroupSize",
                         start_time, end_time);
  }
}

static void zeKernelSuggestGroupSizeOnEnter(
    ze_kernel_suggest_group_size_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelSuggestGroupSizeOnExit(
    ze_kernel_suggest_group_size_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelSuggestGroupSize", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeKernelSuggestGroupSize", start_time, end_time);
  }
}

static void zeKernelSuggestMaxCooperativeGroupCountOnEnter(
    ze_kernel_suggest_max_cooperative_group_count_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelSuggestMaxCooperativeGroupCountOnExit(
    ze_kernel_suggest_max_cooperative_group_count_params_t* params,
    ze_result_t result, void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelSuggestMaxCooperativeGroupCount", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeKernelSuggestMaxCooperativeGroupCount", start_time,
                         end_time);
  }
}

static void zeKernelSetArgumentValueOnEnter(
    ze_kernel_set_argument_value_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelSetArgumentValueOnExit(
    ze_kernel_set_argument_value_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelSetArgumentValue", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeKernelSetArgumentValue", start_time, end_time);
  }
}

static void zeKernelSetIndirectAccessOnEnter(
    ze_kernel_set_indirect_access_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelSetIndirectAccessOnExit(
    ze_kernel_set_indirect_access_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelSetIndirectAccess", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeKernelSetIndirectAccess", start_time, end_time);
  }
}

static void zeKernelGetIndirectAccessOnEnter(
    ze_kernel_get_indirect_access_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelGetIndirectAccessOnExit(
    ze_kernel_get_indirect_access_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelGetIndirectAccess", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeKernelGetIndirectAccess", start_time, end_time);
  }
}

static void zeKernelGetSourceAttributesOnEnter(
    ze_kernel_get_source_attributes_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelGetSourceAttributesOnExit(
    ze_kernel_get_source_attributes_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelGetSourceAttributes", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeKernelGetSourceAttributes", start_time, end_time);
  }
}

static void zeKernelGetPropertiesOnEnter(
    ze_kernel_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelGetPropertiesOnExit(
    ze_kernel_get_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelGetProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeKernelGetProperties", start_time, end_time);
  }
}

static void zeKernelGetNameOnEnter(ze_kernel_get_name_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeKernelGetNameOnExit(ze_kernel_get_name_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeKernelGetName", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeKernelGetName",
                         start_time, end_time);
  }
}

static void zeSamplerCreateOnEnter(ze_sampler_create_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeSamplerCreateOnExit(ze_sampler_create_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeSamplerCreate", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeSamplerCreate",
                         start_time, end_time);
  }
}

static void zeSamplerDestroyOnEnter(ze_sampler_destroy_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeSamplerDestroyOnExit(ze_sampler_destroy_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeSamplerDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeSamplerDestroy",
                         start_time, end_time);
  }
}

static void zePhysicalMemCreateOnEnter(ze_physical_mem_create_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zePhysicalMemCreateOnExit(ze_physical_mem_create_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zePhysicalMemCreate", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zePhysicalMemCreate",
                         start_time, end_time);
  }
}

static void zePhysicalMemDestroyOnEnter(
    ze_physical_mem_destroy_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zePhysicalMemDestroyOnExit(ze_physical_mem_destroy_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zePhysicalMemDestroy", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zePhysicalMemDestroy",
                         start_time, end_time);
  }
}

static void zeMemAllocSharedOnEnter(ze_mem_alloc_shared_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemAllocSharedOnExit(ze_mem_alloc_shared_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemAllocShared", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeMemAllocShared",
                         start_time, end_time);
  }
}

static void zeMemAllocDeviceOnEnter(ze_mem_alloc_device_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemAllocDeviceOnExit(ze_mem_alloc_device_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemAllocDevice", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeMemAllocDevice",
                         start_time, end_time);
  }
}

static void zeMemAllocHostOnEnter(ze_mem_alloc_host_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemAllocHostOnExit(ze_mem_alloc_host_params_t* params,
                                 ze_result_t result, void* global_user_data,
                                 void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemAllocHost", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeMemAllocHost",
                         start_time, end_time);
  }
}

static void zeMemFreeOnEnter(ze_mem_free_params_t* params, ze_result_t result,
                             void* global_user_data,
                             void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemFreeOnExit(ze_mem_free_params_t* params, ze_result_t result,
                            void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemFree", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeMemFree",
                         start_time, end_time);
  }
}

static void zeMemGetAllocPropertiesOnEnter(
    ze_mem_get_alloc_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemGetAllocPropertiesOnExit(
    ze_mem_get_alloc_properties_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemGetAllocProperties", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeMemGetAllocProperties", start_time, end_time);
  }
}

static void zeMemGetAddressRangeOnEnter(
    ze_mem_get_address_range_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemGetAddressRangeOnExit(
    ze_mem_get_address_range_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemGetAddressRange", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeMemGetAddressRange",
                         start_time, end_time);
  }
}

static void zeMemGetIpcHandleOnEnter(ze_mem_get_ipc_handle_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemGetIpcHandleOnExit(ze_mem_get_ipc_handle_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemGetIpcHandle", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeMemGetIpcHandle",
                         start_time, end_time);
  }
}

static void zeMemOpenIpcHandleOnEnter(ze_mem_open_ipc_handle_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemOpenIpcHandleOnExit(ze_mem_open_ipc_handle_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemOpenIpcHandle", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeMemOpenIpcHandle",
                         start_time, end_time);
  }
}

static void zeMemCloseIpcHandleOnEnter(ze_mem_close_ipc_handle_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeMemCloseIpcHandleOnExit(ze_mem_close_ipc_handle_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeMemCloseIpcHandle", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeMemCloseIpcHandle",
                         start_time, end_time);
  }
}

static void zeVirtualMemReserveOnEnter(ze_virtual_mem_reserve_params_t* params,
                                       ze_result_t result,
                                       void* global_user_data,
                                       void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeVirtualMemReserveOnExit(ze_virtual_mem_reserve_params_t* params,
                                      ze_result_t result,
                                      void* global_user_data,
                                      void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeVirtualMemReserve", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeVirtualMemReserve",
                         start_time, end_time);
  }
}

static void zeVirtualMemFreeOnEnter(ze_virtual_mem_free_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeVirtualMemFreeOnExit(ze_virtual_mem_free_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeVirtualMemFree", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeVirtualMemFree",
                         start_time, end_time);
  }
}

static void zeVirtualMemQueryPageSizeOnEnter(
    ze_virtual_mem_query_page_size_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeVirtualMemQueryPageSizeOnExit(
    ze_virtual_mem_query_page_size_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeVirtualMemQueryPageSize", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeVirtualMemQueryPageSize", start_time, end_time);
  }
}

static void zeVirtualMemMapOnEnter(ze_virtual_mem_map_params_t* params,
                                   ze_result_t result, void* global_user_data,
                                   void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeVirtualMemMapOnExit(ze_virtual_mem_map_params_t* params,
                                  ze_result_t result, void* global_user_data,
                                  void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeVirtualMemMap", time);

  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeVirtualMemMap",
                         start_time, end_time);
  }
}

static void zeVirtualMemUnmapOnEnter(ze_virtual_mem_unmap_params_t* params,
                                     ze_result_t result, void* global_user_data,
                                     void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeVirtualMemUnmapOnExit(ze_virtual_mem_unmap_params_t* params,
                                    ze_result_t result, void* global_user_data,
                                    void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeVirtualMemUnmap", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0", "zeVirtualMemUnmap",
                         start_time, end_time);
  }
}

static void zeVirtualMemSetAccessAttributeOnEnter(
    ze_virtual_mem_set_access_attribute_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeVirtualMemSetAccessAttributeOnExit(
    ze_virtual_mem_set_access_attribute_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeVirtualMemSetAccessAttribute", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeVirtualMemSetAccessAttribute", start_time,
                         end_time);
  }
}

static void zeVirtualMemGetAccessAttributeOnEnter(
    ze_virtual_mem_get_access_attribute_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  start_time = collector->GetTimestamp();
}

static void zeVirtualMemGetAccessAttributeOnExit(
    ze_virtual_mem_get_access_attribute_params_t* params, ze_result_t result,
    void* global_user_data, void** instance_user_data) {
  PTI_ASSERT(global_user_data != nullptr);
  ZeApiCollector* collector =
      reinterpret_cast<ZeApiCollector*>(global_user_data);
  uint64_t end_time = collector->GetTimestamp();

  PTI_ASSERT(collector->correlator_ != nullptr);

  uint64_t& start_time = *reinterpret_cast<uint64_t*>(instance_user_data);
  PTI_ASSERT(start_time > 0);
  PTI_ASSERT(start_time < end_time);
  uint64_t time = end_time - start_time;
  collector->AddFunctionTime("zeVirtualMemGetAccessAttribute", time);
  if (collector->callback_ != nullptr) {
    collector->callback_(collector->callback_data_, "0",
                         "zeVirtualMemGetAccessAttribute", start_time,
                         end_time);
  }
}

static void SetTracingAPIs(zel_tracer_handle_t tracer) {
  zet_core_callbacks_t prologue = {};
  zet_core_callbacks_t epilogue = {};

  prologue.Global.pfnInitCb = zeInitOnEnter;
  epilogue.Global.pfnInitCb = zeInitOnExit;
  prologue.Driver.pfnGetCb = zeDriverGetOnEnter;
  epilogue.Driver.pfnGetCb = zeDriverGetOnExit;
  prologue.Driver.pfnGetApiVersionCb = zeDriverGetApiVersionOnEnter;
  epilogue.Driver.pfnGetApiVersionCb = zeDriverGetApiVersionOnExit;
  prologue.Driver.pfnGetPropertiesCb = zeDriverGetPropertiesOnEnter;
  epilogue.Driver.pfnGetPropertiesCb = zeDriverGetPropertiesOnExit;
  prologue.Driver.pfnGetIpcPropertiesCb = zeDriverGetIpcPropertiesOnEnter;
  epilogue.Driver.pfnGetIpcPropertiesCb = zeDriverGetIpcPropertiesOnExit;
  prologue.Driver.pfnGetExtensionPropertiesCb =
      zeDriverGetExtensionPropertiesOnEnter;
  epilogue.Driver.pfnGetExtensionPropertiesCb =
      zeDriverGetExtensionPropertiesOnExit;
  prologue.Device.pfnGetCb = zeDeviceGetOnEnter;
  epilogue.Device.pfnGetCb = zeDeviceGetOnExit;
  prologue.Device.pfnGetSubDevicesCb = zeDeviceGetSubDevicesOnEnter;
  epilogue.Device.pfnGetSubDevicesCb = zeDeviceGetSubDevicesOnExit;
  prologue.Device.pfnGetPropertiesCb = zeDeviceGetPropertiesOnEnter;
  epilogue.Device.pfnGetPropertiesCb = zeDeviceGetPropertiesOnExit;
  prologue.Device.pfnGetComputePropertiesCb =
      zeDeviceGetComputePropertiesOnEnter;
  epilogue.Device.pfnGetComputePropertiesCb =
      zeDeviceGetComputePropertiesOnExit;
  prologue.Device.pfnGetModulePropertiesCb = zeDeviceGetModulePropertiesOnEnter;
  epilogue.Device.pfnGetModulePropertiesCb = zeDeviceGetModulePropertiesOnExit;
  prologue.Device.pfnGetCommandQueueGroupPropertiesCb =
      zeDeviceGetCommandQueueGroupPropertiesOnEnter;
  epilogue.Device.pfnGetCommandQueueGroupPropertiesCb =
      zeDeviceGetCommandQueueGroupPropertiesOnExit;
  prologue.Device.pfnGetMemoryPropertiesCb = zeDeviceGetMemoryPropertiesOnEnter;
  epilogue.Device.pfnGetMemoryPropertiesCb = zeDeviceGetMemoryPropertiesOnExit;
  prologue.Device.pfnGetMemoryAccessPropertiesCb =
      zeDeviceGetMemoryAccessPropertiesOnEnter;
  epilogue.Device.pfnGetMemoryAccessPropertiesCb =
      zeDeviceGetMemoryAccessPropertiesOnExit;
  prologue.Device.pfnGetCachePropertiesCb = zeDeviceGetCachePropertiesOnEnter;
  epilogue.Device.pfnGetCachePropertiesCb = zeDeviceGetCachePropertiesOnExit;
  prologue.Device.pfnGetImagePropertiesCb = zeDeviceGetImagePropertiesOnEnter;
  epilogue.Device.pfnGetImagePropertiesCb = zeDeviceGetImagePropertiesOnExit;
  prologue.Device.pfnGetExternalMemoryPropertiesCb =
      zeDeviceGetExternalMemoryPropertiesOnEnter;
  epilogue.Device.pfnGetExternalMemoryPropertiesCb =
      zeDeviceGetExternalMemoryPropertiesOnExit;
  prologue.Device.pfnGetP2PPropertiesCb = zeDeviceGetP2PPropertiesOnEnter;
  epilogue.Device.pfnGetP2PPropertiesCb = zeDeviceGetP2PPropertiesOnExit;
  prologue.Device.pfnCanAccessPeerCb = zeDeviceCanAccessPeerOnEnter;
  epilogue.Device.pfnCanAccessPeerCb = zeDeviceCanAccessPeerOnExit;
  prologue.Device.pfnGetStatusCb = zeDeviceGetStatusOnEnter;
  epilogue.Device.pfnGetStatusCb = zeDeviceGetStatusOnExit;
  prologue.Context.pfnCreateCb = zeContextCreateOnEnter;
  epilogue.Context.pfnCreateCb = zeContextCreateOnExit;
  prologue.Context.pfnDestroyCb = zeContextDestroyOnEnter;
  epilogue.Context.pfnDestroyCb = zeContextDestroyOnExit;
  prologue.Context.pfnGetStatusCb = zeContextGetStatusOnEnter;
  epilogue.Context.pfnGetStatusCb = zeContextGetStatusOnExit;
  prologue.Context.pfnSystemBarrierCb = zeContextSystemBarrierOnEnter;
  epilogue.Context.pfnSystemBarrierCb = zeContextSystemBarrierOnExit;
  prologue.Context.pfnMakeMemoryResidentCb = zeContextMakeMemoryResidentOnEnter;
  epilogue.Context.pfnMakeMemoryResidentCb = zeContextMakeMemoryResidentOnExit;
  prologue.Context.pfnEvictMemoryCb = zeContextEvictMemoryOnEnter;
  epilogue.Context.pfnEvictMemoryCb = zeContextEvictMemoryOnExit;
  prologue.Context.pfnMakeImageResidentCb = zeContextMakeImageResidentOnEnter;
  epilogue.Context.pfnMakeImageResidentCb = zeContextMakeImageResidentOnExit;
  prologue.Context.pfnEvictImageCb = zeContextEvictImageOnEnter;
  epilogue.Context.pfnEvictImageCb = zeContextEvictImageOnExit;
  prologue.CommandQueue.pfnCreateCb = zeCommandQueueCreateOnEnter;
  epilogue.CommandQueue.pfnCreateCb = zeCommandQueueCreateOnExit;
  prologue.CommandQueue.pfnDestroyCb = zeCommandQueueDestroyOnEnter;
  epilogue.CommandQueue.pfnDestroyCb = zeCommandQueueDestroyOnExit;
  prologue.CommandQueue.pfnExecuteCommandListsCb =
      zeCommandQueueExecuteCommandListsOnEnter;
  epilogue.CommandQueue.pfnExecuteCommandListsCb =
      zeCommandQueueExecuteCommandListsOnExit;
  prologue.CommandQueue.pfnSynchronizeCb = zeCommandQueueSynchronizeOnEnter;
  epilogue.CommandQueue.pfnSynchronizeCb = zeCommandQueueSynchronizeOnExit;
  prologue.CommandList.pfnCreateCb = zeCommandListCreateOnEnter;
  epilogue.CommandList.pfnCreateCb = zeCommandListCreateOnExit;
  prologue.CommandList.pfnCreateImmediateCb =
      zeCommandListCreateImmediateOnEnter;
  epilogue.CommandList.pfnCreateImmediateCb =
      zeCommandListCreateImmediateOnExit;
  prologue.CommandList.pfnDestroyCb = zeCommandListDestroyOnEnter;
  epilogue.CommandList.pfnDestroyCb = zeCommandListDestroyOnExit;
  prologue.CommandList.pfnCloseCb = zeCommandListCloseOnEnter;
  epilogue.CommandList.pfnCloseCb = zeCommandListCloseOnExit;
  prologue.CommandList.pfnResetCb = zeCommandListResetOnEnter;
  epilogue.CommandList.pfnResetCb = zeCommandListResetOnExit;
  prologue.CommandList.pfnAppendWriteGlobalTimestampCb =
      zeCommandListAppendWriteGlobalTimestampOnEnter;
  epilogue.CommandList.pfnAppendWriteGlobalTimestampCb =
      zeCommandListAppendWriteGlobalTimestampOnExit;
  prologue.CommandList.pfnAppendBarrierCb = zeCommandListAppendBarrierOnEnter;
  epilogue.CommandList.pfnAppendBarrierCb = zeCommandListAppendBarrierOnExit;
  prologue.CommandList.pfnAppendMemoryRangesBarrierCb =
      zeCommandListAppendMemoryRangesBarrierOnEnter;
  epilogue.CommandList.pfnAppendMemoryRangesBarrierCb =
      zeCommandListAppendMemoryRangesBarrierOnExit;
  prologue.CommandList.pfnAppendMemoryCopyCb =
      zeCommandListAppendMemoryCopyOnEnter;
  epilogue.CommandList.pfnAppendMemoryCopyCb =
      zeCommandListAppendMemoryCopyOnExit;
  prologue.CommandList.pfnAppendMemoryFillCb =
      zeCommandListAppendMemoryFillOnEnter;
  epilogue.CommandList.pfnAppendMemoryFillCb =
      zeCommandListAppendMemoryFillOnExit;
  prologue.CommandList.pfnAppendMemoryCopyRegionCb =
      zeCommandListAppendMemoryCopyRegionOnEnter;
  epilogue.CommandList.pfnAppendMemoryCopyRegionCb =
      zeCommandListAppendMemoryCopyRegionOnExit;
  prologue.CommandList.pfnAppendMemoryCopyFromContextCb =
      zeCommandListAppendMemoryCopyFromContextOnEnter;
  epilogue.CommandList.pfnAppendMemoryCopyFromContextCb =
      zeCommandListAppendMemoryCopyFromContextOnExit;
  prologue.CommandList.pfnAppendImageCopyCb =
      zeCommandListAppendImageCopyOnEnter;
  epilogue.CommandList.pfnAppendImageCopyCb =
      zeCommandListAppendImageCopyOnExit;
  prologue.CommandList.pfnAppendImageCopyRegionCb =
      zeCommandListAppendImageCopyRegionOnEnter;
  epilogue.CommandList.pfnAppendImageCopyRegionCb =
      zeCommandListAppendImageCopyRegionOnExit;
  prologue.CommandList.pfnAppendImageCopyToMemoryCb =
      zeCommandListAppendImageCopyToMemoryOnEnter;
  epilogue.CommandList.pfnAppendImageCopyToMemoryCb =
      zeCommandListAppendImageCopyToMemoryOnExit;
  prologue.CommandList.pfnAppendImageCopyFromMemoryCb =
      zeCommandListAppendImageCopyFromMemoryOnEnter;
  epilogue.CommandList.pfnAppendImageCopyFromMemoryCb =
      zeCommandListAppendImageCopyFromMemoryOnExit;
  prologue.CommandList.pfnAppendMemoryPrefetchCb =
      zeCommandListAppendMemoryPrefetchOnEnter;
  epilogue.CommandList.pfnAppendMemoryPrefetchCb =
      zeCommandListAppendMemoryPrefetchOnExit;
  prologue.CommandList.pfnAppendMemAdviseCb =
      zeCommandListAppendMemAdviseOnEnter;
  epilogue.CommandList.pfnAppendMemAdviseCb =
      zeCommandListAppendMemAdviseOnExit;
  prologue.CommandList.pfnAppendSignalEventCb =
      zeCommandListAppendSignalEventOnEnter;
  epilogue.CommandList.pfnAppendSignalEventCb =
      zeCommandListAppendSignalEventOnExit;
  prologue.CommandList.pfnAppendWaitOnEventsCb =
      zeCommandListAppendWaitOnEventsOnEnter;
  epilogue.CommandList.pfnAppendWaitOnEventsCb =
      zeCommandListAppendWaitOnEventsOnExit;
  prologue.CommandList.pfnAppendEventResetCb =
      zeCommandListAppendEventResetOnEnter;
  epilogue.CommandList.pfnAppendEventResetCb =
      zeCommandListAppendEventResetOnExit;
  prologue.CommandList.pfnAppendQueryKernelTimestampsCb =
      zeCommandListAppendQueryKernelTimestampsOnEnter;
  epilogue.CommandList.pfnAppendQueryKernelTimestampsCb =
      zeCommandListAppendQueryKernelTimestampsOnExit;
  prologue.CommandList.pfnAppendLaunchKernelCb =
      zeCommandListAppendLaunchKernelOnEnter;
  epilogue.CommandList.pfnAppendLaunchKernelCb =
      zeCommandListAppendLaunchKernelOnExit;
  prologue.CommandList.pfnAppendLaunchCooperativeKernelCb =
      zeCommandListAppendLaunchCooperativeKernelOnEnter;
  epilogue.CommandList.pfnAppendLaunchCooperativeKernelCb =
      zeCommandListAppendLaunchCooperativeKernelOnExit;
  prologue.CommandList.pfnAppendLaunchKernelIndirectCb =
      zeCommandListAppendLaunchKernelIndirectOnEnter;
  epilogue.CommandList.pfnAppendLaunchKernelIndirectCb =
      zeCommandListAppendLaunchKernelIndirectOnExit;
  prologue.CommandList.pfnAppendLaunchMultipleKernelsIndirectCb =
      zeCommandListAppendLaunchMultipleKernelsIndirectOnEnter;
  epilogue.CommandList.pfnAppendLaunchMultipleKernelsIndirectCb =
      zeCommandListAppendLaunchMultipleKernelsIndirectOnExit;
  prologue.Fence.pfnCreateCb = zeFenceCreateOnEnter;
  epilogue.Fence.pfnCreateCb = zeFenceCreateOnExit;
  prologue.Fence.pfnDestroyCb = zeFenceDestroyOnEnter;
  epilogue.Fence.pfnDestroyCb = zeFenceDestroyOnExit;
  prologue.Fence.pfnHostSynchronizeCb = zeFenceHostSynchronizeOnEnter;
  epilogue.Fence.pfnHostSynchronizeCb = zeFenceHostSynchronizeOnExit;
  prologue.Fence.pfnQueryStatusCb = zeFenceQueryStatusOnEnter;
  epilogue.Fence.pfnQueryStatusCb = zeFenceQueryStatusOnExit;
  prologue.Fence.pfnResetCb = zeFenceResetOnEnter;
  epilogue.Fence.pfnResetCb = zeFenceResetOnExit;
  prologue.EventPool.pfnCreateCb = zeEventPoolCreateOnEnter;
  epilogue.EventPool.pfnCreateCb = zeEventPoolCreateOnExit;
  prologue.EventPool.pfnDestroyCb = zeEventPoolDestroyOnEnter;
  epilogue.EventPool.pfnDestroyCb = zeEventPoolDestroyOnExit;
  prologue.EventPool.pfnGetIpcHandleCb = zeEventPoolGetIpcHandleOnEnter;
  epilogue.EventPool.pfnGetIpcHandleCb = zeEventPoolGetIpcHandleOnExit;
  prologue.EventPool.pfnOpenIpcHandleCb = zeEventPoolOpenIpcHandleOnEnter;
  epilogue.EventPool.pfnOpenIpcHandleCb = zeEventPoolOpenIpcHandleOnExit;
  prologue.EventPool.pfnCloseIpcHandleCb = zeEventPoolCloseIpcHandleOnEnter;
  epilogue.EventPool.pfnCloseIpcHandleCb = zeEventPoolCloseIpcHandleOnExit;
  prologue.Event.pfnCreateCb = zeEventCreateOnEnter;
  epilogue.Event.pfnCreateCb = zeEventCreateOnExit;
  prologue.Event.pfnDestroyCb = zeEventDestroyOnEnter;
  epilogue.Event.pfnDestroyCb = zeEventDestroyOnExit;
  prologue.Event.pfnHostSignalCb = zeEventHostSignalOnEnter;
  epilogue.Event.pfnHostSignalCb = zeEventHostSignalOnExit;
  prologue.Event.pfnHostSynchronizeCb = zeEventHostSynchronizeOnEnter;
  epilogue.Event.pfnHostSynchronizeCb = zeEventHostSynchronizeOnExit;
  prologue.Event.pfnQueryStatusCb = zeEventQueryStatusOnEnter;
  epilogue.Event.pfnQueryStatusCb = zeEventQueryStatusOnExit;
  prologue.Event.pfnHostResetCb = zeEventHostResetOnEnter;
  epilogue.Event.pfnHostResetCb = zeEventHostResetOnExit;
  prologue.Event.pfnQueryKernelTimestampCb = zeEventQueryKernelTimestampOnEnter;
  epilogue.Event.pfnQueryKernelTimestampCb = zeEventQueryKernelTimestampOnExit;
  prologue.Image.pfnGetPropertiesCb = zeImageGetPropertiesOnEnter;
  epilogue.Image.pfnGetPropertiesCb = zeImageGetPropertiesOnExit;
  prologue.Image.pfnCreateCb = zeImageCreateOnEnter;
  epilogue.Image.pfnCreateCb = zeImageCreateOnExit;
  prologue.Image.pfnDestroyCb = zeImageDestroyOnEnter;
  epilogue.Image.pfnDestroyCb = zeImageDestroyOnExit;
  prologue.Module.pfnCreateCb = zeModuleCreateOnEnter;
  epilogue.Module.pfnCreateCb = zeModuleCreateOnExit;
  prologue.Module.pfnDestroyCb = zeModuleDestroyOnEnter;
  epilogue.Module.pfnDestroyCb = zeModuleDestroyOnExit;
  prologue.Module.pfnDynamicLinkCb = zeModuleDynamicLinkOnEnter;
  epilogue.Module.pfnDynamicLinkCb = zeModuleDynamicLinkOnExit;
  prologue.Module.pfnGetNativeBinaryCb = zeModuleGetNativeBinaryOnEnter;
  epilogue.Module.pfnGetNativeBinaryCb = zeModuleGetNativeBinaryOnExit;
  prologue.Module.pfnGetGlobalPointerCb = zeModuleGetGlobalPointerOnEnter;
  epilogue.Module.pfnGetGlobalPointerCb = zeModuleGetGlobalPointerOnExit;
  prologue.Module.pfnGetKernelNamesCb = zeModuleGetKernelNamesOnEnter;
  epilogue.Module.pfnGetKernelNamesCb = zeModuleGetKernelNamesOnExit;
  prologue.Module.pfnGetPropertiesCb = zeModuleGetPropertiesOnEnter;
  epilogue.Module.pfnGetPropertiesCb = zeModuleGetPropertiesOnExit;
  prologue.Module.pfnGetFunctionPointerCb = zeModuleGetFunctionPointerOnEnter;
  epilogue.Module.pfnGetFunctionPointerCb = zeModuleGetFunctionPointerOnExit;
  prologue.ModuleBuildLog.pfnDestroyCb = zeModuleBuildLogDestroyOnEnter;
  epilogue.ModuleBuildLog.pfnDestroyCb = zeModuleBuildLogDestroyOnExit;
  prologue.ModuleBuildLog.pfnGetStringCb = zeModuleBuildLogGetStringOnEnter;
  epilogue.ModuleBuildLog.pfnGetStringCb = zeModuleBuildLogGetStringOnExit;
  prologue.Kernel.pfnCreateCb = zeKernelCreateOnEnter;
  epilogue.Kernel.pfnCreateCb = zeKernelCreateOnExit;
  prologue.Kernel.pfnDestroyCb = zeKernelDestroyOnEnter;
  epilogue.Kernel.pfnDestroyCb = zeKernelDestroyOnExit;
  prologue.Kernel.pfnSetCacheConfigCb = zeKernelSetCacheConfigOnEnter;
  epilogue.Kernel.pfnSetCacheConfigCb = zeKernelSetCacheConfigOnExit;
  prologue.Kernel.pfnSetGroupSizeCb = zeKernelSetGroupSizeOnEnter;
  epilogue.Kernel.pfnSetGroupSizeCb = zeKernelSetGroupSizeOnExit;
  prologue.Kernel.pfnSuggestGroupSizeCb = zeKernelSuggestGroupSizeOnEnter;
  epilogue.Kernel.pfnSuggestGroupSizeCb = zeKernelSuggestGroupSizeOnExit;
  prologue.Kernel.pfnSuggestMaxCooperativeGroupCountCb =
      zeKernelSuggestMaxCooperativeGroupCountOnEnter;
  epilogue.Kernel.pfnSuggestMaxCooperativeGroupCountCb =
      zeKernelSuggestMaxCooperativeGroupCountOnExit;
  prologue.Kernel.pfnSetArgumentValueCb = zeKernelSetArgumentValueOnEnter;
  epilogue.Kernel.pfnSetArgumentValueCb = zeKernelSetArgumentValueOnExit;
  prologue.Kernel.pfnSetIndirectAccessCb = zeKernelSetIndirectAccessOnEnter;
  epilogue.Kernel.pfnSetIndirectAccessCb = zeKernelSetIndirectAccessOnExit;
  prologue.Kernel.pfnGetIndirectAccessCb = zeKernelGetIndirectAccessOnEnter;
  epilogue.Kernel.pfnGetIndirectAccessCb = zeKernelGetIndirectAccessOnExit;
  prologue.Kernel.pfnGetSourceAttributesCb = zeKernelGetSourceAttributesOnEnter;
  epilogue.Kernel.pfnGetSourceAttributesCb = zeKernelGetSourceAttributesOnExit;
  prologue.Kernel.pfnGetPropertiesCb = zeKernelGetPropertiesOnEnter;
  epilogue.Kernel.pfnGetPropertiesCb = zeKernelGetPropertiesOnExit;
  prologue.Kernel.pfnGetNameCb = zeKernelGetNameOnEnter;
  epilogue.Kernel.pfnGetNameCb = zeKernelGetNameOnExit;
  prologue.Sampler.pfnCreateCb = zeSamplerCreateOnEnter;
  epilogue.Sampler.pfnCreateCb = zeSamplerCreateOnExit;
  prologue.Sampler.pfnDestroyCb = zeSamplerDestroyOnEnter;
  epilogue.Sampler.pfnDestroyCb = zeSamplerDestroyOnExit;
  prologue.PhysicalMem.pfnCreateCb = zePhysicalMemCreateOnEnter;
  epilogue.PhysicalMem.pfnCreateCb = zePhysicalMemCreateOnExit;
  prologue.PhysicalMem.pfnDestroyCb = zePhysicalMemDestroyOnEnter;
  epilogue.PhysicalMem.pfnDestroyCb = zePhysicalMemDestroyOnExit;
  prologue.Mem.pfnAllocSharedCb = zeMemAllocSharedOnEnter;
  epilogue.Mem.pfnAllocSharedCb = zeMemAllocSharedOnExit;
  prologue.Mem.pfnAllocDeviceCb = zeMemAllocDeviceOnEnter;
  epilogue.Mem.pfnAllocDeviceCb = zeMemAllocDeviceOnExit;
  prologue.Mem.pfnAllocHostCb = zeMemAllocHostOnEnter;
  epilogue.Mem.pfnAllocHostCb = zeMemAllocHostOnExit;
  prologue.Mem.pfnFreeCb = zeMemFreeOnEnter;
  epilogue.Mem.pfnFreeCb = zeMemFreeOnExit;
  prologue.Mem.pfnGetAllocPropertiesCb = zeMemGetAllocPropertiesOnEnter;
  epilogue.Mem.pfnGetAllocPropertiesCb = zeMemGetAllocPropertiesOnExit;
  prologue.Mem.pfnGetAddressRangeCb = zeMemGetAddressRangeOnEnter;
  epilogue.Mem.pfnGetAddressRangeCb = zeMemGetAddressRangeOnExit;
  prologue.Mem.pfnGetIpcHandleCb = zeMemGetIpcHandleOnEnter;
  epilogue.Mem.pfnGetIpcHandleCb = zeMemGetIpcHandleOnExit;
  prologue.Mem.pfnOpenIpcHandleCb = zeMemOpenIpcHandleOnEnter;
  epilogue.Mem.pfnOpenIpcHandleCb = zeMemOpenIpcHandleOnExit;
  prologue.Mem.pfnCloseIpcHandleCb = zeMemCloseIpcHandleOnEnter;
  epilogue.Mem.pfnCloseIpcHandleCb = zeMemCloseIpcHandleOnExit;
  prologue.VirtualMem.pfnReserveCb = zeVirtualMemReserveOnEnter;
  epilogue.VirtualMem.pfnReserveCb = zeVirtualMemReserveOnExit;
  prologue.VirtualMem.pfnFreeCb = zeVirtualMemFreeOnEnter;
  epilogue.VirtualMem.pfnFreeCb = zeVirtualMemFreeOnExit;
  prologue.VirtualMem.pfnQueryPageSizeCb = zeVirtualMemQueryPageSizeOnEnter;
  epilogue.VirtualMem.pfnQueryPageSizeCb = zeVirtualMemQueryPageSizeOnExit;
  prologue.VirtualMem.pfnMapCb = zeVirtualMemMapOnEnter;
  epilogue.VirtualMem.pfnMapCb = zeVirtualMemMapOnExit;
  prologue.VirtualMem.pfnUnmapCb = zeVirtualMemUnmapOnEnter;
  epilogue.VirtualMem.pfnUnmapCb = zeVirtualMemUnmapOnExit;
  prologue.VirtualMem.pfnSetAccessAttributeCb =
      zeVirtualMemSetAccessAttributeOnEnter;
  epilogue.VirtualMem.pfnSetAccessAttributeCb =
      zeVirtualMemSetAccessAttributeOnExit;
  prologue.VirtualMem.pfnGetAccessAttributeCb =
      zeVirtualMemGetAccessAttributeOnEnter;
  epilogue.VirtualMem.pfnGetAccessAttributeCb =
      zeVirtualMemGetAccessAttributeOnExit;

  ze_result_t status = ZE_RESULT_SUCCESS;
  status = zelTracerSetPrologues(tracer, &prologue);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  status = zelTracerSetEpilogues(tracer, &epilogue);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
}
#endif  // ITEX_CORE_PROFILER_TRACING_H_
