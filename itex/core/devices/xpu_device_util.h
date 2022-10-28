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

#ifndef ITEX_CORE_DEVICES_XPU_DEVICE_UTIL_H_
#define ITEX_CORE_DEVICES_XPU_DEVICE_UTIL_H_

#include <string>

#include "itex/core/utils/env_var.h"
#include "itex/core/utils/protobuf/config.pb.h"
#include "itex/core/utils/types.h"

constexpr char DEVICE_XPU_NAME[] = "XPU";

enum ITEX_BACKEND {
  ITEX_BACKEND_GPU,
  ITEX_BACKEND_CPU,
  ITEX_BACKEND_AUTO,
  ITEX_BACKEND_DEFAULT = ITEX_BACKEND_GPU
};

namespace itex {

ITEX_BACKEND itex_get_backend();
ConfigProto itex_get_config();
void itex_set_backend(const char* backend, const ConfigProto& config);
void itex_backend_to_string(ITEX_BACKEND backend, std::string* backend_string);
void itex_freeze_backend(const char* backend, const ConfigProto& config);
void itex_freeze_backend(ITEX_BACKEND backend, const ConfigProto& config);

template <typename Device, typename T>
void DeviceFill(T* ptr, const T& pattern, size_t count, void* stream) {
#ifndef INTEL_CPU_ONLY
  if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    DPCPPStream* dpcpp_stream = static_cast<DPCPPStream*>(stream);
    // pattern must be trivially copyable
    dpcpp_stream->fill<T>(ptr, pattern, count);
    return;
  }
#else
  if (Eigen::internal::is_same<Device, Eigen::ThreadPoolDevice>::value) {
    std::fill(ptr, ptr + count, pattern);
    return;
  }
#endif

  ITEX_CHECK(false) << "Undefined device for fill!";
}

template <typename Device>
void DeviceMemset(void* dst, int value, size_t size, void* stream) {
#ifndef INTEL_CPU_ONLY
  if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    DPCPPStream* dpcpp_stream = static_cast<DPCPPStream*>(stream);
    dpcpp_stream->memset(dst, value, size);
    return;
  }
#else
  if (Eigen::internal::is_same<Device, Eigen::ThreadPoolDevice>::value) {
    memset(dst, value, size);
    return;
  }
#endif

  ITEX_CHECK(false) << "Undefined device for memset!";
}

template <typename Device>
void DeviceMemcpy(void* dst, const void* src, size_t size, void* stream) {
#ifndef INTEL_CPU_ONLY
  if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    DPCPPStream* dpcpp_stream = static_cast<DPCPPStream*>(stream);
    dpcpp_stream->memcpy(dst, src, size);
    return;
  }
#else
  if (Eigen::internal::is_same<Device, Eigen::ThreadPoolDevice>::value) {
    memcpy(dst, src, size);
    return;
  }
#endif

  ITEX_CHECK(false) << "Undefined device for memcpy!";
}

// Get the real backend name of given device.
// @return:
//   1) CPU: DEVICE_CPU;
//   2) GPU: DEVICE_GPU;
//   3) XPU with GPU backend: DEVICE_GPU;
//   4) TODO(itex): XPU with other backend
// Return value will never be nullptr.
const char* GetDeviceBackendName(const std::string& device_name);

}  // namespace itex

// ITEX_TILE_AS_DEVICE
//   True (default behaviour): Tile as an individual device in device list
//   False: Only root device as an individual device in device list
inline bool TileAsDevice() {
  bool tile_as_device;
  if (std::getenv(std::string("ITEX_ENABLE_TILE_AS_DEVICE").c_str()) &&
      !std::getenv(std::string("ITEX_TILE_AS_DEVICE").c_str())) {
    ITEX_CHECK_OK(itex::ReadBoolFromEnvVar("ITEX_ENABLE_TILE_AS_DEVICE", true,
                                           &tile_as_device));
    ITEX_LOG(WARNING) << "`ITEX_ENABLE_TILE_AS_DEVICE` will be deprecated, "
                         "please use `ITEX_TILE_AS_DEVICE` instead.";
  } else {
    ITEX_CHECK_OK(
        itex::ReadBoolFromEnvVar("ITEX_TILE_AS_DEVICE", true, &tile_as_device));
  }
  return tile_as_device;
}

#endif  // ITEX_CORE_DEVICES_XPU_DEVICE_UTIL_H_
