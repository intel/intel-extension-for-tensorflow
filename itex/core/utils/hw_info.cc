/* Copyright (c) 2022 Intel Corporation

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

#include "itex/core/utils/hw_info.h"

#include <string>

#ifndef INTEL_CPU_ONLY
const int32_t XeHPC_id = 0xbd0;
const char* const XeHPC_name = "0x0bd";
const char* const XeHPC_name_new = "Data Center GPU Max";

bool IsXeHPC(sycl::device* device_ptr) {
  if (device_ptr == nullptr) {
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      auto device_list = platform.get_devices();
      for (const auto& device : device_list) {
        if (device.is_gpu()) {
#if defined(SYCL_EXT_INTEL_DEVICE_INFO) && (SYCL_EXT_INTEL_DEVICE_INFO >= 5)
          auto id =
              device.get_info<sycl::ext::intel::info::device::device_id>();
          if ((id & 0xff0) == XeHPC_id) {
            return true;
          }
#else
          std::string name = device.get_info<sycl::info::device::name>();
          if (name.find(XeHPC_name) != std::string::npos ||
              name.find(XeHPC_name_new) != std::string::npos) {
            return true;
          }
#endif
        }
      }
    }
  } else {
#if defined(SYCL_EXT_INTEL_DEVICE_INFO) && (SYCL_EXT_INTEL_DEVICE_INFO >= 5)
    auto id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    if ((id & 0xff0) == XeHPC_id) {
      return true;
    }
#else
    std::string name = device_ptr->get_info<sycl::info::device::name>();
    if (name.find(XeHPC_name) != std::string::npos ||
        name.find(XeHPC_name_new) != std::string::npos) {
      return true;
    }
#endif
  }
  return false;
}
#endif
