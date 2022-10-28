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
const char* const XeHPC_name = "0x0bd5";
const char* const XeHPC_name_448 = "0x0bd6";

bool IsXeHPC(sycl::device* device_ptr) {
  if (device_ptr == nullptr) {
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      auto device_list = platform.get_devices();
      for (const auto& device : device_list) {
        if (device.is_gpu()) {
          std::string name = device.get_info<sycl::info::device::name>();
          if (name.find(XeHPC_name) != std::string::npos ||
              name.find(XeHPC_name_448) != std::string::npos) {
            return true;
          }
        }
      }
    }
  } else {
    std::string name = device_ptr->get_info<sycl::info::device::name>();
    if (name.find(XeHPC_name) != std::string::npos ||
        name.find(XeHPC_name_448) != std::string::npos) {
      return true;
    }
  }
  return false;
}
#endif
