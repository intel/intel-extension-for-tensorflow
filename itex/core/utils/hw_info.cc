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
const int32_t XeHPC_id_2 = 0xb60;

// PVC 1550VG does not have XeMatrix engine, we distinguish it from other PVCs
// by device id.
const int32_t XeHPC_no_xmx_id = 0xbd4;

bool IsXeHPC(sycl::device* device_ptr) {
  if (device_ptr == nullptr) {
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      auto device_list = platform.get_devices();
      for (const auto& device : device_list) {
        if (device.is_gpu()) {
          auto id =
              device.get_info<sycl::ext::intel::info::device::device_id>();
          if ((id & 0xff0) == XeHPC_id || (id & 0xff0) == XeHPC_id_2) {
            return true;
          }
        }
      }
    }
  } else {
    auto id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    if ((id & 0xff0) == XeHPC_id || (id & 0xff0) == XeHPC_id_2) {
      return true;
    }
  }
  return false;
}

// TODO(ITEX): use sycl api like `devices.has(sycl::aspect::ext_intel_matrix)`
// instead of device id once compiler supports XMX query interface.
bool HasXMX(sycl::device* device_ptr) {
  if (device_ptr == nullptr) {
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      auto device_list = platform.get_devices();
      for (const auto& device : device_list) {
        if (device.is_gpu()) {
          auto id =
              device.get_info<sycl::ext::intel::info::device::device_id>();
          if ((id & 0xff0) == XeHPC_id || (id & 0xff0) == XeHPC_id_2) {
            if (id == XeHPC_no_xmx_id) {
              return false;
            } else {
              return true;
            }
          }
        }
      }
    }
  } else {
    auto id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    if ((id & 0xff0) == XeHPC_id || (id & 0xff0) == XeHPC_id_2) {
      if (id == XeHPC_no_xmx_id) {
        return false;
      } else {
        return true;
      }
    }
  }
  return false;
}
#endif
