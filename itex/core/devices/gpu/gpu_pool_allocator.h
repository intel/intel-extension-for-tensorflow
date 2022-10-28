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

#ifndef ITEX_CORE_DEVICES_GPU_GPU_POOL_ALLOCATOR_H_
#define ITEX_CORE_DEVICES_GPU_GPU_POOL_ALLOCATOR_H_

#include <map>

#include "third_party/build_option/dpcpp/runtime/dpcpp_runtime.h"

#include "itex/core/devices/bfc_allocator.h"

namespace itex {

class AllocatorPool {
 public:
  static dpcppError_t getAllocator(DPCPPDevice* device, BFCAllocator** alloc) {
    auto allocs = AllocatorPool::GetAllocatorPool();
    for (auto& [key, value] : allocs) {
      if (key == device) {
        *alloc = value;
        return DPCPP_SUCCESS;
      }
    }
    return DPCPP_ERROR_INVALID_DEVICE;
  }

 private:
  static std::map<DPCPPDevice*, BFCAllocator*>& GetAllocatorPool() {
    static std::once_flag init_alloc_flag;
    static std::map<DPCPPDevice*, BFCAllocator*> allocators;

    std::call_once(init_alloc_flag, []() {
      int device_count = 0;
      dpcppGetDeviceCount(&device_count);
      DPCPPDevice* device = nullptr;
      for (int i = 0; i < device_count; ++i) {
        dpcppGetDevice(&device, i);
        allocators.insert({device, new BFCAllocator(device)});
      }
    });

    return allocators;
  }
};  // class AllocatorPool

// clang-format off
inline dpcppError_t dpcppGetAllocator(DPCPPDevice* device,
                                      BFCAllocator** alloc) {
  return AllocatorPool::getAllocator(device, alloc);
}
// clang-format on

}  // namespace itex
#endif  // ITEX_CORE_DEVICES_GPU_GPU_POOL_ALLOCATOR_H_
