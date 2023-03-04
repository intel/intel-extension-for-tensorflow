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
#include <memory>

#include "itex/core/devices/bfc_allocator.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

namespace itex {

class AllocatorPool {
 public:
  static ITEX_GPUError_t getAllocator(ITEX_GPUDevice* device,
                                      std::shared_ptr<BFCAllocator>* alloc) {
    auto allocs = AllocatorPool::GetAllocatorPool();
    for (const auto& [key, value] : allocs) {
      if (key == device) {
        *alloc = value;
        return ITEX_GPU_SUCCESS;
      }
    }
    return ITEX_GPU_ERROR_INVALID_DEVICE;
  }

 private:
  static std::map<ITEX_GPUDevice*, std::shared_ptr<BFCAllocator>>&
  GetAllocatorPool() {
    static std::once_flag init_alloc_flag;
    // TODO(itex): try impl without objects with static storage duration:
    // https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
    static auto* allocators =
        new std::map<ITEX_GPUDevice*, std::shared_ptr<BFCAllocator>>;

    std::call_once(init_alloc_flag, []() {
      int device_count = 0;
      ITEX_GPUGetDeviceCount(&device_count);
      ITEX_GPUDevice* device = nullptr;
      for (int i = 0; i < device_count; ++i) {
        ITEX_GPUGetDevice(&device, i);
        allocators->insert({device, std::make_shared<BFCAllocator>(device)});
      }
    });

    return *allocators;
  }
};  // class AllocatorPool

inline ITEX_GPUError_t ITEX_GPUGetAllocator(
    ITEX_GPUDevice* device, std::shared_ptr<BFCAllocator>* alloc) {
  return AllocatorPool::getAllocator(device, alloc);
}

}  // namespace itex
#endif  // ITEX_CORE_DEVICES_GPU_GPU_POOL_ALLOCATOR_H_
