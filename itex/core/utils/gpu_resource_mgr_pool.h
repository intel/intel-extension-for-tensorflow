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

#ifndef ITEX_CORE_UTILS_GPU_RESOURCE_MGR_POOL_H_
#define ITEX_CORE_UTILS_GPU_RESOURCE_MGR_POOL_H_

#include <map>
#include <vector>

#include "itex/core/utils/resource_mgr.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

namespace itex {

class ResourceMgr;
class ResourceMgrPool {
 public:
  static ITEX_GPUError_t GetResourceMgr(ITEX_GPUStream* stream,
                                        ResourceMgr** resource_mgr) {
    auto resource_mgr_pool = ResourceMgrPool::GetResourceMgrPool();
    for (auto& [key, value] : resource_mgr_pool) {
      if (key == stream) {
        *resource_mgr = value;
        return ITEX_GPU_SUCCESS;
      }
    }
    return ITEX_GPU_ERROR_INVALID_STREAM;
  }

 private:
  static std::map<ITEX_GPUStream*, ResourceMgr*>& GetResourceMgrPool() {
    static std::once_flag init_flag;
    static std::map<ITEX_GPUStream*, ResourceMgr*> resource_mgr_pool;
    std::call_once(init_flag, []() {
      int device_count = 0;
      ITEX_GPUGetDeviceCount(&device_count);
      ITEX_GPUDevice* device = nullptr;
      for (int i = 0; i < device_count; ++i) {
        ITEX_GPUGetDevice(&device, i);
        std::vector<ITEX_GPUStream*> streams;
        ITEX_GPUGetStreamPool(device, &streams);
        for (int j = 0; j < streams.size(); ++j)
          resource_mgr_pool.insert({streams[j], new ResourceMgr()});
      }
    });

    return resource_mgr_pool;
  }
};  // class ResourceMgrPool

// clang-format off
inline ITEX_GPUError_t GetResourceMgr(ITEX_GPUStream* stream,
                                      ResourceMgr** resource_mgr) {
  return ResourceMgrPool::GetResourceMgr(stream, resource_mgr);
}
// clang-format on

}  // namespace itex
#endif  // ITEX_CORE_UTILS_GPU_RESOURCE_MGR_POOL_H_
