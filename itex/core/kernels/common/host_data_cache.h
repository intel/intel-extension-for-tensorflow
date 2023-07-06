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

#ifndef ITEX_CORE_KERNELS_COMMON_HOST_DATA_CACHE_H_
#define ITEX_CORE_KERNELS_COMMON_HOST_DATA_CACHE_H_

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

typedef Eigen::GpuDevice GPUDevice;
namespace itex {

#ifdef ITEX_ONEDNN_3_0
template <typename Device, typename T>
class HostDataCache {
 public:
  HostDataCache() : gpu_capacity_(0) {}
  T* GetCachedPtr(OpKernelContext* ctx, const T* host_data, size_t count) {
    T* out_ptr = nullptr;
    if (std::is_same<Device, GPUDevice>::value) {
#ifndef INTEL_CPU_ONLY
      GetCachedPtrGPU(ctx, host_data, count, &out_ptr);
#endif
    } else {
      GetCachedPtrCPU(ctx, host_data, count, &out_ptr);
    }
    return out_ptr;
  }

 private:
  bool IsSame(const T* host_data, size_t count) {
    if (last_host_data_.size() != count) return false;
    for (size_t i = 0; i < count; ++i) {
      if (last_host_data_[i] != host_data[i]) return false;
    }
    return true;
  }
  void GetCachedPtrCPU(OpKernelContext* ctx, const T* host_data, size_t count,
                       T** out_ptr) {
    if (!IsSame(host_data, count)) {
      last_host_data_ = std::move(std::vector<T>(host_data, host_data + count));
    }
    *out_ptr = last_host_data_.data();
  }
#ifndef INTEL_CPU_ONLY
  void GetCachedPtrGPU(OpKernelContext* ctx, const T* host_data, size_t count,
                       T** out_ptr) {
    bool need_copy = false;
    if (gpu_capacity_ < count) {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                  TensorShape{static_cast<int64_t>(count)},
                                  &gpu_tensor_));
      gpu_capacity_ = count;
      need_copy = true;
    }
    if (!IsSame(host_data, count)) {
      last_host_data_ = std::move(std::vector<T>(host_data, host_data + count));
      need_copy = true;
    }
    T* gpu_data_ptr = gpu_tensor_.flat<T>().data();

    if (need_copy) {
      ctx->GetDeviceStream()
          ->memcpy(gpu_data_ptr, last_host_data_.data(), count * sizeof(T))
          .wait();
    }
    *out_ptr = gpu_data_ptr;
  }
#endif
  std::vector<T> last_host_data_;
  Tensor gpu_tensor_;
  size_t gpu_capacity_;
};
#endif

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_HOST_DATA_CACHE_H_
