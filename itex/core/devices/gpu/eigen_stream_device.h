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

#ifndef ITEX_CORE_DEVICES_GPU_EIGEN_STREAM_DEVICE_H_
#define ITEX_CORE_DEVICES_GPU_EIGEN_STREAM_DEVICE_H_

#include <iostream>

#include "itex/core/devices/gpu/gpu_info.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/types.h"
#include "tensorflow/c/kernels.h"
#include "third_party/build_option/dpcpp/runtime/eigen_itex_gpu_runtime.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#ifdef USING_NEXTPLUGGABLE_DEVICE
#include "tensorflow/c/experimental/next_pluggable_device/c_api.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"
#endif

using itex::gtl::InlinedVector;

class PluginStreamDevice : public ::Eigen::StreamInterface {
 public:
#ifndef USING_NEXTPLUGGABLE_DEVICE
  PluginStreamDevice(TF_OpKernelContext* ctx, gpuStream_t* strm,
                     InlinedVector<TF_Tensor*, 4>* tmp_tensors)
      : stream_(strm), context_(ctx), tmp_tensors_(tmp_tensors) {
    itex::DeviceInfo* device_info_ = itex::GetDeviceInfo(*strm);
    device_prop_ = device_info_->getGPUDeviceProperties();
  }
#else
  PluginStreamDevice(TF_OpKernelContext* ctx,
                     InlinedVector<TF_Tensor*, 4>* tmp_tensors)
      : context_(ctx), tmp_tensors_(tmp_tensors) {
    TF_Status* tf_status = TF_NewStatus();
    int device_id = TF_GetDeviceId(ctx);
    PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", tf_status);
    stream_ = static_cast<gpuStream_t>(
        ITEXGetStreamFromPjRtDevice(device_id, pjrt_c_client));
    TF_DeleteStatus(tf_status);
    itex::DeviceInfo* device_info_ = itex::GetDeviceInfo(stream_);
    device_prop_ = device_info_->getGPUDeviceProperties();
  }
#endif
  ~PluginStreamDevice() override {}
#ifndef USING_NEXTPLUGGABLE_DEVICE
  const gpuStream_t& stream() const override { return *stream_; }
#else
  const gpuStream_t& stream() const override { return stream_; }
#endif
  void* scratchpad() const override { return nullptr; }
  unsigned int* semaphore() const override { return nullptr; }
  const gpuDeviceProp_t& deviceProperties() const override {
    return device_prop_;
  }
  void* allocate(size_t num_bytes) const override;
  void deallocate(void* buffer) const override {}

 private:
#ifndef USING_NEXTPLUGGABLE_DEVICE
  const gpuStream_t* stream_;  // Not owned.
#else
  gpuStream_t stream_;  // Not owned.
#endif
  gpuDeviceProp_t device_prop_;
  TF_OpKernelContext* context_;
  InlinedVector<TF_Tensor*, 4>* tmp_tensors_;  // Not owned
};

#endif  // ITEX_CORE_DEVICES_GPU_EIGEN_STREAM_DEVICE_H_
