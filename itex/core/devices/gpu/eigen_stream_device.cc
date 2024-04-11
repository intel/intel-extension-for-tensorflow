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

#include "itex/core/devices/gpu/eigen_stream_device.h"

#include <iostream>
#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#ifdef USING_NEXTPLUGGABLE_DEVICE
namespace {
constexpr uintptr_t kTag = 0x1ULL;
}
#endif

struct Deleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

#ifdef USING_NEXTPLUGGABLE_DEVICE
static bool pointer_is_pjrt_tensor(TF_Tensor* tf_tensor) {
  uintptr_t value = reinterpret_cast<uintptr_t>(TF_TensorData(tf_tensor));
  if (value & kTag) {
    return true;
  } else {
    return false;
  }
}
#endif

void* PluginStreamDevice::allocate(size_t num_bytes) const {
  std::unique_ptr<TF_Status, Deleter> status(TF_NewStatus());
  TF_AllocatorAttributes attr;
  attr.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
  attr.on_host = 0;
  TF_Tensor* tmp_tensor = TF_AllocateTemp(
      context_, TF_DataType::TF_UINT8, reinterpret_cast<int64_t*>(&num_bytes),
      1 /*vector*/, &attr, status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    ITEX_LOG(ERROR) << "Error when allocating temporary buffer for eigen!";
    return nullptr;
  }

#ifdef USING_NEXTPLUGGABLE_DEVICE
  if (pointer_is_pjrt_tensor(tmp_tensor)) {
    int device_id = TF_GetDeviceId(context_);
    PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", status.get());

    std::vector<int64_t> dimensions(1);
    dimensions[0] = num_bytes;
    if (npdConfig_.isXlaAutoJitEnabled()) {
      std::vector<int64_t> layout(1);
      std::iota(layout.rbegin(), layout.rend(), 0);
      TF_CreatePjRtBuffer(tmp_tensor,
                          ITEXCreateSEPjRtBuffer(device_id, "uint8", dimensions,
                                                 layout, pjrt_c_client),
                          "XPU", status.get());
    } else {
      TF_CreatePjRtBuffer(tmp_tensor,
                          ITEXCreatePjRtBuffer(device_id, "uint8", &dimensions,
                                               num_bytes, pjrt_c_client),
                          "XPU", status.get());
    }
  }
#endif

  if (tmp_tensors_ != nullptr) {
    tmp_tensors_->push_back(tmp_tensor);
  }

#ifndef USING_NEXTPLUGGABLE_DEVICE
  void* ret = TF_TensorData(tmp_tensor);
#else
  void* ret;
  if (!npdConfig_.IfEnableNextPluggableDevice()) {
    ret = TF_TensorData(tmp_tensor);
  } else {
    void* data_ptr = TF_TensorData(tmp_tensor);
    uintptr_t value = reinterpret_cast<uintptr_t>(data_ptr);
    if (value & kTag) {
      TF_Status* tf_status = TF_NewStatus();
      PJRT_Buffer* pjrt_c_buffer = TF_GetPjRtCBuffer(tmp_tensor, tf_status);
      TF_DeleteStatus(tf_status);
      ret = ITEXOpaqueDataPointerFromPjRtBuffer(pjrt_c_buffer);
    } else {
      ret = data_ptr;
    }
  }
#endif

  if (ret == nullptr) {
    ITEX_LOG(ERROR)
        << "EigenAllocator for GPU ran out of memory when allocating "
        << num_bytes << " bytes. See error logs for more detailed information.";
  }
  return ret;
}
