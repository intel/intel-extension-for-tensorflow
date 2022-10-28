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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

struct Deleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

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

  if (tmp_tensors_ != nullptr) {
    tmp_tensors_->push_back(tmp_tensor);
  }

  void* ret = TF_TensorData(tmp_tensor);

  if (ret == nullptr) {
    ITEX_LOG(ERROR)
        << "EigenAllocator for GPU ran out of memory when allocating "
        << num_bytes << " bytes. See error logs for more detailed information.";
  }
  return ret;
}
