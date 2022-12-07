/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_GPU_DEVICE_ARRAY_H_
#define ITEX_CORE_KERNELS_GPU_GPU_DEVICE_ARRAY_H_

#include "itex/core/utils/allocator.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {

// To decode on the device side, use GetGpuDeviceArrayOnDevice.
// To encode on the host side, use GpuDeviceArrayOnHost.
template <typename ValueType, int MaxInlineValues = 8>
struct GpuDeviceArrayStruct {
  int32 size;
  // used if size <= MaxInlineValues;
  ValueType inline_values[MaxInlineValues];
  ValueType* out_of_line_values = nullptr;  // used if size > MaxInlineValues;
};

template <typename ValueType, int MaxInlineValues = 8>
inline const ValueType* GetGpuDeviceArrayOnDevice(
    const GpuDeviceArrayStruct<ValueType, MaxInlineValues>* data) {
  if (data->size <= MaxInlineValues) {
    return data->inline_values;
  } else {
    return data->out_of_line_values;
  }
}

#define GetPtrByMask                                          \
  if ((Mask >> (ArgsSize / 2)) & 1) {                         \
    *ptr = const_cast<PointerType>(data->inline_values);      \
  } else {                                                    \
    *ptr = const_cast<PointerType>(data->out_of_line_values); \
  }

template <int Mask, typename ValueType, typename PointerType, typename... Ts,
          int ArgsSize = sizeof...(Ts), int MaxInlineValues = 8>
inline typename std::enable_if<ArgsSize != 0, void>::type
GetGpuDeviceArrayOnDeviceWithMask(
    const GpuDeviceArrayStruct<ValueType, MaxInlineValues>* data,
    PointerType* ptr, Ts... Args) {
  static_assert(ArgsSize % 2 == 0, "The size of Args must be even");
  GetPtrByMask;
  GetGpuDeviceArrayOnDeviceWithMask<Mask>(Args...);
}

template <int Mask, typename ValueType, typename PointerType, typename... Ts,
          int ArgsSize = sizeof...(Ts), int MaxInlineValues = 8>
inline typename std::enable_if<ArgsSize == 0, void>::type
GetGpuDeviceArrayOnDeviceWithMask(
    const GpuDeviceArrayStruct<ValueType, MaxInlineValues>* data,
    PointerType* ptr, Ts... Args) {
  GetPtrByMask;
}

#undef GetPtrByMask

template <int DispatchSize, int PreMask, template <int Mask> class Functor>
struct DispatchToGpuDeviceArrayInlined {
  template <typename ValueType, typename... Ts, int MaxInlineValues = 8>
  static void run(
      const GpuDeviceArrayStruct<ValueType, MaxInlineValues>& gpu_device_array,
      Ts... Args) {
    if (gpu_device_array.size <= MaxInlineValues) {
      DispatchToGpuDeviceArrayInlined<DispatchSize - 1, (PreMask << 1) + 1,
                                      Functor>::run(Args..., gpu_device_array);
    } else {
      DispatchToGpuDeviceArrayInlined<DispatchSize - 1, (PreMask << 1),
                                      Functor>::run(Args..., gpu_device_array);
    }
  }
};

template <int PreMask, template <int Mask> class Functor>
struct DispatchToGpuDeviceArrayInlined<1, PreMask, Functor> {
  template <typename ValueType, typename... Ts, int MaxInlineValues = 8>
  static void run(
      const GpuDeviceArrayStruct<ValueType, MaxInlineValues>& gpu_device_array,
      Ts... Args) {
    if (gpu_device_array.size <= MaxInlineValues) {
      Functor<(PreMask << 1) + 1>()(Args..., gpu_device_array);
    } else {
      Functor<(PreMask << 1)>()(Args..., gpu_device_array);
    }
  }
};

// Create an array of value on the host, to be sent to kernel using
// GpuDeviceArrayStruct.
//
// Usage:
//   int size = ...;
//   GpuDeviceArrayOnHost ptrs(context, size);
//   OP_REQUIRES_OK(ptrs.Init());
//   for (int i = 0; i < size; ++i) {
//     ptrs.Set(i, ...);
//   }
//   OP_REQUIRES_OK(ptrs.Finalize());
//   launchKernel(..., ptrs.data, ...);
//
// ValueType must be memcopyable.
template <typename ValueType, int MaxInlineValues = 8>
class GpuDeviceArrayOnHost {
 public:
  GpuDeviceArrayOnHost(OpKernelContext* context, int32 size)
      : context_(context),
        total_bytes_(static_cast<int64>(size) * sizeof(ValueType)) {
    data_.size = size;
  }

  Status Init() {
    if (inlined()) {
      values_ = data_.inline_values;
      return Status::OK();
    }

    // Out-of-line: allocate data that will be memcopied.
    AllocatorAttributes attr;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(
        context_->allocate_temp(DT_INT8, TensorShape{total_bytes_},
                                &out_of_line_values_on_host_, attr));
    values_ = reinterpret_cast<ValueType*>(
        out_of_line_values_on_host_.flat<int8>().data());
    return Status::OK();
  }

  void Set(int index, ValueType val) {
    ITEX_DCHECK(values_);  // ensure Init was called.
    ITEX_DCHECK_LT(index, data_.size);
    *(values_ + index) = val;
  }

  Status Finalize() {
    if (inlined()) {
      return Status::OK();
    }

    // Out-of-line - copy pointers to device.
    TF_RETURN_IF_ERROR(context_->allocate_temp(
        DT_INT8, TensorShape{total_bytes_}, &out_of_line_values_on_gpu_));
    ITEX_GPUMemcpyHtoDAsync(out_of_line_values_on_gpu_.flat<int8>().data(),
                            out_of_line_values_on_host_.flat<int8>().data(),
                            total_bytes_, context_->GetDeviceStream());
    data_.out_of_line_values = reinterpret_cast<ValueType*>(
        out_of_line_values_on_gpu_.flat<int8>().data());
    return Status::OK();
  }

  const GpuDeviceArrayStruct<ValueType, MaxInlineValues>& data() const {
    // Ensure Finalize is called.
    ITEX_DCHECK(inlined() || out_of_line_values_on_gpu_.IsInitialized());
    return data_;
  }

 private:
  bool inlined() const { return data_.size <= MaxInlineValues; }
  OpKernelContext* const context_;
  const int64 total_bytes_;  // total size of all pointers.
  ValueType* values_ = nullptr;
  GpuDeviceArrayStruct<ValueType, MaxInlineValues> data_;

  Tensor out_of_line_values_on_host_;
  Tensor out_of_line_values_on_gpu_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuDeviceArrayOnHost);
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_GPU_DEVICE_ARRAY_H_
