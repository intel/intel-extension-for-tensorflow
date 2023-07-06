/* Copyright (c) 2023 Intel Corporation

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

#ifndef THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_ITEX_GPU_RUNTIME_H_
#define THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_ITEX_GPU_RUNTIME_H_

#include <string>
#include <vector>

#include "absl/strings/ascii.h"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

enum ITEX_GPUError_t {
  ITEX_GPU_SUCCESS,
  ITEX_GPU_ERROR_NO_DEVICE,
  ITEX_GPU_ERROR_NOT_READY,
  ITEX_GPU_ERROR_INVALID_DEVICE,
  ITEX_GPU_ERROR_INVALID_POINTER,
  ITEX_GPU_ERROR_INVALID_STREAM,
  ITEX_GPU_ERROR_DESTROY_DEFAULT_STREAM,
};

typedef int DeviceOrdinal;

using ITEX_GPUDevice = sycl::device;
using ITEX_GPUStream = sycl::queue;
using ITEX_GPUEvent = sycl::event;

inline bool IsMultipleStreamEnabled() {
  bool is_multiple_stream_enabled = false;
  const char* env = std::getenv("ITEX_ENABLE_MULTIPLE_STREAM");
  if (env == nullptr) {
    return is_multiple_stream_enabled;
  }

  std::string str_value = absl::AsciiStrToLower(env);
  if (str_value == "0" || str_value == "false") {
    is_multiple_stream_enabled = false;
  } else if (str_value == "1" || str_value == "true") {
    is_multiple_stream_enabled = true;
  }

  return is_multiple_stream_enabled;
}

const char* ITEX_GPUGetErrorName(ITEX_GPUError_t error);

ITEX_GPUError_t ITEX_GPUGetDeviceCount(int* count);

ITEX_GPUError_t ITEX_GPUGetDevice(ITEX_GPUDevice** device, int device_ordinal);

ITEX_GPUError_t ITEX_GPUGetDeviceOrdinal(const ITEX_GPUDevice& device,
                                         DeviceOrdinal* device_ordinal);

ITEX_GPUError_t ITEX_GPUGetCurrentDeviceOrdinal(DeviceOrdinal* ordinal);

ITEX_GPUError_t ITEX_GPUSetCurrentDeviceOrdinal(DeviceOrdinal ordinal);

ITEX_GPUError_t ITEX_GPUCreateStream(ITEX_GPUDevice* device_handle,
                                     ITEX_GPUStream** stream);

ITEX_GPUError_t ITEX_GPUGetDefaultStream(ITEX_GPUDevice* device_handle,
                                         ITEX_GPUStream** stream);

ITEX_GPUError_t ITEX_GPUDestroyStream(ITEX_GPUDevice* device_handle,
                                      ITEX_GPUStream* stream);

ITEX_GPUError_t ITEX_GPUGetStreamPool(ITEX_GPUDevice* device_handle,
                                      std::vector<ITEX_GPUStream*>* streams);

ITEX_GPUError_t ITEX_GPUCreateEvent(ITEX_GPUDevice* device_handle,
                                    ITEX_GPUEvent* event);
ITEX_GPUError_t ITEX_GPUDestroyEvent(ITEX_GPUDevice* device_handle,
                                     ITEX_GPUEvent event);

ITEX_GPUError_t ITEX_GPUStreamWaitEvent(ITEX_GPUStream* stream,
                                        ITEX_GPUEvent event);

ITEX_GPUError_t ITEX_GPUStreamWaitStream(ITEX_GPUStream* dependent,
                                         ITEX_GPUStream* other);

ITEX_GPUError_t ITEX_GPUCtxSynchronize(ITEX_GPUDevice* device_handle);

ITEX_GPUError_t ITEX_GPUStreamSynchronize(ITEX_GPUStream* stream);

ITEX_GPUError_t ITEX_GPUMemcpyDtoH(void* dstHost, const void* srcDevice,
                                   size_t ByteCount, ITEX_GPUDevice* device);

ITEX_GPUError_t ITEX_GPUMemcpyHtoD(void* dstDevice, const void* srcHost,
                                   size_t ByteCount, ITEX_GPUDevice* device);

ITEX_GPUError_t ITEX_GPUMemcpyDtoD(void* dstDevice, const void* srcDevice,
                                   size_t ByteCount, ITEX_GPUDevice* device);

ITEX_GPUError_t ITEX_GPUMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                        size_t ByteCount,
                                        ITEX_GPUStream* stream);

ITEX_GPUError_t ITEX_GPUMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                        size_t ByteCount,
                                        ITEX_GPUStream* stream);

ITEX_GPUError_t ITEX_GPUMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                        size_t ByteCount,
                                        ITEX_GPUStream* stream);

ITEX_GPUError_t ITEX_GPUMemsetD8(void* dstDevice, unsigned char uc, size_t N,
                                 ITEX_GPUDevice* device);

ITEX_GPUError_t ITEX_GPUMemsetD8Async(void* dstDevice, unsigned char uc,
                                      size_t N, ITEX_GPUStream* stream);

ITEX_GPUError_t ITEX_GPUMemsetD32(void* dstDevice, unsigned int ui, size_t N,
                                  ITEX_GPUDevice* device);

ITEX_GPUError_t ITEX_GPUMemsetD32Async(void* dstDevice, unsigned int ui,
                                       size_t N, ITEX_GPUStream* stream);

void* ITEX_GPUMalloc(ITEX_GPUDevice* device, size_t ByteCount);

void* ITEX_GPUMallocHost(size_t ByteCount);

void ITEX_GPUFree(ITEX_GPUDevice* device, void* ptr);
#endif  // THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_ITEX_GPU_RUNTIME_H_
