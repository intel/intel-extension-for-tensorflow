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

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "tensorflow/c/c_api_experimental.h"

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

#ifdef USING_NEXTPLUGGABLE_DEVICE

typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_Client PJRT_Client;

void* ITEXOpaqueDataPointerFromPjRtBuffer(PJRT_Buffer*);

PJRT_Buffer* ITEXCreatePjRtBuffer(int device_id, std::string data_type,
                                  std::vector<int64_t>* dimentions, size_t size,
                                  PJRT_Client* pjrt_c_client);

PJRT_Buffer* ITEXCreateSEPjRtBuffer(int device_id, std::string datatype,
                                    std::vector<int64_t> dimentions,
                                    std::vector<int64_t> layout, PJRT_Client*);

void* ITEXGetStreamFromPjRtDevice(int device_id, PJRT_Client*);

PJRT_Buffer* ITEXSameDevicePjRtBufferCopy(PJRT_Buffer* src_buffer,
                                          PJRT_Client* c_client,
                                          bool xla_enabled);

void ITEXXlaShapeToDeviceShapeRepresentation(void* serialized_xla_shape,
                                             void* serialized_device_shape);

void* ITEXBFCAllocateOnSyclDevice(const sycl::device& device,
                                  PJRT_Client* pjrt_c_client, size_t n);
void ITEXBFCDeallocateOnSyclDevice(const sycl::device& device,
                                   PJRT_Client* pjrt_c_client, void* addr);

class ITEXNpdConfig {
 public:
  static ITEXNpdConfig& getNpdConfig() {
    static ITEXNpdConfig npdConfig;
    return npdConfig;
  }
  bool isXlaAutoJitEnabled() const { return isXlaAutoJitEnabled_; }
  bool ifUsingNextPluggableDevice() const {
    return isNextPluggableDeviceEnabled_;
  }
  bool IfEnableNextPluggableDevice() {
    if (isXlaAutoJitEnabled() || ifUsingNextPluggableDevice()) {
      return true;
    }
    return false;
  }

 private:
  ITEXNpdConfig() {
    const char* npdEnv = std::getenv("ITEX_ENABLE_NEXTPLUGGABLE_DEVICE");
    if (npdEnv != nullptr) {
      std::string env_value = absl::AsciiStrToLower(npdEnv);
      isNextPluggableDeviceEnabled_ =
          (env_value == "1" || env_value == "true") ? true : false;
    }
    if ((isXlaAutoJitEnabled_ = static_cast<bool>(TF_GetXlaAutoJitEnabled()))) {
      setenv("ITEX_REMAPPER", "0", 0);
      setenv("ITEX_LAYOUT_OPT", "0", 0);
      setenv("ITEX_ENABLE_MULTIPLE_STREAM", "1", 0);
    }
  }
  ITEXNpdConfig(ITEXNpdConfig const&) = delete;
  void operator=(ITEXNpdConfig const&) = delete;

  bool isNextPluggableDeviceEnabled_ = false;
  bool isXlaAutoJitEnabled_ = false;
};

#endif  // USING_NEXTPLUGGABLE_DEVICE

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
