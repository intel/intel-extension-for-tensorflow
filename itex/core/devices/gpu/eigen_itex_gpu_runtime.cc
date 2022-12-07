#include "third_party/build_option/dpcpp/runtime/eigen_itex_gpu_runtime.h"

#include <cassert>
#include <iostream>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "itex/core/utils/logging.h"

#define REQUIRE_SUCCESS(func)                                               \
  do {                                                                      \
    ITEX_GPUError_t error = func;                                           \
    if (error != ITEX_GPU_SUCCESS) {                                        \
      ITEX_LOG(ERROR) << "Error call the function " << #func << " because " \
                      << dpruntimeGetErrorName(error);                      \
      return error;                                                         \
    }                                                                       \
  } while (0)

ITEX_GPUError_t dpruntimeGetDeviceCount(int* count) {
  return ITEX_GPUGetDeviceCount(count);
}

ITEX_GPUError_t dpruntimeGetCurrentDevice(DeviceOrdinal* device) {
  return ITEX_GPUGetCurrentDeviceOrdinal(device);
}

ITEX_GPUError_t dpruntimeGetDevice(DeviceOrdinal* device, int device_ordinal) {
  *device = device_ordinal;
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeGetRealITEX_GPUDevice(ITEX_GPUDevice* device,
                                               int device_ordinal) {
  ITEX_GPUDevice* real_device;
  REQUIRE_SUCCESS(ITEX_GPUGetDevice(&real_device, device_ordinal));
  *device = *real_device;
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeSetDevice(int device_ordinal) {
  return ITEX_GPUSetCurrentDeviceOrdinal(device_ordinal);
}

static ITEX_GPUError_t getCurrentDevice(ITEX_GPUDevice** device) {
  DeviceOrdinal ordinal;
  REQUIRE_SUCCESS(ITEX_GPUGetCurrentDeviceOrdinal(&ordinal));
  REQUIRE_SUCCESS(ITEX_GPUGetDevice(device, ordinal));
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeCreateStream(ITEX_GPUStream** stream) {
  ITEX_GPUDevice* device;
  REQUIRE_SUCCESS(getCurrentDevice(&device));
  return ITEX_GPUCreateStream(device, stream);
}

ITEX_GPUError_t dpruntimeDestroyStream(ITEX_GPUStream* stream) {
  ITEX_GPUDevice* device;
  REQUIRE_SUCCESS(getCurrentDevice(&device));
  return ITEX_GPUDestroyStream(device, stream);
}

ITEX_GPUError_t dpruntimeStreamWaitEvent(ITEX_GPUStream* stream,
                                         ITEX_GPUEvent* event) {
  return ITEX_GPUStreamWaitEvent(stream, event);
}

ITEX_GPUError_t dpruntimeCtxSynchronize() {
  ITEX_GPUDevice* device;
  REQUIRE_SUCCESS(getCurrentDevice(&device));
  return ITEX_GPUCtxSynchronize(device);
}

ITEX_GPUError_t dpruntimeStreamSynchronize(ITEX_GPUStream* stream) {
  return ITEX_GPUStreamSynchronize(stream);
}

ITEX_GPUError_t dpruntimeMemcpyDtoH(void* dstHost, const void* srcDevice,
                                    size_t ByteCount) {
  ITEX_GPUDevice* device;
  REQUIRE_SUCCESS(getCurrentDevice(&device));
  REQUIRE_SUCCESS(ITEX_GPUMemcpyDtoH(dstHost, srcDevice, ByteCount, device));
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemcpyHtoD(void* dstDevice, const void* srcHost,
                                    size_t ByteCount) {
  ITEX_GPUDevice* device;
  REQUIRE_SUCCESS(getCurrentDevice(&device));
  REQUIRE_SUCCESS(ITEX_GPUMemcpyHtoD(dstDevice, srcHost, ByteCount, device));

  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemcpyDtoD(void* dstDevice, const void* srcDevice,
                                    size_t ByteCount) {
  ITEX_GPUDevice* device;
  REQUIRE_SUCCESS(getCurrentDevice(&device));
  REQUIRE_SUCCESS(ITEX_GPUMemcpyDtoD(dstDevice, srcDevice, ByteCount, device));

  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                         size_t ByteCount,
                                         ITEX_GPUStream* stream) {
  REQUIRE_SUCCESS(
      ITEX_GPUMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, stream));

  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                         size_t ByteCount,
                                         ITEX_GPUStream* stream) {
  REQUIRE_SUCCESS(
      ITEX_GPUMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, stream));

  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                         size_t ByteCount,
                                         ITEX_GPUStream* stream) {
  REQUIRE_SUCCESS(
      ITEX_GPUMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, stream));

  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemsetD8(void* dstDevice, unsigned char uc, size_t N) {
  ITEX_GPUDevice* device;
  REQUIRE_SUCCESS(getCurrentDevice(&device));
  REQUIRE_SUCCESS(ITEX_GPUMemsetD8(dstDevice, uc, N, device));
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemsetD8Async(void* dstDevice, unsigned char uc,
                                       size_t N, ITEX_GPUStream* stream) {
  REQUIRE_SUCCESS(ITEX_GPUMemsetD8Async(dstDevice, uc, N, stream));
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemsetD32(void* dstDevice, unsigned int ui, size_t N) {
  ITEX_GPUDevice* device;
  REQUIRE_SUCCESS(getCurrentDevice(&device));
  REQUIRE_SUCCESS(ITEX_GPUMemsetD32(dstDevice, ui, N, device));

  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t dpruntimeMemsetD32Async(void* dstDevice, unsigned int ui,
                                        size_t N, ITEX_GPUStream* stream) {
  REQUIRE_SUCCESS(ITEX_GPUMemsetD32Async(dstDevice, ui, N, stream));
  return ITEX_GPU_SUCCESS;
}

void* dpruntimeMalloc(size_t ByteCount) {
  ITEX_GPUDevice* device;
  if (getCurrentDevice(&device) != ITEX_GPU_SUCCESS) {
    return nullptr;
  }

  return ITEX_GPUMalloc(device, ByteCount);
}

void dpruntimeFree(void* ptr) {
  ITEX_GPUDevice* device;
  if (getCurrentDevice(&device) != ITEX_GPU_SUCCESS) {
    return;
  }

  ITEX_GPUFree(device, ptr);
}

const char* dpruntimeGetErrorName(ITEX_GPUError_t error) {
  return ITEX_GPUGetErrorName(error);
}

#undef REQUIRE_SUCCESS
