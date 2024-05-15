#ifndef THIRD_PARTY_BUILD_OPTION_ITEX_GPU_RUNTIME_EIGEN_ITEX_GPU_RUNTIME_H_
#define THIRD_PARTY_BUILD_OPTION_ITEX_GPU_RUNTIME_EIGEN_ITEX_GPU_RUNTIME_H_

#include "third_party/build_option/sycl/runtime/itex_gpu_runtime.h"

// for usage in eigen
using dpruntimeStream_t = ITEX_GPUStream*;
using ITEX_GPUdevice_st = ITEX_GPUDevice;

ITEX_GPUError_t dpruntimeGetDeviceCount(int* count);

ITEX_GPUError_t dpruntimeGetCurrentDevice(DeviceOrdinal* device);

ITEX_GPUError_t dpruntimeGetDevice(DeviceOrdinal* device, int device_ordinal);

ITEX_GPUError_t dpruntimeGetRealDPCPPDevice(ITEX_GPUDevice* device,
                                            int device_ordinal);

ITEX_GPUError_t dpruntimeSetDevice(int device_ordinal);

const char* dpruntimeGetErrorName(ITEX_GPUError_t error);

ITEX_GPUError_t dpruntimeCreateStream(ITEX_GPUStream** stream);

ITEX_GPUError_t dpruntimeDestroyStream(ITEX_GPUStream* stream);

ITEX_GPUError_t dpruntimeStreamWaitEvent(ITEX_GPUStream* stream,
                                         ITEX_GPUEvent* event);

ITEX_GPUError_t dpruntimeCtxSynchronize();

ITEX_GPUError_t dpruntimeStreamSynchronize(ITEX_GPUStream* stream);

ITEX_GPUError_t dpruntimeMemcpyDtoH(void* dstHost, const void* srcDevice,
                                    size_t ByteCount);

ITEX_GPUError_t dpruntimeMemcpyHtoD(void* dstDevice, const void* srcHost,
                                    size_t ByteCount);

ITEX_GPUError_t dpruntimeMemcpyDtoD(void* dstDevice, const void* srcDevice,
                                    size_t ByteCount);

ITEX_GPUError_t dpruntimeMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                         size_t ByteCount,
                                         ITEX_GPUStream* stream);

ITEX_GPUError_t dpruntimeMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                         size_t ByteCount,
                                         ITEX_GPUStream* stream);

ITEX_GPUError_t dpruntimeMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                         size_t ByteCount,
                                         ITEX_GPUStream* stream);

ITEX_GPUError_t dpruntimeMemsetD8(void* dstDevice, unsigned char uc, size_t N);

ITEX_GPUError_t dpruntimeMemsetD8Async(void* dstDevice, unsigned char uc,
                                       size_t N, ITEX_GPUStream* stream);

ITEX_GPUError_t dpruntimeMemsetD32(void* dstDevice, unsigned int ui, size_t N);

ITEX_GPUError_t dpruntimeMemsetD32Async(void* dstDevice, unsigned int ui,
                                        size_t N, ITEX_GPUStream* stream);

void* dpruntimeMalloc(size_t ByteCount);

void dpruntimeFree(void* ptr);

#endif  // THIRD_PARTY_BUILD_OPTION_ITEX_GPU_RUNTIME_EIGEN_ITEX_GPU_RUNTIME_H_
