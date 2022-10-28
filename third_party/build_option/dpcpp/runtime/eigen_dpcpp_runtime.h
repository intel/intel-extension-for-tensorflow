#ifndef THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_EIGEN_DPCPP_RUNTIME_H_
#define THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_EIGEN_DPCPP_RUNTIME_H_

#include "third_party/build_option/dpcpp/runtime/dpcpp_runtime.h"

// for usage in eigen
using dpruntimeStream_t = DPCPPStream*;
using DPCPPdevice_st = DPCPPDevice;

dpcppError_t dpruntimeGetDeviceCount(int* count);

dpcppError_t dpruntimeGetCurrentDevice(DeviceOrdinal* device);

dpcppError_t dpruntimeGetDevice(DeviceOrdinal* device, int device_ordinal);

dpcppError_t dpruntimeGetRealDPCPPDevice(DPCPPDevice* device,
                                         int device_ordinal);

dpcppError_t dpruntimeSetDevice(int device_ordinal);

const char* dpruntimeGetErrorName(dpcppError_t error);

dpcppError_t dpruntimeCreateStream(DPCPPStream** stream);

dpcppError_t dpruntimeDestroyStream(DPCPPStream* stream);

dpcppError_t dpruntimeStreamWaitEvent(DPCPPStream* stream, DPCPPEvent* event);

dpcppError_t dpruntimeCtxSynchronize();

dpcppError_t dpruntimeStreamSynchronize(DPCPPStream* stream);

dpcppError_t dpruntimeMemcpyDtoH(void* dstHost, const void* srcDevice,
                                 size_t ByteCount);

dpcppError_t dpruntimeMemcpyHtoD(void* dstDevice, const void* srcHost,
                                 size_t ByteCount);

dpcppError_t dpruntimeMemcpyDtoD(void* dstDevice, const void* srcDevice,
                                 size_t ByteCount);

dpcppError_t dpruntimeMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                      size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpruntimeMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                      size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpruntimeMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                      size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpruntimeMemsetD8(void* dstDevice, unsigned char uc, size_t N);

dpcppError_t dpruntimeMemsetD8Async(void* dstDevice, unsigned char uc, size_t N,
                                    DPCPPStream* stream);

dpcppError_t dpruntimeMemsetD32(void* dstDevice, unsigned int ui, size_t N);

dpcppError_t dpruntimeMemsetD32Async(void* dstDevice, unsigned int ui, size_t N,
                                     DPCPPStream* stream);

void* dpruntimeMalloc(size_t ByteCount);

void dpruntimeFree(void* ptr);

#endif  // THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_EIGEN_DPCPP_RUNTIME_H_
