#ifndef THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_DPCPP_RUNTIME_H_
#define THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_DPCPP_RUNTIME_H_

#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

enum dpcppError_t {
  DPCPP_SUCCESS,
  DPCPP_ERROR_NO_DEVICE,
  DPCPP_ERROR_NOT_READY,
  DPCPP_ERROR_INVALID_DEVICE,
  DPCPP_ERROR_INVALID_POINTER,
  DPCPP_ERROR_INVALID_STREAM,
  DPCPP_ERROR_DESTROY_DEFAULT_STREAM,
};

typedef int DeviceOrdinal;

using DPCPPDevice = sycl::device;
using DPCPPStream = sycl::queue;
using DPCPPEvent = sycl::event;

const char* dpcppGetErrorName(dpcppError_t error);

dpcppError_t dpcppGetDeviceCount(int* count);

dpcppError_t dpcppGetDevice(DPCPPDevice** device, int device_ordinal);

dpcppError_t dpcppGetDeviceOrdinal(const DPCPPDevice& device,
                                   DeviceOrdinal* device_ordinal);

dpcppError_t dpcppGetCurrentDeviceOrdinal(DeviceOrdinal* ordinal);

dpcppError_t dpcppSetCurrentDeviceOrdinal(DeviceOrdinal ordinal);

dpcppError_t dpcppCreateStream(DPCPPDevice* device_handle,
                               DPCPPStream** stream);

dpcppError_t dpcppGetDefaultStream(DPCPPDevice* device_handle,
                                   DPCPPStream** stream);

dpcppError_t dpcppDestroyStream(DPCPPDevice* device_handle,
                                DPCPPStream* stream);

dpcppError_t dpcppGetStreamPool(DPCPPDevice* device_handle,
                                std::vector<DPCPPStream*>* streams);

dpcppError_t dpcppCreateEvent(DPCPPDevice* device_handle, DPCPPEvent** event);
dpcppError_t dpcppDestroyEvent(DPCPPDevice* device_handle, DPCPPEvent* event);

dpcppError_t dpcppStreamWaitEvent(DPCPPStream* stream, DPCPPEvent* event);

dpcppError_t dpcppStreamWaitStream(DPCPPStream* dependent, DPCPPStream* other);

dpcppError_t dpcppCtxSynchronize(DPCPPDevice* device_handle);

dpcppError_t dpcppStreamSynchronize(DPCPPStream* stream);

dpcppError_t dpcppMemcpyDtoH(void* dstHost, const void* srcDevice,
                             size_t ByteCount, DPCPPDevice* device);

dpcppError_t dpcppMemcpyHtoD(void* dstDevice, const void* srcHost,
                             size_t ByteCount, DPCPPDevice* device);

dpcppError_t dpcppMemcpyDtoD(void* dstDevice, const void* srcDevice,
                             size_t ByteCount, DPCPPDevice* device);

dpcppError_t dpcppMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                  size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpcppMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                  size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpcppMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                  size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpcppMemsetD8(void* dstDevice, unsigned char uc, size_t N,
                           DPCPPDevice* device);

dpcppError_t dpcppMemsetD8Async(void* dstDevice, unsigned char uc, size_t N,
                                DPCPPStream* stream);

dpcppError_t dpcppMemsetD32(void* dstDevice, unsigned int ui, size_t N,
                            DPCPPDevice* device);

dpcppError_t dpcppMemsetD32Async(void* dstDevice, unsigned int ui, size_t N,
                                 DPCPPStream* stream);

void* dpcppMalloc(DPCPPDevice* device, size_t ByteCount);

void* dpcppMallocHost(size_t ByteCount);

void dpcppFree(DPCPPDevice* device, void* ptr);
#endif  // THIRD_PARTY_BUILD_OPTION_DPCPP_RUNTIME_DPCPP_RUNTIME_H_
