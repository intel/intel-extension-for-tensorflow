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

#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "itex/core/devices/xpu_device_util.h"

namespace {

inline bool RunOnLevelZero() {
  char* sycl_device_filter = getenv("SYCL_DEVICE_FILTER");
  // Current default backend platform is Level-Zero
  if (sycl_device_filter == nullptr) return true;
  auto filter_device = std::string(sycl_device_filter);
  std::transform(filter_device.begin(), filter_device.end(),
                 filter_device.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return filter_device.find("level_zero") != std::string::npos;
}

bool hasDevice() {
  int count = 0;
  ITEX_GPUError_t error = ITEX_GPUGetDeviceCount(&count);
  if (error != ITEX_GPU_SUCCESS) {
    ITEX_LOG(ERROR) << "Error to get the device count because "
                    << ITEX_GPUGetErrorName(error);
    return false;
  }

  if (count == 0) {
    ITEX_LOG(ERROR) << "Error. The device count is 0, "
                    << ITEX_GPUGetErrorName(ITEX_GPU_ERROR_NO_DEVICE);
    return false;
  }

  return true;
}

bool isValidDevice(DeviceOrdinal ordinal) {
  int count = 0;
  ITEX_GPUGetDeviceCount(&count);

  if (ordinal > count) {
    return false;
  }

  return true;
}

class DevicePool {
 public:
  DevicePool() : current_ordinal_(0) {}

  static sycl::context& getDeviceContext() {
    static sycl::context context(DevicePool::GetDevicesPool());
    return context;
  }

  static ITEX_GPUError_t getDeviceCount(int* count) {
    *count = DevicePool::GetDevicesPool().size();
    return ITEX_GPU_SUCCESS;
  }

  static ITEX_GPUError_t getDevice(ITEX_GPUDevice** device,
                                   int device_ordinal) {
    // absl::ReaderMutexLock lock(&mu_);
    if (device_ordinal >= DevicePool::GetDevicesPool().size()) {
      return ITEX_GPU_ERROR_INVALID_DEVICE;
    } else {
      *device = &DevicePool::GetDevicesPool()[device_ordinal];
      return ITEX_GPU_SUCCESS;
    }
  }

  ITEX_GPUError_t getDeviceOrdinal(const ITEX_GPUDevice& device,
                                   DeviceOrdinal* device_ordinal) {
    const auto& devices = DevicePool::GetDevicesPool();
    auto it = std::find(devices.begin(), devices.end(), device);
    if (it != devices.end()) {
      *device_ordinal = it - devices.begin();
      return ITEX_GPU_SUCCESS;
    } else {
      return ITEX_GPU_ERROR_INVALID_DEVICE;
    }
  }

  ITEX_GPUError_t setCurrentDeviceOrdinal(DeviceOrdinal ordinal);

  ITEX_GPUError_t getCurrentDeviceOrdinal(DeviceOrdinal* ordinal);

  static DevicePool* GetInstance();

 private:
  static std::vector<ITEX_GPUDevice>& GetDevicesPool() {
    static std::once_flag init_device_flag;
    static std::vector<ITEX_GPUDevice> devices;

    std::call_once(init_device_flag, []() {
      std::vector<ITEX_GPUDevice> root_devices;
      // Get root device list from platform list.
      auto platform_list = sycl::platform::get_platforms();
      for (const auto& platform : platform_list) {
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        bool is_level_zero =
            platform_name.find("Level-Zero") != std::string::npos;
        // Add device in these two scenarios:
        // true == true means need Level-Zero and the backend platform is
        // Level-Zero.
        // false == false mean need OCL and the backend platform is OCL.
        if (is_level_zero == RunOnLevelZero()) {
          ITEX_LOG(INFO) << "Selected platform: " << platform_name;
          auto device_list = platform.get_devices();
          for (const auto& device : device_list) {
            if (device.is_gpu()) {
              root_devices.push_back(device);
            }
          }
        }
      }

      if (TileAsDevice()) {
        // If ITEX_TILE_AS_DEVICE is true.
        // Create sub devices from root devices:
        //   If succ, add sub devices into devices list
        //   If fail, add root devices into devices list
        constexpr auto partition_by_affinity =
            sycl::info::partition_property::partition_by_affinity_domain;
        constexpr auto next_partitionable =
            sycl::info::partition_affinity_domain::next_partitionable;
        for (const auto& root_device : root_devices) {
          std::vector<ITEX_GPUDevice> sub_devices;
          auto max_sub_devices =
              root_device
                  .get_info<sycl::info::device::partition_max_sub_devices>();
          if (max_sub_devices == 0) {
            ITEX_LOG(INFO) << "number of sub-devices is zero, expose root "
                              "device.";
            devices.push_back(root_device);
          } else {
            sub_devices = root_device.create_sub_devices<partition_by_affinity>(
                next_partitionable);
            devices.insert(devices.end(), sub_devices.begin(),
                           sub_devices.end());
          }
        }
      } else {
        // If ITEX_TILE_AS_DEVICE is false.
        // Only set root device as device list.
        devices = std::move(root_devices);
      }

      size_t num_device = devices.size();

      if (num_device <= 0) {
        ITEX_LOG(ERROR) << "Can not found any devices. "
                        << "To check runtime environment on your host, "
                        << "please run itex/itex/tools/env_check.sh.";
      }
      assert((num_device > 0));
    });

    return devices;
  }

  DeviceOrdinal current_ordinal_;
  static absl::Mutex mu_;
  static DevicePool* instance_;
};

/* static */ absl::Mutex DevicePool::mu_{absl::kConstInit};
/* static */ DevicePool* DevicePool::instance_{nullptr};

DevicePool* DevicePool::GetInstance() {
  absl::MutexLock lock(&mu_);
  if (instance_ == nullptr) {
    instance_ = new DevicePool();
  }

  return instance_;
}

ITEX_GPUError_t DevicePool::setCurrentDeviceOrdinal(DeviceOrdinal ordinal) {
  absl::MutexLock lock(&mu_);

  if (!hasDevice()) {
    return ITEX_GPU_ERROR_NO_DEVICE;
  }

  if (!isValidDevice(ordinal)) {
    return ITEX_GPU_ERROR_INVALID_DEVICE;
  }

  current_ordinal_ = ordinal;

  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t DevicePool::getCurrentDeviceOrdinal(DeviceOrdinal* ordinal) {
  absl::MutexLock lock(&mu_);

  if (!hasDevice()) {
    return ITEX_GPU_ERROR_NO_DEVICE;
  }

  *ordinal = current_ordinal_;
  return ITEX_GPU_SUCCESS;
}
}  // namespace

/******************* ITEX_GPU context management**************************/
static sycl::async_handler ITEX_GPUAsyncHandler = [](sycl::exception_list eL) {
  for (auto& e : eL) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception& e) {
      ITEX_LOG(ERROR) << "DPC++ Exception: " << e.what()
                      << ", file = " << __FILE__ << ", line = " << __LINE__
                      << ".";
    }
  }
};

class StreamPool {
 public:
  static ITEX_GPUError_t getDefaultStream(ITEX_GPUDevice* device_handle,
                                          ITEX_GPUStream** stream_p) {
    *stream_p = StreamPool::GetStreamsPool(device_handle)[0].get();
    return ITEX_GPU_SUCCESS;
  }

  static ITEX_GPUError_t createStream(ITEX_GPUDevice* device_handle,
                                      ITEX_GPUStream** stream_p) {
    *stream_p = StreamPool::GetStreamsPool(device_handle).back().get();
    return ITEX_GPU_SUCCESS;
  }

  static ITEX_GPUError_t syncStream(ITEX_GPUStream* stream) {
    stream->wait();
    return ITEX_GPU_SUCCESS;
  }

  static ITEX_GPUError_t syncContext(ITEX_GPUDevice* device_handle) {
    for (auto stream : StreamPool::GetStreamsPool(device_handle)) {
      stream->wait();
    }
    return ITEX_GPU_SUCCESS;
  }

  static ITEX_GPUError_t destroyStream(ITEX_GPUDevice* device_handle,
                                       ITEX_GPUStream* stream_handle) {
    if (stream_handle == nullptr) return ITEX_GPU_ERROR_INVALID_STREAM;
    auto stream_pool = StreamPool::GetStreamsPool(device_handle);
    for (int i = 0; i < stream_pool.size(); i++) {
      if (stream_pool[i].get() == stream_handle) {
        stream_pool.erase(stream_pool.begin() + i);
        return ITEX_GPU_SUCCESS;
      }
    }
    return ITEX_GPU_ERROR_INVALID_STREAM;
  }

  static ITEX_GPUError_t getStreams(ITEX_GPUDevice* device_handle,
                                    std::vector<ITEX_GPUStream*>* streams) {
    auto stream_pool = StreamPool::GetStreamsPool(device_handle);
    for (int i = 0; i < stream_pool.size(); i++) {
      streams->push_back(stream_pool[i].get());
    }
    return ITEX_GPU_SUCCESS;
  }

 private:
  static std::vector<std::shared_ptr<ITEX_GPUStream>>& GetStreamsPool(
      ITEX_GPUDevice* device_handle) {
    static std::unordered_map<ITEX_GPUDevice*,
                              std::vector<std::shared_ptr<ITEX_GPUStream>>>
        stream_pool_map;
    auto iter = stream_pool_map.find(device_handle);
    if (iter != stream_pool_map.end()) return iter->second;
    sycl::property_list propList{sycl::property::queue::in_order()};
    std::vector<std::shared_ptr<ITEX_GPUStream>> stream_pool = {
        std::make_shared<ITEX_GPUStream>(DevicePool::getDeviceContext(),
                                         *device_handle, ITEX_GPUAsyncHandler,
                                         propList)};
    stream_pool_map.insert(std::make_pair(device_handle, stream_pool));
    return stream_pool_map[device_handle];
  }
};

ITEX_GPUError_t ITEX_GPUGetDeviceCount(int* count) {
  return DevicePool::getDeviceCount(count);
}

ITEX_GPUError_t ITEX_GPUGetDevice(ITEX_GPUDevice** device, int device_ordinal) {
  return DevicePool::getDevice(device, device_ordinal);
}

ITEX_GPUError_t ITEX_GPUGetCurrentDeviceOrdinal(DeviceOrdinal* ordinal) {
  return DevicePool::GetInstance()->getCurrentDeviceOrdinal(ordinal);
}

ITEX_GPUError_t ITEX_GPUGetDeviceOrdinal(const ITEX_GPUDevice& device,
                                         DeviceOrdinal* device_ordinal) {
  return DevicePool::GetInstance()->getDeviceOrdinal(device, device_ordinal);
}

ITEX_GPUError_t ITEX_GPUSetCurrentDeviceOrdinal(DeviceOrdinal ordinal) {
  return DevicePool::GetInstance()->setCurrentDeviceOrdinal(ordinal);
}

ITEX_GPUError_t ITEX_GPUCreateStream(ITEX_GPUDevice* device_handle,
                                     ITEX_GPUStream** stream_p) {
  return StreamPool::createStream(device_handle, stream_p);
}

ITEX_GPUError_t ITEX_GPUGetDefaultStream(ITEX_GPUDevice* device_handle,
                                         ITEX_GPUStream** stream) {
  return StreamPool::getDefaultStream(device_handle, stream);
}

ITEX_GPUError_t ITEX_GPUDestroyStream(ITEX_GPUDevice* device_handle,
                                      ITEX_GPUStream* stream_handle) {
  return StreamPool::destroyStream(device_handle, stream_handle);
}

ITEX_GPUError_t ITEX_GPUGetStreamPool(ITEX_GPUDevice* device_handle,
                                      std::vector<ITEX_GPUStream*>* streams) {
  return StreamPool::getStreams(device_handle, streams);
}

ITEX_GPUError_t ITEX_GPUCreateEvent(ITEX_GPUDevice* device_handle,
                                    ITEX_GPUEvent** event_handle) {
  *event_handle = new sycl::event();
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t ITEX_GPUDestroyEvent(ITEX_GPUDevice* device_handle,
                                     ITEX_GPUEvent* event_handle) {
  delete event_handle;
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t ITEX_GPUStreamWaitEvent(ITEX_GPUStream* stream,
                                        ITEX_GPUEvent* event) {
  // TODO(itex): queue.wait_for(event)?
  stream->wait();
  // event->wait();
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t ITEX_GPUStreamWaitStream(ITEX_GPUStream* dependent,
                                         ITEX_GPUStream* other) {
  // TODO(itex): queue.wait_for(event)?
  dependent->wait();
  other->wait();
  // event->wait();
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t ITEX_GPUCtxSynchronize(ITEX_GPUDevice* device_handle) {
  return StreamPool::syncContext(device_handle);
}

ITEX_GPUError_t ITEX_GPUStreamSynchronize(ITEX_GPUStream* stream_handle) {
  return StreamPool::syncStream(stream_handle);
}

/************************* ITEX_GPU memory management
 * ***************************/

static void memcpyHostToDevice(void* dstDevice, const void* srcHost,
                               size_t ByteCount, bool async,
                               ITEX_GPUStream* stream) {
  if (ByteCount == 0) return;

  auto event = stream->memcpy(dstDevice, srcHost, ByteCount);
  if (!async) {
    event.wait();
  }
}

static void memcpyDeviceToHost(void* dstHost, const void* srcDevice,
                               size_t ByteCount, bool async,
                               ITEX_GPUStream* stream) {
  if (ByteCount == 0) return;

  auto event = stream->memcpy(dstHost, srcDevice, ByteCount);

  if (!async) {
    event.wait();
  }
}

static void memcpyDeviceToDevice(void* dstDevice, const void* srcDevice,
                                 size_t ByteCount, bool async,
                                 ITEX_GPUStream* stream) {
  if (ByteCount == 0) return;

  auto event = stream->memcpy(dstDevice, srcDevice, ByteCount);

  if (!async) {
    event.wait();
  }
}

static void memsetDeviceD8(void* dstDevice, unsigned char value, size_t n,
                           bool async, ITEX_GPUStream* stream) {
  if (n == 0) return;

  auto event = stream->memset(dstDevice, value, n * sizeof(uint8_t));
  if (!async) {
    event.wait();
  }
}

static void memsetDeviceD32(void* dstDevice, int value, size_t n, bool async,
                            ITEX_GPUStream* stream) {
  if (n == 0) return;

  auto event = stream->memset(dstDevice, value, n * sizeof(uint32_t));

  if (!async) {
    event.wait();
  }
}

ITEX_GPUError_t ITEX_GPUMemcpyDtoH(void* dstHost, const void* srcDevice,
                                   size_t ByteCount, ITEX_GPUDevice* device) {
  ITEX_GPUStream* stream;
  auto res = StreamPool::getDefaultStream(device, &stream);
  memcpyDeviceToHost(dstHost, srcDevice, ByteCount, false, stream);
  return res;
}

ITEX_GPUError_t ITEX_GPUMemcpyHtoD(void* dstDevice, const void* srcHost,
                                   size_t ByteCount, ITEX_GPUDevice* device) {
  ITEX_GPUStream* stream;
  auto res = StreamPool::getDefaultStream(device, &stream);
  memcpyHostToDevice(dstDevice, srcHost, ByteCount, false, stream);
  return res;
}

ITEX_GPUError_t ITEX_GPUMemcpyDtoD(void* dstDevice, const void* srcDevice,
                                   size_t ByteCount, ITEX_GPUDevice* device) {
  ITEX_GPUStream* stream;
  auto res = StreamPool::getDefaultStream(device, &stream);
  memcpyDeviceToDevice(dstDevice, srcDevice, ByteCount, false, stream);
  return res;
}

ITEX_GPUError_t ITEX_GPUMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                        size_t ByteCount,
                                        ITEX_GPUStream* stream) {
  // TODO(itex): set async = True when we support
  memcpyDeviceToHost(dstHost, srcDevice, ByteCount, false, stream);
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t ITEX_GPUMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                        size_t ByteCount,
                                        ITEX_GPUStream* stream) {
  // TODO(itex): set async = True when we support
  memcpyHostToDevice(dstDevice, srcHost, ByteCount, false, stream);
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t ITEX_GPUMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                        size_t ByteCount,
                                        ITEX_GPUStream* stream) {
  // TODO(itex): set async = True when we support
  memcpyDeviceToDevice(dstDevice, srcDevice, ByteCount, true, stream);
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t ITEX_GPUMemsetD8(void* dstDevice, unsigned char uc, size_t N,
                                 ITEX_GPUDevice* device) {
  ITEX_GPUStream* stream;
  auto res = StreamPool::getDefaultStream(device, &stream);
  memsetDeviceD8(dstDevice, uc, N, false, stream);
  return res;
}

ITEX_GPUError_t ITEX_GPUMemsetD8Async(void* dstDevice, unsigned char uc,
                                      size_t N, ITEX_GPUStream* stream) {
  memsetDeviceD8(dstDevice, uc, N, true, stream);
  return ITEX_GPU_SUCCESS;
}

ITEX_GPUError_t ITEX_GPUMemsetD32(void* dstDevice, unsigned int ui, size_t N,
                                  ITEX_GPUDevice* device) {
  ITEX_GPUStream* stream;
  auto res = StreamPool::getDefaultStream(device, &stream);
  memsetDeviceD32(dstDevice, ui, N, false, stream);
  return res;
}

ITEX_GPUError_t ITEX_GPUMemsetD32Async(void* dstDevice, unsigned int ui,
                                       size_t N, ITEX_GPUStream* stream) {
  memsetDeviceD32(dstDevice, ui, N, true, stream);
  return ITEX_GPU_SUCCESS;
}

void* ITEX_GPUMalloc(ITEX_GPUDevice* device, size_t ByteCount) {
  ITEX_GPUStream* stream;
  StreamPool::getDefaultStream(device, &stream);

  // Always use default 0 stream to allocate mem
  auto ptr = aligned_alloc_device(64, ByteCount, *stream);
  return static_cast<void*>(ptr);
}

void* ITEX_GPUMallocHost(size_t ByteCount) {
  ITEX_GPUStream* stream;
  ITEX_GPUDevice* device;
  DeviceOrdinal device_ordinal;
  ITEX_GPUGetCurrentDeviceOrdinal(&device_ordinal);
  ITEX_GPUGetDevice(&device, device_ordinal);
  StreamPool::getDefaultStream(device, &stream);

  // Always use default 0 stream to allocate mem
  auto ptr = aligned_alloc_host(/*alignment=*/64, ByteCount, *stream);
  return static_cast<void*>(ptr);
}

void ITEX_GPUFree(ITEX_GPUDevice* device, void* ptr) {
  ITEX_GPUStream* stream;
  StreamPool::getDefaultStream(device, &stream);

  // Always use default 0 stream to free mem
  sycl::free(ptr, *stream);
}

const char* ITEX_GPUGetErrorName(ITEX_GPUError_t error) {
  switch (error) {
    case ITEX_GPU_SUCCESS:
      return "DPC++ succeed.";
    case ITEX_GPU_ERROR_NO_DEVICE:
      return "DPC++ did not find the device.";
    case ITEX_GPU_ERROR_INVALID_DEVICE:
      return "DPC++ got invalid device id.";
    case ITEX_GPU_ERROR_INVALID_POINTER:
      return "DPC++ got invalid pointer.";
    case ITEX_GPU_ERROR_INVALID_STREAM:
      return "DPC++ got invalid stream.";
    case ITEX_GPU_ERROR_DESTROY_DEFAULT_STREAM:
      return "DPC++ cannot destroy default stream.";
    default:
      return "DPC++ got invalid error code.";
  }
}  // namespace
