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

#include "itex/core/devices/gpu/gpu_device_plugin.h"

#include <memory>
#include <string>

#include "itex/core/devices/gpu/gpu_pool_allocator.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

namespace itex {

void gpu_device_count(const SP_Platform* platform, int* device_count,
                      TF_Status* status) {
  ITEX_GPUGetDeviceCount(device_count);
}

void gpu_create_device(const SP_Platform* platform,
                       SE_CreateDeviceParams* params, TF_Status* const status) {
  params->device->struct_size = SP_DEVICE_STRUCT_SIZE;
  ITEX_GPUDevice* device_h;
  ITEX_GPUGetDevice(&device_h, params->ordinal);
  params->device->device_handle = static_cast<void*>(device_h);
  params->device->ordinal = params->ordinal;
}

void gpu_destroy_device(const SP_Platform* platform, SP_Device* device) {
  device->device_handle = nullptr;
  device->ordinal = -1;
}

void gpu_create_device_fns(const SP_Platform* platform,
                           SE_CreateDeviceFnsParams* params,
                           TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  params->device_fns->struct_size = {SP_DEVICE_FNS_STRUCT_SIZE};
}
void gpu_destroy_device_fns(const SP_Platform* platform,
                            SP_DeviceFns* device_fns) {}

/*StreamExecutor Backend Impl*/
void gpu_allocate(const SP_Device* device, uint64_t size, int64_t memory_space,
                  SP_DeviceMemoryBase* mem) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
  std::shared_ptr<BFCAllocator> alloc;
  auto status = ITEX_GPUGetAllocator(device_handle, &alloc);
  ITEX_CHECK(status == ITEX_GPU_SUCCESS)
      << "Failed to get device allocator, device handle: " << device_handle;
  mem->opaque = alloc->AllocateRaw(size);
  mem->size = size;
}

void gpu_deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  std::shared_ptr<BFCAllocator> alloc;
  auto status = ITEX_GPUGetAllocator(device_handle, &alloc);
  ITEX_CHECK(status == ITEX_GPU_SUCCESS)
      << "Failed to get device allocator, device handle: " << device_handle;
  alloc->DeallocateRaw(mem->opaque);
  mem->opaque = nullptr;
  mem->size = 0;
}

void* gpu_host_memory_allocate(const SP_Device* device, uint64_t size) {
  void* ptr = nullptr;
  ptr = ITEX_GPUMallocHost(size);
  return ptr;
}

void gpu_host_memory_deallocate(const SP_Device* device, void* mem) {
  auto device_handle = static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUStream* stream;
  // Always use default 0 stream to free mem
  ITEX_GPUGetDefaultStream(device_handle, &stream);
  sycl::free(mem, *stream);
}

TF_Bool gpu_get_allocator_stats(const SP_Device* device,
                                SP_AllocatorStats* stats) {
  stats->struct_size = SP_ALLOCATORSTATS_STRUCT_SIZE;
  stats->bytes_in_use = 123;
  ITEX_LOG(ERROR) << "Get allocator stats not implemented!";
  return true;
}

TF_Bool gpu_device_memory_usage(const SP_Device* device, int64_t* free,
                                int64_t* total) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  *free =
      device_handle->template get_info<sycl::info::device::global_mem_size>();
  *total =
      device_handle->template get_info<sycl::info::device::global_mem_size>();
  return true;
}

void gpu_create_stream(const SP_Device* device, SP_Stream* stream,
                       TF_Status* status) {
  ITEX_GPUStream* stream_handle;
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUCreateStream(device_handle, &stream_handle);
  *stream = new SP_Stream_st(stream_handle);
}

// Destroys SP_Stream and deallocates any underlying resources.
void gpu_destroy_stream(const SP_Device* device, SP_Stream stream) {
  ITEX_GPUStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUDestroyStream(device_handle, stream_handle);
  delete stream;
}

void gpu_create_stream_dependency(const SP_Device* device, SP_Stream dependent,
                                  SP_Stream other, TF_Status* status) {
  ITEX_GPUStream* stream_handle1 =
      static_cast<SP_Stream_st*>(dependent)->stream_handle;
  ITEX_GPUStream* stream_handle2 =
      static_cast<SP_Stream_st*>(other)->stream_handle;
  ITEX_GPUStreamWaitStream(stream_handle1, stream_handle2);
}

// Without blocking the device, retrieve the current stream status.
void gpu_get_stream_status(const SP_Device* device, SP_Stream stream,
                           TF_Status* status) {
  // TF_SetStatus(status, TF_OK, "");
  ITEX_LOG(ERROR) << "DPC++: gpu_get_stream_status not implemented.";
}

void gpu_create_event(const SP_Device* device, SP_Event* event,
                      TF_Status* status) {
  ITEX_GPUEvent event_handle;
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUCreateEvent(device_handle, &event_handle);
  *event = new SP_Event_st(event_handle);
}

// Destroy SE_Event and perform any platform-specific deallocation and
// cleanup of an event.
void gpu_destroy_event(const SP_Device* device, SP_Event event) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUEvent event_handle = static_cast<SP_Event_st*>(event)->event_handle;
  ITEX_GPUDestroyEvent(device_handle, event_handle);
  delete event;
}

// Requests the current status of the event from the underlying platform.
SE_EventStatus gpu_get_event_status(const SP_Device* device, SP_Event event) {
  if (IsMultipleStreamEnabled()) {
    ITEX_GPUEvent event_handle = static_cast<SP_Event_st*>(event)->event_handle;
    auto event_status =
        event_handle
            .get_info<cl::sycl::info::event::command_execution_status>();
    switch (event_status) {
      case cl::sycl::info::event_command_status::submitted:
      case cl::sycl::info::event_command_status::running:
        return SE_EVENT_PENDING;
      case cl::sycl::info::event_command_status::complete:
        return SE_EVENT_COMPLETE;
      default:
        return SE_EVENT_UNKNOWN;
    }
  }
  return SE_EVENT_COMPLETE;
}

// Inserts the specified event at the end of the specified stream.
void gpu_record_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                      TF_Status* status) {
  if (IsMultipleStreamEnabled()) {
    ITEX_GPUStream* stream_handle =
        static_cast<SP_Stream_st*>(stream)->stream_handle;
    ITEX_GPUEvent recorded_event = stream_handle->ext_oneapi_submit_barrier();
    event->event_handle = recorded_event;
  }
}

// Wait for the specified event at the end of the specified stream.
void gpu_wait_for_event(const SP_Device* const device, SP_Stream stream,
                        SP_Event event, TF_Status* const status) {
  ITEX_GPUStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  ITEX_GPUEvent event_handle = static_cast<SP_Event_st*>(event)->event_handle;
  ITEX_GPUStreamWaitEvent(stream_handle, event_handle);
}

/*** TIMER CALLBACKS ***/
// Creates SP_Timer. Allocates timer resources on the underlying platform
// and initializes its internals, setting `timer` output variable. Sets
// values in `timer_fns` struct.
void gpu_create_timer(const SP_Device* device, SP_Timer* timer,
                      TF_Status* status) {
  ITEX_LOG(ERROR) << "DPC++: create_timer not implemented.";
}

// Destroy timer and deallocates timer resources on the underlying platform.
void gpu_destroy_timer(const SP_Device* device, SP_Timer timer) {
  ITEX_LOG(ERROR) << "DPC++: destroy_timer not implemented.";
}

// Records a start event for an interval timer.
void gpu_start_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                     TF_Status* status) {
  ITEX_LOG(ERROR) << "DPC++: start_timer not implemented.";
}

// Records a stop event for an interval timer.
void gpu_stop_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                    TF_Status* status) {
  ITEX_LOG(ERROR) << "DPC++: stop_timer not implemented.";
}

/*** MEMCPY CALLBACKS ***/
// Enqueues a memcpy operation onto stream, with a host destination location
// `host_dst` and a device memory source, with target size `size`.
void gpu_memcpy_dtoh(const SP_Device* device, SP_Stream stream, void* host_dst,
                     const SP_DeviceMemoryBase* device_src, uint64_t size,
                     TF_Status* status) {
  ITEX_GPUStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  ITEX_GPUMemcpyDtoHAsync(host_dst, device_src->opaque, size, stream_handle);
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a host memory source, with target size `size`.
void gpu_memcpy_htod(const SP_Device* device, SP_Stream stream,
                     SP_DeviceMemoryBase* device_dst, const void* host_src,
                     uint64_t size, TF_Status* status) {
  ITEX_GPUStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  ITEX_GPUMemcpyHtoDAsync(device_dst->opaque, host_src, size, stream_handle);
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a device memory source, with target size `size`.
void gpu_memcpy_dtod(const SP_Device* device, SP_Stream stream,
                     SP_DeviceMemoryBase* device_dst,
                     const SP_DeviceMemoryBase* device_src, uint64_t size,
                     TF_Status* status) {
  ITEX_GPUStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  ITEX_GPUMemcpyDtoDAsync(device_dst->opaque, device_src->opaque, size,
                          stream_handle);
}

// Blocks the caller while a data segment of the given size is
// copied from the device source to the host destination.
void gpu_sync_memcpy_dtoh(const SP_Device* device, void* host_dst,
                          const SP_DeviceMemoryBase* device_src, uint64_t size,
                          TF_Status* status) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUMemcpyDtoH(host_dst, device_src->opaque, size, device_handle);
}

// Blocks the caller while a data segment of the given size is
// copied from the host source to the device destination.
void gpu_sync_memcpy_htod(const SP_Device* device,
                          SP_DeviceMemoryBase* device_dst, const void* host_src,
                          uint64_t size, TF_Status* status) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUMemcpyHtoD(device_dst->opaque, host_src, size, device_handle);
}

// Blocks the caller while a data segment of the given size is copied from the
// device source to the device destination.
void gpu_sync_memcpy_dtod(const SP_Device* device,
                          SP_DeviceMemoryBase* device_dst,
                          const SP_DeviceMemoryBase* device_src, uint64_t size,
                          TF_Status* status) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUMemcpyDtoD(device_dst->opaque, device_src->opaque, size,
                     device_handle);
}

// Causes the host code to synchronously wait for the event to complete.
void gpu_block_host_for_event(const SP_Device* device, SP_Event event,
                              TF_Status* status) {
  event->event_handle.wait();
}

void gpu_block_host_until_done(const SP_Device* device, SP_Stream stream,
                               TF_Status* status) {
  ITEX_GPUStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  stream_handle->wait();
}

// Synchronizes all activity occurring in the StreamExecutor's context (most
// likely a whole device).
void gpu_synchronize_all_activity(const SP_Device* device, TF_Status* status) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUCtxSynchronize(device_handle);
}

void gpu_mem_zero(const SP_Device* device, SP_Stream stream,
                  SP_DeviceMemoryBase* location, uint64_t size,
                  TF_Status* status) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUMemsetD8(location->opaque, 0, size, device_handle);
}

void gpu_memset(const SP_Device* device, SP_Stream stream,
                SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
                TF_Status* status) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUMemsetD8(location->opaque, pattern, size, device_handle);
}

void gpu_memset32(const SP_Device* device, SP_Stream stream,
                  SP_DeviceMemoryBase* location, uint32_t pattern,
                  uint64_t size, TF_Status* status) {
  ITEX_GPUDevice* device_handle =
      static_cast<ITEX_GPUDevice*>(device->device_handle);
  ITEX_GPUMemsetD32(location->opaque, pattern, size, device_handle);
}

// Enqueues on a stream a user-specified function to be run on the host.
// `callback_arg` should be passed as the first argument to `callback_fn`.
TF_Bool gpu_host_callback(const SP_Device* device, SP_Stream stream,
                          SE_StatusCallbackFn callback_fn, void* callback_arg) {
  ITEX_GPUStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  stream_handle->submit([&](auto& cgh) {
    auto host_task = [&]() {
      TF_Status* tf_status = TF_NewStatus();
      callback_fn(callback_arg, tf_status);
      if (TF_GetCode(tf_status) != TF_OK) {
        ITEX_LOG(WARNING) << "DPC++: Host callback failed: "
                          << std::string(TF_Message(tf_status));
      }
      TF_DeleteStatus(tf_status);
    };
    cgh.host_task(host_task);
  });
  return 1;
}

/*Timer Backer Impl*/
uint64_t nanoseconds(SP_Timer timer) { return timer->timer_handle; }

void gpu_create_timer_fns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                          TF_Status* const status) {
  timer_fns->nanoseconds = nanoseconds;
}

void gpu_destroy_timer_fns(const SP_Platform* platform,
                           SP_TimerFns* timer_fns) {}

void gpu_destroy_stream_executor(const SP_Platform* platform,
                                 SP_StreamExecutor* stream_executor) {}

void gpu_destroy_platform(SP_Platform* const platform) {}
void gpu_destroy_platform_fns(SP_PlatformFns* const platform_fns) {}

}  // namespace itex
