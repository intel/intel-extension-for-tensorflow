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
#ifndef USING_NEXTPLUGGABLE_DEVICE
#ifndef CC_BUILD
#include "itex/core/devices/xpu_device.h"
#endif

#include "itex/core/devices/device_backend_util.h"
#include "itex/core/utils/types.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#ifndef INTEL_CPU_ONLY
#include "itex/core/devices/gpu/gpu_device_plugin.h"
#endif  // INTEL_CPU_ONLY

namespace itex {
#define REPORT_UNIMPLMENTED(status)                            \
  {                                                            \
    std::ostringstream err;                                    \
    err << "Umplemented backend";                              \
    TF_SetStatus(status, TF_UNIMPLEMENTED, err.str().c_str()); \
  }

void xpu_device_count(const SP_Platform* platform, int* device_count,
                      TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_device_count(platform, device_count, status);
      return;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_device_count BACKEND CPU is not support yet";
      return;
    default:
      ITEX_LOG(ERROR) << "xpu_device_count BACKEND UNKOWN";
      return;
  }
}

void xpu_create_device(const SP_Platform* platform,
                       SE_CreateDeviceParams* params, TF_Status* const status) {
  ITEX_BACKEND backend = itex_get_backend();
  itex_freeze_backend(backend);
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_create_device(platform, params, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_create_device BACKEND CPU is not support yet";
      REPORT_UNIMPLMENTED(status);
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_create_device BACKEND UNKOWN";
      REPORT_UNIMPLMENTED(status);
      break;
  }
}

void xpu_destroy_device(const SP_Platform* platform, SP_Device* device) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_device(platform, device);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_destroy_device BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_device BACKEND UNKOWN";
      break;
  }
}

void xpu_create_device_fns(const SP_Platform* platform,
                           SE_CreateDeviceFnsParams* params,
                           TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_create_device_fns(platform, params, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_create_device_fns BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_create_device_fns BACKEND UNKOWN";
      break;
  }
}

void xpu_destroy_device_fns(const SP_Platform* platform,
                            SP_DeviceFns* device_fns) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_device_fns(platform, device_fns);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_destroy_device_fns BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_device_fns BACKEND UNKOWN";
      break;
  }
}

void xpu_allocate(const SP_Device* device, uint64_t size, int64_t memory_space,
                  SP_DeviceMemoryBase* mem) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_allocate(device, size, memory_space, mem);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_allocate BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_device_fns BACKEND UNKOWN";
      break;
  }
}

void xpu_deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_deallocate(device, mem);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_deallocate BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_device_fns BACKEND UNKOWN";
      break;
  }
}

void* xpu_host_memory_allocate(const SP_Device* device, uint64_t size) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      return itex::gpu_host_memory_allocate(device, size);
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_host_memory_allocate BACKEND CPU is not support yet";
      return nullptr;
    default:
      ITEX_LOG(ERROR) << "xpu_host_memory_allocate BACKEND UNKOWN";
      return nullptr;
  }
}

void xpu_host_memory_deallocate(const SP_Device* device, void* mem) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_host_memory_deallocate(device, mem);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_host_memory_deallocate BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_host_memory_deallocate BACKEND UNKOWN";
      break;
  }
}

TF_Bool xpu_get_allocator_stats(const SP_Device* device,
                                SP_AllocatorStats* stats) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      return itex::gpu_get_allocator_stats(device, stats);
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_get_allocator_stats BACKEND CPU is not support yet";
      return true;
    default:
      ITEX_LOG(ERROR) << "xpu_get_allocator_stats BACKEND UNKOWN";
      return true;
  }
}

TF_Bool xpu_device_memory_usage(const SP_Device* device, int64_t* free,
                                int64_t* total) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      return itex::gpu_device_memory_usage(device, free, total);
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_device_memory_usage BACKEND CPU is not support yet";
      return true;
    default:
      ITEX_LOG(ERROR) << "xpu_get_allocator_stats BACKEND UNKOWN";
      return true;
  }
}

void xpu_create_stream(const SP_Device* device, SP_Stream* stream,
                       TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_create_stream(device, stream, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_create_stream BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_create_stream BACKEND UNKOWN";
      break;
  }
}

void xpu_destroy_stream(const SP_Device* device, SP_Stream stream) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_stream(device, stream);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_destroy_stream BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_stream BACKEND UNKOWN";
      break;
  }
}

void xpu_create_stream_dependency(const SP_Device* device, SP_Stream dependent,
                                  SP_Stream other, TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_create_stream_dependency(device, dependent, other, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_create_stream_dependency BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_create_stream_dependency BACKEND UNKOWN";
      break;
  }
}

void xpu_get_stream_status(const SP_Device* device, SP_Stream stream,
                           TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_get_stream_status(device, stream, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_get_stream_status BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_get_stream_status BACKEND UNKOWN";
      break;
  }
}

void xpu_create_event(const SP_Device* device, SP_Event* event,
                      TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_create_event(device, event, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_create_event BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_create_event BACKEND UNKOWN";
      break;
  }
}

void xpu_destroy_event(const SP_Device* device, SP_Event event) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_event(device, event);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_destroy_event BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_event  BACKEND UNKOWN";
      break;
  }
}

SE_EventStatus xpu_get_event_status(const SP_Device* device, SP_Event event) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      return itex::gpu_get_event_status(device, event);
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_get_event_status BACKEND CPU is not support yet";
      return SE_EVENT_COMPLETE;
    default:
      ITEX_LOG(ERROR) << "xpu_get_event_status BACKEND UNKOWN";
      return SE_EVENT_COMPLETE;
  }
}

void xpu_record_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                      TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_record_event(device, stream, event, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_record_event BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_record_event BACKEND UNKOWN";
      break;
  }
}

void xpu_wait_for_event(const SP_Device* const device, SP_Stream stream,
                        SP_Event event, TF_Status* const status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_wait_for_event(device, stream, event, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_wait_for_event BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_wait_for_event BACKEND UNKOWN";
      break;
  }
}

void xpu_create_timer(const SP_Device* device, SP_Timer* timer,
                      TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_create_timer(device, timer, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_create_timer BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_create_timer BACKEND UNKOWN";
      break;
  }
}

void xpu_destroy_timer(const SP_Device* device, SP_Timer timer) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_timer(device, timer);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_destroy_timer BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_timer BACKEND UNKOWN";
      break;
  }
}

// Records a start event for an interval timer.
void xpu_start_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                     TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_start_timer(device, stream, timer, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_start_timer BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_start_timer BACKEND UNKOWN";
      break;
  }
}

// Records a stop event for an interval timer.
void xpu_stop_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                    TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_stop_timer(device, stream, timer, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_stop_timer BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_stop_timer BACKEND UNKOWN";
      break;
  }
}

void xpu_memcpy_dtoh(const SP_Device* device, SP_Stream stream, void* host_dst,
                     const SP_DeviceMemoryBase* device_src, uint64_t size,
                     TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_memcpy_dtoh(device, stream, host_dst, device_src, size, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_memcpy_dtoh BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_memcpy_dtoh BACKEND UNKOWN";
      break;
  }
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a host memory source, with target size `size`.
void xpu_memcpy_htod(const SP_Device* device, SP_Stream stream,
                     SP_DeviceMemoryBase* device_dst, const void* host_src,
                     uint64_t size, TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_memcpy_htod(device, stream, device_dst, host_src, size, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_memcpy_htod BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_memcpy_htod BACKEND UNKOWN";
      break;
  }
}
// Enqueues a memcpy operation onto stream, with a device destination
// location and a device memory source, with target size `size`.
void xpu_memcpy_dtod(const SP_Device* device, SP_Stream stream,
                     SP_DeviceMemoryBase* device_dst,
                     const SP_DeviceMemoryBase* device_src, uint64_t size,
                     TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_memcpy_dtod(device, stream, device_dst, device_src, size,
                            status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_memcpy_dtoh BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_memcpy_dtoh BACKEND UNKOWN";
      break;
  }
}

// Blocks the caller while a data segment of the given size is
// copied from the device source to the host destination.
void xpu_sync_memcpy_dtoh(const SP_Device* device, void* host_dst,
                          const SP_DeviceMemoryBase* device_src, uint64_t size,
                          TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_sync_memcpy_dtoh(device, host_dst, device_src, size, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_sync_memcpy_dtoh BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_sync_memcpy_dtoh BACKEND UNKOWN";
      break;
  }
}

// Blocks the caller while a data segment of the given size is
// copied from the host source to the device destination.
void xpu_sync_memcpy_htod(const SP_Device* device,
                          SP_DeviceMemoryBase* device_dst, const void* host_src,
                          uint64_t size, TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_sync_memcpy_htod(device, device_dst, host_src, size, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_sync_memcpy_dtoh BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_sync_memcpy_dtoh BACKEND UNKOWN";
      break;
  }
}

// Blocks the caller while a data segment of the given size is copied from the
// device source to the device destination.
void xpu_sync_memcpy_dtod(const SP_Device* device,
                          SP_DeviceMemoryBase* device_dst,
                          const SP_DeviceMemoryBase* device_src, uint64_t size,
                          TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_sync_memcpy_dtod(device, device_dst, device_src, size, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_sync_memcpy_dtod BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_sync_memcpy_dtod BACKEND UNKOWN";
      break;
  }
}

// Causes the host code to synchronously wait for the event to complete.
void xpu_block_host_for_event(const SP_Device* device, SP_Event event,
                              TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_block_host_for_event(device, event, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_sync_memcpy_dtod BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_sync_memcpy_dtod BACKEND UNKOWN";
      break;
  }
}

void xpu_block_host_until_done(const SP_Device* device, SP_Stream stream,
                               TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_block_host_until_done(device, stream, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_block_host_until_done BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_block_host_until_done BACKEND UNKOWN";
      break;
  }
}

// Synchronizes all activity occurring in the StreamExecutor's context (most
// likely a whole device).
void xpu_synchronize_all_activity(const SP_Device* device, TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_synchronize_all_activity(device, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_synchronize_all_activity BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_synchronize_all_activity BACKEND UNKOWN";
      break;
  }
}

void xpu_mem_zero(const SP_Device* device, SP_Stream stream,
                  SP_DeviceMemoryBase* location, uint64_t size,
                  TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_mem_zero(device, stream, location, size, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_mem_zero BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_mem_zero BACKEND UNKOWN";
  }
}

void xpu_memset(const SP_Device* device, SP_Stream stream,
                SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
                TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_memset(device, stream, location, size, pattern, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_memset BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_memset BACKEND UNKOWN";
  }
}

void xpu_memset32(const SP_Device* device, SP_Stream stream,
                  SP_DeviceMemoryBase* location, uint32_t pattern,
                  uint64_t size, TF_Status* status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_memset32(device, stream, location, size, pattern, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_memset32 BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_memset32 BACKEND UNKOWN";
  }
}

// Enqueues on a stream a user-specified function to be run on the host.
// `callback_arg` should be passed as the first argument to `callback_fn`.
TF_Bool xpu_host_callback(const SP_Device* device, SP_Stream stream,
                          SE_StatusCallbackFn callback_fn, void* callback_arg) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      return itex::gpu_host_callback(device, stream, callback_fn, callback_arg);
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_host_callback BACKEND CPU is not support yet";
      return true;
    default:
      ITEX_LOG(ERROR) << "xpu_host_callback BACKEND UNKOWN";
      return true;
  }
}

void xpu_create_stream_executor(const SP_Platform* platform,
                                SE_CreateStreamExecutorParams* params,
                                TF_Status* const status) {
  params->stream_executor->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  params->stream_executor->allocate = xpu_allocate;
  params->stream_executor->deallocate = xpu_deallocate;
  params->stream_executor->host_memory_allocate = xpu_host_memory_allocate;
  params->stream_executor->host_memory_deallocate = xpu_host_memory_deallocate;
  params->stream_executor->get_allocator_stats = xpu_get_allocator_stats;
  params->stream_executor->device_memory_usage = xpu_device_memory_usage;
  params->stream_executor->create_stream = xpu_create_stream;
  params->stream_executor->destroy_stream = xpu_destroy_stream;
  params->stream_executor->create_stream_dependency =
      xpu_create_stream_dependency;
  params->stream_executor->get_stream_status = xpu_get_stream_status;
  params->stream_executor->create_event = xpu_create_event;
  params->stream_executor->destroy_event = xpu_destroy_event;
  params->stream_executor->get_event_status = xpu_get_event_status;
  params->stream_executor->record_event = xpu_record_event;
  params->stream_executor->wait_for_event = xpu_wait_for_event;
  params->stream_executor->create_timer = xpu_create_timer;
  params->stream_executor->destroy_timer = xpu_destroy_timer;
  params->stream_executor->start_timer = xpu_start_timer;
  params->stream_executor->stop_timer = xpu_stop_timer;
  params->stream_executor->memcpy_dtoh = xpu_memcpy_dtoh;
  params->stream_executor->memcpy_htod = xpu_memcpy_htod;
  params->stream_executor->memcpy_dtod = xpu_memcpy_dtod;
  params->stream_executor->sync_memcpy_dtoh = xpu_sync_memcpy_dtoh;
  params->stream_executor->sync_memcpy_htod = xpu_sync_memcpy_htod;
  params->stream_executor->sync_memcpy_dtod = xpu_sync_memcpy_dtod;
  params->stream_executor->block_host_until_done = xpu_block_host_until_done;
  params->stream_executor->block_host_for_event = xpu_block_host_for_event;
  params->stream_executor->synchronize_all_activity =
      xpu_synchronize_all_activity;
  params->stream_executor->mem_zero = xpu_mem_zero;
  params->stream_executor->memset = xpu_memset;
  params->stream_executor->memset32 = xpu_memset32;
  params->stream_executor->host_callback = xpu_host_callback;
}

void xpu_destroy_stream_executor(const SP_Platform* platform,
                                 SP_StreamExecutor* stream_executor) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_stream_executor(platform, stream_executor);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_destroy_stream_executor BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_stream_executor BACKEND UNKOWN";
  }
}

void xpu_create_timer_fns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                          TF_Status* const status) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_create_timer_fns(platform, timer_fns, status);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_create_timer_fns BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_create_timer_fns BACKEND UNKOWN";
      break;
  }
}

void xpu_destroy_timer_fns(const SP_Platform* platform,
                           SP_TimerFns* timer_fns) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_timer_fns(platform, timer_fns);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_destroy_timer_fns BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_timer_fns BACKEND UNKOWN";
  }
}

void xpu_destroy_platform(SP_Platform* const platform) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_platform(platform);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR) << "xpu_destroy_platform BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_platform BACKEND UNKOWN";
  }
}

void xpu_destroy_platform_fns(SP_PlatformFns* const platform_fns) {
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
      itex::gpu_destroy_platform_fns(platform_fns);
      break;
    case ITEX_BACKEND_CPU:
      ITEX_LOG(ERROR)
          << "xpu_destroy_platform_fns BACKEND CPU is not support yet";
      break;
    default:
      ITEX_LOG(ERROR) << "xpu_destroy_platform_fns BACKEND UNKOWN";
  }
}

void SE_InitXPUPluginFns(SE_PlatformRegistrationParams* const params,
                         TF_Status* const status) {
  params->platform_fns->get_device_count = xpu_device_count;
  params->platform_fns->create_device = xpu_create_device;
  params->platform_fns->destroy_device = xpu_destroy_device;
  params->platform_fns->create_device_fns = xpu_create_device_fns;
  params->platform_fns->destroy_device_fns = xpu_destroy_device_fns;
  params->platform_fns->create_stream_executor = xpu_create_stream_executor;
  params->platform_fns->destroy_stream_executor = xpu_destroy_stream_executor;
  params->platform_fns->create_timer_fns = xpu_create_timer_fns;
  params->platform_fns->destroy_timer_fns = xpu_destroy_timer_fns;
  params->destroy_platform = xpu_destroy_platform;
  params->destroy_platform_fns = xpu_destroy_platform_fns;
}

}  // namespace itex

#ifndef CC_BUILD
void SE_InitPlugin_Internal(SE_PlatformRegistrationParams* const params,
                            TF_Status* const status) {
#else
void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
#endif
  params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  params->platform->name = DEVICE_XPU_NAME;
  params->platform->type = itex::DEVICE_XPU;
  // TODO(itex): check whether we need to turn on this setting
  // params->platform->supports_unified_memory = true;
  // params->platform->use_bfc_allocator = true;
  itex::SE_InitXPUPluginFns(params, status);
}
#endif  // USING_NEXTPLUGGABLE_DEVICE
