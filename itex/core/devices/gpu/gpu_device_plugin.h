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

#ifndef ITEX_CORE_DEVICES_GPU_GPU_DEVICE_PLUGIN_H_
#define ITEX_CORE_DEVICES_GPU_GPU_DEVICE_PLUGIN_H_

#ifndef USING_NEXTPLUGGABLE_DEVICE
#include "itex/core/utils/logging.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

struct SP_Stream_st {
  explicit SP_Stream_st(ITEX_GPUStream* stream_h) : stream_handle(stream_h) {}
  ITEX_GPUStream* stream_handle;
};

struct SP_Event_st {
  explicit SP_Event_st(ITEX_GPUEvent event_h) : event_handle(event_h) {}
  ITEX_GPUEvent event_handle;
};

struct SP_Timer_st {
  explicit SP_Timer_st(int id) : timer_handle(id) {}
  int timer_handle;
};

namespace itex {
void SE_InitGPUPluginFns(SE_PlatformRegistrationParams* const params,
                         TF_Status* const status);
void gpu_device_count(const SP_Platform* platform, int* device_count,
                      TF_Status* status);
void gpu_create_device(const SP_Platform* platform,
                       SE_CreateDeviceParams* params, TF_Status* const status);
void gpu_destroy_device(const SP_Platform* platform, SP_Device* device);
void gpu_create_device_fns(const SP_Platform* platform,
                           SE_CreateDeviceFnsParams* params, TF_Status* status);
void gpu_destroy_device_fns(const SP_Platform* platform,
                            SP_DeviceFns* device_fns);
void gpu_destroy_stream_executor(const SP_Platform* platform,
                                 SP_StreamExecutor* stream_executor);
void gpu_create_timer_fns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                          TF_Status* const status);
void gpu_destroy_timer_fns(const SP_Platform* platform, SP_TimerFns* timer_fns);
void gpu_destroy_platform(SP_Platform* const platform);
void gpu_destroy_platform_fns(SP_PlatformFns* const platform_fns);

void gpu_allocate(const SP_Device* device, uint64_t size, int64_t memory_space,
                  SP_DeviceMemoryBase* mem);
void gpu_deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem);
void* gpu_host_memory_allocate(const SP_Device* device, uint64_t size);
void gpu_host_memory_deallocate(const SP_Device* device, void* mem);
TF_Bool gpu_get_allocator_stats(const SP_Device* device,
                                SP_AllocatorStats* stats);
TF_Bool gpu_device_memory_usage(const SP_Device* device, int64_t* free,
                                int64_t* total);
void gpu_create_stream(const SP_Device* device, SP_Stream* stream,
                       TF_Status* status);
void gpu_destroy_stream(const SP_Device* device, SP_Stream stream);
void gpu_create_stream_dependency(const SP_Device* device, SP_Stream dependent,
                                  SP_Stream other, TF_Status* status);
void gpu_get_stream_status(const SP_Device* device, SP_Stream stream,
                           TF_Status* status);
void gpu_create_event(const SP_Device* device, SP_Event* event,
                      TF_Status* status);
void gpu_destroy_event(const SP_Device* device, SP_Event event);
SE_EventStatus gpu_get_event_status(const SP_Device* device, SP_Event event);
void gpu_record_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                      TF_Status* status);
void gpu_wait_for_event(const SP_Device* const device, SP_Stream stream,
                        SP_Event event, TF_Status* const status);
void gpu_create_timer(const SP_Device* device, SP_Timer* timer,
                      TF_Status* status);
void gpu_destroy_timer(const SP_Device* device, SP_Timer timer);
void gpu_start_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                     TF_Status* status);
void gpu_stop_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                    TF_Status* status);
void gpu_memcpy_dtoh(const SP_Device* device, SP_Stream stream, void* host_dst,
                     const SP_DeviceMemoryBase* device_src, uint64_t size,
                     TF_Status* status);
void gpu_memcpy_htod(const SP_Device* device, SP_Stream stream,
                     SP_DeviceMemoryBase* device_dst, const void* host_src,
                     uint64_t size, TF_Status* status);
void gpu_memcpy_dtod(const SP_Device* device, SP_Stream stream,
                     SP_DeviceMemoryBase* device_dst,
                     const SP_DeviceMemoryBase* device_src, uint64_t size,
                     TF_Status* status);
void gpu_sync_memcpy_dtoh(const SP_Device* device, void* host_dst,
                          const SP_DeviceMemoryBase* device_src, uint64_t size,
                          TF_Status* status);
void gpu_sync_memcpy_htod(const SP_Device* device,
                          SP_DeviceMemoryBase* device_dst, const void* host_src,
                          uint64_t size, TF_Status* status);
void gpu_sync_memcpy_dtod(const SP_Device* device,
                          SP_DeviceMemoryBase* device_dst,
                          const SP_DeviceMemoryBase* device_src, uint64_t size,
                          TF_Status* status);
void gpu_block_host_for_event(const SP_Device* device, SP_Event event,
                              TF_Status* status);
void gpu_block_host_until_done(const SP_Device* device, SP_Stream stream,
                               TF_Status* status);
void gpu_synchronize_all_activity(const SP_Device* device, TF_Status* status);
void gpu_mem_zero(const SP_Device* device, SP_Stream stream,
                  SP_DeviceMemoryBase* location, uint64_t size,
                  TF_Status* status);
void gpu_memset(const SP_Device* device, SP_Stream stream,
                SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
                TF_Status* status);
void gpu_memset32(const SP_Device* device, SP_Stream stream,
                  SP_DeviceMemoryBase* location, uint32_t pattern,
                  uint64_t size, TF_Status* status);
TF_Bool gpu_host_callback(const SP_Device* device, SP_Stream stream,
                          SE_StatusCallbackFn callback_fn, void* callback_arg);
}  // namespace itex
#endif  // USING_NEXTPLUGGABLE_DEVICE
#endif  // ITEX_CORE_DEVICES_GPU_GPU_DEVICE_PLUGIN_H_
