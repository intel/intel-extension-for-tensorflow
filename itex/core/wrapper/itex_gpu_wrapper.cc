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

#include <cpuid.h>
#include <dlfcn.h>
#include <stdlib.h>

#include "itex/core/devices/device_backend_util.h"
#include "itex/core/utils/logging.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#ifdef USING_NEXTPLUGGABLE_DEVICE
#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#endif

static void* handle;
static void* LoadGpuLibrary() __attribute__((constructor));
static void UnloadGpuLibrary() __attribute__((destructor));

void* LoadGpuLibrary() {
  setenv("ENABLE_PJRT_COMPATIBILITY", "1", 0);
  handle = dlopen("libitex_gpu_internal.so", RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    itex_freeze_backend(ITEX_BACKEND_CPU);
    const char* error_msg = dlerror();
    ITEX_LOG(WARNING) << "Could not load dynamic library: " << error_msg;
  } else {
    itex_freeze_backend(ITEX_BACKEND_GPU);
    ITEX_LOG(INFO) << "Intel Extension for Tensorflow* GPU backend is loaded.";
  }
  return handle;
}

void UnloadGpuLibrary() {
  if (handle) {
    dlclose(handle);
  }
}

// Fake functions
void xpu_device_count(const SP_Platform* platform, int* device_count,
                      TF_Status* status) {
  ITEX_LOG(ERROR) << "Could not load Intel Extension for Tensorflow* GPU "
                     "backend, GPU will not be used.";
}

void xpu_create_device(const SP_Platform* platform,
                       SE_CreateDeviceParams* params, TF_Status* const status) {
}

void xpu_destroy_device(const SP_Platform* platform, SP_Device* device) {}

void xpu_create_device_fns(const SP_Platform* platform,
                           SE_CreateDeviceFnsParams* params,
                           TF_Status* status) {}

void xpu_destroy_device_fns(const SP_Platform* platform,
                            SP_DeviceFns* device_fns) {}

void xpu_allocate(const SP_Device* device, uint64_t size, int64_t memory_space,
                  SP_DeviceMemoryBase* mem) {}

void xpu_deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem) {}

void* xpu_host_memory_allocate(const SP_Device* device, uint64_t size) {
  return nullptr;
}

void xpu_host_memory_deallocate(const SP_Device* device, void* mem) {}

TF_Bool xpu_get_allocator_stats(const SP_Device* device,
                                SP_AllocatorStats* stats) {
  return false;
}

TF_Bool xpu_device_memory_usage(const SP_Device* device, int64_t* free,
                                int64_t* total) {
  return false;
}

void xpu_create_stream(const SP_Device* device, SP_Stream* stream,
                       TF_Status* status) {}

void xpu_destroy_stream(const SP_Device* device, SP_Stream stream) {}

void xpu_create_stream_dependency(const SP_Device* device, SP_Stream dependent,
                                  SP_Stream other, TF_Status* status) {}

void xpu_get_stream_status(const SP_Device* device, SP_Stream stream,
                           TF_Status* status) {}

void xpu_create_event(const SP_Device* device, SP_Event* event,
                      TF_Status* status) {}

void xpu_destroy_event(const SP_Device* device, SP_Event event) {}

SE_EventStatus xpu_get_event_status(const SP_Device* device, SP_Event event) {
  return SE_EVENT_COMPLETE;
}

void xpu_record_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                      TF_Status* status) {}

void xpu_wait_for_event(const SP_Device* const device, SP_Stream stream,
                        SP_Event event, TF_Status* const status) {}

void xpu_create_timer(const SP_Device* device, SP_Timer* timer,
                      TF_Status* status) {}

void xpu_destroy_timer(const SP_Device* device, SP_Timer timer) {}

void xpu_start_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                     TF_Status* status) {}

void xpu_stop_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                    TF_Status* status) {}

void xpu_memcpy_dtoh(const SP_Device* device, SP_Stream stream, void* host_dst,
                     const SP_DeviceMemoryBase* device_src, uint64_t size,
                     TF_Status* status) {}

void xpu_memcpy_htod(const SP_Device* device, SP_Stream stream,
                     SP_DeviceMemoryBase* device_dst, const void* host_src,
                     uint64_t size, TF_Status* status) {}

void xpu_memcpy_dtod(const SP_Device* device, SP_Stream stream,
                     SP_DeviceMemoryBase* device_dst,
                     const SP_DeviceMemoryBase* device_src, uint64_t size,
                     TF_Status* status) {}

void xpu_sync_memcpy_dtoh(const SP_Device* device, void* host_dst,
                          const SP_DeviceMemoryBase* device_src, uint64_t size,
                          TF_Status* status) {}

void xpu_sync_memcpy_htod(const SP_Device* device,
                          SP_DeviceMemoryBase* device_dst, const void* host_src,
                          uint64_t size, TF_Status* status) {}

void xpu_sync_memcpy_dtod(const SP_Device* device,
                          SP_DeviceMemoryBase* device_dst,
                          const SP_DeviceMemoryBase* device_src, uint64_t size,
                          TF_Status* status) {}

void xpu_block_host_for_event(const SP_Device* device, SP_Event event,
                              TF_Status* status) {}

void xpu_block_host_until_done(const SP_Device* device, SP_Stream stream,
                               TF_Status* status) {}

void xpu_synchronize_all_activity(const SP_Device* device, TF_Status* status) {}

void xpu_mem_zero(const SP_Device* device, SP_Stream stream,
                  SP_DeviceMemoryBase* location, uint64_t size,
                  TF_Status* status) {}

void xpu_memset(const SP_Device* device, SP_Stream stream,
                SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
                TF_Status* status) {}

void xpu_memset32(const SP_Device* device, SP_Stream stream,
                  SP_DeviceMemoryBase* location, uint32_t pattern,
                  uint64_t size, TF_Status* status) {}

TF_Bool xpu_host_callback(const SP_Device* device, SP_Stream stream,
                          SE_StatusCallbackFn callback_fn, void* callback_arg) {
  return false;
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
                                 SP_StreamExecutor* stream_executor) {}

struct SP_Timer_st {
  explicit SP_Timer_st(int id) : timer_handle(id) {}
  int timer_handle;
};

uint64_t nanoseconds(SP_Timer timer) { return timer->timer_handle; }

void xpu_create_timer_fns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                          TF_Status* const status) {
  timer_fns->nanoseconds = nanoseconds;
}

void xpu_destroy_timer_fns(const SP_Platform* platform,
                           SP_TimerFns* timer_fns) {}

void xpu_destroy_platform(SP_Platform* const platform) {}

void xpu_destroy_platform_fns(SP_PlatformFns* const platform_fns) {}

void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf,
                        const TF_GrapplerItem* tf_item,
                        TF_Buffer* optimized_graph_buf, TF_Status* tf_status) {
  const char* device_name = "XPU";
  ITEX_VLOG(2) << "Optimizer_Optimize  device is " << device_name;
}

void gpu_start(const TP_Profiler* profiler, TF_Status* status) {}
void gpu_stop(const TP_Profiler* profiler, TF_Status* status) {}

void gpu_collect_data_xspace(const TP_Profiler* profiler, uint8_t* buffer,
                             size_t* size_in_bytes, TF_Status* status) {}

void gpu_destroy_profiler(TP_Profiler* profiler) {}

void gpu_destroy_profiler_fns(TP_ProfilerFns* profiler_fns) {}

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
  typedef void (*se_initplugin_internal)(SE_PlatformRegistrationParams*,
                                         TF_Status*);

  if (handle) {
    auto se_initplugin = reinterpret_cast<se_initplugin_internal>(
        dlsym(handle, "SE_InitPlugin_Internal"));
    if (se_initplugin != nullptr) {
      se_initplugin(params, status);
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  } else {
    // fill SE_PlatformRegistrationParams with fake functions to pass the
    // parameter validate.
    params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
    params->platform->name = DEVICE_XPU_NAME;
    params->platform->type = itex::DEVICE_XPU;
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
}

void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status) {
  typedef void (*tf_initgraph_internal)(TP_OptimizerRegistrationParams*,
                                        TF_Status*);

  // void* handle = LoadGpuLibrary();
  if (handle) {
    auto tf_initgraph = reinterpret_cast<tf_initgraph_internal>(
        dlsym(handle, "TF_InitGraph_Internal"));
    if (tf_initgraph != nullptr) {
      tf_initgraph(params, status);
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  } else {
    // fill TP_OptimizerRegistrationParams with fake functions to pass the
    // parameter validate.
    params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
    params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
    params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
    params->device_type = itex::DEVICE_XPU;
    params->optimizer->optimize_func = Optimizer_Optimize;
  }
}

void TF_InitKernel() {
  typedef void (*tf_initkernel_internal)();

  // void* handle = LoadGpuLibrary();
  if (handle) {
    auto tf_initkernel = reinterpret_cast<tf_initkernel_internal>(
        dlsym(handle, "TF_InitKernel_Internal"));
    if (*tf_initkernel != nullptr) {
      tf_initkernel();
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  }
}

void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status* status) {
  typedef void (*tf_initptofiler_internal)(TF_ProfilerRegistrationParams*,
                                           TF_Status*);

  // void* handle = LoadGpuLibrary();
  if (handle) {
    auto tf_initptofiler = reinterpret_cast<tf_initptofiler_internal>(
        dlsym(handle, "TF_InitProfiler_Internal"));
    if (*tf_initptofiler != nullptr) {
      tf_initptofiler(params, status);
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  } else {
    // fill TF_ProfilerRegistrationParams with fake functions to pass the
    // parameter validate.
    params->struct_size = TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE;
    params->profiler->struct_size = TP_PROFILER_STRUCT_SIZE;
    params->profiler_fns->struct_size = TP_PROFILER_FNS_STRUCT_SIZE;
    params->profiler->device_type = "XPU";
    params->profiler_fns->start = gpu_start;
    params->profiler_fns->stop = gpu_stop;
    params->profiler_fns->collect_data_xspace = gpu_collect_data_xspace;
    params->destroy_profiler = gpu_destroy_profiler;
    params->destroy_profiler_fns = gpu_destroy_profiler_fns;
  }
}

#ifdef USING_NEXTPLUGGABLE_DEVICE
int32_t tfnpd_get_device_count(TF_Status* status) {
  ITEX_LOG(ERROR) << "Could not load Intel Extension for Tensorflow* GPU "
                     "backend, GPU will not be used.";
  return 0;
}

void tfnpd_init_plugin_internal_device_states(TF_Status* status) {
  // TF_CreateAndSetPjRtCApiClient("XPU", status);
}

void tfnpd_xla_shape_to_device_shape_representation(
    XLA_Shape* serialized_xla_shape, int data_type, bool use_fast_memory,
    XLA_LayoutPreference layout_preference, XLA_Shape* serialized_device_shape,
    TF_Status* tf_status) {}

const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params,
                                  TF_Status* tf_status) {
  typedef const TFNPD_Api* (*tfnpd_init_internal)(TFNPD_PluginParams*,
                                                  TF_Status*);
  if (handle) {
    auto tfnpd_init = reinterpret_cast<tfnpd_init_internal>(
        dlsym(handle, "TFNPD_InitPlugin_Internal"));
    if (*tfnpd_init != nullptr) {
      return tfnpd_init(params, tf_status);
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  } else {
    params->struct_size = TFNPD_PLUGIN_PARAMS_STRUCT_SIZE;
    params->device_type = "XPU";
    params->compilation_device_name = "XLA_GPU_JIT";
    params->is_pluggable_device = true;
    params->use_pjrt_on_demand_compile = false;
    params->priority = 0;
    static TFNPD_Api tfnpd_api;

    tfnpd_api.TFNPD_GetDeviceCount = tfnpd_get_device_count;
    tfnpd_api.TFNPD_InitPluginInternalDeviceStates =
        tfnpd_init_plugin_internal_device_states;
    tfnpd_api.TFNPD_XlaShapeToDeviceShapeRepresentation =
        tfnpd_xla_shape_to_device_shape_representation;
    return &tfnpd_api;
  }
}

#ifdef __cplusplus
extern "C" {
#endif
const PJRT_Api* GetPjrtApi();
#ifdef __cplusplus
}
#endif

const PJRT_Api* GetPjrtApi() {
  typedef const PJRT_Api* (*get_pjrt_api_internal)();
  if (handle) {
    auto get_pjrt_api = reinterpret_cast<get_pjrt_api_internal>(
        dlsym(handle, "GetPjrtApi_Internal"));
    if (*get_pjrt_api != nullptr) {
      return get_pjrt_api();
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
      return nullptr;
    }
  } else {
    return nullptr;
  }
}
#endif  // USING_NEXTPLUGGABLE_DEVICE
