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

#ifndef CC_BUILD
#include "itex/core/profiler/gpu_profiler.h"
#endif

#include <string>
#include <vector>

#include "itex/core/profiler/gpu_collector.h"
#include "itex/core/profiler/utils.h"
#include "itex/core/profiler/utils/xplane_utils.h"
#include "itex/core/profiler/ze_tracer.h"
#include "itex/core/profiler/ze_utils.h"
#include "itex/core/utils/annotation_stack.h"
#include "itex/core/utils/hw_info.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/strcat.h"
#include "protos/xplane.pb.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

inline std::string GpuPlaneName(int32_t device_ordinal) {
  return itex::strings::StrCat("/device:GPU:", device_ordinal);
}

static ZeTracer* tracer = nullptr;

static bool IsItexProfilerEnabled() {
  std::string enable_trace_layer = utils::GetEnv("ZE_ENABLE_TRACING_LAYER");
  std::string use_cycles_per_second = utils::GetEnv("UseCyclesPerSecondTimer");
  std::string enable_tf_profiler = utils::GetEnv("ENABLE_TF_PROFILER");
  if (enable_trace_layer == "1" && use_cycles_per_second == "1" &&
      enable_tf_profiler == "1") {
    return true;
  } else {
    return false;
  }
}

void EnableProfiling() {
  assert(zeInit(ZE_INIT_FLAG_GPU_ONLY) == ZE_RESULT_SUCCESS);
  std::string enable_immediate_commmand_list =
      utils::GetEnv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS");
  if (enable_immediate_commmand_list == "0") {
    utils::ImmediateCommandListDisabled();
  } else if (enable_immediate_commmand_list.empty()) {
    if (!IsXeHPC()) {
      utils::ImmediateCommandListDisabled();
    }
  }

  uint32_t flags = 0;
  flags |= (1 << TRACE_DEVICE_TIMING);
  flags |= (1 << TRACE_HOST_RUNTIME_TIMING);
  std::string log_file = std::string();
  tracer = ZeTracer::Create(TraceOptions(flags, log_file));
  itex::AnnotationStack::Enable(true);
}

void DisableProfiling() {
  if (tracer != nullptr) {
    delete tracer;
  }
}

void __attribute__((destructor)) Unload() { DisableProfiling(); }

void gpu_start(const TP_Profiler* profiler, TF_Status* status) {
  if (!IsItexProfilerEnabled()) {
    ITEX_LOG(WARNING)
        << "******************************Intel Extension For TensorFlow "
           "profiler "
           "Warning***************************************************";
    ITEX_LOG(WARNING)
        << "Intel Extension For TensorFlow profiler not enabled, if you want "
           "to enable it, please set "
           "environment as :\nexport ZE_ENABLE_TRACING_LAYER=1 \nexport "
           "UseCyclesPerSecondTimer=1\nexport ENABLE_TF_PROFILER=1";
    ITEX_LOG(WARNING) << "*****************************************************"
                         "*****************"
                         "********************************";
  }
  if (tracer != nullptr) {
    tracer->Start();
  }
}
void gpu_stop(const TP_Profiler* profiler, TF_Status* status) {
  if (tracer != nullptr) {
    tracer->Stop();
  }
}

static void NormalizeTimeStamps(itex::profiler::XPlaneBuilder* plane,
                                uint64_t start_walltime_ns) {
  plane->ForEachLine([&](itex::profiler::XLineBuilder line) {
    line.SetTimestampNs(start_walltime_ns);
  });
}

void gpu_collect_data_xspace(const TP_Profiler* profiler, uint8_t* buffer,
                             size_t* size_in_bytes, TF_Status* status) {
  int device_count = 0;
  ITEX_GPUGetDeviceCount(&device_count);

  std::vector<itex::profiler::PerDeviceCollector> per_device_collector;
  itex::XSpace space;

  if (IsItexProfilerEnabled()) {
    for (int i = 0; i < device_count; i++) {
      per_device_collector.emplace_back(i, tracer->GetStartWallTime(),
                                        tracer->GetGPUStartTime());
      std::string name = GpuPlaneName(i);
      itex::profiler::XPlaneBuilder device_plane(
          itex::profiler::FindOrAddMutablePlaneWithName(&space, name));
      device_plane.SetId(i);
      per_device_collector[i].Flush(&device_plane);
      NormalizeTimeStamps(&device_plane, tracer->GetStartWallTime());
    }
  }

  *size_in_bytes = space.ByteSizeLong();
  if (buffer == nullptr) {
    return;
  }
  space.SerializeToArray(buffer, space.ByteSizeLong());
}

void gpu_destroy_profiler(TP_Profiler* profiler) {}

void gpu_destroy_profiler_fns(TP_ProfilerFns* profiler_fns) {}

#ifndef CC_BUILD
void TF_InitProfiler_Internal(TF_ProfilerRegistrationParams* params,
                              TF_Status* status) {
#else
void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status* status) {
#endif
  params->struct_size = TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->profiler->struct_size = TP_PROFILER_STRUCT_SIZE;
  params->profiler_fns->struct_size = TP_PROFILER_FNS_STRUCT_SIZE;

  params->profiler->device_type = "XPU";

  params->profiler_fns->start = gpu_start;
  params->profiler_fns->stop = gpu_stop;
  params->profiler_fns->collect_data_xspace = gpu_collect_data_xspace;
  params->destroy_profiler = gpu_destroy_profiler;
  params->destroy_profiler_fns = gpu_destroy_profiler_fns;
  if (IsItexProfilerEnabled()) {
    EnableProfiling();
  }
}
