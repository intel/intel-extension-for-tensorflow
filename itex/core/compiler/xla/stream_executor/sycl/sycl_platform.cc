/* Copyright (c) 2023 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/stream_executor/sycl/sycl_platform.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/const_init.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "itex/core/compiler/xla/stream_executor/lib/error.h"
#include "itex/core/compiler/xla/stream_executor/lib/initialize.h"
#include "itex/core/compiler/xla/stream_executor/lib/status.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_executor.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor {
namespace gpu {
namespace {

// Synchronize with spinlocks.
const char kScheduleSpinString[] = "spin";
// Synchronize with spinlocks that also call CPU yield instructions.
const char kScheduleYieldString[] = "yield";
// Synchronize with a "synchronization primitive" (e.g. mutex).
const char kScheduleBlockingSyncString[] = "blocking_sync";

const DeviceOptions GetDeviceOptionsFromEnv() {
  const char* gpu_schedule_string =
      std::getenv("TF_CUDA_PLATFORM_GPU_DEVICE_SCHEDULE");

  if (gpu_schedule_string == nullptr) {
    return DeviceOptions::Default();
  }

  unsigned device_flags = 0;
  if (strcmp(kScheduleSpinString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleSpin;
  } else if (strcmp(kScheduleYieldString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleYield;
  } else if (strcmp(kScheduleBlockingSyncString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleBlockingSync;
  } else {
    ITEX_LOG(QFATAL) << "Unknown option for environment variable "
                        "TF_CUDA_PLATFORM_GPU_DEVICE_SCHEDULE "
                     << gpu_schedule_string << " should be one of {"
                     << kScheduleBlockingSyncString << ", "
                     << kScheduleSpinString << ", " << kScheduleYieldString
                     << "}";
  }

  return DeviceOptions(device_flags);
}

}  // namespace

SyclPlatform::SyclPlatform()
    : name_("sycl"), min_numa_node_(0), limit_numa_node_(0) {}

SyclPlatform::~SyclPlatform() {}

// Due to legacy issues in user code, we can't currently call InpectNumaNodes
// at module initialization time, because non-GPU programs still include this
// plugin via various methods, so instead, it has to be init-on-reference.
void SyclPlatform::InspectNumaNodes() {
  // To get NUMA node information, we need to create all executors, so we can
  // examine their device descriptions to see their bus assignments.
  static absl::once_flag once;
  absl::call_once(once, [&] {
    for (int i = 0; i < VisibleDeviceCount(); i++) {
      StreamExecutor* exec = *ExecutorForDevice(i);
      if (i == 0) {
        // NUMA nodes may not start at 0, so set the minimum node  based on the
        // first executor we see.
        min_numa_node_ = exec->GetDeviceDescription().numa_node();
        limit_numa_node_ = min_numa_node_ + 1;
      } else {
        min_numa_node_ =
            std::min(min_numa_node_, exec->GetDeviceDescription().numa_node());
        limit_numa_node_ = std::max(
            limit_numa_node_, exec->GetDeviceDescription().numa_node() + 1);
      }
    }
  });
}

int SyclPlatform::BusCount() {
  InspectNumaNodes();
  return limit_numa_node_ - min_numa_node_;
}

int SyclPlatform::DeviceToBus(int device_ordinal) {
  StreamExecutor* exec = *ExecutorForDevice(device_ordinal);
  return exec->GetDeviceDescription().numa_node() - min_numa_node_;
}

port::StatusOr<StreamExecutor*> SyclPlatform::FirstExecutorForBus(
    int bus_ordinal) {
  InspectNumaNodes();
  ITEX_CHECK_LT(bus_ordinal, BusCount())
      << "bus ordinal out of available range";
  for (int i = 0; i < VisibleDeviceCount(); i++) {
    if (DeviceToBus(i) == bus_ordinal) {
      return *ExecutorForDevice(i);
    }
  }

  return port::Status(
      itex::error::NOT_FOUND,
      absl::StrFormat("Executor for bus %d not found.", bus_ordinal));
}

Platform::Id SyclPlatform::id() const { return gpu::kSyclPlatformId; }

int SyclPlatform::VisibleDeviceCount() const {
  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.
  int device_count;
  ITEX_GPUGetDeviceCount(&device_count);
  return device_count;
}

const std::string& SyclPlatform::Name() const { return name_; }

port::StatusOr<std::unique_ptr<DeviceDescription>>
SyclPlatform::DescriptionForDevice(int ordinal) const {
  return GpuExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<StreamExecutor*> SyclPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = GetDeviceOptionsFromEnv();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> SyclPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = GetDeviceOptionsFromEnv();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> SyclPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  if (config.gpu_stream) {
    // If the GPU stream was provided, it's not possible to get-or-create a
    // stream with a required pointer: so we are looking for previously
    // allocated streams.
    return executor_cache_.Get(config);
  }
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
SyclPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = std::make_unique<StreamExecutor>(
      this, std::make_unique<GpuExecutor>(config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status(
        itex::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for SYCL device ordinal %d: %s",
            config.ordinal, init_status.ToString()));
  }

  return std::move(executor);
}

void SyclPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  ITEX_LOG(FATAL) << "not yet implemented: register SYCL trace listener";
}

void SyclPlatform::UnregisterTraceListener(TraceListener* listener) {
  ITEX_LOG(FATAL) << "not yet implemented: unregister SYCL trace listener";
}

}  // namespace gpu

static void InitializeSyclPlatform() {
  // Disabling leak checking, MultiPlatformManager does not destroy its
  // registered platforms.
  std::unique_ptr<gpu::SyclPlatform> platform(new gpu::SyclPlatform);
  SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(sycl_platform,
                            stream_executor::InitializeSyclPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(sycl_platform, multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     sycl_platform);
