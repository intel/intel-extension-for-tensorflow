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

#ifndef ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_H_
#define ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "itex/core/compiler/xla/stream_executor/executor_cache.h"
#include "itex/core/compiler/xla/stream_executor/lib/statusor.h"
#include "itex/core/compiler/xla/stream_executor/multi_platform_manager.h"
#include "itex/core/compiler/xla/stream_executor/platform.h"
#include "itex/core/compiler/xla/stream_executor/platform/port.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor_internal.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "itex/core/compiler/xla/stream_executor/trace_listener.h"

namespace stream_executor {
namespace gpu {

extern const Platform::Id kSyclPlatformId;
// Cuda-specific platform plugin, registered as a singleton value via module
// initializer.
class SyclPlatform : public Platform {
 public:
  SyclPlatform();
  ~SyclPlatform() override;

  // CudaPlatform-specific functionality
  // Returns the number of distinct buses / NUMA nodes on the machine.
  int BusCount();

  // Returns the bus/NUMA node for the specified device ordinal.
  int DeviceToBus(int device_ordinal);

  // Returns the lowest-ordinal-number StreamExecutor on the specified bus.
  port::StatusOr<StreamExecutor*> FirstExecutorForBus(int bus_ordinal);

  // Platform interface implementation:
  // Returns the same value as kCudaPlatform above.
  Platform::Id id() const override;

  // Returns -1 as a sentinel on internal failure (and logs the error).
  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  port::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

  port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& config) override;

  port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;

  port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override;

  void UnregisterTraceListener(TraceListener* listener) override;

 private:
  // Determines the number of NUMA nodes and the assignment of executor to each.
  void InspectNumaNodes();

  // This platform's name.
  std::string name_;

  // Cache of created executors.
  ExecutorCache executor_cache_;

  // The smallest NUMA node value for any device managed by this machine
  // manager. Used, along with limit_numa_node_, to convert NUMA nodes into bus
  // ordinals. The NUMA node space occupied by GPUs is assumed to be dense./
  int min_numa_node_;

  // Larger than the NUMA node value for any device managed by this machine
  // manager.
  int limit_numa_node_;

  SE_DISALLOW_COPY_AND_ASSIGN(SyclPlatform);
};

}  // namespace gpu

}  // namespace stream_executor

#endif  // ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_H_
