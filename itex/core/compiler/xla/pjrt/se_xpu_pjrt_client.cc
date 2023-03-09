/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/compiler/xla/pjrt/se_xpu_pjrt_client.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "itex/core/compiler/xla/client/client_library.h"
#include "itex/core/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "itex/core/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "itex/core/compiler/xla/service/platform_util.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/compiler/xla/stream_executor/device_memory.h"
#include "itex/core/devices/gpu/gpu_pool_allocator.h"

namespace itex_xla {
namespace {

class StreamExecutorXpuClient : public itex_xla::PjRtStreamExecutorClient {
 public:
  using itex_xla::PjRtStreamExecutorClient::PjRtStreamExecutorClient;

  itex_xla::StatusOr<itex_xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;
};

itex_xla::StatusOr<itex_xla::DeviceAssignment>
StreamExecutorXpuClient::GetDefaultDeviceAssignment(int num_replicas,
                                                    int num_partitions) const {
  if (num_partitions == 1 && num_replicas <= addressable_devices().size()) {
    itex_xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = addressable_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

// Builds a LocalDeviceState for each GPU present.
StatusOr<std::map<int, std::unique_ptr<LocalDeviceState>>>
BuildLocalDeviceStates(LocalClient* xla_client, bool asynchronous) {
  std::map<int, std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<LocalDeviceState>(
            executor, xla_client, LocalDeviceState::kComputeSynchronized,
            /*max_inflight_computations=*/32,
            /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& ordinal_and_device : local_device_states) {
    const se::DeviceDescription& description =
        ordinal_and_device.second->executor()->GetDeviceDescription();
    auto device = std::make_unique<StreamExecutorXpuDevice>(
        ordinal_and_device.first, std::move(ordinal_and_device.second),
        description.name(), description.device_vendor(),
        /*node_id=*/0);
    devices.push_back(std::move(device));
  }
  return devices;
}
}  // namespace

StreamExecutorXpuDevice::StreamExecutorXpuDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state,
    std::string device_kind, std::string device_vendor, int node_id)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               std::move(device_kind), node_id),
      device_vendor_(std::move(device_vendor)) {
  attributes_ = {
      {"device_vendor", "Intel"},
  };
  to_string_ = absl::StrFormat(
      "StreamExecutorXpuDevice(id=%i, process_index=%i)", id, process_index());
}

absl::string_view StreamExecutorXpuDevice::device_vendor() {
  return device_vendor_;
}

absl::string_view StreamExecutorXpuDevice::ToString() const {
  return to_string_;
}

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorXpuClient(
    bool asynchronous, int node_id,
    const std::optional<std::set<int>>& allowed_devices,
    std::optional<std::string> platform_name) {
  TF_ASSIGN_OR_RETURN(LocalClient * xla_client,
                      GetXpuXlaClient(platform_name, allowed_devices));
  std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(local_device_states,
                      BuildLocalDeviceStates(xla_client, asynchronous));
  // EnablePeerAccess(xla_client->backend().stream_executors());
  // TF_ASSIGN_OR_RETURN(
  //     auto allocator,
  //     GetStreamExecutorXpuDeviceAllocator(
  //         xla_client->platform(), local_device_states));

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
  // if (distributed_client) {
  //   TF_RETURN_IF_ERROR(BuildDistributedDevices(
  //       std::move(local_device_states), std::move(distributed_client),
  //       node_id, &devices, gpu_run_options.get()));
  // } else {
  devices = BuildLocalDevices(std::move(local_device_states));
  // }
  return std::unique_ptr<PjRtClient>(std::make_unique<StreamExecutorXpuClient>(
      XpuName(), xla_client, std::move(devices),
      /*node_id=*/node_id, nullptr, nullptr,
      /*should_stage_host_to_device_transfers=*/true,
      /*gpu_run_options=*/std::move(gpu_run_options)));
}

}  // namespace itex_xla
