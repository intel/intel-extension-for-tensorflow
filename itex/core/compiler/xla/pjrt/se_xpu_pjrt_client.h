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

#ifndef ITEX_CORE_COMPILER_XLA_PJRT_SE_XPU_PJRT_CLIENT_H_
#define ITEX_CORE_COMPILER_XLA_PJRT_SE_XPU_PJRT_CLIENT_H_
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "itex/core/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "itex/core/compiler/xla/pjrt/xpu_helpers.h"
#include "itex/core/compiler/xla/statusor.h"

namespace itex_xla {

class StreamExecutorXpuDevice : public PjRtStreamExecutorDevice {
 public:
  StreamExecutorXpuDevice(int id,
                          std::unique_ptr<LocalDeviceState> local_device_state,
                          std::string device_kind, std::string device_vendor,
                          int node_id);

  absl::string_view device_vendor();
  absl::string_view ToString() const override;

 private:
  std::string device_vendor_;
  std::string to_string_;
};

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorXpuClient(
    bool asynchronous, int node_id,
    const std::optional<std::set<int>>& allowed_devices = std::nullopt,
    std::optional<std::string> platform_name = std::nullopt);

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_PJRT_SE_XPU_PJRT_CLIENT_H_
