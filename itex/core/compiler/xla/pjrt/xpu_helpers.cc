/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/pjrt/xpu_helpers.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "itex/core/compiler/xla/client/client_library.h"
#include "itex/core/compiler/xla/service/platform_util.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/env_var.h"

namespace itex_xla {

// Builds an xla::LocalClient for the GPU platform.
StatusOr<LocalClient*> GetXpuXlaClient(
    const std::optional<std::string>& platform_name,
    const std::optional<std::set<int>>& allowed_devices) {
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      PlatformUtil::GetPlatform(platform_name ? *platform_name : "sycl"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("No visible XPU devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  options.set_allowed_devices(allowed_devices);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

}  // namespace itex_xla
