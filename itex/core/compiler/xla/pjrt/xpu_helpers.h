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

#ifndef ITEX_CORE_COMPILER_XLA_PJRT_XPU_HELPERS_H_
#define ITEX_CORE_COMPILER_XLA_PJRT_XPU_HELPERS_H_

#include <memory>
#include <optional>
#include <set>
#include <string>

#include "absl/types/span.h"
#include "itex/core/compiler/xla/client/local_client.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor.h"
#include "itex/core/compiler/xla/types.h"

namespace itex_xla {

// Builds an xla::LocalClient for the GPU platform.
StatusOr<LocalClient*> GetXpuXlaClient(
    const std::optional<std::string>& platform_name,
    const std::optional<std::set<int>>& allowed_devices);

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_PJRT_XPU_HELPERS_H_
