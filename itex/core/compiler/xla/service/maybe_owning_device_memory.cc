/* Copyright (c) 2023 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/service/maybe_owning_device_memory.h"

#include <utility>

#include "absl/types/variant.h"

namespace itex_xla {

se::DeviceMemoryBase MaybeOwningDeviceMemory::AsDeviceMemoryBase() const {
  if (HasOwnership()) {
    return *absl::get<se::OwningDeviceMemory>(mem_);
  } else {
    return absl::get<se::DeviceMemoryBase>(mem_);
  }
}

bool MaybeOwningDeviceMemory::HasOwnership() const {
  return absl::holds_alternative<se::OwningDeviceMemory>(mem_);
}

absl::optional<se::OwningDeviceMemory> MaybeOwningDeviceMemory::Release() {
  if (!HasOwnership()) {
    return {};
  }
  return std::move(absl::get<se::OwningDeviceMemory>(mem_));
}

const se::OwningDeviceMemory* MaybeOwningDeviceMemory::AsOwningDeviceMemory()
    const {
  return HasOwnership() ? &absl::get<se::OwningDeviceMemory>(mem_) : nullptr;
}

}  // namespace itex_xla
