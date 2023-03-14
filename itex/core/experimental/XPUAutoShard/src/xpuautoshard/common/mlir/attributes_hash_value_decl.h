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

#pragma once

// NOTE: Include this header before llvm/ADT/Hashing.h

// Declare custom hash_value functions earlier enough for
// later uses to resolve. This is due to the fact that LLVM
// doesn't define a match-for-all templatized hash_value function,
// which prevents the compiler from resolving custom hash_value
// functions declared and defined after Hashing.h is included.

#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/ref_base.h"
#include "xpuautoshard/common/sharding_property.h"

namespace llvm {
class hash_code;
hash_code hash_value(const float& v);
hash_code hash_value(const ::as::ShardingPropertyRef& prop);
hash_code hash_value(const ::as::DeviceInfo& info);
hash_code hash_value(const ::as::Device& device);
}  // namespace llvm

#include "llvm/ADT/Hashing.h"
