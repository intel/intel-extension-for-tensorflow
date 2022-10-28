/* Copyright (c) 2022 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_PROFILER_UTILS_CONTEXT_TYPES_H_
#define ITEX_CORE_PROFILER_UTILS_CONTEXT_TYPES_H_

#include <cstdint>

namespace itex {
namespace profiler {

enum class ContextType : int {
  kGeneric = 0,
  kLegacy,
  kTfExecutor,
  kTfrtExecutor,
  kSharedBatchScheduler,
  kPjRt,
  kAdaptiveSharedBatchScheduler,
  kTfrtTpuRuntime,
  kTpuEmbeddingEngine,
  kGpuLaunch,
  kBatcher,
  kLastContextType = ContextType::kBatcher,
};

// In XFlow we encode context type as flow category as 6 bits.
static_assert(static_cast<int>(ContextType::kLastContextType) < 64,
              "Should have less than 64 categories.");

const char* GetContextTypeString(ContextType context_type);

inline ContextType GetSafeContextType(uint32_t context_type) {
  if (context_type > static_cast<uint32_t>(ContextType::kLastContextType)) {
    return ContextType::kGeneric;
  }
  return static_cast<ContextType>(context_type);
}

}  // namespace profiler
}  // namespace itex

#endif  // ITEX_CORE_PROFILER_UTILS_CONTEXT_TYPES_H_
