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

#include "itex/core/utils/tracing.h"

#include <stdlib.h>

#include <array>
#include <atomic>

#include "itex/core/utils/hash.h"

namespace itex {
namespace tracing {
namespace {
std::atomic<uint64> unique_arg{1};
}  // namespace

const char* GetEventCategoryName(EventCategory category) {
  switch (category) {
    case EventCategory::kScheduleClosure:
      return "ScheduleClosure";
    case EventCategory::kRunClosure:
      return "RunClosure";
    case EventCategory::kCompute:
      return "Compute";
    default:
      return "Unknown";
  }
}

std::array<const EventCollector*, GetNumEventCategories()>
    EventCollector::instances_;

void SetEventCollector(EventCategory category,
                       const EventCollector* collector) {
  EventCollector::instances_[static_cast<unsigned>(category)] = collector;
}

uint64 GetUniqueArg() {
  return unique_arg.fetch_add(1, std::memory_order_relaxed);
}

uint64 GetArgForName(StringPiece name) {
  return Hash64(name.data(), name.size());
}

}  // namespace tracing
}  // namespace itex

namespace itex {
namespace tracing {
namespace {
bool TryGetEnv(const char* name, const char** value) {
  *value = getenv(name);
  return *value != nullptr && (*value)[0] != '\0';
}
}  // namespace

void EventCollector::SetCurrentThreadName(const char*) {}

const char* GetLogDir() {
  const char* dir;
  if (TryGetEnv("TEST_TMPDIR", &dir)) return dir;
  if (TryGetEnv("TMP", &dir)) return dir;
  if (TryGetEnv("TMPDIR", &dir)) return dir;
#ifndef PLATFORM_WINDOWS
  dir = "/tmp";
  if (access(dir, R_OK | W_OK | X_OK) == 0) return dir;
#endif
  return ".";  // Default to current directory.
}
}  // namespace tracing
}  // namespace itex
