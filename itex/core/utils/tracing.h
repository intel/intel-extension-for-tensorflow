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

#ifndef ITEX_CORE_UTILS_TRACING_H_
#define ITEX_CORE_UTILS_TRACING_H_

// Tracing interface

#include <array>

#include "itex/core/utils/macros.h"
#include "itex/core/utils/platform.h"
#include "itex/core/utils/stringpiece.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace tracing {

// This enumeration contains the identifiers of all TensorFlow CPU profiler
// events. It must be kept in sync with the code in GetEventCategoryName().
enum struct EventCategory : unsigned {
  kScheduleClosure = 0,
  kRunClosure = 1,
  kCompute = 2,
  kNumCategories = 3  // sentinel - keep last
};
constexpr unsigned GetNumEventCategories() {
  return static_cast<unsigned>(EventCategory::kNumCategories);
}
const char* GetEventCategoryName(EventCategory);

// Interface for CPU profiler events.
class EventCollector {
 public:
  virtual ~EventCollector() {}
  virtual void RecordEvent(uint64 arg) const = 0;
  virtual void StartRegion(uint64 arg) const = 0;
  virtual void StopRegion() const = 0;

  // Annotates the current thread with a name.
  static void SetCurrentThreadName(const char* name);
  // Returns whether event collection is enabled.
  static bool IsEnabled();

 private:
  friend void SetEventCollector(EventCategory, const EventCollector*);
  friend const EventCollector* GetEventCollector(EventCategory);

  static std::array<const EventCollector*, GetNumEventCategories()> instances_;
};
// Set the callback for RecordEvent and ScopedRegion of category.
// Not thread safe. Only call while EventCollector::IsEnabled returns false.
void SetEventCollector(EventCategory category, const EventCollector* collector);

// Returns the callback for RecordEvent and ScopedRegion of category if
// EventCollector::IsEnabled(), otherwise returns null.
inline const EventCollector* GetEventCollector(EventCategory category) {
  if (EventCollector::IsEnabled()) {
    return EventCollector::instances_[static_cast<unsigned>(category)];
  }
  return nullptr;
}

// Returns a unique id to pass to RecordEvent/ScopedRegion. Never returns zero.
uint64 GetUniqueArg();

// Returns an id for name to pass to RecordEvent/ScopedRegion.
uint64 GetArgForName(StringPiece name);

// Records an atomic event through the currently registered EventCollector.
inline void RecordEvent(EventCategory category, uint64 arg) {
  if (auto collector = GetEventCollector(category)) {
    collector->RecordEvent(arg);
  }
}

// Records an event for the duration of the instance lifetime through the
// currently registered EventCollector.
class ScopedRegion {
 public:
  ScopedRegion(ScopedRegion&& other) noexcept  // Move-constructible.
      : collector_(other.collector_) {
    other.collector_ = nullptr;
  }

  ScopedRegion(EventCategory category, uint64 arg)
      : collector_(GetEventCollector(category)) {
    if (collector_) {
      collector_->StartRegion(arg);
    }
  }

  // Same as ScopedRegion(category, GetUniqueArg()), but faster if
  // EventCollector::IsEnabled() returns false.
  explicit ScopedRegion(EventCategory category)
      : collector_(GetEventCollector(category)) {
    if (collector_) {
      collector_->StartRegion(GetUniqueArg());
    }
  }

  // Same as ScopedRegion(category, GetArgForName(name)), but faster if
  // EventCollector::IsEnabled() returns false.
  ScopedRegion(EventCategory category, StringPiece name)
      : collector_(GetEventCollector(category)) {
    if (collector_) {
      collector_->StartRegion(GetArgForName(name));
    }
  }

  ~ScopedRegion() {
    if (collector_) {
      collector_->StopRegion();
    }
  }

  bool IsEnabled() const { return collector_ != nullptr; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ScopedRegion);

  const EventCollector* collector_;
};

// Return the pathname of the directory where we are writing log files.
const char* GetLogDir();

}  // namespace tracing
}  // namespace itex

// Stub implementations of tracing functionality.

// Definitions that do nothing for platforms that don't have underlying thread
// tracing support.
#define TRACELITERAL(a) \
  do {                  \
  } while (0)
#define TRACESTRING(s) \
  do {                 \
  } while (0)
#define TRACEPRINTF(format, ...) \
  do {                           \
  } while (0)

namespace itex {
namespace tracing {

inline bool EventCollector::IsEnabled() { return false; }

}  // namespace tracing
}  // namespace itex

#endif  // ITEX_CORE_UTILS_TRACING_H_
