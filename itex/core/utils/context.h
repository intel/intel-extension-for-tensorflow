/* Copyright (c) 2023 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_UTILS_CONTEXT_H_
#define ITEX_CORE_UTILS_CONTEXT_H_

#include "itex/core/utils/platform.h"

namespace itex {

enum class ContextKind {
  // Initial state with default (empty) values.
  kDefault,
  // Initial state inherited from the creating or scheduling thread.
  kThread,
};

// Context is a container for request-specific information that should be passed
// to threads that perform related work. The default constructor should capture
// all relevant context.
class Context {
 public:
  Context() {}
  explicit Context(const ContextKind kind) {}

  bool operator==(const Context& other) const { return true; }
};

// Scoped object that sets the current thread's context until the object is
// destroyed.
class WithContext {
 public:
  explicit WithContext(const Context& x) {}
  ~WithContext() {}
};

}  // namespace itex

#endif  // ITEX_CORE_UTILS_CONTEXT_H_
