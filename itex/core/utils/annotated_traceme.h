/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_ANNOTATED_TRACEME_H_
#define ITEX_CORE_UTILS_ANNOTATED_TRACEME_H_

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/scoped_annotation.h"
#include "itex/core/utils/traceme.h"
#include "itex/core/utils/types.h"

namespace itex {

// Combination of TraceMe and ScopedAnnotation which share the same label.
// Optimization are done to ensure the label generation are done once.
class AnnotatedTraceMe {
 public:
  template <typename NameGeneratorT>
  explicit AnnotatedTraceMe(NameGeneratorT&& name_generator, int level = 1) {
    ITEX_DCHECK_GE(level, 1);
    bool annotation_enabled = ScopedAnnotation::IsEnabled();
    bool traceme_enabled = TraceMe::Active(level);
    if (ITEX_PREDICT_FALSE(annotation_enabled || traceme_enabled)) {
      string name = std::forward<NameGeneratorT>(name_generator)();
      if (annotation_enabled) {
        scoped_annotation_.emplace(absl::string_view(name));
      }
      if (ITEX_PREDICT_TRUE(traceme_enabled)) {
        trace_me_.emplace([&name] { return std::move(name); }, level);
      }
    }
  }

 private:
  absl::optional<TraceMe> trace_me_;
  absl::optional<ScopedAnnotation> scoped_annotation_;
};

}  // namespace itex

#endif  // ITEX_CORE_UTILS_ANNOTATED_TRACEME_H_
