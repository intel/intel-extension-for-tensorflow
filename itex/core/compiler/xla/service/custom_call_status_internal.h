/* Copyright (c) 2023 Intel Corporation

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_CUSTOM_CALL_STATUS_INTERNAL_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_CUSTOM_CALL_STATUS_INTERNAL_H_

#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "itex/core/compiler/xla/service/custom_call_status.h"

struct ItexXlaCustomCallStatus_ {
  // The message being present means "failure". Otherwise means "success".
  absl::optional<std::string> message;
};

namespace itex_xla {

// Get a view of the internal error message of the ItexXlaCustomCallStatus. Only
// lives as long as the ItexXlaCustomCallStatus. Returns an empty optional if
// the result was "success".
absl::optional<absl::string_view> CustomCallStatusGetMessage(
    const ItexXlaCustomCallStatus* status);

}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_CUSTOM_CALL_STATUS_INTERNAL_H_
