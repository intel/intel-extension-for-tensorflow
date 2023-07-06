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
#include "itex/core/compiler/xla/service/custom_call_status.h"

#include <string>

#include "itex/core/compiler/xla/service/custom_call_status_internal.h"

namespace itex_xla {
// Internal functions

absl::optional<absl::string_view> CustomCallStatusGetMessage(
    const ItexXlaCustomCallStatus* status) {
  return status->message;
}

}  // namespace itex_xla

void ItexXlaCustomCallStatusSetSuccess(ItexXlaCustomCallStatus* status) {
  status->message = absl::nullopt;
}

void ItexXlaCustomCallStatusSetFailure(ItexXlaCustomCallStatus* status,
                                       const char* message,
                                       size_t message_len) {
  status->message = std::string(message, strnlen(message, message_len));
}
