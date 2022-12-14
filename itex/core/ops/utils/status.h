/* Copyright (c) 2021 Intel Corporation

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

#ifndef ITEX_CORE_OPS_UTILS_STATUS_H_
#define ITEX_CORE_OPS_UTILS_STATUS_H_

#include <memory>

#include "tensorflow/c/tf_status.h"

namespace itex {

struct StatusDeleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

using StatusUniquePtr = std::unique_ptr<TF_Status, StatusDeleter>;

}  // namespace itex

#endif  // ITEX_CORE_OPS_UTILS_STATUS_H_
