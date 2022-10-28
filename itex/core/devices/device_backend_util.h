/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef ITEX_CORE_DEVICES_DEVICE_BACKEND_UTIL_H_
#define ITEX_CORE_DEVICES_DEVICE_BACKEND_UTIL_H_

#include <string>

#include "itex/core/utils/protobuf/config.pb.h"
#include "itex/core/utils/types.h"

constexpr char DEVICE_XPU_NAME[] = "XPU";

enum ITEX_BACKEND {
  ITEX_BACKEND_GPU,
  ITEX_BACKEND_CPU,
  ITEX_BACKEND_AUTO,
  ITEX_BACKEND_DEFAULT = ITEX_BACKEND_GPU
};

namespace itex {

ITEX_BACKEND itex_get_backend();
ConfigProto itex_get_config();
void itex_set_backend(const char* backend, const ConfigProto& config);
void itex_backend_to_string(ITEX_BACKEND backend, std::string* backend_string);
void itex_freeze_backend(const char* backend, const ConfigProto& config);
void itex_freeze_backend(ITEX_BACKEND backend, const ConfigProto& config);

// Get the real backend name of given device.
// @return:
//   1) CPU: DEVICE_CPU;
//   2) GPU: DEVICE_GPU;
//   3) XPU with GPU backend: DEVICE_GPU;
//   4) TODO(itex): XPU with other backend
// Return value will never be nullptr.
const char* GetDeviceBackendName(const std::string& device_name);

}  // namespace itex

#endif  // ITEX_CORE_DEVICES_DEVICE_BACKEND_UTIL_H_
