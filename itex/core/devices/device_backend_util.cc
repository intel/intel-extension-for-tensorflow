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

#include "itex/core/devices/device_backend_util.h"

#include <cstring>

static const char* frozen_backend = nullptr;
static bool backend_is_frozen = false;

void itex_freeze_backend_internal(const char* backend) {
  if (strcasecmp(backend, "GPU") == 0) {
    frozen_backend = itex::DEVICE_GPU;
  } else if (strcasecmp(backend, "CPU") == 0) {
    frozen_backend = itex::DEVICE_CPU;
  } else if (strcasecmp(backend, "AUTO") == 0) {
    frozen_backend = itex::DEVICE_AUTO;
  } else {
    ITEX_LOG(FATAL) << "Invalid ITEX_BACKEND: " << backend
                    << ", please select from CPU, GPU, AUTO";
  }

  backend_is_frozen = true;
}

void itex_freeze_backend(ITEX_BACKEND backend) {
  std::string backend_s;
  switch (backend) {
    case ITEX_BACKEND_GPU:
      frozen_backend = itex::DEVICE_GPU;
      break;
    case ITEX_BACKEND_CPU:
      frozen_backend = itex::DEVICE_CPU;
      break;
    case ITEX_BACKEND_AUTO:
      frozen_backend = itex::DEVICE_AUTO;
      break;
    default:
      ITEX_LOG(FATAL) << "Invalid ITEX_BACKEND: " << backend
                      << ", please select from CPU, GPU, AUTO";
  }
}

ITEX_BACKEND itex_get_backend() {
  const char* backend = nullptr;
  if (backend_is_frozen) {
    backend = frozen_backend;
  } else {
    backend = std::getenv("ITEX_BACKEND");
    if (backend == nullptr) return ITEX_BACKEND_DEFAULT;
  }

  if (strcasecmp(backend, "GPU") == 0) {
    return ITEX_BACKEND_GPU;
  } else if (strcasecmp(backend, "CPU") == 0) {
    return ITEX_BACKEND_CPU;
  } else if (strcasecmp(backend, "AUTO") == 0) {
    return ITEX_BACKEND_AUTO;
  } else {
    ITEX_LOG(FATAL) << "Invalid ITEX_BACKEND: " << backend
                    << ", please select from CPU, GPU, AUTO";
    return ITEX_BACKEND_DEFAULT;
  }
}

void itex_set_backend(const char* backend) {
  if (backend_is_frozen && (strcasecmp(backend, frozen_backend) != 0)) {
    ITEX_LOG(INFO) << "ITEX backend is already set as " << frozen_backend
                   << ", setting backend as " << backend << " is ignored";
    return;
  }

  itex_freeze_backend_internal(backend);
}

const char* itex_backend_to_string(ITEX_BACKEND backend) {
  const char* backend_string;
  switch (backend) {
    case ITEX_BACKEND_GPU:
      backend_string = const_cast<char*>("GPU");
      break;
    case ITEX_BACKEND_CPU:
      backend_string = const_cast<char*>("CPU");
      break;
    case ITEX_BACKEND_AUTO:
      backend_string = const_cast<char*>("AUTO");
      break;
    default:
      ITEX_LOG(INFO) << "Unkown ITEX_BACKEND: " << backend;
      backend_string = const_cast<char*>("");
      break;
  }
  return backend_string;
}

const char* GetDeviceBackendName(const char* device_name) {
  if (strstr(device_name, itex::DEVICE_XPU) != nullptr) {
    ITEX_BACKEND backend = itex_get_backend();
    switch (backend) {
      case ITEX_BACKEND_GPU:
        return itex::DEVICE_GPU;
      case ITEX_BACKEND_CPU:
        return itex::DEVICE_CPU;
      case ITEX_BACKEND_AUTO:
        return itex::DEVICE_AUTO;
      default:
        return "";
    }

  } else if (strstr(device_name, itex::DEVICE_GPU) != nullptr) {
    return itex::DEVICE_GPU;
  } else if (strstr(device_name, itex::DEVICE_CPU) != nullptr) {
    return itex::DEVICE_CPU;
  } else {
    ITEX_CHECK(false) << "Unsupported device type: " << device_name;
  }
}
