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

#include "itex/core/wrapper/itex_cpu_wrapper.h"

#include <cpuid.h>
#include <dlfcn.h>

#include "itex/core/devices/device_backend_util.h"
#include "itex/core/utils/cpu_info.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/types.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

static void* handle;
static void* LoadCpuLibrary() __attribute__((constructor));
static void UnloadCpuLibrary() __attribute__((destructor));

void* LoadCpuLibrary() {
  bool enable_omp;
  if (itex_get_backend() == ITEX_BACKEND_DEFAULT) {
    itex_freeze_backend(ITEX_BACKEND_CPU);
  }
  ITEX_CHECK_OK(
      itex::ReadBoolFromEnvVar("ITEX_OMP_THREADPOOL", true, &enable_omp));
  if (enable_omp) {
    onednn_handle = dlopen("libonednn_cpu_so.so", RTLD_NOW | RTLD_GLOBAL);
    if (!onednn_handle) {
      ITEX_LOG(FATAL) << dlerror();
    }
  } else {
    onednn_handle = dlopen("libonednn_cpu_eigen_so.so", RTLD_NOW | RTLD_GLOBAL);
    if (!onednn_handle) {
      ITEX_LOG(FATAL) << dlerror();
    }
  }

  if (itex::port::CPUIDAVX512()) {
    handle = dlopen("libitex_cpu_internal_avx512.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char* error_msg = dlerror();
      ITEX_LOG(WARNING)
          << "AVX512 CPU library is loaded failed, try to load AVX2.";
      ITEX_LOG(WARNING) << error_msg;
    } else {
      ITEX_LOG(INFO)
          << "Intel Extension for Tensorflow* AVX512 CPU backend is loaded.";
      return handle;
    }
  }
  handle = dlopen("libitex_cpu_internal_avx2.so", RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char* error_msg = dlerror();
    ITEX_LOG(FATAL) << "Could not load dynamic library: " << error_msg;
  }
  ITEX_LOG(INFO)
      << "Intel Extension for Tensorflow* AVX2 CPU backend is loaded.";
  return handle;
}

void UnloadCpuLibrary() {
  if (handle) {
    dlclose(handle);
  }
}

void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status) {
  typedef void (*tf_initgraph_internal)(TP_OptimizerRegistrationParams*,
                                        TF_Status*);

  if (handle) {
    auto tf_initgraph = reinterpret_cast<tf_initgraph_internal>(
        dlsym(handle, "TF_InitGraph_Internal"));
    if (tf_initgraph != nullptr) {
      tf_initgraph(params, status);
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  } else {
    ITEX_LOG(WARNING) << "Graph module not found.";
  }
}

void TF_InitKernel() {
  typedef void (*tf_initkernel_internal)();

  if (handle) {
    auto tf_initkernel = reinterpret_cast<tf_initkernel_internal>(
        dlsym(handle, "TF_InitKernel_Internal"));
    if (*tf_initkernel != nullptr) {
      tf_initkernel();
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  }
}
