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

#include <string>

#include "itex/core/devices/device_backend_util.h"
#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common.h"
#include "tensorflow/c/kernels.h"

#ifndef INTEL_CPU_ONLY
#include "itex/core/kernels/gpu/gpu_kernel_init.h"
#else
#include "itex/core/kernels/cpu/cpu_kernel_init.h"
#endif  // INTEL_CPU_ONLY

void TF_InitKernel() {
  // Register generic GPU kernels.
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_GPU:
#ifndef INTEL_CPU_ONLY
      RegisterGPUKernels(itex::DEVICE_XPU);
#else
      ITEX_LOG(ERROR) << "XPU-GPU kernel not supported.";
#endif  // INTEL_CPU_ONLY
      break;
    case ITEX_BACKEND_AUTO:
      ITEX_LOG(ERROR) << "XPU-AUTO kernel not supported.";
      break;
    default:
      ITEX_LOG(ERROR) << "backend not supported.";
      break;
  }

  // Register op definitions.
  CallOnce_RegisterOps();

#ifdef INTEL_CPU_ONLY
  // Register generic CPU kernels.
  RegisterCPUKernels(itex::DEVICE_CPU);
#endif
}
