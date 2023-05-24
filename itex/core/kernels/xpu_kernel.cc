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

#if defined(INTEL_CPU_ONLY) && !defined(CC_BUILD)
#include "itex/core/kernels/xpu_kernel.h"
#endif

#include <string>

#include "Python.h"
#include "itex/core/devices/device_backend_util.h"
#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common.h"
#ifndef INTEL_CPU_ONLY
#include "itex/core/kernels/gpu/gpu_kernel_init.h"
#else
#include "itex/core/kernels/cpu/cpu_kernel_init.h"
#endif  // INTEL_CPU_ONLY
#if !defined(INTEL_CPU_ONLY) || defined(CC_BUILD)
#include "tensorflow/c/kernels.h"
#endif

#if defined(INTEL_CPU_ONLY) && !defined(CC_BUILD)
void TF_InitKernel_Internal() {
#else
void TF_InitKernel() {
#endif
  // Register generic GPU kernels.
  ITEX_BACKEND backend = itex_get_backend();
  switch (backend) {
    case ITEX_BACKEND_CPU:
      break;
    case ITEX_BACKEND_GPU:
#ifndef INTEL_CPU_ONLY
      RegisterGPUKernels(itex::DEVICE_XPU);
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
#endif  // INTEL_CPU_ONLY

#ifndef CC_BUILD
  bool ops_override = false;
  ITEX_CHECK_OK(
      itex::ReadBoolFromEnvVar("ITEX_OPS_OVERRIDE", false, &ops_override));
  if (ops_override) {
    PyRun_SimpleString("import intel_extension_for_tensorflow as itex;\n");
    PyRun_SimpleString("itex.experimental_ops_override();\n");
  }
#endif  // CC_BUILD
}
