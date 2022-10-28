/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/base/internal/sysinfo.h"

#if defined(__linux__)
#include <sched.h>
#else
#include <sys/syscall.h>
#endif

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#endif

#include <cstdio>
#if defined(__FreeBSD__)
#include <thread>  // NOLINT(build/c++11)
#endif

#include "itex/core/utils/cpu_info.h"

namespace itex {
namespace port {

int NumSchedulableCPUs() {
#if defined(__linux__)
  cpu_set_t cpuset;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
    return CPU_COUNT(&cpuset);
  }
  perror("sched_getaffinity");
#endif
#if defined(__FreeBSD__)
  unsigned int count = std::thread::hardware_concurrency();
  if (count > 0) return static_cast<int>(count);
#endif
  const int kDefaultCores = 4;  // Semi-conservative guess
  fprintf(stderr, "can't determine number of CPU cores: assuming %d\n",
          kDefaultCores);
  return kDefaultCores;
}

int NumHyperthreadsPerCore() {
  static const int ht_per_core = itex::port::CPUIDNumSMT();
  return (ht_per_core > 0) ? ht_per_core : 1;
}

int MaxParallelism() { return NumSchedulableCPUs(); }

int NumTotalCPUs() {
  int count = absl::base_internal::NumCPUs();
  return (count <= 0) ? kUnknownCPU : count;
}

}  // namespace port
}  // namespace itex
