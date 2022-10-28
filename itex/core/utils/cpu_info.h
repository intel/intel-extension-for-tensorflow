/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_UTILS_CPU_INFO_H_
#define ITEX_CORE_UTILS_CPU_INFO_H_

namespace itex {
namespace port {

// Returns an estimate of the number of schedulable CPUs for this
// process.  Usually, it's constant throughout the lifetime of a
// process, but it might change if the underlying cluster management
// software can change it dynamically.  If the underlying call fails, a default
// value (e.g. `4`) may be returned.
int NumSchedulableCPUs();

// Returns an estimate for the maximum parallelism for this process.
// Applications should avoid running more than this number of threads with
// intensive workloads concurrently to avoid performance degradation and
// contention.
// This value is either the number of schedulable CPUs, or a value specific to
// the underlying cluster management. Applications should assume this value can
// change throughout the lifetime of the process. This function must not be
// called during initialization, i.e., before main() has started.
int MaxParallelism();

// Returns the total number of CPUs on the system.  This number should
// not change even if the underlying cluster management software may
// change the number of schedulable CPUs.  Unlike `NumSchedulableCPUs`, if the
// underlying call fails, an invalid value of -1 will be returned;
// the user must check for validity.
static constexpr int kUnknownCPU = -1;
int NumTotalCPUs();

// Returns an estimate of the number of hyperthreads per physical core
// on the CPU
int NumHyperthreadsPerCore();

// Returns num of hyperthreads per physical core
int CPUIDNumSMT();

}  // namespace port
}  // namespace itex

#endif  // ITEX_CORE_UTILS_CPU_INFO_H_
