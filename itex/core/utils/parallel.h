/* Copyright (c) 2023 Intel Corporation
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

#ifndef ITEX_CORE_UTILS_PARALLEL_H_
#define ITEX_CORE_UTILS_PARALLEL_H_

#include "itex/core/utils/parallel_openmp.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

// Returns the maximum number of threads that may be used in a parallel region
inline int GetNumThreads() {
  bool is_omp = true;
  ITEX_CHECK_OK(ReadBoolFromEnvVar("ITEX_OMP_THREADPOOL", true, &is_omp));
  if (is_omp) {
    return GetOmpNumThreads();
  } else {
    const Eigen::ThreadPoolDevice& device =
        OpKernelContext::eigen_cpu_device_singleton();
    return device.numThreadsInPool();
  }
}

// Returns the current thread number (starting from 0)
// in the current parallel region, or 0 in the sequential region.
// NOTE: if Eigen thread pool is used, the thread num can be -1 (when task
// amount is less than the num threads, one task can be executed in the main
// thread, which returns -1 as thread id).
inline int GetThreadNum() {
  bool is_omp = true;
  ITEX_CHECK_OK(ReadBoolFromEnvVar("ITEX_OMP_THREADPOOL", true, &is_omp));
  if (is_omp) {
    return GetOmpThreadNum();
  } else {
    const Eigen::ThreadPoolDevice& device =
        OpKernelContext::eigen_cpu_device_singleton();
    return device.currentThreadId();
  }
}

template <typename F>
inline void ParallelFor(int64_t n, const Eigen::TensorOpCost& cost,
                        const F& f) {
  bool is_omp = true;
  ITEX_CHECK_OK(ReadBoolFromEnvVar("ITEX_OMP_THREADPOOL", true, &is_omp));

  if (is_omp) {
    OmpParallelFor(0, n, 1, f);
  } else {
    const Eigen::ThreadPoolDevice& device =
        OpKernelContext::eigen_cpu_device_singleton();
    device.parallelFor(n, cost, f);
  }
}
}  // namespace itex

#endif  // ITEX_CORE_UTILS_PARALLEL_H_
