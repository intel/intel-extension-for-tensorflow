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

#ifndef ITEX_CORE_UTILS_PARALLEL_OPENMP_H_
#define ITEX_CORE_UTILS_PARALLEL_OPENMP_H_

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace itex {

int get_num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

bool in_parallel_region() {
#ifdef _OPENMP
  return omp_in_parallel();
#else
  return false;
#endif
}

inline int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

#ifdef _OPENMP
template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size,
                         const F& f) {
  if (begin >= end) {
    return;
  }

  const auto numiter = end - begin;
  const bool use_parallel = (numiter > grain_size && numiter > 1 &&
                             !in_parallel_region() && get_num_threads() > 1);
  if (!use_parallel) {
    f(begin, end);
    return;
  }

#pragma omp parallel
  {
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      f(begin_tid, std::min(end, chunk_size + begin_tid));
    }
  }
}
#endif  // _OPENMP

}  // namespace itex

#endif  // ITEX_CORE_UTILS_PARALLEL_OPENMP_H_
