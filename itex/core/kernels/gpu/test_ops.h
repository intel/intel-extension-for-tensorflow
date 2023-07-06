/* Copyright (c) 2021-2023 Intel Corporation
Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ITEX_CORE_KERNELS_GPU_TEST_OPS_H_
#define ITEX_CORE_KERNELS_GPU_TEST_OPS_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/time_utils.h"

namespace itex {
void sleep_kernel(int seconds) {
  int64_t nanoseconds = int64_t{seconds} * 1'000'000'000;
  // Passing too high a number to __nanosleep makes it sleep for much less time
  // than the passed-in number. So only pass 1,000,000 and keep calling
  // __nanosleep in a loop.
  for (int64_t i = 0; i < nanoseconds; i += 1'000'000) {
    profiler::SleepForNanos(1'000'000);
  }
}

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_TEST_OPS_H_
