/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_STATEFUL_RANDOM_OPS_H_
#define ITEX_CORE_KERNELS_GPU_STATEFUL_RANDOM_OPS_H_

#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

using StateElementType = int64;
static constexpr DataType STATE_ELEMENT_DTYPE = DT_INT64;
static constexpr DataType ALGORITHM_DTYPE = STATE_ELEMENT_DTYPE;

// A per-device helper function that does the actual work for
// `UpdateVariableAndFill`.
// Reason to use functor: C++ doesn't allow function-template partial
// specialization.
template <typename Device, typename Distribution>
struct UpdateVariableAndFill_Philox;

template <typename Device>
struct RngSkip_Philox;

using GPUDevice = Eigen::GpuDevice;

// Declares the partially GPU-specialized functor structs.
template <typename Distribution>
struct UpdateVariableAndFill_Philox<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& device,
                  Distribution dist, int64 output_size, int64 alg_tag_skip,
                  Tensor* state_tensor,
                  typename Distribution::ResultElementType* output_data);
};

template <>
struct RngSkip_Philox<GPUDevice> {
  void operator()(const GPUDevice& device, const StateElementType* in_data,
                  uint64 delta, StateElementType* out_data);
};
}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_STATEFUL_RANDOM_OPS_H_
