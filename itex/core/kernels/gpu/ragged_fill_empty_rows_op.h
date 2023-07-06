/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_RAGGED_FILL_EMPTY_ROWS_OP_H_
#define ITEX_CORE_KERNELS_GPU_RAGGED_FILL_EMPTY_ROWS_OP_H_

#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_types.h"

namespace itex {

namespace functor {

template <typename Device, typename T, typename Tindex, bool RaggedOperands>
struct FillEmptyRows {
  // Note that the done callback is only used by the GPU implementation.
  Status operator()(OpKernelContext* context, const Tensor& default_value_t,
                    const Tensor& indices_t, const Tensor& values_t,
                    const Tensor& dense_shape_t);
};

}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_RAGGED_FILL_EMPTY_ROWS_OP_H_
