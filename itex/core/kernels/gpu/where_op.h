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

#ifndef ITEX_CORE_KERNELS_GPU_WHERE_OP_H_
#define ITEX_CORE_KERNELS_GPU_WHERE_OP_H_

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;
namespace functor {

template <typename T, typename TIndex>
struct InputCumSum {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* context, typename TTypes<T>::ConstFlat input,
      typename TTypes<TIndex>::Vec input_cumsum, TIndex num_elems);
};

template <typename Device, int NDIM, typename T, typename TIndex>
struct Where {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* context, const Device& d,
      typename TTypes<T, NDIM>::ConstTensor input,
      typename TTypes<int64>::Matrix output, TIndex* found_true);
};

}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_WHERE_OP_H_
