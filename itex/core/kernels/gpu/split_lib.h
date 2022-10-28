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

#ifndef ITEX_CORE_KERNELS_GPU_SPLIT_LIB_H_
#define ITEX_CORE_KERNELS_GPU_SPLIT_LIB_H_

#include "itex/core/kernels/gpu/gpu_device_array.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename T, int NDims>
struct Split {
  void operator()(const Eigen::GpuDevice& d,
                  typename TTypes<T, NDims>::Tensor output,
                  typename TTypes<T, NDims>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_sizes);
};

template <typename T>
struct SplitGpuFunctor {
  void operator()(const Eigen::GpuDevice& d, const T* input,
                  int prefix_dim_size, int split_dim_size, int suffix_dim_size,
                  int split_dim_output_size, int offset,
                  const GpuDeviceArrayStruct<T*>& output_ptr_data);
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SPLIT_LIB_H_
