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

#ifndef ITEX_CORE_KERNELS_GPU_IN_TOPK_OP_H_
#define ITEX_CORE_KERNELS_GPU_IN_TOPK_OP_H_

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// InTopK argument can be passed either via mode attribute (InTopK op), or as an
// input tensor (InTopKV2 op).
struct TopKArg {
  int64 k_value = -1;
  const Tensor* k_tensor = nullptr;
};

template <typename Device, typename T, typename TargetT>
struct InTopKFunctor {
  template <int ndims>
  using Dims = Eigen::DSizes<Eigen::Index, ndims>;

  void operator()(OpKernelContext* context,
                  typename TTypes<T, 2>::ConstTensor predictions,
                  typename TTypes<TargetT>::ConstVec targets, const TopKArg k,
                  typename TTypes<bool>::Vec output) {}
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_IN_TOPK_OP_H_
