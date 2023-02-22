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

#ifndef ITEX_CORE_KERNELS_GPU_TOPK_OP_H_
#define ITEX_CORE_KERNELS_GPU_TOPK_OP_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename IndexT>
struct TopKFunctor;

template <typename T, typename IndexT>
struct TopKFunctor<GPUDevice, T, IndexT> {
  void operator()(OpKernelContext* context,
                  typename TTypes<T, 2>::ConstTensor input,
                  typename TTypes<T, 2>::Tensor values,
                  typename TTypes<IndexT, 2>::Tensor indices, bool sorted,
                  int num_topk);
};

template <typename KeyT, typename ValueT>
void DispatchToFallBackRadixSort(const gpuStream_t& stream,
                                 const KeyT* key_array, KeyT* key_src,
                                 KeyT* key_dst, ValueT* value_src,
                                 ValueT* value_dst, const int num_rows,
                                 const int num_cols, const int max_group_size);

}  // end namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_TOPK_OP_H_
