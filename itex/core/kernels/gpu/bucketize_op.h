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

#ifndef ITEX_CORE_KERNELS_GPU_BUCKETIZE_OP_H_
#define ITEX_CORE_KERNELS_GPU_BUCKETIZE_OP_H_

#include <vector>
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename Device, typename T>
struct BucketizeFunctor {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& input,
                        const std::vector<float>& boundaries_vector,
                        const typename TTypes<int32, 1>::Tensor& output);
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_BUCKETIZE_OP_H_
