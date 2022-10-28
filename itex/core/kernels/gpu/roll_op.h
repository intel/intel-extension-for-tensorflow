/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_ROLL_OP_H_
#define ITEX_CORE_KERNELS_GPU_ROLL_OP_H_

#include "itex/core/utils/numeric_types.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename Device, typename T>
struct Roll {
  // dim_size - the size of each dimension
  // dim_range - the number of indices over in the flattened tensor
  //    you need to skip in order to make it over from one side of a dimension
  //    to the other. Used to make the shifts wrap around after a threshold.
  // threshold - the index for each dimension that the roll starts to wrap
  //    back to the front
  // isd - inner shift dimension
  void operator()(const OpKernelContext* context, const int64 num_elements,
                  const int num_dims, const gtl::ArraySlice<int32> dim_size,
                  const T* input, T* output,
                  const gtl::ArraySlice<int32> threshold,
                  const gtl::ArraySlice<int64> dim_range, const int64 isd);
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_ROLL_OP_H_
