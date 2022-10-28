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

#ifndef ITEX_CORE_KERNELS_GPU_REDUCTION_UTILS_H_
#define ITEX_CORE_KERNELS_GPU_REDUCTION_UTILS_H_

#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/status.h"

namespace itex {
class ReductionHelper {
 public:
  ReductionHelper() : reduce_first_axis_(false) {}

  Status Simplify(const Tensor& data, const Tensor& axis, const bool keep_dims);

  // We need to do roughly:
  //   tmp_out = allocate(out_reshape())
  //   tmp_out.reshape(out_reshape) = data.reshape(data_reshape).reduce(axes)
  //   out = tmp_out.reshape(out_shape)

  // The reduction result must be allocated with this shape.
  TensorShape out_reshape() const;

  // The final output shape must be allocated with this shape.
  TensorShape out_shape() const;

  // The reduction is on a reshaped tensor of this rank.
  int ndims() const { return data_reshape_.size(); }

  // True if need to reduce the 0-th dimension.
  bool reduce_first_axis() const { return reduce_first_axis_; }

  // The output is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::Tensor out(Tensor* out) {
    return out->shaped<T, N>(out_reshape_);
  }

  // The input is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::ConstTensor in(const Tensor& data) {
    return data.shaped<T, N>(data_reshape_);
  }

  // Shape of shuffled input
  TensorShape data_reshape() const {
    TensorShape shape;
    for (auto s : data_reshape_) shape.AddDim(s);
    return shape;
  }

  // Shape with all reduction dimensions at the end
  TensorShape shuffled_shape();

  // Permutation of reduced dims needed to put reduction dimensions at the end
  gtl::InlinedVector<int32, 8> permutation();

 private:
  bool reduce_first_axis_;  // True if need to reduce the 0-th dimension.
  gtl::InlinedVector<int64, 4> data_reshape_;  // Reshape data before reduction.
  gtl::InlinedVector<int64, 4> out_shape_;     // The final output shape.
  gtl::InlinedVector<int64, 4> out_reshape_;   // Reshape output for reduction.
};
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_REDUCTION_UTILS_H_
