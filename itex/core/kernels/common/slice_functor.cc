/* Copyright (c) 2022 Intel Corporation

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

#include "itex/core/kernels/common/slice_functor.h"

namespace itex {

void IntTensorToInt64Vec(const Tensor& tensor,
                         gtl::InlinedVector<int64, 4>* out) {
  out->resize(tensor.NumElements());
  int64* out_ptr = out->data();
  if (tensor.dtype() == DT_INT32) {
    const int32* tensor_ptr = tensor.flat<int32>().data();
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out_ptr[i] = tensor_ptr[i];
    }
  } else if (tensor.dtype() == DT_INT64) {
    const int64* tensor_ptr = tensor.flat<int64>().data();
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out_ptr[i] = tensor_ptr[i];
    }
  } else {
    ITEX_LOG(FATAL) << "begin must be either int32 or int64";
  }
}

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
void SharedSliceValidation(OpKernelContext* context,
                           const TensorShape& src_tf_shape,
                           TensorShape* dst_tf_shape, bool* is_identity,
                           bool* slice_dim0,
                           gtl::InlinedVector<int64, 4>* begin,
                           gtl::InlinedVector<int64, 4>* size) {
  const Tensor& begin_tensor = context->input(1);
  const Tensor& size_tensor = context->input(2);

  OP_REQUIRES(
      context,
      TensorShapeUtils::IsVector(begin_tensor.shape()) &&
          TensorShapeUtils::IsVector(size_tensor.shape()) &&
          begin_tensor.NumElements() == src_tf_shape.dims() &&
          size_tensor.NumElements() == src_tf_shape.dims(),
      errors::InvalidArgument(
          "Expected begin and size arguments to be 1-D tensors of size ",
          src_tf_shape.dims(), ", but got shapes ",
          begin_tensor.shape().DebugString(), " and ",
          size_tensor.shape().DebugString(), " instead."));

  const int input_dims = src_tf_shape.dims();
  IntTensorToInt64Vec(begin_tensor, begin);
  IntTensorToInt64Vec(size_tensor, size);
  for (int i = 0; i < input_dims; ++i) {
    if ((*size)[i] == -1) {
      // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
      (*size)[i] = src_tf_shape.dim_size(i) - (*begin)[i];
    }
  }

  *is_identity = true;
  *slice_dim0 = true;
  for (int i = 0; i < input_dims; ++i) {
    int64 b = (*begin)[i];
    int64 s = (*size)[i];
    if (src_tf_shape.dim_size(i) == 0) {
      OP_REQUIRES(
          context, b == 0 && s == 0,
          errors::InvalidArgument("Expected begin[", i, "] == 0 (got ", b,
                                  ") and size[", i, "] == 0 ", "(got ", s,
                                  ") when ", "input.dim_size(", i, ") == 0"));
    } else {
      OP_REQUIRES(
          context, 0 <= b && b <= src_tf_shape.dim_size(i),
          errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                  src_tf_shape.dim_size(i), "], but got ", b));
      OP_REQUIRES(context, 0 <= s && b + s <= src_tf_shape.dim_size(i),
                  errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                          src_tf_shape.dim_size(i) - b,
                                          "], but ", "got ", s));
    }
    dst_tf_shape->AddDim(s);
    const bool take_all = (b == 0) && (s == src_tf_shape.dim_size(i));
    (*is_identity) &= take_all;
    (*slice_dim0) &= (i == 0) || take_all;
  }
}

}  // namespace itex
