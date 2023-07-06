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

#include "itex/core/kernels/gpu/concat_lib.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

template <typename Device, typename T, AxisArgumentName AxisArgName>
class ConcatBaseOp : public OpKernel {
 public:
  explicit ConcatBaseOp(OpKernelConstruction* c)
      : OpKernel(c),
        is_v2_(AxisArgName == NAME_IS_AXIS),
        axis_attribute_name_(is_v2_ ? "axis" : "concat_dim") {}

  void Compute(OpKernelContext* c) override {
    const int num_inputs = c->num_inputs();
    OP_REQUIRES(
        c, num_inputs > 2,
        errors::InvalidArgument("Number of values must larger than 1, but got ",
                                num_inputs - 1));
    // "Concat" and "ConcatV2" have different input order
    if (is_v2_) {
      values_input_start_index_ = 0;
      values_input_end_index_ = num_inputs - 2;
      axis_input_index_ = num_inputs - 1;
    } else {
      axis_input_index_ = 0;
      values_input_start_index_ = 1;
      values_input_end_index_ = num_inputs - 1;
    }

    const Tensor& concat_dim_tensor = c->input(axis_input_index_);

    OP_REQUIRES(c,
                (TensorShapeUtils::IsScalar(concat_dim_tensor.shape()) ||
                 (TensorShapeUtils::IsVector(concat_dim_tensor.shape()) &&
                  concat_dim_tensor.shape().dim_size(0) == 1)),
                errors::InvalidArgument(
                    axis_attribute_name_,
                    " tensor should be a scalar integer, but got shape ",
                    concat_dim_tensor.shape().DebugString()));
    int64 concat_dim;
    // In case of ConcatV2, "axis" could be int32 or int64
    if (is_v2_) {
      OP_REQUIRES(
          c,
          (concat_dim_tensor.dtype() == DT_INT32 ||
           concat_dim_tensor.dtype() == DT_INT64),
          errors::InvalidArgument(axis_attribute_name_,
                                  " tensor should be int32 or int64, but got ",
                                  DataTypeString(concat_dim_tensor.dtype())));
    } else {
      OP_REQUIRES(c, (concat_dim_tensor.dtype() == DT_INT32),
                  errors::InvalidArgument(
                      axis_attribute_name_, " tensor should be int32, but got ",
                      DataTypeString(concat_dim_tensor.dtype())));
    }

    if (concat_dim_tensor.dtype() == DT_INT32) {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int32>()());
    } else {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int64>()());
    }

    const int N = values_input_end_index_ - values_input_start_index_ + 1;
    const Tensor& first_input = c->input(values_input_start_index_);
    const int input_dims = first_input.dims();
    const TensorShape& input_shape = first_input.shape();

    int32 axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    // concat_dim==0 allows concatenating a list of scalars into a vector.
    OP_REQUIRES(c, (0 <= axis && axis < input_dims) || concat_dim == 0,
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));
    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
        ConstMatrixVector;
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(N);
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < axis; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int64 output_concat_dim = 0;
    for (int i = 0; i < N; ++i) {
      const auto& in = c->input(values_input_start_index_ + i);
      OP_REQUIRES(
          c, in.dims() == input_dims,
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == axis) {
          continue;
        }
        OP_REQUIRES(
            c, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.DebugString(), " vs. shape[", i,
                "] = ", in.shape().DebugString()));
      }
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.template shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      output_concat_dim += in.dims() > 0 ? in.dim_size(axis) : 1;
    }

    TensorShape output_shape(input_shape);
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(axis, output_concat_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() > 0) {
      int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
      auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
      Concat<T>(c, inputs_flat, &output_flat);
    }
  }

 private:
  bool is_v2_;
  const char* const axis_attribute_name_;
  int values_input_start_index_;
  int values_input_end_index_;
  int axis_input_index_;
};

template <typename Device, typename T>
using ConcatOp = ConcatBaseOp<Device, T, NAME_IS_CONCAT_DIM>;
template <typename Device, typename T>
using ConcatV2Op = ConcatBaseOp<Device, T, NAME_IS_AXIS>;

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Concat")                 \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ConcatOp<GPUDevice, type>)     \
  REGISTER_KERNEL_BUILDER(Name("ConcatV2")               \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("axis"),       \
                          ConcatV2Op<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU

}  // namespace itex
