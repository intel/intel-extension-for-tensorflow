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

#include "itex/core/kernels/gpu/dilation_ops.h"

#include <cfloat>
#include <vector>

#include "itex/core/kernels/gpu/cast_op.h"
#include "itex/core/utils/common_shape_fns.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gtl/array_slice.h"
#include "itex/core/utils/numeric_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/padding.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

void ParseAttributes(OpKernelConstruction* context, std::vector<int32>* strides,
                     std::vector<int32>* rates, Padding* padding) {
  OP_REQUIRES_OK(context, context->GetAttr("strides", strides));
  OP_REQUIRES(context, strides->size() == 4,
              errors::InvalidArgument("Sliding window stride field must "
                                      "specify 4 dimensions"));
  OP_REQUIRES(context, (*strides)[0] == 1 && (*strides)[3] == 1,
              errors::Unimplemented(
                  "Stride is only supported across spatial dimensions."));

  OP_REQUIRES_OK(context, context->GetAttr("rates", rates));
  OP_REQUIRES(context, rates->size() == 4,
              errors::InvalidArgument("Input stride (atrous rate) field "
                                      "must specify 4 dimensions"));
  OP_REQUIRES(context, (*rates)[0] == 1 && (*rates)[3] == 1,
              errors::Unimplemented(
                  "Rate is only supported across spatial dimensions."));
  OP_REQUIRES_OK(context, context->GetAttr("padding", padding));
}

void ParseSizes(OpKernelContext* context, const std::vector<int32>& strides,
                const std::vector<int32>& rates, const Padding& padding,
                int* stride_rows, int* stride_cols, int* rate_rows,
                int* rate_cols, int64* pad_top, int64* pad_left,
                int64* out_rows, int64* out_cols) {
  // Input tensor is of the following dimensions:
  // [ batch, input_rows, input_cols, depth ]
  const Tensor& input = context->input(0);
  OP_REQUIRES(context, input.dims() == 4,
              errors::InvalidArgument("input must be 4-dimensional",
                                      input.shape().DebugString()));
  const int input_rows = input.dim_size(1);
  const int input_cols = input.dim_size(2);
  const int depth = input.dim_size(3);

  // For now we take the stride and rate from the second and third dimensions
  // only (we do not support striding on the batch or depth dimension).
  *stride_rows = strides[1];
  *stride_cols = strides[2];
  *rate_rows = rates[1];
  *rate_cols = rates[2];

  // Input filter is of the following dimensions:
  // [ filter_rows, filter_cols, depth ]
  const Tensor& filter = context->input(1);
  OP_REQUIRES(context, filter.dims() == 3,
              errors::InvalidArgument("filter must be 3-dimensional: ",
                                      filter.shape().DebugString()));
  const int filter_rows = filter.dim_size(0);
  const int filter_cols = filter.dim_size(1);
  OP_REQUIRES(context, depth == filter.dim_size(2),
              errors::InvalidArgument(
                  "input and filter must have the same depth: ", depth, " vs ",
                  filter.dim_size(2)));

  // Effective filter size, after introducing rate - 1 zeros between each
  // non-zero filter element.
  const int filter_rows_eff =
      filter_rows + (filter_rows - 1) * (*rate_rows - 1);
  const int filter_cols_eff =
      filter_cols + (filter_cols - 1) * (*rate_cols - 1);

  OP_REQUIRES_OK(
      context, GetWindowedOutputSize(input_rows, filter_rows_eff, *stride_rows,
                                     padding, out_rows, pad_top));
  OP_REQUIRES_OK(
      context, GetWindowedOutputSize(input_cols, filter_cols_eff, *stride_cols,
                                     padding, out_cols, pad_left));
}

template <typename Device, typename T>
class DilationOp : public OpKernel {
 public:
  explicit DilationOp(OpKernelConstruction* context) : OpKernel(context) {
    ParseAttributes(context, &strides_, &rates_, &padding_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);

    // Determine relevant sizes from input and filters.
    int stride_rows = 0, stride_cols = 0;
    int rate_rows = 0, rate_cols = 0;
    int64 pad_top = 0, pad_left = 0;
    int64 out_rows = 0, out_cols = 0;
    ParseSizes(context, strides_, rates_, padding_, &stride_rows, &stride_cols,
               &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows,
               &out_cols);

    // Output tensor is of the following dimensions:
    // [ batch, out_rows, out_cols, depth ]
    const int batch = input.dim_size(0);
    const int depth = input.dim_size(3);
    const std::vector<int64> out_sizes = {batch, out_rows, out_cols, depth};
    TensorShape out_shape(out_sizes);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    functor::Dilation<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 4>(),
        filter.tensor<T, 3>(), stride_rows, stride_cols, rate_rows, rate_cols,
        pad_top, pad_left, output->tensor<T, 4>());
  }

  std::vector<int32> strides_;
  std::vector<int32> rates_;
  Padding padding_;
};

namespace {
template <typename Device, typename T>
struct CastFloatToSmallFloat {
  void operator()(const Device& d, typename TTypes<float>::ConstFlat input,
                  typename TTypes<T>::Flat output);
};

template <>
struct CastFloatToSmallFloat<GPUDevice, float> {
  void operator()(const GPUDevice& d, typename TTypes<float>::ConstFlat input,
                  typename TTypes<float>::Flat output) {}
};

template <>
struct CastFloatToSmallFloat<GPUDevice, double> {
  void operator()(const GPUDevice& d, typename TTypes<float>::ConstFlat input,
                  typename TTypes<double>::Flat output) {}
};

template <typename T>
struct CastFloatToSmallFloat<GPUDevice, T> {
  typename std::enable_if<std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, Eigen::bfloat16>::value,
                          void>::type
  operator()(const GPUDevice& d, typename TTypes<float>::ConstFlat input,
             typename TTypes<T>::Flat output) {
    // Use existing cast functor instead of directly casting Eigen tensor, as
    // otherwise we need to instantiate the cast function in a .cu.cc file
    functor::CastFunctor<GPUDevice, T, float> cast;
    cast(d, output, input);
  }
};
}  // namespace

template <typename Device, typename T>
class DilationBackpropInputOp : public OpKernel {
 public:
  explicit DilationBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    ParseAttributes(context, &strides_, &rates_, &padding_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // Determine relevant sizes from input and filters.
    int stride_rows = 0, stride_cols = 0;
    int rate_rows = 0, rate_cols = 0;
    int64 pad_top = 0, pad_left = 0;
    int64 out_rows = 0, out_cols = 0;
    ParseSizes(context, strides_, rates_, padding_, &stride_rows, &stride_cols,
               &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows,
               &out_cols);

    // Verify that the incoming gradient tensor has the expected size
    // [ batch, out_rows, out_cols, depth ]
    const int batch = input.dim_size(0);
    const int depth = input.dim_size(3);
    OP_REQUIRES(context,
                batch == out_backprop.dim_size(0) &&
                    out_rows == out_backprop.dim_size(1) &&
                    out_cols == out_backprop.dim_size(2) &&
                    depth == out_backprop.dim_size(3),
                errors::InvalidArgument("out_backprop has incompatible size."));

    // The computed in_backprop has the same dimensions as the input:
    // [ batch, input_rows, input_cols, depth ]
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &in_backprop));

    // If there is nothing to compute, return.
    if (input.shape().num_elements() == 0) {
      return;
    }
    if (std::is_same<T, float>::value) {
      functor::DilationBackpropInput<Device, T>()(
          context->eigen_device<Device>(), input.tensor<T, 4>(),
          filter.tensor<T, 3>(), out_backprop.tensor<T, 4>(), stride_rows,
          stride_cols, rate_rows, rate_cols, pad_top, pad_left,
          in_backprop->tensor<float, 4>());

    } else if (std::is_same<T, Eigen::half>::value ||
               std::is_same<T, Eigen::bfloat16>::value) {
      Tensor output_grad;
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(DT_FLOAT, in_backprop->shape(), &output_grad));
      functor::DilationBackpropInput<Device, T>()(
          context->eigen_device<Device>(), input.tensor<T, 4>(),
          filter.tensor<T, 3>(), out_backprop.tensor<T, 4>(), stride_rows,
          stride_cols, rate_rows, rate_cols, pad_top, pad_left,
          output_grad.tensor<float, 4>());

      const Tensor& output_grad_const = output_grad;
      CastFloatToSmallFloat<Device, T>{}(
          context->eigen_gpu_device(), output_grad_const.template flat<float>(),
          in_backprop->template flat<T>());
    } else if (std::is_same<T, double>::value) {
      functor::DilationBackpropInput<Device, T, T>()(
          context->eigen_device<Device>(), input.tensor<T, 4>(),
          filter.tensor<T, 3>(), out_backprop.tensor<T, 4>(), stride_rows,
          stride_cols, rate_rows, rate_cols, pad_top, pad_left,
          in_backprop->tensor<T, 4>());
    }
  }

  std::vector<int32> strides_;
  std::vector<int32> rates_;
  Padding padding_;
};

template <typename Device, typename T>
class DilationBackpropFilterOp : public OpKernel {
 public:
  explicit DilationBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    ParseAttributes(context, &strides_, &rates_, &padding_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // Determine relevant sizes from input and filters.
    int stride_rows = 0, stride_cols = 0;
    int rate_rows = 0, rate_cols = 0;
    int64 pad_top = 0, pad_left = 0;
    int64 out_rows = 0, out_cols = 0;
    ParseSizes(context, strides_, rates_, padding_, &stride_rows, &stride_cols,
               &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows,
               &out_cols);

    // Verify that the incoming gradient tensor has the expected size
    // [ batch, out_rows, out_cols, depth ]
    const int batch = input.dim_size(0);
    const int depth = input.dim_size(3);
    OP_REQUIRES(context,
                batch == out_backprop.dim_size(0) &&
                    out_rows == out_backprop.dim_size(1) &&
                    out_cols == out_backprop.dim_size(2) &&
                    depth == out_backprop.dim_size(3),
                errors::InvalidArgument("out_backprop has incompatible size."));

    // The computed filter_backprop has the same dimensions as the filter:
    // [ batch, input_rows, input_cols, depth ]
    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, filter.shape(), &filter_backprop));

    // If there is nothing to compute, return.
    if (filter.shape().num_elements() == 0) {
      return;
    }

    if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
      functor::DilationBackpropFilter<Device, T, T>()(
          context->eigen_device<Device>(), input.tensor<T, 4>(),
          filter.tensor<T, 3>(), out_backprop.tensor<T, 4>(), stride_rows,
          stride_cols, rate_rows, rate_cols, pad_top, pad_left,
          filter_backprop->tensor<T, 3>());
    } else if (std::is_same<T, Eigen::half>::value ||
               std::is_same<T, Eigen::bfloat16>::value) {
      Tensor output_grad;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_FLOAT, filter_backprop->shape(),
                                            &output_grad));
      functor::DilationBackpropFilter<Device, T>()(
          context->eigen_device<Device>(), input.tensor<T, 4>(),
          filter.tensor<T, 3>(), out_backprop.tensor<T, 4>(), stride_rows,
          stride_cols, rate_rows, rate_cols, pad_top, pad_left,
          output_grad.tensor<float, 3>());

      const Tensor& output_grad_const = output_grad;
      CastFloatToSmallFloat<Device, T>{}(
          context->eigen_gpu_device(), output_grad_const.template flat<float>(),
          filter_backprop->template flat<T>());
    }
  }

  std::vector<int32> strides_;
  std::vector<int32> rates_;
  Padding padding_;
};

#define REGISTER(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Dilation2D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DilationOp<GPUDevice, T>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_float(REGISTER);
TF_CALL_bfloat16(REGISTER);
TF_CALL_half(REGISTER);

#undef REGISTER

#define REGISTER_GRAD(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("Dilation2DBackpropInput")         \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T"),            \
                          DilationBackpropInputOp<GPUDevice, T>); \
                                                                  \
  REGISTER_KERNEL_BUILDER(Name("Dilation2DBackpropFilter")        \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T"),            \
                          DilationBackpropFilterOp<GPUDevice, T>);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GRAD);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_float(REGISTER_GRAD);
TF_CALL_bfloat16(REGISTER_GRAD);
TF_CALL_half(REGISTER_GRAD);

#undef REGISTER_GRAD
}  // namespace itex
