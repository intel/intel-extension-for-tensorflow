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

#include "itex/core/kernels/gpu/bias_op.h"

#include <limits>
#include <string>

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
struct Constants {
  // Derive Index type. int (32-bit) or long (64-bit) depending on the
  // compile-time configuration. "float" here is not relevant.
  typedef TTypes<float>::Tensor::Index Index;
  Eigen::array<Index, 1> kZero;
  Eigen::array<Index, 1> kOne;

  Constants() {
    kZero[0] = 0;
    kOne[0] = 1;
  }
};

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width, int32* depth,
                      int32* channel) {
  *batch = 1;
  *height = 1;
  *width = 1;
  *depth = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32 channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    *batch = static_cast<int32>(value_tensor.dim_size(0));
    *channel = static_cast<int32>(value_tensor.dim_size(1));
    *height = static_cast<int32>(value_tensor.dim_size(2));
    if (value_tensor.dims() > 3) {
      *width = static_cast<int32>(value_tensor.dim_size(3));
    }
    if (value_tensor.dims() > 4) {
      *depth = static_cast<int32>(value_tensor.dim_size(4));
    }
  }
}

template <class T>
struct AccumulatorType {
  typedef T type;
};

// float is faster on the CPU than half, and also more precise,
// so use float for the temporary accumulators.
template <>
struct AccumulatorType<Eigen::half> {
  typedef float type;
};

template <typename Device, typename T>
class BiasOp;

template <typename T>
class BiasOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit BiasOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));

    // Added by intel_tf to support NCHW on CPU regardless of oneDNN used or
    // not.
    size_t channel_dim;
    if (data_format_ == FORMAT_NCHW) {
      channel_dim = 1;  // NCHW always have channel dim in 1 (with 3, 4, 5
                        // dimensions data).
    } else {
      channel_dim = input.shape().dims() - 1;  // End of code by intel_tf.
    }

    OP_REQUIRES(
        context,
        bias.shape().dim_size(0) == input.shape().dim_size(channel_dim),
        errors::InvalidArgument(
            "Must provide as many biases as the last dimension "
            "of the input tensor: ",
            bias.shape().DebugString(), " vs. ", input.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    // Added by intel_tf to support NCHW on CPU regardless of oneDNN used or
    // not.
    if (data_format_ == FORMAT_NCHW) {
      int32 batch, height, width, depth, channel;
      GetBiasValueDims(input, data_format_, &batch, &height, &width, &depth,
                       &channel);
      switch (input.shape().dims()) {
        case 3: {
          Eigen::DSizes<int32, 3> three_dims(1, channel, 1);
          Eigen::DSizes<int32, 3> broad_cast_dims(batch, 1, height);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 3>().device(d) =
              input.tensor<T, 3>() + bias.tensor<T, 1>()
                                         .reshape(three_dims)
                                         .broadcast(broad_cast_dims);
        } break;
        case 4: {
          Eigen::DSizes<int32, 4> four_dims(1, channel, 1, 1);
          Eigen::DSizes<int32, 4> broad_cast_dims(batch, 1, height, width);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 4>().device(d) =
              input.tensor<T, 4>() +
              bias.tensor<T, 1>().reshape(four_dims).broadcast(broad_cast_dims);
        } break;
        case 5: {
          Eigen::DSizes<int32, 5> five_dims(1, channel, 1, 1, 1);
          Eigen::DSizes<int32, 5> broad_cast_dims(batch, 1, height, width,
                                                  depth);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 5>().device(d) =
              input.tensor<T, 5>() +
              bias.tensor<T, 1>().reshape(five_dims).broadcast(broad_cast_dims);
        } break;
        default:
          OP_REQUIRES(context, false,
                      errors::InvalidArgument("Only ranks up to 5 supported: ",
                                              input.shape().DebugString()));
      }
      return;
    }  // End of code by intel_tf.

    switch (input.shape().dims()) {
      case 2:
        Compute<2>(context, input, bias, output);
        break;
      case 3:
        Compute<3>(context, input, bias, output);
        break;
      case 4:
        Compute<4>(context, input, bias, output);
        break;
      case 5:
        Compute<5>(context, input, bias, output);
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Only ranks up to 5 supported: ",
                                            input.shape().DebugString()));
    }
  }

  // Add biases for an input matrix of rank Dims, by using the Bias.
  template <int Dims>
  void Compute(OpKernelContext* ctx, const Tensor& input, const Tensor& bias,
               Tensor* output) {
    functor::Bias<Device, T, Dims> functor;
    functor(ctx->eigen_device<Device>(), input.tensor<T, Dims>(), bias.vec<T>(),
            output->tensor<T, Dims>());
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      BiasOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_GPU_KERNEL(int32);

template <typename Device, typename T>
class BiasGradOp : public OpKernel {
 public:
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);
    typedef Eigen::internal::SumReducer<T> SumReducer;
    typedef functor::ReduceFunctor<SumReducer> SumFunctor;
    SumReducer reducer;
    Constants<GPUDevice> constants;

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));

    OP_REQUIRES(
        context,
        FastBoundsCheck(output_backprop.NumElements(),
                        std::numeric_limits<int32>::max()),
        errors::InvalidArgument("BiasGrad requires tensor size <= int32 max"));

    int32 batch, height, width, depth, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &depth, &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (channel == 0) {
      return;  // Nothing to do
    } else if (output_backprop.NumElements() == 0) {
      // Eigen often crashes by design on empty tensors, SetZeroFunctor is safe
      functor::SetZeroFunctor<Device, T> fill;
      fill(context->eigen_gpu_device(), output->flat<T>());
    } else {
      // Added by intel_tf to support NCHW on CPU regardless of oneDNN used or
      // not.
      if (data_format_ == FORMAT_NCHW) {
        TensorShape three_dims_shape{batch, channel, height * width * depth};
        TensorShape shuffled_shape{channel, batch, height * width * depth};
        const int64_t num_reduced = batch * height * width * depth;

        Tensor backprop_reshaped;
        OP_REQUIRES(
            context,
            backprop_reshaped.CopyFrom(output_backprop, three_dims_shape),
            errors::Internal("Error during reduction copy."));
        Tensor shuffled;
        gtl::InlinedVector<int32, 8> perm({1, 0, 2});
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::value,
                                              shuffled_shape, &shuffled));
        OP_REQUIRES_OK(
            context, DoTranspose(context->eigen_gpu_device(), backprop_reshaped,
                                 perm, &shuffled));
        const Tensor& const_shuffled = shuffled;
        SumFunctor::Reduce(context, output->flat<T>(),
                           const_shuffled.shaped<T, 2>({channel, num_reduced}),
                           constants.kOne, reducer);
      } else {
        const int64_t num_reduced = batch * height * width * depth;
        TensorShape two_dims_shape{num_reduced, channel};

        Tensor backprop_reshaped;
        OP_REQUIRES(context,
                    backprop_reshaped.CopyFrom(output_backprop, two_dims_shape),
                    errors::Internal("Error during reduction copy."));
        SumFunctor::Reduce(
            context, output->flat<T>(),
            backprop_reshaped.shaped<T, 2>({num_reduced, channel}),
            constants.kZero, reducer);
      }
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasGradOp<GPUDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_KERNEL(float);
REGISTER_KERNEL(Eigen::bfloat16);
REGISTER_KERNEL(Eigen::half);
#undef REGISTER_KERNEL

}  // namespace itex
