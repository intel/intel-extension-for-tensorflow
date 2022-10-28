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

#include "itex/core/kernels/gpu/image/mirror_pad_op.h"

#include <string>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/mirror_pad_mode.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
Status GetAttrForMirrorPadMode(OpKernelConstruction* context,
                               StringPiece attr_name, MirrorPadMode* value) {
  std::string str_value;
  Status status = context->GetAttr<std::string>(attr_name, &str_value);
  if (str_value == "REFLECT") {
    *value = MirrorPadMode::REFLECT;
  } else if (str_value == "SYMMETRIC") {
    *value = MirrorPadMode::SYMMETRIC;
  } else {
    return errors::NotFound(str_value, " is not an allowed padding mode.");
  }
  return status;
}

template <typename Device, typename T, typename Tpaddings>
class MirrorPadOp : public OpKernel {
 public:
  explicit MirrorPadOp(OpKernelConstruction* context) : OpKernel(context) {
    MirrorPadMode mode;
    OP_REQUIRES_OK(context, GetAttrForMirrorPadMode(context, "mode", &mode));

    switch (mode) {
      case MirrorPadMode::SYMMETRIC: {
        offset_ = 0;
        break;
      }
      case MirrorPadMode::REFLECT: {
        offset_ = 1;
        break;
      }
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "mode must be either REFLECT or SYMMETRIC."));
    }
  }

  ~MirrorPadOp() override = default;

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();
    constexpr int kMinDims = 0;
    constexpr int kMaxDims = 5;
    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));
    OP_REQUIRES(
        context, dims == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), ", ", in0.shape().DebugString()));

    // Compute the shape of the output tensor, and allocate it.
    TensorShape output_shape;
    typename TTypes<Tpaddings>::ConstMatrix paddings = in1.matrix<Tpaddings>();
    for (int d = 0; d < dims; ++d) {
      const Tpaddings before = paddings(d, 0);  // Pad before existing elements.
      const Tpaddings after = paddings(d, 1);   // Pad after existing elements.
      OP_REQUIRES(context, before >= 0 && after >= 0,
                  errors::InvalidArgument(
                      "paddings must be non-negative: ", before, " ", after));
      if (offset_ == 0) {  // SYMMETRIC mode.
        OP_REQUIRES(context,
                    before <= in0.dim_size(d) && after <= in0.dim_size(d),
                    errors::InvalidArgument("paddings must be no greater "
                                            "than the dimension size: ",
                                            before, ", ", after,
                                            " greater than ", in0.dim_size(d)));
      } else if (offset_ == 1) {  // REFLECT mode.
        OP_REQUIRES(
            context, before < in0.dim_size(d) && after < in0.dim_size(d),
            errors::InvalidArgument("paddings must be less than"
                                    " the dimension size: ",
                                    before, ", ", after, " not less than ",
                                    in0.dim_size(d)));
      }

      output_shape.AddDim(before + in0.dim_size(d) + after);
    }

    if (output_shape.num_elements() == in0.NumElements()) {
      // When num_elements == 0, shape may have changed.
      Tensor out;
      ITEX_CHECK(out.CopyFrom(in0, output_shape));
      context->set_output(0, out);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define MIRROR_PAD_CASE(i)                                            \
  case i: {                                                           \
    functor::MirrorPad<Device, T, Tpaddings, i>()(                    \
        context->eigen_gpu_device(), To32Bit(output->tensor<T, i>()), \
        To32Bit(in0.tensor<T, i>()), paddings, offset_);              \
    break;                                                            \
  }

    // Invoke the dims-specific implementation.
    switch (dims) {
      MIRROR_PAD_CASE(1)
      MIRROR_PAD_CASE(2)
      MIRROR_PAD_CASE(3)
      MIRROR_PAD_CASE(4)
      MIRROR_PAD_CASE(5)
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Unsupported rank: ",
                                            in0.shape().DebugString()));
    }
#undef MIRROR_PAD_CASE
  }

 private:
  int offset_;
};

using GpuDevice = Eigen::GpuDevice;

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("MirrorPad")                       \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MirrorPadOp<GpuDevice, T, int32>);      \
  REGISTER_KERNEL_BUILDER(Name("MirrorPad")                       \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int64>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MirrorPadOp<GpuDevice, T, int64>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_GPU_KERNEL

// Gradient op.
template <typename Device, typename T, typename Tpaddings>
class MirrorPadGradOp : public OpKernel {
 public:
  explicit MirrorPadGradOp(OpKernelConstruction* context) : OpKernel(context) {
    MirrorPadMode mode;
    OP_REQUIRES_OK(context, GetAttrForMirrorPadMode(context, "mode", &mode));

    switch (mode) {
      case MirrorPadMode::SYMMETRIC: {
        offset_ = 0;
        break;
      }
      case MirrorPadMode::REFLECT: {
        offset_ = 1;
        break;
      }
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "mode must be either REFLECT or SYMMETRIC."));
    }
  }

  ~MirrorPadGradOp() override = default;

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();
    constexpr int kMinDims = 0;
    constexpr int kMaxDims = 5;
    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));
    OP_REQUIRES(
        context, dims == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), " ", in0.shape().DebugString()));

    // Compute the shape of the output tensor, and allocate it.
    TensorShape output_shape;
    typename TTypes<Tpaddings>::ConstMatrix paddings = in1.matrix<Tpaddings>();
    for (int d = 0; d < dims; ++d) {
      const Tpaddings before = paddings(d, 0);  // Pad before existing elements.
      const Tpaddings after = paddings(d, 1);   // Pad after existing elements.
      OP_REQUIRES(context, before >= 0 && after >= 0,
                  errors::InvalidArgument(
                      "Paddings must be non-negative: ", before, ", ", after));

      const int64 out_size = in0.dim_size(d) - (before + after);
      if (offset_ == 0) {  // SYMMETRIC mode.
        OP_REQUIRES(context, before <= out_size && after <= out_size,
                    errors::InvalidArgument("paddings must be no greater "
                                            "than the output dimension size: ",
                                            before, ", ", after,
                                            " greater than ", out_size));
      } else if (offset_ == 1) {  // REFLECT mode.
        OP_REQUIRES(context, before < out_size && after < out_size,
                    errors::InvalidArgument("paddings must be less than"
                                            " the output dimension size: ",
                                            before, ", ", after,
                                            " not less than ", out_size));
      }
      output_shape.AddDim(out_size);
    }

    if (output_shape == in0.shape()) {
      context->set_output(0, in0);
      return;
    }

    Tensor scratch;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   in0.shape(), &scratch));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define MIRROR_PAD_GRAD_CASE(k)                                       \
  case k: {                                                           \
    functor::MirrorPadGrad<Device, T, Tpaddings, k>()(                \
        context->eigen_gpu_device(), To32Bit(output->tensor<T, k>()), \
        To32Bit(in0.tensor<T, k>()), paddings, offset_,               \
        To32Bit(scratch.tensor<T, k>()));                             \
    break;                                                            \
  }

    // Invoke the dims-specific implementation.
    switch (dims) {
      MIRROR_PAD_GRAD_CASE(1);
      MIRROR_PAD_GRAD_CASE(2);
      MIRROR_PAD_GRAD_CASE(3);
      MIRROR_PAD_GRAD_CASE(4);
      MIRROR_PAD_GRAD_CASE(5);
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Unsupported rank: ",
                                            in0.shape().DebugString()));
    }
#undef MIRROR_PAD_GRAD_CASE
  }

 private:
  int offset_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("MirrorPadGrad")                   \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MirrorPadGradOp<GpuDevice, T, int32>);  \
  REGISTER_KERNEL_BUILDER(Name("MirrorPadGrad")                   \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int64>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MirrorPadGradOp<GpuDevice, T, int64>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_GPU_KERNEL

}  // namespace itex
