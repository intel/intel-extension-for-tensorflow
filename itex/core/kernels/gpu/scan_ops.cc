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

#include "itex/core/kernels/gpu/scan_ops.h"

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, class T, typename Reducer, typename Tidx>
class ScanOp : public OpKernel {
 public:
  explicit ScanOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reverse", &reverse_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("exclusive", &exclusive_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& tensor_axis = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_axis.shape()),
                errors::InvalidArgument("ScanOp: axis must be a scalar, not ",
                                        tensor_axis.shape().DebugString()));

    const Tidx axis_arg =
        internal::SubtleMustCopy(tensor_axis.scalar<Tidx>()());
    const Tidx axis = (axis_arg < 0) ? input.dims() + axis_arg : axis_arg;
    OP_REQUIRES(ctx, FastBoundsCheck(axis, input.dims()),
                errors::InvalidArgument(
                    "ScanOp: Expected scan axis in the range [", -input.dims(),
                    ", ", input.dims(), "), but got ", axis));

    const TensorShape& output_shape = input.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // Exit early if there's nothing to compute
    if (output_shape.num_elements() == 0) return;

    Reducer reducer;

    // Dim reduction.
    int64 reduced_shape[3] = {1, 1, 1};
    for (Tidx i = 0; i < axis; ++i) {
      reduced_shape[0] *= input.dim_size(i);
    }
    reduced_shape[1] = input.dim_size(axis);
    for (Tidx i = axis + 1; i < input.dims(); ++i) {
      reduced_shape[2] *= input.dim_size(i);
    }

    functor::Scan<Reducer, T>()(ctx, input.shaped<T, 3>(reduced_shape),
                                output->shaped<T, 3>(reduced_shape), reducer,
                                reverse_, exclusive_, reduced_shape[0],
                                reduced_shape[1], reduced_shape[2]);
  }

 private:
  bool reverse_;
  bool exclusive_;
};

#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_GPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int32>("Tidx")                                  \
          .HostMemory("axis"),                                            \
      ScanOp<GPUDevice, type, Eigen::internal::ProdReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_GPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int64>("Tidx")                                  \
          .HostMemory("axis"),                                            \
      ScanOp<GPUDevice, type, Eigen::internal::ProdReducer<type>, int64>)

TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int32>("Tidx")                                 \
          .HostMemory("axis"),                                           \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int64>("Tidx")                                 \
          .HostMemory("axis"),                                           \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>, int64>)

TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

#define REGISTER_GPU_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("CumulativeLogsumexp")                                      \
          .Device(DEVICE_GPU)                                          \
          .TypeConstraint<type>("T")                                   \
          .TypeConstraint<int32>("Tidx")                               \
          .HostMemory("axis"),                                         \
      ScanOp<GPUDevice, type, functor::LogSumExpReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("CumulativeLogsumexp")                                      \
          .Device(DEVICE_GPU)                                          \
          .TypeConstraint<type>("T")                                   \
          .TypeConstraint<int64>("Tidx")                               \
          .HostMemory("axis"),                                         \
      ScanOp<GPUDevice, type, functor::LogSumExpReducer<type>, int64>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

}  // namespace itex
