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

#include "itex/core/kernels/gpu/segment_reduction_ops.h"

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

// Static check routines not in the templated class to reduce code size
static void UnsortedSegmentReductionValidation(OpKernelContext* context,
                                               const Tensor& data,
                                               const Tensor& segment_ids,
                                               const Tensor& num_segments) {
  OP_REQUIRES(
      context, num_segments.shape().dims() == 0,
      errors::InvalidArgument("num_segments should be a scalar, not shape ",
                              num_segments.shape().DebugString()));
  OP_REQUIRES(
      context, TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape()),
      errors::InvalidArgument("data.shape = ", data.shape().DebugString(),
                              " does not start with segment_ids.shape = ",
                              segment_ids.shape().DebugString()));
}

static bool UnsortedSegmentReductionDoValidation(OpKernelContext* context,
                                                 const Tensor& data,
                                                 const Tensor& segment_ids,
                                                 const Tensor& num_segments) {
  UnsortedSegmentReductionValidation(context, data, segment_ids, num_segments);
  return context->status().ok();
}

template <typename T, typename Index, typename DeviceReductionFunctor>
class UnsortedSegmentReductionOp : public OpKernel {
 public:
  explicit UnsortedSegmentReductionOp(OpKernelConstruction* context)
      : OpKernel(context), reduction_functor_(DeviceReductionFunctor()) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& segment_ids = context->input(1);
    const Tensor& num_segments = context->input(2);
    if (!UnsortedSegmentReductionDoValidation(context, data, segment_ids,
                                              num_segments)) {
      return;
    }
    const auto segment_flat = segment_ids.flat<Index>();
    const Index output_rows =
        internal::SubtleMustCopy(num_segments.scalar<int32>()());
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("Input num_segments == ", output_rows,
                                        " must not be negative."));
    TensorShape output_shape;
    output_shape.AddDim(output_rows);
    for (int i = segment_ids.dims(); i < data.dims(); i++) {
      output_shape.AddDim(data.dim_size(i));
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_flat = output->flat_outer_dims<T>();
    Tensor output_fp32;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                   output_shape, &output_fp32));
    auto output_fp32_flat = output_fp32.flat_outer_dims<float>();

    auto data_ptr = data.template flat<T>().data();
    reduction_functor_(context, output_rows, segment_ids.shape(), segment_flat,
                       data.NumElements(), data_ptr, output_flat,
                       output_fp32_flat);
  }

 protected:
  DeviceReductionFunctor reduction_functor_;
};

#define REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                 \
    name, type, index_type, initial_value_functor, reduction_kernel_functor) \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name(name)                                                             \
          .Device(DEVICE_GPU)                                                \
          .HostMemory("num_segments")                                        \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tindices"),                           \
      UnsortedSegmentReductionOp<                                            \
          type, index_type,                                                  \
          functor::UnsortedSegmentFunctor<GPUDevice, type, index_type,       \
                                          initial_value_functor,             \
                                          reduction_kernel_functor> >)

// sum is the only op that supports all input types currently
#define REGISTER_REAL_GPU_UNSORTED_KERNELS(type, reduction_type, index_type) \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                       \
      "UnsortedSegmentMax", type, index_type, itex::functor::Lowest<type>,   \
      itex::functor::MaxOpGpu<reduction_type>);                              \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                       \
      "UnsortedSegmentMin", type, index_type, itex::functor::Highest<type>,  \
      itex::functor::MinOpGpu<reduction_type>);                              \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                       \
      "UnsortedSegmentProd", type, index_type,                               \
      itex::functor::One<reduction_type>,                                    \
      itex::functor::ProdOpGpu<reduction_type>);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS(type, reduction_type, index_type) \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                      \
      "UnsortedSegmentSum", type, index_type,                               \
      itex::functor::Zero<reduction_type>,                                  \
      itex::functor::SumOpGpu<reduction_type>);

#define REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL(type)           \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(type, type, itex::int32); \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(type, type, itex::int64);

#define REGISTER_REAL_GPU_UNSORTED_KERNELS_BF16()                          \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(Eigen::bfloat16, float, itex::int32); \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(Eigen::bfloat16, float, itex::int64);

#define REGISTER_REAL_GPU_UNSORTED_KERNELS_HALF()                      \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(Eigen::half, float, itex::int32); \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(Eigen::half, float, itex::int64);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL(type)           \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, type, itex::int32); \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, type, itex::int64);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS_BF16()                          \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(Eigen::bfloat16, float, itex::int32); \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(Eigen::bfloat16, float, itex::int64);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS_HALF()                      \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(Eigen::half, float, itex::int32); \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(Eigen::half, float, itex::int64);

TF_CALL_float(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_int32(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_float(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_int32(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_complex64(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_complex128(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);

TF_CALL_double(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
#endif

REGISTER_REAL_GPU_UNSORTED_KERNELS_BF16();
REGISTER_SUM_GPU_UNSORTED_KERNELS_BF16();
REGISTER_REAL_GPU_UNSORTED_KERNELS_HALF();
REGISTER_SUM_GPU_UNSORTED_KERNELS_HALF();

#undef REGISTER_GPU_KERNEL_UNSORTEDSEGMENT
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS_BF16
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS_BF16
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS_HALF
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS_HALF

template <class T, class Index, class SegmentReductionFunctor>
class SegmentReductionGPUOp : public OpKernel {
 public:
  explicit SegmentReductionGPUOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& segment_ids = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));

    const int64 num_indices = segment_ids.NumElements();
    OP_REQUIRES(context, num_indices == input.dim_size(0),
                errors::InvalidArgument(
                    "segment_ids should be the same size as dimension 0 of"
                    " input."));

    if (num_indices == 0) {
      TensorShape output_shape = input.shape();
      output_shape.set_dim(0, 0);

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
      return;
    }
    auto output_rows_device =
        const_cast<Tensor&>(segment_ids).template flat<Index>().data() +
        (num_indices - 1);

    auto* stream = context->GetDeviceStream();

    OP_REQUIRES(context, stream != nullptr,
                errors::Internal("No GPU stream available."));

    const GPUDevice& d = context->eigen_device<GPUDevice>();

    Index output_rows;
    d.memcpyDeviceToHost(&output_rows, output_rows_device, sizeof(Index));
    output_rows++;

    SegmentReductionFunctor functor_;
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    /*
    // The determinism check is here, rather than inside the functor (as it is
    // for the unsorted segment reduction ops) because the done callback
    // (required for OP_REQUIRES_ASYNC) is not available inside the functor.
    bool determinism_requirement_met =
        SegmentReductionFunctor::atomic_reduction_is_associative ||
        !RequireDeterminism() ||
        DisableSegmentReductionOpDeterminismExceptions();
    OP_REQUIRES_ASYNC(
        context, determinism_requirement_met,
        errors::Unimplemented(
            "Deterministic GPU implementation of sorted segment reduction op"
            " not available."),
        done);
    */
    auto output_flat = output->flat_outer_dims<T>();
    Tensor output_fp32;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                   output_shape, &output_fp32));
    auto output_fp32_flat = output_fp32.flat_outer_dims<float>();

    auto data_ptr = input.template flat<T>().data();
    auto segment_flat = segment_ids.flat<Index>();
    functor_(context, output_rows, segment_ids.shape(), segment_flat,
             input.NumElements(), data_ptr, output_flat, output_fp32_flat);
  }
};

#define REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                   \
    name, type, index_type, initial_value_functor, reduction_kernel_functor, \
    atomic_reduction_kernel_functor)                                         \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name(name)                                                             \
          .Device(DEVICE_GPU)                                                \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tindices"),                           \
      SegmentReductionGPUOp<                                                 \
          type, index_type,                                                  \
          itex::functor::SegmentReductionFunctor<                            \
              type, index_type, initial_value_functor,                       \
              reduction_kernel_functor, atomic_reduction_kernel_functor> >)

#define REGISTER_GPU_SORTED_KERNELS(type, reduction_type, index_type)         \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                          \
      "SegmentSum", type, index_type, itex::functor::Zero<reduction_type>,    \
      itex::functor::NonAtomicSumOpGpu<reduction_type>,                       \
      itex::functor::SumOpGpu<reduction_type>);                               \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                          \
      "SegmentProd", type, index_type, itex::functor::One<reduction_type>,    \
      itex::functor::NonAtomicProdOpGpu<reduction_type>,                      \
      itex::functor::ProdOpGpu<reduction_type>);                              \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                          \
      "SegmentMin", type, index_type, itex::functor::Highest<reduction_type>, \
      itex::functor::NonAtomicMinOpGpu<reduction_type>,                       \
      itex::functor::MinOpGpu<reduction_type>);                               \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                          \
      "SegmentMax", type, index_type, itex::functor::Lowest<reduction_type>,  \
      itex::functor::NonAtomicMaxOpGpu<reduction_type>,                       \
      itex::functor::MaxOpGpu<reduction_type>);

#define REGISTER_GPU_SORTED_KERNELS_ALL(type)           \
  REGISTER_GPU_SORTED_KERNELS(type, type, itex::int32); \
  REGISTER_GPU_SORTED_KERNELS(type, type, itex::int64);

#define REGISTER_GPU_SORTED_KERNELS_BF16()                          \
  REGISTER_GPU_SORTED_KERNELS(Eigen::bfloat16, float, itex::int32); \
  REGISTER_GPU_SORTED_KERNELS(Eigen::bfloat16, float, itex::int64);

#define REGISTER_GPU_SORTED_KERNELS_HALF()                      \
  REGISTER_GPU_SORTED_KERNELS(Eigen::half, float, itex::int32); \
  REGISTER_GPU_SORTED_KERNELS(Eigen::half, float, itex::int64);

TF_CALL_float(REGISTER_GPU_SORTED_KERNELS_ALL);
TF_CALL_int32(REGISTER_GPU_SORTED_KERNELS_ALL);
REGISTER_GPU_SORTED_KERNELS_BF16();
REGISTER_GPU_SORTED_KERNELS_HALF();
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_GPU_SORTED_KERNELS(double, float, itex::int32);
REGISTER_GPU_SORTED_KERNELS(double, float, itex::int64);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNEL_SORTEDSEGMENT
#undef REGISTER_GPU_SORTED_KERNELS
#undef REGISTER_GPU_SORTED_KERNELS_ALL
#undef REGISTER_GPU_SORTED_KERNELS_BF16
#undef REGISTER_GPU_SORTED_KERNELS_HALF
};  // namespace itex
