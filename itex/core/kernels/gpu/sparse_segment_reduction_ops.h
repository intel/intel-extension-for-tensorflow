/* Copyright (c) 2023 Intel Corporation

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef ITEX_CORE_KERNELS_GPU_SPARSE_SEGMENT_REDUCTION_OPS_H_
#define ITEX_CORE_KERNELS_GPU_SPARSE_SEGMENT_REDUCTION_OPS_H_

#include "itex/core/kernels/gpu/segment_reduction_ops.h"
#include "itex/core/kernels/gpu/sparse_segment_reduction_util.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/op_requires.h"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionMeanOp : public OpKernel {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : OpKernel(context) {}
};

template <typename T, typename Index, typename SegmentId>
class SparseSegmentReductionMeanOp<GPUDevice, T, Index, SegmentId>
    : public OpKernel {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : OpKernel(context),
        // TODO(itex): Read this value in the forth input when official TF
        // support this functionality.
        has_num_segments_(false),
        default_value_(T(0)) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    OP_REQUIRES_OK(
        context, ValidateSparseSegmentReduction(
                     context, input, indices, segment_ids, has_num_segments_));

    auto device = context->eigen_gpu_device();
    auto stream = device.stream();
    SegmentId last_segment_id_host;
    const Index num_indices = static_cast<Index>(indices.NumElements());
    auto last_segment_id_device =
        const_cast<Tensor&>(segment_ids).template flat<SegmentId>().data() +
        (num_indices - 1);
    stream
        ->memcpy(&last_segment_id_host, last_segment_id_device,
                 sizeof(SegmentId))
        .wait();
    SegmentId output_rows = last_segment_id_host + 1;
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    Tensor output_fp32;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                   output_shape, &output_fp32));

    auto input_flat = input.flat_outer_dims<T>();
    const auto indices_vec = indices.vec<Index>();
    const auto segment_ids_vec = segment_ids.vec<SegmentId>();

    // Allocate and compute segment_offsets.
    Tensor segment_offsets;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Index>::value,
                                          TensorShape({output_rows + 1}),
                                          &segment_offsets));
    auto segment_offsets_flat = segment_offsets.vec<Index>();
    functor::SparseSegmentReductionFunctor<T, Index, SegmentId,
                                           functor::NonAtomicSumOpGpu<float>,
                                           functor::SumOpGpu<float>>
        functor;
    // Atomic operators only support fp32 now.
    auto output_fp32_flat = output_fp32.flat_outer_dims<float>();
    OP_REQUIRES_OK(context,
                   functor(context, true, false, default_value_, input_flat,
                           input.NumElements(), indices_vec, segment_ids_vec,
                           segment_offsets_flat, output_fp32_flat));
    ConvertFromFp32<GPUDevice, T>(device, output->NumElements(),
                                  static_cast<float*>(output_fp32.data()),
                                  static_cast<T*>(output->data()));
  }

 private:
  const bool has_num_segments_;
  const T default_value_;
};

template <typename Index, typename SegmentId>
class SparseSegmentReductionMeanOp<GPUDevice, float, Index, SegmentId>
    : public OpKernel {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : OpKernel(context),
        // TODO(itex): Read this value in the forth input when official TF
        // support this functionality.
        has_num_segments_(false),
        default_value_(static_cast<float>(0)) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    OP_REQUIRES_OK(
        context, ValidateSparseSegmentReduction(
                     context, input, indices, segment_ids, has_num_segments_));

    auto device = context->eigen_gpu_device();
    auto stream = device.stream();
    SegmentId last_segment_id_host;
    const Index num_indices = static_cast<Index>(indices.NumElements());
    auto last_segment_id_device =
        const_cast<Tensor&>(segment_ids).template flat<SegmentId>().data() +
        (num_indices - 1);
    stream
        ->memcpy(&last_segment_id_host, last_segment_id_device,
                 sizeof(SegmentId))
        .wait();
    SegmentId output_rows = last_segment_id_host + 1;
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto input_flat = input.flat_outer_dims<float>();
    const auto indices_vec = indices.vec<Index>();
    const auto segment_ids_vec = segment_ids.vec<SegmentId>();
    auto output_flat = output->flat_outer_dims<float>();

    // Allocate and compute segment_offsets.
    Tensor segment_offsets;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Index>::value,
                                          TensorShape({output_rows + 1}),
                                          &segment_offsets));
    auto segment_offsets_flat = segment_offsets.vec<Index>();
    functor::SparseSegmentReductionFunctor<float, Index, SegmentId,
                                           functor::NonAtomicSumOpGpu<float>,
                                           functor::SumOpGpu<float>>
        functor;
    // Atomic operators only support fp32 now.
    OP_REQUIRES_OK(context,
                   functor(context, true, false, default_value_, input_flat,
                           input.NumElements(), indices_vec, segment_ids_vec,
                           segment_offsets_flat, output_flat));
  }

 private:
  const bool has_num_segments_;
  const float default_value_;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SPARSE_SEGMENT_REDUCTION_OPS_H_
