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
#include "itex/core/kernels/gpu/sparse_segment_reduction_util.h"

#include "itex/core/kernels/gpu/segment_reduction_ops.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T, typename Index, typename SegmentId, typename ReductionF,
          typename AtomicReductionF>
Status SparseSegmentReductionFunctor<T, Index, SegmentId, ReductionF,
                                     AtomicReductionF>::
operator()(OpKernelContext* context, bool is_mean, bool is_sqrtn,
           T default_value, typename TTypes<T, 2>::ConstTensor input,
           const Index data_size, typename TTypes<Index>::ConstVec indices,
           typename TTypes<SegmentId>::ConstVec segment_ids,
           typename TTypes<Index>::Vec segment_offsets,
           typename TTypes<float, 2>::Tensor output) {
  const GPUDevice& d = context->eigen_gpu_device();
  // atomic operators only support fp32 now.
  output.device(d) = output.constant(static_cast<float>(default_value));

  const Index output_rows = static_cast<Index>(output.dimension(0));
  const Index input_outer_dim_size =
      static_cast<Index>(segment_ids.dimension(0));
  const Index input_inner_dim_size = output.size() / output_rows;

  TF_RETURN_IF_ERROR(LaunchSegmentOffsetsKernel<Index, SegmentId>()(
      d, input_outer_dim_size, output_rows, segment_ids.data(),
      segment_offsets.data()));

  const int OuterDimTileSize = 8;
  const Index input_outer_dim_num_stripe =
      Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));
  const Index total_stripe_count =
      input_inner_dim_size * input_outer_dim_num_stripe;
  TF_RETURN_IF_ERROR(
      LaunchSortedSegmentKernel<T, Index, SegmentId, OuterDimTileSize,
                                ReductionF, AtomicReductionF>()(
          d, input_outer_dim_size, input_inner_dim_size, output_rows,
          indices.data(), segment_ids.data(), segment_offsets.data(),
          input.data(), output.data(), total_stripe_count,
          static_cast<float>(default_value), is_mean, is_sqrtn));

  return Status::OK();
}

#define DEFINE_SORTED_GPU_SPECS_INDEX_SEGMENTID(T, T_Reduction, Index, \
                                                SegmentId)             \
  template struct SparseSegmentReductionFunctor<                       \
      T, Index, SegmentId, functor::NonAtomicSumOpGpu<T_Reduction>,    \
      functor::SumOpGpu<T_Reduction>>;

#define DEFINE_SORTED_GPU_SPECS_INDEX(T, T_Reduction, Index)             \
  DEFINE_SORTED_GPU_SPECS_INDEX_SEGMENTID(T, T_Reduction, Index, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX_SEGMENTID(T, T_Reduction, Index, int64);

#define DEFINE_SORTED_GPU_SPECS(T)            \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, T, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, T, int64);

#define DEFINE_SORTED_GPU_SPECS_BF16()                          \
  DEFINE_SORTED_GPU_SPECS_INDEX(Eigen::bfloat16, float, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(Eigen::bfloat16, float, int64);

#define DEFINE_SORTED_GPU_SPECS_HALF()                      \
  DEFINE_SORTED_GPU_SPECS_INDEX(Eigen::half, float, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(Eigen::half, float, int64);

TF_CALL_float(DEFINE_SORTED_GPU_SPECS);
DEFINE_SORTED_GPU_SPECS_BF16();
DEFINE_SORTED_GPU_SPECS_HALF();
#ifdef ITEX_ENABLE_DOUBLE
DEFINE_SORTED_GPU_SPECS_INDEX(double, float, int32);
DEFINE_SORTED_GPU_SPECS_INDEX(double, float, int64);
#endif  // ITEX_ENABLE_DOUBLE
#undef DEFINE_SORTED_GPU_SPECS
#undef DEFINE_SORTED_GPU_SPECS_BF16
#undef DEFINE_SORTED_GPU_SPECS_HALF
#undef DEFINE_SORTED_GPU_SPECS_INDEX
#undef DEFINE_SORTED_GPU_SPECS_INDEX_SEGMENTID
}  // namespace functor

// Static routines not in the templated class to reduce code size
Status ValidateSegmentReduction(OpKernelContext* context, const Tensor& input,
                                const Tensor& segment_ids) {
  if (!TensorShapeUtils::IsVectorOrHigher(input.shape())) {
    return errors::InvalidArgument("input must be at least rank 1");
  }
  if (!TensorShapeUtils::IsVector(segment_ids.shape())) {
    return errors::InvalidArgument("segment_ids should be a vector.");
  }
  const int64_t num_indices = segment_ids.NumElements();
  if (num_indices != input.dim_size(0)) {
    return errors::InvalidArgument(
        "segment_ids should be the same size as dimension 0 of"
        " input.");
  }

  return Status::OK();
}

// check routines not in the templated class to reduce code size
Status ValidateUnsortedSegmentReduction(OpKernel* op_kernel,
                                        OpKernelContext* context,
                                        const Tensor& data,
                                        const Tensor& segment_ids,
                                        const Tensor& num_segments) {
  if (!TensorShapeUtils::IsScalar(num_segments.shape())) {
    return errors::InvalidArgument(
        "num_segments should be a scalar, not shape ",
        num_segments.shape().DebugString());
  }

  if (!TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape())) {
    return errors::InvalidArgument("data.shape = ", data.shape().DebugString(),
                                   " does not start with segment_ids.shape = ",
                                   segment_ids.shape().DebugString());
  }

  return Status::OK();
}

Status ValidateSparseSegmentReduction(OpKernelContext* context,
                                      const Tensor& input,
                                      const Tensor& indices,
                                      const Tensor& segment_ids,
                                      bool has_num_segments) {
  if (has_num_segments) {
    const Tensor& num_segments_t = context->input(3);
    if (!TensorShapeUtils::IsScalar(num_segments_t.shape())) {
      return errors::InvalidArgument(
          "num_segments should be a scalar, not shape ",
          num_segments_t.shape().DebugString());
    }
    int64_t output_rows =
        internal::SubtleMustCopy(num_segments_t.dtype() == DT_INT32
                                     ? num_segments_t.scalar<int32>()()
                                     : num_segments_t.scalar<int64_t>()());
    if (output_rows < 0) {
      return errors::InvalidArgument("segment ids must be >= 0");
    }
  }

  if (!TensorShapeUtils::IsVector(indices.shape())) {
    return errors::InvalidArgument("indices should be a vector.");
  }

  if (!TensorShapeUtils::IsVector(segment_ids.shape())) {
    return errors::InvalidArgument("segment_ids should be a vector.");
  }

  const int64_t num_indices = indices.NumElements();
  if (num_indices != segment_ids.NumElements()) {
    return errors::InvalidArgument(
        "segment_ids and indices should have same size.");
  }

  if (input.dims() < 1) {
    return errors::InvalidArgument("Shape must be at least rank 1");
  }

  return Status::OK();
}

}  // namespace itex
