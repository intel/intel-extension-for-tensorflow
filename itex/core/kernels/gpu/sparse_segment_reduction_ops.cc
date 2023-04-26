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

#include "itex/core/kernels/gpu/sparse_segment_reduction_ops.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
namespace itex {

#define REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, index_type) \
  REGISTER_GPU_SPARSE_KERNELS(type, index_type, int32)                         \
  REGISTER_GPU_SPARSE_KERNELS(type, index_type, int64_t)
#define REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type)       \
  REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int32) \
  REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int64_t)

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type, segment_ids_type)       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSum")                                                \
          .Device(DEVICE_GPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumOp<GPUDevice, type, index_type,                \
                                  segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSumWithNumSegments")                                 \
          .Device(DEVICE_GPU)                                                 \
          .HostMemory("num_segments")                                         \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumWithNumSegmentsOp<GPUDevice, type, index_type, \
                                                 segment_ids_type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_SPARSE_KERNELS

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMean")                                         \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionMeanOp<GPUDevice, type, index_type,         \
                                   segment_ids_type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_SPARSE_KERNELS

}  // namespace itex
