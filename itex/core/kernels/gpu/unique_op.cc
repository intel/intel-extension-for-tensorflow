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

#include "itex/core/kernels/gpu/unique_op.h"

#include <limits>

#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/str_util.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

// The algorithm implemented here is as follows:
// input = [3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6]
// 1) Sort the input to group equal values together in segments.
//      sorted_input, sorted_input_inds = sort(input)
// sorted_input:
//   [1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 8, 8, 9]
// sorted_input_inds:
//   [4, 17, 0, 2, 9, 3, 5, 14, 16, 1, 10, 18, 8, 15, 19, 11, 7, 12, 13, 6]
// 2) Identify the boundaries between segments and use prefix sum to
//    compute the unique ID for each sorted value.
//      sorted_input_unique_ids = prefix_sum(indicator(sorted_input))
// indicator(sorted_input):
//   [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
// sorted_input_unique_ids:
//   [0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 7, 7, 7, 8]
// 3) Extract the input index of the first occurrence of each unique value.
//    If counts are required, also extract the end index of each segment.
//      unique_input_inds[sorted_input_unique_ids] =
//          sorted_input_inds (@ indicator)
//      segment_ends[sorted_input_unique_ids[i] - 1] = i (@ indicator)
// unique_input_inds: [4, 17, 0, 3, 1, 8, 11, 7, 6]
// segment_ends: [1, 2, 5, 9, 12, 15, 16, 19, 20]
// 4) Sort the extracted unique input indices to put them in order of
//    first appearance.
//      sorted_unique_input_inds, sorted_unique_perm =
//          sort(unique_input_inds)
// sorted_unique_input_inds: [0, 1, 3, 4, 6, 7, 8, 11, 17]
// sorted_unique_perm: [2, 4, 3, 0, 8, 7, 5, 6, 1]
// 5) Gather the sorted unique input values to produce output, and invert
//    the second sort permutation to produce an inverse ID mapping. If
//    counts are required, also take the adjacent difference between
//    segment_ends indices to produce counts.
//      output = input[sorted_unique_input_inds]
//      inv_sorted_unique_perm[sorted_unique_perm[i]] = i
//      counts = adjacent_difference(segment_ends)
// output: [3, 5, 4, 1, 9, 8, 6, 7, 2]
// inv_sorted_unique_perm: [3, 8, 0, 2, 1, 6, 7, 5, 4]
// counts: [3, 3, 4, 1, 1, 3, 3, 1, 1]
// 6) Look up unique IDs via the inverse ID mapping and scatter them using
//    the original sort permutation to produce the indices output.
//      idx[sorted_input_inds] =
//          inv_sorted_unique_perm[sorted_input_unique_ids]
// idx: [0, 1, 0, 2, 3, 2, 4, 5, 6, 0, 1, 7, 5, 5, 2, 6, 2, 8, 1, 6]

// This only supports Unique[WithCounts], not Unique[WithCounts]V2.
template <typename KeyT, typename ValueT>
class UniqueOp : public OpKernel {
 public:
  explicit UniqueOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    sycl::queue* stream = device.stream();

    const Tensor& input = context->input(0);
    TensorShape in_shape = input.shape();
    const KeyT* const_input_ptr = input.flat<KeyT>().data();
    KeyT* input_ptr = const_cast<KeyT*>(const_input_ptr);
    // vectors to support large tensors.
    OP_REQUIRES(context,
                input.NumElements() <= std::numeric_limits<int32>::max(),
                errors::InvalidArgument(
                    "unique does not support input tensors larger than ",
                    std::numeric_limits<int32>::max(), " elements"));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument("unique expects a 1D vector."));

    int64_t input_size = input.NumElements();
    bool has_count_output = context->num_outputs();

    if (input_size == 0) {
      // Early exit for trivial case.
      Tensor* t = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({0}), &t));
      OP_REQUIRES_OK(context,
                     context->allocate_output(1, TensorShape({0}), &t));
      if (has_count_output) {
        OP_REQUIRES_OK(context,
                       context->allocate_output(2, TensorShape({0}), &t));
      }
      return;
    }

    Tensor sorted_input;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<KeyT>::value,
                                                   in_shape, &sorted_input));
    KeyT* sorted_input_ptr = sorted_input.flat<KeyT>().data();

    Tensor sorted_input_inds;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<ValueT>::value, in_shape,
                                        &sorted_input_inds));
    ValueT* sorted_input_inds_ptr = sorted_input_inds.flat<ValueT>().data();
    // step 1: sort the input pairs
    OP_REQUIRES_OK(
        context,
        impl::DispatchRadixSort<KeyT, ValueT, /*KEYS_PER_ITEM=*/8,
                                /*GROUP_SIZE=*/256, /*SUBGROUP_SIZE*/ 16>(
            context, input_size,
            /*keys_in = */ input_ptr,
            /*indices_in = */ static_cast<ValueT*>(nullptr),
            /*keys_out = */ sorted_input_ptr,
            /*indices_out = */ sorted_input_inds_ptr,
            /*num_bits = */ 30));

    Tensor sorted_input_unique_ids;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<ValueT>::value, in_shape,
                                        &sorted_input_unique_ids));
    ValueT* sorted_input_unique_ids_ptr =
        sorted_input_unique_ids.flat<ValueT>().data();
    // create a fancy input iterator to indicate segment boundaries.
    CountIterator<ValueT> counting_iter(0);
    TransformIterator<ValueT, SegmentIndicatorFunctor<KeyT>,
                      CountIterator<ValueT>>
        segment_indicator_iter(counting_iter, {sorted_input_ptr});
    // step 2: identify the boundaries between segments and use prefix sum to
    //        compute the unique ID for each sorted value.
    using BinaryOp = sycl::plus<ValueT>;
    impl::DispatchScan<decltype(segment_indicator_iter), ValueT*, BinaryOp>(
        context, input_size, segment_indicator_iter,
        sorted_input_unique_ids_ptr, (const KeyT)0, false, false, BinaryOp());

    int uniq_size = sorted_input_unique_ids_ptr[input_size - 1] + 1;

    Tensor unique_input_inds;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<ValueT>::value,
                                TensorShape({uniq_size}), &unique_input_inds));
    ValueT* unique_input_inds_ptr = unique_input_inds.flat<ValueT>().data();

    Tensor segment_ends;
    ValueT* segment_ends_ptr = nullptr;
    if (has_count_output) {
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<ValueT>::value,
                                  TensorShape({uniq_size}), &segment_ends));
      segment_ends_ptr = segment_ends.flat<ValueT>().data();
    }
    // step 3: extract the input index of the first occurrence of each unique
    // value.
    OP_REQUIRES_OK(context,
                   impl::ExtractFirstOccurrenceIndices<ValueT>(
                       stream, input_size, uniq_size, sorted_input_inds_ptr,
                       sorted_input_unique_ids_ptr, unique_input_inds_ptr,
                       segment_ends_ptr));

    Tensor sorted_unique_input_inds;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<ValueT>::value,
                                          TensorShape({uniq_size}),
                                          &sorted_unique_input_inds));
    ValueT* sorted_unique_input_inds_ptr =
        sorted_unique_input_inds.flat<ValueT>().data();
    Tensor sorted_unique_perm;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<ValueT>::value,
                                TensorShape({uniq_size}), &sorted_unique_perm));
    ValueT* sorted_unique_perm_ptr = sorted_unique_perm.flat<ValueT>().data();

    // step4: sort by input index so that output is in order of appearance.
    OP_REQUIRES_OK(
        context,
        impl::DispatchRadixSort<ValueT, ValueT, /*KEYS_PER_ITEM*/ 8,
                                /*GROUP_SIZE*/ 256, /*SUBGROUP_SIZE*/ 16>(
            context, uniq_size, unique_input_inds_ptr,
            static_cast<ValueT*>(nullptr), sorted_unique_input_inds_ptr,
            sorted_unique_perm_ptr, impl::Log2Ceiling(input_size)));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({uniq_size}), &output));
    KeyT* output_ptr = output->flat<KeyT>().data();

    Tensor inv_sorted_unique_perm;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<ValueT>::value,
                                          TensorShape({uniq_size}),
                                          &inv_sorted_unique_perm));
    ValueT* inv_sorted_unique_perm_ptr =
        inv_sorted_unique_perm.flat<ValueT>().data();

    ValueT* count_ptr = nullptr;
    if (has_count_output) {
      Tensor* count = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  2, TensorShape({uniq_size}), &count));
      count_ptr = count->flat<ValueT>().data();
    }

    // Compute output and counts (if necessary).
    OP_REQUIRES_OK(
        context, impl::GatherOutputsAndInvertPermutation<KeyT, ValueT>(
                     stream, uniq_size, input_ptr, sorted_unique_input_inds_ptr,
                     sorted_unique_perm_ptr, segment_ends_ptr, output_ptr,
                     inv_sorted_unique_perm_ptr, count_ptr));

    Tensor* idx = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(1, TensorShape({input_size}), &idx));
    ValueT* idx_ptr = idx->flat<ValueT>().data();

    // Compute indices output.
    OP_REQUIRES_OK(context, impl::LookupAndScatterUniqueIds<ValueT>(
                                stream, input_size, sorted_input_inds_ptr,
                                sorted_input_unique_ids_ptr,
                                inv_sorted_unique_perm_ptr, idx_ptr));
  }
};

// Register with int32 out_idx.
#define REGISTER_UNIQUE_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueOp<type, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueOp<type, int32>)

TF_CALL_float(REGISTER_UNIQUE_GPU);
#undef REGISTER_UNIQUE_GPU

}  // namespace itex
