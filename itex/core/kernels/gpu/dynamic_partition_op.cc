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

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/kernels/gpu/gather_functor.h"
#include "itex/core/kernels/gpu/unique_op.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "itex/core/utils/util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace {

struct CountPartitionKernel {
  CountPartitionKernel(const int32* keys, int32* unique_ids,
                       int32* segment_ends, int32* out, int32 out_size,
                       int32 num_partitions)
      : keys_(keys),
        unique_ids_(unique_ids),
        segment_ends_(segment_ends),
        out_(out),
        out_size_(out_size),
        num_partitions_(num_partitions) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= out_size_) {
      return;
    }

    int32 unique_id = unique_ids_[id];
    int32 key = keys_[unique_id];
    int32 segment_end = segment_ends_[id];
    int32 segment_begin = 0;
    if (id > 0) {
      segment_begin = segment_ends_[id - 1];
    }
    if (FastBoundsCheck(key, num_partitions_)) {
      out_[key] = segment_end - segment_begin;
    }
  }

 private:
  const int32* keys_;
  int32* unique_ids_;
  int32* segment_ends_;
  int32* out_;
  int32 out_size_;
  int32 num_partitions_;
};

void CountPartition(const GPUDevice& d, const int32* keys, int32* unique_ids,
                    int32* segment_ends, int32* out, int32 uniq_size,
                    int32 num_partitions) {
  // We launch the kernel with size = min(num_partitions, uniq_size).
  // This is valid for correct inputs, because then num_partitions >= uniq_size.
  // For wrong inputs, we may have num_partitions < uniq_size. In this case we
  // will only handle the first num_partitions values.
  int32 out_size = std::min(uniq_size, num_partitions);
  auto stream = d.stream();
  int workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  sycl::range<1> local_range(workgroup_size);
  const int num_wg = (out_size + workgroup_size - 1) / workgroup_size;
  sycl::range<1> global_range(num_wg * workgroup_size);
  stream->submit([&](sycl::handler& cgh) {
    CountPartitionKernel task(keys, unique_ids, segment_ends, out, out_size,
                              num_partitions);
    cgh.parallel_for<CountPartitionKernel>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
}

}  // namespace

template <class T>
class DynamicPartitionOp : public OpKernel {
 public:
  explicit DynamicPartitionOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES(c, num_partitions_ >= 1,
                errors::InvalidArgument("num_partitions must be at least 1"));
  }

  void AllocateTempSpace(OpKernelContext* c, int32 N, Tensor* partitions_out,
                         Tensor* indices_out) {
    OP_REQUIRES_OK(
        c, c->allocate_temp(DT_INT32, TensorShape({N}), partitions_out));
    OP_REQUIRES_OK(c,
                   c->allocate_temp(DT_INT32, TensorShape({N}), indices_out));
  }

  void AllocateOutputs(OpKernelContext* c, const Tensor* data,
                       const Tensor* partitions, const Tensor* partition_count,
                       OpOutputList* Tout) {
    auto e_part_count = partition_count->flat<int32>();
    // Allocate output tensors of the right size
    OP_REQUIRES_OK(c, c->output_list("outputs", Tout));
    for (int p = 0; p < num_partitions_; p++) {
      TensorShape shape;
      shape.AddDim(e_part_count(p));
      for (int i = partitions->dims(); i < data->dims(); i++) {
        shape.AddDim(data->dim_size(i));
      }
      Tensor* out;
      OP_REQUIRES_OK(c, Tout->allocate(p, shape, &out));
    }
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& data = c->input(0);
    const Tensor& partitions = c->input(1);

    OP_REQUIRES(c,
                TensorShapeUtils::StartsWith(data.shape(), partitions.shape()),
                errors::InvalidArgument(
                    "data.shape must start with partitions.shape, ",
                    "got data.shape = ", data.shape().DebugString(),
                    ", partitions.shape = ", partitions.shape().DebugString()));

    Tensor partition_count;
    // We must handle the case of empty partitions separately,
    // because kernels don't work with 0-sized tensors.
    if (partitions.NumElements() == 0) {
      AllocatorAttributes alloc_attr;
      alloc_attr.set_on_host(true);
      OP_REQUIRES_OK(c,
                     c->allocate_temp(DT_INT32, TensorShape({num_partitions_}),
                                      &partition_count, alloc_attr));
      auto e_part_count = partition_count.flat<int32>();
      for (int i = 0; i < num_partitions_; i++) e_part_count(i) = 0;
      OpOutputList outputs;
      this->AllocateOutputs(c, &data, &partitions, &partition_count, &outputs);
      return;
    }

    // Prepare for counting.
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT32, TensorShape({num_partitions_}),
                                       &partition_count));
    Tensor indices_out;
    // Count how many times each partition index occurs.
    // Also sort the info in partitions and output it in indices_out,
    // in preparation for the next step.
    this->CountAndSortParts(c, &partitions, &partition_count, &indices_out);
    if (!c->status().ok()) return;

    // In order to allocate the output tensor we have to move partition_count
    // to CPU.
    auto* stream = c->GetDeviceStream();
    OP_REQUIRES(c, stream, errors::Internal("No GPU stream available."));
    Tensor cpu_tensor;
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(
        c, c->allocate_temp(partition_count.dtype(), partition_count.shape(),
                            &cpu_tensor, alloc_attr));

    // TF use async copy and set the subsequent series of operations as a
    // callback function, so the output buffers can be populated after copy has
    // been done. We first use synchronous copy to ensure functionality, and
    // plan to do the same to improve performance in the future.
    stream
        ->memcpy(cpu_tensor.flat<int32>().data(),
                 partition_count.flat<int32>().data(),
                 num_partitions_ * sizeof(int))
        .wait();

    OpOutputList outputs;
    this->AllocateOutputs(c, &data, &partitions, &cpu_tensor, &outputs);
    if (!c->status().ok()) {
      return;
    }
    int32 N = partitions.NumElements();
    int64 slice_size = data.NumElements() / N;
    this->GatherSlices(c, &data, &indices_out, N, slice_size, outputs);
  }

 protected:
  void RadixSort(OpKernelContext* c, const Tensor* partitions,
                 Tensor* partitions_out, Tensor* indices_out, int input_size) {
    OP_REQUIRES_OK(
        c, impl::DispatchRadixSort<int32, int32, /*KEYS_PER_ITEM=*/8,
                                   /*GROUP_SIZE=*/256, /*SUBGROUP_SIZE*/ 16>(
               c, input_size,
               /*keys_in = */
               const_cast<Tensor*>(partitions)->flat<int32>().data(),
               /*indices_in = */ static_cast<int32*>(nullptr),
               /*keys_out = */ partitions_out->flat<int32>().data(),
               /*indices_out = */ indices_out->flat<int32>().data()));
  }

  void CountAndSortParts(OpKernelContext* c, const Tensor* partitions,
                         Tensor* partition_count, Tensor* indices_out) {
    const GPUDevice& device = c->eigen_device<GPUDevice>();
    const auto& stream = c->GetDeviceStream();
    int32 N = partitions->NumElements();
    Tensor partitions_out;

    // Allocate memory for Radix-Sort.
    this->AllocateTempSpace(c, N, &partitions_out, indices_out);
    if (!c->status().ok()) return;
    this->RadixSort(c, partitions, &partitions_out, indices_out, N);
    if (!c->status().ok()) return;
    // We will now apply a reduce operation to count how many times
    // each index appears in partitions.

    // Zero-out the partition_count tensor.
    functor::SetZeroFunctor<GPUDevice, int32> zero_functor;
    zero_functor(device, partition_count->flat<int32>());

    // Currently, sycl don't support cooperatively kernel,
    // so we need use different implementation compared with cub
    // for calculating partition counts.
    int32* partitions_out_ptr = partitions_out.flat<int32>().data();
    Tensor sorted_unique_ids;
    OP_REQUIRES_OK(
        c, c->allocate_temp(DataTypeToEnum<int32>::value, partitions->shape(),
                            &sorted_unique_ids));
    int32* sorted_unique_ids_ptr = sorted_unique_ids.flat<int32>().data();
    // create a fancy input iterator to indicate segment boundaries.
    CountIterator<int32> counting_iter(0);
    TransformIterator<int32, SegmentIndicatorFunctor<int32>,
                      CountIterator<int32>>
        segment_indicator_iter(counting_iter, {partitions_out_ptr});
    // Identify the boundaries between segments and use prefix sum to
    // compute the unique ID for each sorted value.
    using BinaryOp = sycl::plus<int32>;
    impl::DispatchScan<decltype(segment_indicator_iter), int32*, BinaryOp>(
        c, N, segment_indicator_iter, sorted_unique_ids_ptr, (const int32)0,
        false, false, BinaryOp());

    int32 uniq_size;
    stream->memcpy(&uniq_size, &(sorted_unique_ids_ptr[N - 1]), sizeof(int))
        .wait();
    uniq_size += 1;

    Tensor unique_ids_out;
    OP_REQUIRES_OK(c,
                   c->allocate_temp(DataTypeToEnum<int32>::value,
                                    TensorShape({uniq_size}), &unique_ids_out));
    int32* unique_ids_out_ptr = unique_ids_out.flat<int32>().data();

    Tensor segment_ends;
    int32* segment_ends_ptr = nullptr;
    OP_REQUIRES_OK(c,
                   c->allocate_temp(DataTypeToEnum<int32>::value,
                                    TensorShape({uniq_size}), &segment_ends));
    segment_ends_ptr = segment_ends.flat<int32>().data();

    int32* indices_out_ptr = indices_out->flat<int32>().data();

    // Extract the input index of the first occurrence of each unique
    // value.
    OP_REQUIRES_OK(
        c, impl::ExtractFirstOccurrenceIndices<int32>(
               stream, N, uniq_size, indices_out_ptr, sorted_unique_ids_ptr,
               unique_ids_out_ptr, segment_ends_ptr));

    // We are not done yet. unique_out only contains the indices that appeared
    // at least once in partitions, and segment ends records each segment range.
    // We get partition_count based on them. This will handle possibly empty
    // parts.
    const int32* partitions_ptr = partitions->flat<int32>().data();
    CountPartition(device, partitions_ptr, unique_ids_out_ptr, segment_ends_ptr,
                   partition_count->flat<int32>().data(), uniq_size,
                   num_partitions_);
  }

  void GatherSlices(OpKernelContext* c, const Tensor* data,
                    const Tensor* indices, int32 N, int64 slice_size,
                    OpOutputList& outs) {  // NOLINT
    const GPUDevice& device = c->eigen_device<GPUDevice>();
    const int32* ind_base = indices->flat<int32>().data();
    const T* data_base = data->flat<T>().data();

    for (int p = 0; p < num_partitions_; p++) {
      int32 indices_size = outs[p]->dim_size(0);
      int64 out_size = outs[p]->NumElements();
      T* out_base = outs[p]->flat<T>().data();
      if (out_size > 0)
        LaunchGatherKernel<T, int32, /* is_axis_zero = */ true,
                           /*can_use_same_index = */ true>(
            device, data_base, ind_base, out_base, N, indices_size, slice_size,
            out_size);
      ind_base += indices_size;
    }
  }

  int32 num_partitions_;
};

#define REGISTER_DYNAMIC_PARTITION_GPU(T)                                 \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DynamicPartition").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DynamicPartitionOp<T>)

TF_CALL_int32(REGISTER_DYNAMIC_PARTITION_GPU);
TF_CALL_int64(REGISTER_DYNAMIC_PARTITION_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_PARTITION_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_DYNAMIC_PARTITION_GPU);
#undef REGISTER_DYNAMIC_PARTITION_GPU

}  // namespace itex
