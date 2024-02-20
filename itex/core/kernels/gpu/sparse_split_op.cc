/* Copyright (c) 2021-2023 Intel Corporation

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

#include "itex/core/kernels/gpu/sparse_split_op.h"

#include <algorithm>
#include <vector>

#include "itex/core/kernels/gpu/unique_op.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {

template <typename T>
using DeviceWriteAcc = sycl::accessor<T, 1, sycl::access::mode::write,
                                      sycl::access::target::device>;

template <typename Index>
inline Index GetSliceIndex(const Index index, const Index split_size,
                           const Index residual) {
  if (residual == 0) return index / split_size;
  const Index offset = residual * (split_size + Index(1));
  if (index < offset) {
    return index / (split_size + Index(1));
  } else {
    return residual + ((index - offset) / split_size);
  }
}

template <typename Index>
inline Index GetDimensionInSlice(const Index index, const Index split_size,
                                 const Index residual) {
  if (residual == 0) return index % split_size;
  const Index offset = residual * (split_size + 1);
  if (index < offset) {
    return index % (split_size + 1);
  } else {
    return (index - offset) % split_size;
  }
}

template <typename Index>
inline Index GetSliceShape(const Index slice_index, const Index split_size,
                           const Index residual) {
  if (residual == 0) return split_size;
  if (slice_index < residual) {
    return split_size + 1;
  } else {
    return split_size;
  }
}

template <typename Index>
struct SliceIndexer {
  SliceIndexer(const Index split_dim_size, const Index num_split)
      : split_size_(split_dim_size / num_split),
        residual_(split_dim_size % num_split) {}

  inline Index GetSliceIndex(const Index index) const {
    return itex::functor::GetSliceIndex(index, split_size_, residual_);
  }

  inline Index GetIndexInSlice(const Index index) const {
    return GetDimensionInSlice(index, split_size_, residual_);
  }

  inline Index GetSliceSize(const Index slice_index) const {
    return GetSliceShape(slice_index, split_size_, residual_);
  }

 private:
  const Index split_size_;
  const Index residual_;
};

template <typename Index>
struct SparseSplitSliceIndexesKernel {
  SparseSplitSliceIndexesKernel(Index input_nnz, int rank, int axis,
                                SliceIndexer<Index> slice_indexer,
                                const Index* input_indices, int* slice_indexes)
      : input_nnz_(input_nnz),
        rank_(rank),
        axis_(axis),
        slice_indexer_(slice_indexer),
        input_indices_(input_indices),
        slice_indexes_(slice_indexes) {}

  void operator()(sycl::nd_item<1> item) const {
    auto input_nz = item.get_global_linear_id();
    if (Index(input_nz) < input_nnz_) {
      slice_indexes_[input_nz] = slice_indexer_.GetSliceIndex(
          input_indices_[input_nz * rank_ + axis_]);
    }
  }

 private:
  Index input_nnz_;
  int rank_;
  int axis_;
  SliceIndexer<Index> slice_indexer_;
  const Index* input_indices_;
  int* slice_indexes_;
};

template <typename Index>
Status LaunchSparseSplitSliceIndexesKernel(const GPUDevice& device,
                                           Index input_nnz, int rank, int axis,
                                           SliceIndexer<Index> slice_indexer,
                                           const Index* input_indices,
                                           int* slice_indexes) {
  if (input_nnz == 0) return Status::OK();
  device.stream()->submit([&](sycl::handler& cgh) {
    auto max_group_size =
        device.stream()
            ->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    auto num_work_group = (input_nnz + max_group_size - 1) / max_group_size;
    sycl::range<1> local_range(max_group_size);
    sycl::range<1> global_range(max_group_size * num_work_group);

    SparseSplitSliceIndexesKernel<Index> task(
        input_nnz, rank, axis, slice_indexer, input_indices, slice_indexes);
    cgh.parallel_for<SparseSplitSliceIndexesKernel<Index>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

template <typename Index>
struct SparseSplitFindSliceEndsKernel {
  SparseSplitFindSliceEndsKernel(Index input_nnz, int num_split,
                                 const int* sorted_slice_indexes,
                                 Index* slice_ends)
      : input_nnz_(input_nnz),
        num_split_(num_split),
        sorted_slice_indexes_(sorted_slice_indexes),
        slice_ends_(slice_ends) {}

  void operator()(sycl::nd_item<1> item) const {
    auto slice_index = item.get_global_linear_id();
    if (Index(slice_index) < num_split_) {
      slice_ends_[slice_index] =
          std::upper_bound(sorted_slice_indexes_,
                           sorted_slice_indexes_ + input_nnz_, slice_index) -
          sorted_slice_indexes_;
    }
  }

 private:
  Index input_nnz_;
  int num_split_;
  const int* sorted_slice_indexes_;
  Index* slice_ends_;
};

template <typename Index>
Status LaunchSparseSplitFindSliceEndsKernel(const GPUDevice& device,
                                            Index input_nnz, int num_split,
                                            const int* sorted_slice_indexes,
                                            Index* slice_ends) {
  device.stream()->submit([&](sycl::handler& cgh) {
    auto max_group_size =
        device.stream()
            ->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    auto num_work_group = (num_split + max_group_size - 1) / max_group_size;
    sycl::range<1> local_range(max_group_size);
    sycl::range<1> global_range(max_group_size * num_work_group);

    SparseSplitFindSliceEndsKernel<Index> task(
        input_nnz, num_split, sorted_slice_indexes, slice_ends);
    cgh.parallel_for<SparseSplitFindSliceEndsKernel<Index>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

template <typename T, typename Index>
struct SparseSplitScatterKernel {
  SparseSplitScatterKernel(Index input_nnz, int rank, int axis,
                           SliceIndexer<Index> slice_indexer,
                           const Index* sort_permutation,
                           const Index* slice_ends, const Index* input_indices,
                           const T* input_values,
                           DeviceWriteAcc<Index*> output_indices_acc,
                           DeviceWriteAcc<T*> output_values_acc)
      : input_nnz_(input_nnz),
        rank_(rank),
        axis_(axis),
        slice_indexer_(slice_indexer),
        sort_permutation_(sort_permutation),
        slice_ends_(slice_ends),
        input_indices_(input_indices),
        input_values_(input_values),
        output_indices_acc_(output_indices_acc),
        output_values_acc_(output_values_acc) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (Index(id) >= input_nnz_) return;

    Index sorted_input_nz = Index(id);
    Index input_nz = sort_permutation_[sorted_input_nz];
    int slice_index =
        slice_indexer_.GetSliceIndex(input_indices_[input_nz * rank_ + axis_]);
    Index slice_nz =
        sorted_input_nz -
        (slice_index == 0 ? Index(0) : slice_ends_[slice_index - 1]);
    output_values_acc_[slice_index][slice_nz] = input_values_[input_nz];
    for (int dim = 0; dim < rank_; ++dim) {
      Index input_index = input_indices_[input_nz * rank_ + dim];
      output_indices_acc_[slice_index][slice_nz * rank_ + dim] =
          (dim == axis_) ? slice_indexer_.GetIndexInSlice(input_index)
                         : input_index;
    }
  }

 private:
  Index input_nnz_;
  int rank_;
  int axis_;
  SliceIndexer<Index> slice_indexer_;
  const Index* sort_permutation_;
  const Index* slice_ends_;
  const Index* input_indices_;
  const T* input_values_;
  DeviceWriteAcc<Index*> output_indices_acc_;
  DeviceWriteAcc<T*> output_values_acc_;
};

template <typename T, typename Index>
Status LaunchSparseSplitScatterKernel(
    const GPUDevice& device, Index input_nnz, int rank, int axis,
    SliceIndexer<Index> slice_indexer, const Index* sort_permutation,
    const Index* slice_ends, const Index* input_indices, const T* input_values,
    sycl::buffer<Index*>* output_indices, sycl::buffer<T*>* output_values) {
  if (input_nnz == 0) return Status::OK();
  device.stream()->submit([&](sycl::handler& cgh) {
    auto max_group_size =
        device.stream()
            ->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    auto num_work_group = (input_nnz + max_group_size - 1) / max_group_size;
    sycl::range<1> local_range(max_group_size);
    sycl::range<1> global_range(max_group_size * num_work_group);
    sycl::accessor output_indices_acc(*output_indices, cgh, sycl::write_only);
    sycl::accessor output_values_acc(*output_values, cgh, sycl::write_only);

    SparseSplitScatterKernel<T, Index> task(
        input_nnz, rank, axis, slice_indexer, sort_permutation, slice_ends,
        input_indices, input_values, output_indices_acc, output_values_acc);
    cgh.parallel_for<SparseSplitScatterKernel<T, Index>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

}  // namespace

template <typename T>
struct SparseSplitFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const TensorShape& dense_shape,
                  const int64_t axis, const int num_split) {
    using Index = int64_t;

    const Index input_nnz = input_indices.dim_size(0);
    const Index split_dim_size = dense_shape.dim_size(static_cast<int>(axis));
    const int rank = dense_shape.dims();

    const Index* input_indices_ptr = input_indices.matrix<Index>().data();
    const T* input_values_ptr = input_values.vec<T>().data();

    const SliceIndexer<Index> slice_indexer(split_dim_size, num_split);

    auto& device = context->eigen_gpu_device();
    auto stream = device.stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    Tensor sort_permutation;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Index>::value,
                                                   TensorShape({input_nnz}),
                                                   &sort_permutation));
    Index* sort_permutation_ptr = sort_permutation.vec<Index>().data();

    Tensor slice_ends;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<Index>::value,
                                        TensorShape({num_split}), &slice_ends));
    Index* slice_ends_ptr = slice_ends.vec<Index>().data();

    // First we compute the slice index for each element, sort them, and use a
    // binary search to find the end of each slice.
    {
      Tensor slice_indexes;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT32, TensorShape({input_nnz}),
                                            &slice_indexes));
      int* slice_indexes_ptr = slice_indexes.vec<int>().data();

      OP_REQUIRES_OK(context, LaunchSparseSplitSliceIndexesKernel(
                                  device, input_nnz, rank, axis, slice_indexer,
                                  input_indices_ptr, slice_indexes_ptr));

      Tensor sorted_slice_indexes;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT32, TensorShape({input_nnz}),
                                            &sorted_slice_indexes));
      int* sorted_slice_indexes_ptr = sorted_slice_indexes.vec<int>().data();

      if (input_nnz > 0) {
        OP_REQUIRES_OK(
            context,
            impl::DispatchRadixSort<int, Index, /*KEYS_PER_ITEM=*/8,
                                    /*GROUP_SIZE=*/256, /*SUBGROUP_SIZE*/ 16>(
                context, input_nnz,
                /*keys_in = */ slice_indexes_ptr,
                /*indices_in = */ static_cast<Index*>(nullptr),
                /*keys_out = */ sorted_slice_indexes_ptr,
                /*indices_out = */ sort_permutation_ptr));
      }

      OP_REQUIRES_OK(context, LaunchSparseSplitFindSliceEndsKernel(
                                  device, input_nnz, num_split,
                                  sorted_slice_indexes_ptr, slice_ends_ptr));
    }

    // Copy the slice ends to the host so that we can compute the output shapes.
    std::vector<Index> slice_ends_host(num_split);
    stream
        ->memcpy(const_cast<Index*>(slice_ends_host.data()), slice_ends_ptr,
                 sizeof(Index) * num_split)
        .wait();

    // ouptput_indices and output_values are buffers on GPU. We need accessor to
    // access them on host.
    sycl::buffer<Index*> output_indices(num_split);
    sycl::buffer<T*> output_values(num_split);
    // Allocate output indices and values
    {
      sycl::host_accessor output_indices_acc(output_indices, sycl::write_only);
      sycl::host_accessor output_values_acc(output_values, sycl::write_only);
      for (int slice_index = 0; slice_index < num_split; ++slice_index) {
        Index slice_nnz =
            slice_ends_host[slice_index] -
            (slice_index == 0 ? Index(0) : slice_ends_host[slice_index - 1]);

        Tensor* output_inds = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(slice_index, {slice_nnz, rank},
                                                &output_inds));
        output_indices_acc[slice_index] = output_inds->matrix<Index>().data();
        Tensor* output_vals = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(num_split + slice_index,
                                                {slice_nnz}, &output_vals));
        output_values_acc[slice_index] = output_vals->vec<T>().data();
        Tensor* output_shape = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(num_split * 2 + slice_index,
                                                {rank}, &output_shape));
        for (int dim = 0; dim < rank; ++dim) {
          output_shape->vec<int64_t>()(dim) =
              (dim == axis) ? slice_indexer.GetSliceSize(slice_index)
                            : dense_shape.dim_size(dim);
        }
      }
    }

    // Launch Scatter kernel
    OP_REQUIRES_OK(context,
                   LaunchSparseSplitScatterKernel(
                       device, input_nnz, rank, axis, slice_indexer,
                       sort_permutation_ptr, slice_ends_ptr, input_indices_ptr,
                       input_values_ptr, &output_indices, &output_values));
    stream->wait();
  }
};

}  // namespace functor

namespace {

template <typename T>
void SparseSplitOpImpl(OpKernelContext* context, int num_split) {
  const Tensor& input_axis = context->input(0);
  const Tensor& input_indices = context->input(1);
  const Tensor& input_values = context->input(2);
  const Tensor& input_shape = context->input(3);

  OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_axis.shape()),
              errors::InvalidArgument(
                  "Input axis should be a scalar but received shape ",
                  input_axis.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
              errors::InvalidArgument(
                  "Input indices should be a matrix but received shape ",
                  input_indices.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values.shape()),
              errors::InvalidArgument(
                  "Input values should be a vector but received shape ",
                  input_indices.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape.shape()),
              errors::InvalidArgument(
                  "Input shape should be a vector but received shape ",
                  input_shape.shape().DebugString()));

  const int64_t axis_input = input_axis.scalar<int64_t>()();
  const int64_t input_rank = input_shape.vec<int64_t>().size();
  const int64_t axis = (axis_input < 0) ? input_rank + axis_input : axis_input;

  OP_REQUIRES(
      context, axis >= 0 && axis < input_rank,
      errors::InvalidArgument("Input axis should be in range [", -input_rank,
                              ", ", input_rank, "), got ", axis_input));

  OP_REQUIRES(context,
              num_split >= 1 && num_split <= input_shape.vec<int64_t>()(axis),
              errors::InvalidArgument("Input num_split should be between 1 "
                                      "and the splitting dimension size (",
                                      input_shape.vec<int64_t>()(axis),
                                      "), got ", num_split));

  // Prevent overflow by constructing the dense shape separately
  TensorShape dense_shape;

  const auto input_shape_flat = input_shape.flat<int64_t>();
  for (int i = 0; i < input_shape.NumElements(); i++) {
    dense_shape.AddDim(input_shape_flat(i));
  }

  functor::SparseSplitFunctor<GPUDevice, T>()(
      context, input_indices, input_values, dense_shape, axis, num_split);
}

}  // anonymous namespace

template <typename T>
class SparseSplitOp : public OpKernel {
 public:
  explicit SparseSplitOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_split", &num_split_));
  }

  void Compute(OpKernelContext* context) override {
    SparseSplitOpImpl<T>(context, num_split_);
  }

 private:
  int num_split_;
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseSplit")             \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("split_dim")    \
                              .HostMemory("shape")        \
                              .HostMemory("output_shape") \
                              .TypeConstraint<type>("T"), \
                          SparseSplitOp<type>)

TF_CALL_INTEGRAL_TYPES(REGISTER_KERNELS);
TF_CALL_GPU_ALL_TYPES(REGISTER_KERNELS);
TF_CALL_complex64(REGISTER_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNELS);
TF_CALL_complex128(REGISTER_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNELS

}  // namespace itex
