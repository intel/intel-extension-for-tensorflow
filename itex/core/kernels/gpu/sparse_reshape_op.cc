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

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "itex/core/kernels/common/reshape_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {
using GPUDevice = Eigen::GpuDevice;

template <typename Tindex>
struct ReshapeSparseReshapeTensorKernel {
  ReshapeSparseReshapeTensorKernel(
      const Tindex nnz, const Tindex input_rank, const Tindex output_rank,
      const Tindex input_dims_size[SPARSE_RESHAPE_MAX_SHAPE_DIMS],
      const Tindex output_dims_size[SPARSE_RESHAPE_MAX_SHAPE_DIMS],
      const Tindex* input_indices, Tindex* output_indices)
      : nnz_(nnz),
        input_rank_(input_rank),
        output_rank_(output_rank),
        input_indices_(input_indices),
        output_indices_(output_indices) {
    for (int i = 0; i < input_rank_; ++i) input_shape_[i] = input_dims_size[i];
    for (int i = 0; i < output_rank_; ++i)
      output_shape_[i] = output_dims_size[i];
  }
  inline void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= nnz_) return;

    const Tindex* input_index = &input_indices_[id * input_rank_];
    Tindex* output_index = &output_indices_[id * output_rank_];
    int64 dense_index = 0;  // int64 to avoid overflow if Tindex is int32
// Flatten input index from slowest- to fastest-changing dimension.
#pragma unroll
    for (int i = 0; i < input_rank_; ++i) {
      dense_index = dense_index * input_shape_[i] + input_index[i];
    }
// Compute output index from fastest- to slowest-changing dimension.
#pragma unroll
    for (int i = output_rank_ - 1; i >= 0; --i) {
      Tindex output_size = output_shape_[i];
      output_index[i] = dense_index % output_size;
      dense_index /= output_size;
    }
  }

 private:
  const Tindex nnz_, input_rank_, output_rank_;
  const Tindex* input_indices_;
  Tindex input_shape_[SPARSE_RESHAPE_MAX_SHAPE_DIMS];
  Tindex output_shape_[SPARSE_RESHAPE_MAX_SHAPE_DIMS];
  Tindex* output_indices_;
};

namespace functor {

template <>
Status ReshapeSparseTensorFunctor<GPUDevice>::operator()(
    OpKernelContext* context, const TensorShape& input_shape,
    const TensorShape& output_shape,
    typename TTypes<int64_t>::ConstMatrix input_indices,
    typename TTypes<int64_t>::Matrix output_indices) const {
  const int64_t input_rank = input_shape.dims();
  const int64_t output_rank = output_shape.dims();
  const int64_t nnz = input_indices.dimension(0);
  // We copy input_shape and output_shape to the GPU and then launch a kernel
  // to compute output_indices.
  Tensor input_shape_gpu_t;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT64, TensorShape({input_rank}),
                                            &input_shape_gpu_t));
  Tensor output_shape_gpu_t;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT64, TensorShape({output_rank}), &output_shape_gpu_t));
  int64_t input_shape_host[SPARSE_RESHAPE_MAX_SHAPE_DIMS];
  int64_t output_shape_host[SPARSE_RESHAPE_MAX_SHAPE_DIMS];
  for (int i = 0; i < input_rank; ++i)
    input_shape_host[i] = input_shape.dim_size(i);
  for (int i = 0; i < output_rank; ++i)
    output_shape_host[i] = output_shape.dim_size(i);

  auto stream = context->eigen_gpu_device().stream();
  stream->submit([&](sycl::handler& cgh) {
    ReshapeSparseReshapeTensorKernel<int64> task(
        nnz, input_rank, output_rank, input_shape_host, output_shape_host,
        input_indices.data(), output_indices.data());
    auto max_group_size =
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    const int workgroup_count = (nnz + max_group_size - 1) / max_group_size;
    sycl::range<1> global = workgroup_count * max_group_size;
    sycl::range<1> local = max_group_size;
    cgh.parallel_for<ReshapeSparseReshapeTensorKernel<int64>>(
        sycl::nd_range<1>(global, local), task);
  });

  return Status::OK();
}

}  // namespace functor

template <typename Device>
class SparseReshapeOp : public OpKernel {
 public:
  explicit SparseReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_indices_in = context->input(0);
    const Tensor& input_shape_in = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices_in.shape()),
                errors::InvalidArgument("Input must be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
                errors::InvalidArgument("Input shape must be a vector."));
    OP_REQUIRES(context,
                input_indices_in.dim_size(1) == input_shape_in.dim_size(0),
                errors::InvalidArgument(
                    "Input tensor rank must match input shape length."));
    ReshapeSparseTensor<Device>(context, context->input(0), context->input(1),
                                context->input(2), 0 /* output indices index */,
                                1 /* output shape index */);
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseReshape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_shape")
                            .HostMemory("new_shape")
                            .HostMemory("output_shape"),
                        SparseReshapeOp<GPUDevice>)

}  // namespace itex
