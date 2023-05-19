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

#include "itex/core/kernels/gpu/sparse_slice_op.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "itex/core/kernels/gpu/where_op.h"

namespace itex {
namespace functor {
struct SparseSliceSelectKernel {
  SparseSliceSelectKernel(const int dims, const int input_size,
                          const int64_t* input_start_data,
                          const int64_t* input_size_data,
                          const int64_t* input_indices, int* nonzeros)
      : dims_(dims),
        input_size_(input_size),
        input_start_data_(input_start_data),
        input_size_data_(input_size_data),
        input_indices_(input_indices),
        nonzeros_(nonzeros) {}

  // Set nonzeros[index] to 1 iff input_indices[index] within the slice volumn.
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= input_size_) return;
    nonzeros_[id] = 1;
    for (int dim = 0; dim < dims_; ++dim) {
      int64_t index = input_indices_[id * dims_ + dim];
      int64_t slice_start = input_start_data_[dim];
      int64_t slice_end = slice_start + input_size_data_[dim];
      if (index < slice_start || index >= slice_end) {
        // Set to 0 if not in slice volumn and return immediately.
        nonzeros_[id] = 0;
        return;
      }
    }
  }

 private:
  const int dims_;
  const int input_size_;
  const int64_t* input_start_data_;
  const int64_t* input_size_data_;
  const int64_t* input_indices_;
  int* nonzeros_;
};

Status LaunchSparseSliceSelectKernel(const GPUDevice& d, const int dims,
                                     const int64_t input_size,
                                     const int64_t* input_start_data,
                                     const int64_t* input_size_data,
                                     const int64_t* input_indices,
                                     int* nonzeros) {
  auto compute = [=, &d](sycl::handler& cgh) {
    auto max_group_size =
        d.stream()
            ->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_group = (input_size + max_group_size - 1) / max_group_size;
    sycl::range<1> local_range(max_group_size);
    sycl::range<1> global_range(max_group_size * num_work_group);

    d.stream()->submit([&](sycl::handler& cgh) {
      SparseSliceSelectKernel task(dims, input_size, input_start_data,
                                   input_size_data, input_indices, nonzeros);
      cgh.parallel_for<SparseSliceSelectKernel>(
          sycl::nd_range<1>(global_range, local_range), task);
    });
  };

  d.stream()->submit(std::move(compute));
  return Status::OK();
}

// Gathers selected indices and values from input into output.
template <typename T>
struct SparseSliceGatherKernel {
  SparseSliceGatherKernel(const int dims, const int64_t input_size,
                          const int64_t* input_start_data,
                          const int64_t* input_indices, const T* input_values,
                          const int* nonzeros, const int64_t* input_cumsum,
                          int64_t* output_indices, T* output_values)
      : dims_(dims),
        input_size_(input_size),
        input_start_data_(input_start_data),
        input_indices_(input_indices),
        input_values_(input_values),
        nonzeros_(nonzeros),
        input_cumsum_(input_cumsum),
        output_indices_(output_indices),
        output_values_(output_values) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id < input_size_ && nonzeros_[id] != 0) {
      auto output_nz = input_cumsum_[id] - 1;
      auto input_nz = id;
      output_values_[output_nz] = input_values_[input_nz];
      // Copy indices.
      for (int dim = 0; dim < dims_; ++dim) {
        output_indices_[output_nz * dims_ + dim] =
            input_indices_[input_nz * dims_ + dim] - input_start_data_[dim];
      }
    }
  }

 private:
  const int dims_;
  const int64_t input_size_;
  const int64_t* input_start_data_;
  const int64_t* input_indices_;
  const T* input_values_;
  const int* nonzeros_;
  const int64_t* input_cumsum_;
  int64_t* output_indices_;
  T* output_values_;
};

template <typename T>
Status LaunchSparseSliceGather(const GPUDevice& d, const int dims,
                               const int64_t input_size,
                               const int64_t* input_start_data,
                               const int64_t* input_indices,
                               const T* input_values, const int* nonzeros,
                               const int64_t* input_cumsum,
                               int64_t* output_indices, T* output_values) {
  auto compute = [=, &d](sycl::handler& cgh) {
    auto max_group_size =
        d.stream()
            ->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    auto num_work_group = (input_size + max_group_size - 1) / max_group_size;
    sycl::range<1> local_range(max_group_size);
    sycl::range<1> global_range(max_group_size * num_work_group);

    SparseSliceGatherKernel<T> task(
        dims, input_size, input_start_data, input_indices, input_values,
        nonzeros, input_cumsum, output_indices, output_values);
    cgh.parallel_for<SparseSliceGatherKernel<T>>(
        sycl::nd_range<1>(global_range, local_range), task);
  };

  d.stream()->submit(std::move(compute));
  return Status::OK();
}

template <typename T>
struct SparseSliceFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const Tensor& input_shape,
                  const Tensor& input_start, const Tensor& input_size) const {
    auto& d = context->eigen_gpu_device();
    auto stream = d.stream();
    const int dims = input_shape.NumElements();

    std::vector<int64_t> input_start_host(dims);
    std::vector<int64_t> input_size_host(dims);

    // Allocate and compute output shape.
    Tensor* output_shape = nullptr;
    int64_t output_volume = 1;
    OP_REQUIRES_OK(context, context->allocate_output(2, {dims}, &output_shape));

    for (int dim = 0; dim < dims; ++dim) {
      int64_t input_dimsize = input_shape.vec<int64_t>()(dim);
      int64_t slice_start = input_start.vec<int64_t>()(dim);
      int64_t slice_size = input_size.vec<int64_t>()(dim);
      input_start_host[dim] = slice_start;
      input_size_host[dim] = slice_size;
      int64_t output_size = std::max(
          std::min(slice_start + slice_size, input_dimsize) - slice_start,
          int64_t(0));
      output_shape->vec<int64_t>()(dim) = output_size;
      output_volume *= output_size;
    }

    // Copy input_start and input_size to device memory.
    Tensor input_start_gpu;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64_t>::v(),
                                                   {dims}, &input_start_gpu));
    int64_t* input_start_ptr = input_start_gpu.vec<int64_t>().data();
    stream
        ->memcpy(input_start_ptr, input_start_host.data(),
                 sizeof(int64_t) * dims)
        .wait();

    Tensor input_size_gpu;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64_t>::v(),
                                                   {dims}, &input_size_gpu));
    int64_t* input_size_ptr = input_size_gpu.vec<int64_t>().data();
    stream
        ->memcpy(input_size_ptr, input_size_host.data(), sizeof(int64_t) * dims)
        .wait();

    int64_t input_nnz = input_indices.dim_size(0);

    // Early exit for empty input or output shape.
    if (input_nnz == 0 || output_volume == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, {0, dims}, &output_indices));
      Tensor* output_values = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, {0}, &output_values));
      return;
    }

    // Allocate flags tensor and launch select kernel to compute.
    Tensor nonzeros;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::v(),
                                                   {input_nnz}, &nonzeros));

    const int64_t* input_indices_ptr = input_indices.matrix<int64_t>().data();
    const T* input_values_ptr = input_values.vec<T>().data();
    int* nonzeros_ptr = nonzeros.vec<int>().data();

    Status s = LaunchSparseSliceSelectKernel(d, dims, input_nnz,
                                             input_start_ptr, input_size_ptr,
                                             input_indices_ptr, nonzeros_ptr);
    OP_REQUIRES_OK(context, s);

    // Compute a cumsum to then get the counter of true elements seen so far.
    Tensor input_cumsum;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64_t>::v(),
                                                   {input_nnz}, &input_cumsum));
    auto input_cumsum_t = input_cumsum.vec<int64_t>();

    // Push kernel to stream to get number of true elements.
    const Tensor& const_nonzeros = nonzeros;
    s = functor::InputCumSum<int, int64_t>::Compute(
        context, const_nonzeros.flat<int>(), input_cumsum_t, input_nnz);
    OP_REQUIRES_OK(context, s);

    // Copy num_true to host, which is the number of output indices.
    int64_t output_nnz;
    stream
        ->memcpy(&output_nnz, input_cumsum_t.data() + input_nnz - 1,
                 sizeof(int64_t))
        .wait();

    // Allocate output indices and values.
    Tensor* output_indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {output_nnz, dims},
                                                     &output_indices));
    int64_t* output_indices_ptr = output_indices->matrix<int64_t>().data();

    Tensor* output_values = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {output_nnz}, &output_values));
    T* output_values_ptr = output_values->vec<T>().data();

    if (output_nnz == 0) {
      return;
    }

    // Gather input to output.
    s = LaunchSparseSliceGather<T>(d, dims, input_nnz, input_start_ptr,
                                   input_indices_ptr, input_values_ptr,
                                   nonzeros_ptr, input_cumsum_t.data(),
                                   output_indices_ptr, output_values_ptr);
    OP_REQUIRES_OK(context, s);
  }
};

}  // namespace functor

namespace {

template <typename T>
void SparseSliceOpImpl(OpKernelContext* context) {
  const Tensor& input_indices = context->input(0);
  const Tensor& input_values = context->input(1);
  const Tensor& input_shape = context->input(2);
  const Tensor& input_start = context->input(3);
  const Tensor& input_size = context->input(4);

  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
              errors::InvalidArgument(
                  "Input indices should be a matrix but received shape ",
                  input_indices.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values.shape()),
              errors::InvalidArgument(
                  "Input values should be a vector but received shape ",
                  input_values.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape.shape()),
              errors::InvalidArgument(
                  "Input shape should be a vector but received shape ",
                  input_shape.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_start.shape()),
              errors::InvalidArgument(
                  "Input start should be a vector but received shape ",
                  input_start.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_size.shape()),
              errors::InvalidArgument(
                  "Input size should be a vector but received shape ",
                  input_size.shape().DebugString()));

  const int input_dims = input_shape.NumElements();
  OP_REQUIRES(context, input_dims == input_start.NumElements(),
              errors::InvalidArgument(
                  "Expected start to be a vector of length ", input_dims,
                  " but got length ", input_start.NumElements()));

  OP_REQUIRES(context, input_dims == input_size.NumElements(),
              errors::InvalidArgument("Expected size to be a vector of length ",
                                      input_dims, " but got length ",
                                      input_size.NumElements()));

  functor::SparseSliceFunctor<GPUDevice, T>()(context, input_indices,
                                              input_values, input_shape,
                                              input_start, input_size);
}

}  // anonymous namespace

template <typename T>
class SparseSliceOp : public OpKernel {
 public:
  explicit SparseSliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    SparseSliceOpImpl<T>(context);
  }
};

#define REGISTER_SPARSE_SLICE_OP(T)                       \
  REGISTER_KERNEL_BUILDER(Name("SparseSlice")             \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shape")        \
                              .HostMemory("start")        \
                              .HostMemory("size")         \
                              .HostMemory("output_shape") \
                              .TypeConstraint<T>("T"),    \
                          SparseSliceOp<T>)

TF_CALL_INTEGRAL_TYPES(REGISTER_SPARSE_SLICE_OP)
TF_CALL_GPU_ALL_TYPES(REGISTER_SPARSE_SLICE_OP)
TF_CALL_complex64(REGISTER_SPARSE_SLICE_OP)
#ifdef ITEX_ENABLE_DOUBLE
    TF_CALL_double(REGISTER_SPARSE_SLICE_OP);
TF_CALL_complex128(REGISTER_SPARSE_SLICE_OP);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_SPARSE_SLICE_OP

}  // namespace itex
