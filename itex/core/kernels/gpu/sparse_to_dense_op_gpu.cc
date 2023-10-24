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

#include "itex/core/kernels/gpu/sparse_to_dense_op_gpu.h"

#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace itex {

namespace {

template <typename T, typename Index>
struct SparseToDenseKernel {
  SparseToDenseKernel(const Index* indices, const T* vals, int nnz,
                      int num_vals, const Index* dims, int ndims, T* dense)
      : indices(indices),
        vals(vals),
        nnz(nnz),
        num_vals(num_vals),
        dims(dims),
        ndims(ndims),
        dense(dense) {}

  void operator()(sycl::nd_item<1> item) const {
    auto global_id = item.get_global_id(0);
    auto global_range = item.get_global_range(0);
    for (int thread_idx = global_id; thread_idx < nnz;
         thread_idx += global_range) {
      eigen_assert(ndims >= 1);
      int64 output_idx = indices[thread_idx * ndims + ndims - 1];
      Index strides = 1;
      for (int i = ndims - 2; i >= 0; i--) {
        strides *= dims[i + 1];
        output_idx += indices[thread_idx * ndims + i] * strides;
      }
      // If num_vals == 1, broadcast the scalar to the positions for non-zeros.
      dense[output_idx] = vals[(num_vals == 1) ? 0 : thread_idx];
    }
  }

 private:
  const Index* indices;
  const T* vals;
  int nnz;
  int num_vals;
  const Index* dims;
  int ndims;
  T* dense;
};

template <typename T, typename Index>
struct SetDefaultValueKernel {
  SetDefaultValueKernel(T default_value, int64 dense_size, T* dense)
      : default_value(default_value), dense_size(dense_size), dense(dense) {}

  void operator()(sycl::nd_item<1> item) const {
    auto global_id = item.get_global_id(0);
    auto global_range = item.get_global_range(0);
    for (int32 i = global_id, step = global_range; i < dense_size; i += step) {
      dense[i] = default_value;
    }
  }

 private:
  T default_value;
  int64 dense_size;
  T* dense;
};

template <typename Index>
struct CheckIndicesValidKernel {
  CheckIndicesValidKernel(const Index* indices, const int nnz,
                          const Index* dims, const int ndims, int* status)
      : indices(indices), nnz(nnz), dims(dims), ndims(ndims), status(status) {}

  void operator()(sycl::nd_item<1> item) const {
    auto global_id = item.get_global_id(0);
    auto global_range = item.get_global_range(0);

    for (int32 thread_idx = global_id, step = global_range; thread_idx < nnz;
         thread_idx += step) {
      bool increasing = true;
      bool different = false;
      bool valid = true;

      if (thread_idx == 0) {
        for (int di = 0; di < ndims; di++) {
          Index curr_idx = indices[thread_idx * ndims + di];
          if (curr_idx < 0 || curr_idx >= dims[di]) valid = false;
        }
        different = true;
      } else {
        for (int di = 0; di < ndims; di++) {
          Index curr_idx = indices[thread_idx * ndims + di];
          if (curr_idx < 0 || curr_idx >= dims[di]) valid = false;
          Index prev_idx = indices[(thread_idx - 1) * ndims + di];
          Index diff = curr_idx - prev_idx;
          if (diff > 0) different = true;
          if (!different && diff < 0) increasing = false;
        }
      }
      if (!valid) {
        ItexAtomicMin(&status[0], thread_idx);
      }
      if (!increasing) {
        ItexAtomicMin(&status[1], thread_idx);
      }
      if (!different) {
        ItexAtomicMin(&status[2], thread_idx);
      }
    }
  }

 private:
  const Index* indices;
  const int nnz;
  const Index* dims;
  const int ndims;
  int* status;
};
// IndicesValidStatus contains three status for the out-of-bound check, the
// sorted check, and the repeat check. If the value equals to INT_MAX, the
// check passes. Otherwise, it represents the first detected position of the
// invalid index for the check.
struct IndicesValidStatus {
  int valid;
  int increasing;
  int different;
};

template <typename T, typename Index>
Status LaunchComputeKernels(OpKernelContext* c, const int64 dense_size,
                            const T default_value, const Index* indices,
                            const T* values, const int num_elems,
                            const int num_values, const Index* shape,
                            const int num_dims, T* dense) {
  const Eigen::GpuDevice& device = c->eigen_gpu_device();
  if (dense_size > 0) {
    device.stream()->submit([&](sycl::handler& cgh) {
      auto max_group_size =
          device.stream()
              ->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();

      auto num_work_group = (dense_size + max_group_size - 1) / max_group_size;
      sycl::range<1> local_range(max_group_size);
      sycl::range<1> global_range(max_group_size * num_work_group);

      SetDefaultValueKernel<T, Index> task(default_value, dense_size, dense);
      cgh.parallel_for<SetDefaultValueKernel<T, Index>>(
          sycl::nd_range<1>(global_range, local_range), task);
    });
  }

  if (num_elems > 0) {
    device.stream()->submit([&](sycl::handler& cgh) {
      auto max_group_size =
          device.stream()
              ->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();

      auto num_work_group = (num_elems + max_group_size - 1) / max_group_size;
      sycl::range<1> local_range(max_group_size);
      sycl::range<1> global_range(max_group_size * num_work_group);

      SparseToDenseKernel<T, Index> task(indices, values, num_elems, num_values,
                                         shape, num_dims, dense);
      cgh.parallel_for<SparseToDenseKernel<T, Index>>(
          sycl::nd_range<1>(global_range, local_range), task);
    });
  }
  return OkStatus();
}

}  // namespace

namespace functor {

template <typename T, typename Index>
void LaunchSparseToDense<T, Index>::operator()(
    OpKernelContext* c, bool validate_indices, const Tensor& indices,
    const Tensor& values, const Tensor& shape, const T default_value,
    Tensor* dense) {
  const Eigen::GpuDevice& device = c->eigen_gpu_device();
  auto* stream = device.stream();
  const Index* indices_ptr = indices.flat<Index>().data();
  const T* values_ptr = values.flat<T>().data();
  const Index* shape_ptr = shape.flat<Index>().data();
  T* dense_ptr = dense->flat<T>().data();
  const int64 dense_size = dense->NumElements();
  const int64 num_values = values.NumElements();
  const int64 num_elems = indices.dims() > 0 ? indices.dim_size(0) : 1;
  const int64 num_dims = indices.dims() > 1 ? indices.dim_size(1) : 1;
  if (validate_indices && num_elems != 0) {
    ITEX_VLOG(1)
        << "SparseToDense will be performed on GPUs. For performance "
           "reasons, it is suggested to pass False to validate_indices.";

    IndicesValidStatus valid_status;
    int valid_status_size = sizeof(valid_status) / sizeof(int);

    Tensor valid_status_tensor;
    OP_REQUIRES_OK(c,
                   c->allocate_temp(DT_INT32, TensorShape({valid_status_size}),
                                    &valid_status_tensor));

    auto status_ptr = valid_status_tensor.template flat<int>().data();
    stream->memset(status_ptr, INT_MAX, valid_status_size).wait();

    device.stream()->submit([&](sycl::handler& cgh) {
      auto max_group_size =
          device.stream()
              ->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();

      auto num_work_group = (num_elems + max_group_size - 1) / max_group_size;
      sycl::range<1> local_range(max_group_size);
      sycl::range<1> global_range(max_group_size * num_work_group);

      CheckIndicesValidKernel<Index> task(indices_ptr, num_elems, shape_ptr,
                                          num_dims, status_ptr);
      cgh.parallel_for<CheckIndicesValidKernel<Index>>(
          sycl::nd_range<1>(global_range, local_range), task);
    });
    stream
        ->memcpy(reinterpret_cast<int*>(&valid_status),
                 reinterpret_cast<int*>(status_ptr), valid_status_size)
        .wait();

    OP_REQUIRES_OK(
        c, LaunchComputeKernels(
               c, dense_size, default_value, indices_ptr, values_ptr, num_elems,
               num_values, shape.flat<Index>().data(), num_dims, dense_ptr));
  } else {
    OP_REQUIRES_OK(
        c, LaunchComputeKernels(c, dense_size, default_value, indices_ptr,
                                values_ptr, num_elems, num_values, shape_ptr,
                                num_dims, dense_ptr));
  }
}

}  // namespace functor

#define DEFINE_GPU_SPEC(T)                                \
  template struct functor::LaunchSparseToDense<T, int64>; \
  template struct functor::LaunchSparseToDense<T, int32>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);
TF_CALL_INTEGRAL_TYPES(DEFINE_GPU_SPEC);
DEFINE_GPU_SPEC(bool);

}  // namespace itex
