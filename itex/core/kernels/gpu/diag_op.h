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

#ifndef ITEX_CORE_KERNELS_GPU_DIAG_OP_H_
#define ITEX_CORE_KERNELS_GPU_DIAG_OP_H_

#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

typedef Eigen::GpuDevice GPUDevice;

namespace itex {

template <typename T>
struct DiagKernel {
  DiagKernel(size_t num_work_items, int64 size, const T* in_ptr, T* out_ptr)
      : num_work_items(num_work_items),
        size(size),
        in_ptr(in_ptr),
        out_ptr(out_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }
    if (id % (1 + size) == 0) {
      out_ptr[id] = in_ptr[id / (1 + size)];
    } else {
      out_ptr[id] = T(0);
    }
  }

 private:
  size_t num_work_items;
  int64 size;
  const T* in_ptr;
  T* out_ptr;
};

template <typename T>
struct DiagPartKernel {
  DiagPartKernel(size_t num_work_items, int64 size, const T* in_ptr, T* out_ptr)
      : num_work_items(num_work_items),
        size(size),
        in_ptr(in_ptr),
        out_ptr(out_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }
    out_ptr[id] = in_ptr[(1 + size) * id];
  }

 private:
  size_t num_work_items;
  int64 size;
  const T* in_ptr;
  T* out_ptr;
};

namespace functor {

template <typename Device, typename T>
struct DiagFunctor {
  Status operator()(OpKernelContext* context, const int64 size, const T* in,
                    T* out);
};

template <typename T>
struct DiagFunctor<GPUDevice, T> {
  EIGEN_ALWAYS_INLINE Status operator()(OpKernelContext* ctx, const int64 size,
                                        const T* in, T* out) {
    if (size == 0) {
      return Status::OK();
    }

    auto stream = ctx->eigen_gpu_device().stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = size * size;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      auto in_ptr = in;
      auto out_ptr = out;
      DiagKernel<T> task(num_work_items, size, in_ptr, out_ptr);
      cgh.parallel_for<DiagKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          task);
    });

    return Status::OK();
  }
};

template <typename Device, typename T>
struct DiagPartFunctor {
  Status operator()(OpKernelContext* context, const int64 size, const T* in,
                    T* out);
};

template <typename T>
struct DiagPartFunctor<GPUDevice, T> {
  EIGEN_ALWAYS_INLINE Status operator()(OpKernelContext* ctx, const int64 size,
                                        const T* in, T* out) {
    if (size == 0) {
      return Status::OK();
    }

    auto stream = ctx->eigen_gpu_device().stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = size;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      auto in_ptr = in;
      auto out_ptr = out;
      DiagPartKernel<T> task(num_work_items, size, in_ptr, out_ptr);
      cgh.parallel_for<DiagPartKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          task);
    });
    return Status::OK();
  }
};

}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_DIAG_OP_H_
