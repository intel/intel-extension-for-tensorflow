/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_INPLACE_OPS_FUNCTOR_H_
#define ITEX_CORE_KERNELS_GPU_INPLACE_OPS_FUNCTOR_H_

#include "itex/core/utils/plugin_tensor.h"

namespace itex {
namespace functor {
typedef Eigen::GpuDevice Device;

template <typename T>
void DoParallelConcatUpdate(const T* src, T* dst, int work_items, int32 loc,
                            const int64 rows, const int64 cols,
                            sycl::nd_item<1> item) {
  auto idx = item.get_global_linear_id();
  if (idx >= work_items) return;

  int64 r = (loc % rows + rows) % rows;
  int64 c = idx % cols;

  dst[r * cols + c] = src[idx];
}

template <typename T>
struct DoParallelConcatKernel {
  DoParallelConcatKernel(const T* src, T* dst, int64_t nelem, int32_t loc,
                         int64_t nrows, int64_t ncols)
      : src(src),
        dst(dst),
        nelem(nelem),
        loc(loc),
        nrows(nrows),
        ncols(ncols) {}
  void operator()(sycl::nd_item<1> item) const {
    DoParallelConcatUpdate<T>(src, dst, nelem, loc, nrows, ncols, item);
  }

 private:
  const T* src;
  T* dst;
  int64_t nelem;
  int32_t loc;
  int64_t nrows;
  int64_t ncols;
};

template <typename T>
void DoParallelConcat(const Device& d, const Tensor& value, int32 loc,
                      Tensor* output) {
  const int64 nelem = value.NumElements();
  auto Toutput = output->flat_outer_dims<T>();
  const int64 nrows = Toutput.dimension(0);
  const int64 ncols = Toutput.dimension(1);

  auto& stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (nelem + group_size - 1) / group_size;

  const T* src = value.flat<T>().data();

  if (nelem == 0) {
    // When params_size is 0, the data pointer of params tensor maybe a host
    // pointer. If we use a host pointer in dpcpp kernel even if the code is
    // in impossible condition branch, we will get an error -50
    // (CL_INVALID_ARG_VALUE). Here we workaround this case. All indices will
    // be out of range in this condition, so the output value will be zero
    // according to definition of GatherV2.

    d.stream()->memset(output->flat<T>().data(), 0,
                       output->NumElements() * sizeof(T));
    return;
  }
  T* dst = output->flat<T>().data();

  stream->submit([&](sycl::handler& cgh) {
    DoParallelConcatKernel<T> task(src, dst, nelem, loc, nrows, ncols);
    cgh.parallel_for<DoParallelConcatKernel<T>>(
        sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                          sycl::range<1>(group_size)),
        task);
  });
}

// Inplace update/add/sub values in 'y'. It computes
//   y[i, :] = v if op is I_UPDATE
//   y[i, :] += v if op is I_ADD
//   y[i, :] -= v if op is I_SUB
// Returns an error if the operation fails.
enum InplaceOpType : unsigned int {
  I_UPDATE,  // x = y
  I_ADD,     // x += y
  I_SUB,     // x -= y
};

template <typename Device>
Status DoInplace(const Device& device, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y);

// Copies x into y.
template <typename Device>
Status DoCopy(const Device& device, const Tensor& x, Tensor* y);
}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_INPLACE_OPS_FUNCTOR_H_
