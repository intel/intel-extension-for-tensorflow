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

#ifndef ITEX_CORE_KERNELS_GPU_TILE_FUNCTOR_GPU_H_
#define ITEX_CORE_KERNELS_GPU_TILE_FUNCTOR_GPU_H_

#include "itex/core/kernels/gpu/tile_functor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "itex/core/utils/util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace internal {

template <typename T>
struct TileKernel {
  TileKernel(int nthreads, const T* src, const int32* buf, const int32 ndims,
             T* dst)
      : nthreads_(nthreads), src_(src), buf_(buf), ndims_(ndims), dst_(dst) {}
  void operator()(sycl::nd_item<1> item) const {
    const auto in_strides = buf_;
    const auto out_strides = buf_ + ndims_;
    const auto in_dim_sizes = buf_ + ndims_ * 2;
    auto o_idx = item.get_global_linear_id();
    if (o_idx < nthreads_) {
      int32 i_idx = 0;
      int32 t = o_idx;
      for (int i = 0; i < ndims_; ++i) {
        i_idx += t / out_strides[i] % in_dim_sizes[i] * in_strides[i];
        t %= out_strides[i];
      }
      dst_[o_idx] = src_[i_idx];
    }
  }

 private:
  int nthreads_;
  const T* src_;
  const int32* buf_;
  const int32 ndims_;
  T* dst_;
};

template <typename T>
class TileDummyKernel;

template <typename T>
void TileSimple(const Eigen::GpuDevice& d, Tensor* out, const Tensor& in) {
  // Ensures we can use 32-bit index.
  const int64 in_nelem = in.NumElements();
  ITEX_CHECK_LT(in_nelem, kint32max) << "Tensor too large to transpose on GPU";
  const int64 out_nelem = out->NumElements();
  ITEX_CHECK_LT(out_nelem, kint32max) << "Tensor too large to transpose on GPU";
  // Pack strides and input dimension sizes into one buffer.
  const int32 ndims = in.dims();
  gtl::InlinedVector<int32, 24> host_buf(ndims * 3);
  gtl::InlinedVector<int32, 8> in_strides = ComputeStride<int32>(in.shape());
  gtl::InlinedVector<int32, 8> out_strides = ComputeStride<int32>(out->shape());
  for (int i = 0; i < ndims; ++i) {
    host_buf[i] = in_strides[i];
    host_buf[ndims + i] = out_strides[i];
    host_buf[ndims * 2 + i] = in.dim_size(i);
  }
  // Copies the input strides, output strides and input dimension sizes to the
  // device.
  auto num_bytes = sizeof(int64) * host_buf.size();
  auto dev_buf = d.allocate(num_bytes);
  // NOTE: host_buf is not allocated by CudaHostAllocator, and
  // therefore we are doing a sync copy effectively.
  d.memcpyHostToDevice(dev_buf, host_buf.data(), num_bytes);
  // Launch kernel to q[...] = p[...].

  auto& stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (out_nelem + group_size - 1) / group_size;
  stream->submit([&](sycl::handler& cgh) {
    TileKernel<T> task(out_nelem, in.flat<T>().data(),
                       reinterpret_cast<const int32*>(dev_buf), ndims,
                       out->flat<T>().data());
    cgh.parallel_for<TileDummyKernel<T>>(
        sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                          sycl::range<1>(group_size)),
        task);
  });

  // Safe to deallocate immediately after the kernel launch.
  d.deallocate(dev_buf);
}

}  // end namespace internal
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_TILE_FUNCTOR_GPU_H_
