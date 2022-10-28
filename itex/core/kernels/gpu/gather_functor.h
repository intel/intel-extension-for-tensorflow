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

#ifndef ITEX_CORE_KERNELS_GPU_GATHER_FUNCTOR_H_
#define ITEX_CORE_KERNELS_GPU_GATHER_FUNCTOR_H_

#include <limits>

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename ValueOrVec, typename Index, bool is_axis_zero,
          bool can_use_same_index>
struct GatherOpKernel {
  GatherOpKernel(const ValueOrVec* params, const Index* indices,
                 ValueOrVec* out, int64 gather_dim_size, int64 indices_size,
                 int64 slice_size, int64 out_size)
      : params(params),
        indices(indices),
        out(out),
        gather_dim_size_(gather_dim_size),
        indices_size_(indices_size),
        slice_size_(slice_size),
        out_size_(out_size) {}

  void operator()(sycl::nd_item<1> item) const {
    typedef typename Eigen::internal::conditional<can_use_same_index, Index,
                                                  int64>::type CastType;
    auto i = item.get_global_id()[0];
    if (i >= out_size_) return;
    Index batch_i = 0;
    Index indices_i = 0;
    Index slice_i = 0;
    auto i_cast = static_cast<CastType>(i);
    auto slice_size_cast = static_cast<CastType>(slice_size_);
    auto indices_size_cast = static_cast<CastType>(indices_size_);
    auto gather_dim_size_cast = static_cast<CastType>(gather_dim_size_);
    if (is_axis_zero) {
      indices_i = i_cast / slice_size_cast;
      slice_i = i_cast - indices_i * slice_size_cast;
    } else {
      Index batch_indices_i = i_cast / slice_size_cast;
      // The batch index into params to use for i.
      batch_i = batch_indices_i / indices_size_cast;
      // The index into indices to use for i.
      indices_i = batch_indices_i - batch_i * indices_size_cast;
      // Index into the current slice in params to use for i.
      slice_i = i_cast - batch_indices_i * slice_size_cast;
    }

    // Index into the gather axis to use for i.
    // Index gather_i = ldg(indices + indices_i);
    Index gather_i = indices[indices_i];

    // Check gather_i is in [0, gather_dim_size_).
    if (!FastBoundsCheck(gather_i, gather_dim_size_)) {
      // Set indices out of range to zero
      // TODO(fpmc): Log an error for transfer back to host.
      out[i] = ValueOrVec(0);
    } else {
      // params is a [batch_size, gather_dim_size_, slice_size_] tensor. Read
      // params[batch_i, gather_i, slice_i] and write it to the i'th position in
      // out.
      Index params_i =
          (batch_i * gather_dim_size_cast + gather_i) * slice_size_cast +
          slice_i;
      out[i] = params[params_i];
    }
  }

 private:
  const ValueOrVec* params;
  const Index* indices;
  ValueOrVec* out;
  int64 gather_dim_size_;
  int64 indices_size_;
  int64 slice_size_;
  int64 out_size_;
};

namespace detail {

template <bool is_axis_zero, bool can_use_same_index>
struct LaunchGatherKernelVectorized {
  template <int vec_size>
  struct Impl {
    template <typename T, typename Index>
    void operator()(const GPUDevice& d, const T* params, const Index* indices,
                    T* out, int64 gather_dim_size, int64 indices_size,
                    int64 slice_size, int64 out_size) {
      ITEX_DCHECK_EQ(slice_size % vec_size, 0);
      ITEX_DCHECK_EQ(out_size % vec_size, 0);
      ITEX_DCHECK_EQ(reinterpret_cast<std::uintptr_t>(params) % vec_size, 0);
      ITEX_DCHECK_EQ(reinterpret_cast<std::uintptr_t>(out) % vec_size, 0);
      int64 out_size_vec = out_size / vec_size;
      int64 slice_size_vec = slice_size / vec_size;
      using Tvec = AlignedVector<T, vec_size>;
      const Tvec* params_vec = reinterpret_cast<const Tvec*>(params);
      Tvec* out_vec = reinterpret_cast<Tvec*>(out);

      auto& stream = d.stream();
      auto workgroup_size =
          (*stream)
              .get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      auto num_workgroups =
          (out_size_vec + workgroup_size - 1) / workgroup_size;

      stream->submit([&](sycl::handler& cgh) {
        GatherOpKernel<Tvec, Index, is_axis_zero, can_use_same_index> task(
            params_vec, indices, out_vec, gather_dim_size, indices_size,
            slice_size_vec, out_size_vec);
        cgh.parallel_for<
            GatherOpKernel<Tvec, Index, is_axis_zero, can_use_same_index>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    }
  };
};  // namespace detail

}  // namespace detail

template <typename T, typename Index, bool is_axis_zero,
          bool can_use_same_index>
void LaunchGatherKernel(const GPUDevice& d, const T* params,
                        const Index* indices, T* out, int64 gather_dim_size,
                        const int64 indices_size, const int64 slice_size,
                        const int64 out_size) {
  // Note that the GPU memory allocator always returns aligned buffers, so the
  // alignment of data pointers is expected to be deterministic.
  // There will be performance cliffs when slice_size is not aligned, but there
  // is no easy way to handle the misalignment because each row will be aligned
  // differently.
  DispatchToVectorized<T, detail::LaunchGatherKernelVectorized<
                              is_axis_zero, can_use_same_index>::template Impl>(
      MinAlignmentOf(params, out, slice_size), d, params, indices, out,
      gather_dim_size, indices_size, slice_size, out_size);
}

namespace functor {

template <typename Device, typename T, typename Index>
struct GatherFunctor {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out);
};

template <typename T, typename Index>
struct GatherFunctor<GPUDevice, T, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out) {
    const GPUDevice& d = ctx->eigen_gpu_device();
    const int64 out_size = out.size();
    if (out_size == 0) {
      // We need a check here since the CPU version does useful error checking
      // work if there are nonempty indices but empty slices, so the kernel is
      // executed in that case.  In the GPU case we don't know how to do error
      // checking, so we skip the loop entirely.
      return -1;
    }
    const bool is_axis_zero = params.dimension(0) == 1;
    const int64 gather_dim_size = params.dimension(1);
    const int64 indices_size = indices.size();
    const int64 slice_size = params.dimension(2);
    const bool all_in_index_range =
        out_size <= std::numeric_limits<Index>::max();

    if (is_axis_zero && all_in_index_range) {
      LaunchGatherKernel<T, Index, true, true>(
          d, params.data(), indices.data(), out.data(), gather_dim_size,
          indices_size, slice_size, out_size);
    } else if (is_axis_zero && !all_in_index_range) {
      LaunchGatherKernel<T, Index, true, false>(
          d, params.data(), indices.data(), out.data(), gather_dim_size,
          indices_size, slice_size, out_size);
    } else if (!is_axis_zero && all_in_index_range) {
      LaunchGatherKernel<T, Index, false, true>(
          d, params.data(), indices.data(), out.data(), gather_dim_size,
          indices_size, slice_size, out_size);
    } else {
      LaunchGatherKernel<T, Index, false, false>(
          d, params.data(), indices.data(), out.data(), gather_dim_size,
          indices_size, slice_size, out_size);
    }
    // TODO(fpmc): enable indices validation on GPU.
    // Right now checking for indicies out of bound in the kernel would
    // require copying code between GPU/CPU, and thus slow.
    return -1;
  }
};

// TODO(itex): remove this specification once Eigen::half is replaced with
// sycl::half
template <typename Index>
struct GatherFunctor<GPUDevice, Eigen::half, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<Eigen::half, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<Eigen::half, 3>::Tensor out) {
    const GPUDevice& d = ctx->eigen_gpu_device();
    const int64 out_size = out.size();
    if (out_size == 0) {
      // We need a check here since the CPU version does useful error checking
      // work if there are nonempty indices but empty slices, so the kernel is
      // executed in that case.  In the GPU case we don't know how to do error
      // checking, so we skip the loop entirely.
      return -1;
    }
    const bool is_axis_zero = params.dimension(0) == 1;
    const int64 gather_dim_size = params.dimension(1);
    const int64 indices_size = indices.size();
    const int64 slice_size = params.dimension(2);
    const bool all_in_index_range =
        out_size <= std::numeric_limits<Index>::max();

    if (is_axis_zero && all_in_index_range) {
      LaunchGatherKernel<sycl::half, Index, true, true>(
          d, reinterpret_cast<const sycl::half*>(params.data()), indices.data(),
          reinterpret_cast<sycl::half*>(out.data()), gather_dim_size,
          indices_size, slice_size, out_size);
    } else if (is_axis_zero && !all_in_index_range) {
      LaunchGatherKernel<sycl::half, Index, true, false>(
          d, reinterpret_cast<const sycl::half*>(params.data()), indices.data(),
          reinterpret_cast<sycl::half*>(out.data()), gather_dim_size,
          indices_size, slice_size, out_size);
    } else if (!is_axis_zero && all_in_index_range) {
      LaunchGatherKernel<sycl::half, Index, false, true>(
          d, reinterpret_cast<const sycl::half*>(params.data()), indices.data(),
          reinterpret_cast<sycl::half*>(out.data()), gather_dim_size,
          indices_size, slice_size, out_size);
    } else {
      LaunchGatherKernel<sycl::half, Index, false, false>(
          d, reinterpret_cast<const sycl::half*>(params.data()), indices.data(),
          reinterpret_cast<sycl::half*>(out.data()), gather_dim_size,
          indices_size, slice_size, out_size);
    }
    // TODO(fpmc): enable indices validation on GPU.
    // Right now checking for indicies out of bound in the kernel would
    // require copying code between GPU/CPU, and thus slow.
    return -1;
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_GATHER_FUNCTOR_H_
