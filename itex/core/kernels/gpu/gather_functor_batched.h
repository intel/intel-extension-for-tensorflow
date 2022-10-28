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

#ifndef ITEX_CORE_KERNELS_GPU_GATHER_FUNCTOR_BATCHED_H_
#define ITEX_CORE_KERNELS_GPU_GATHER_FUNCTOR_BATCHED_H_

#include "itex/core/utils/bounds_check.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename ValueOrVec, typename Index, bool is_axis_zero,
          bool is_batch_dims_zero>
struct GatherBatchOpKernel {
  GatherBatchOpKernel(const ValueOrVec* __restrict__ params_,
                      const Index* __restrict__ indices_,
                      ValueOrVec* __restrict__ out_, int64 outer_size_,
                      int64 gather_dim_size_, int64 indices_size_,
                      int64 slice_size_, int64 out_size_)
      : params(params_),
        indices(indices_),
        out(out_),
        outer_size(outer_size_),
        gather_dim_size(gather_dim_size_),
        indices_size(indices_size_),
        slice_size(slice_size_),
        out_size(out_size_) {}

  // params is a tensor of shape
  // [batch_size, outer_size, gather_dim_size, slice_size].
  void operator()(sycl::nd_item<1> item) const {
    int i = item.get_global_linear_id();
    if (i >= out_size) return;
    Index batch_i = 0;    // The batch index into params to use for i.
    Index outer_i = 0;    // The outer index into params to use for i.
    Index indices_i = 0;  // The index into indices to use for i.
    Index slice_i = 0;  // Index into the current slice in params to use for i.

    const Index slices_count = i / slice_size;
    if (is_batch_dims_zero) {
      if (is_axis_zero) {
        indices_i = slices_count;
      } else {
        outer_i = slices_count / indices_size;
        indices_i = slices_count - outer_i * indices_size;
      }
    } else {
      const Index entries_count = slices_count / indices_size;
      if (is_axis_zero) {
        batch_i = entries_count;
      } else {
        batch_i = entries_count / outer_size;
        outer_i = entries_count - batch_i * outer_size;
      }
      indices_i = slices_count - entries_count * indices_size;
    }
    slice_i = i - slices_count * slice_size;

    // Index into the gather axis to use for i.
    Index gather_i = *(indices + batch_i * indices_size + indices_i);

    // Check gather_i is in [0, gather_dim_size).
    if (!FastBoundsCheck(gather_i, gather_dim_size)) {
      // Set indices out of range to zero
      // TODO(fpmc): Log an error for transfer back to host.
      out[i] = ValueOrVec(0);
    } else {
      // Read params[batch_i, outer_i, gather_i, slice_i] and write it to the
      // i'th position in out.
      Index params_i =
          ((batch_i * outer_size + outer_i) * gather_dim_size + gather_i) *
              slice_size +
          slice_i;
      out[i] = params[params_i];
    }
  }

 private:
  const ValueOrVec* params;
  const Index* indices;
  ValueOrVec* out;
  int64 outer_size;
  int64 gather_dim_size;
  int64 indices_size;
  int64 slice_size;
  int64 out_size;
};

namespace detail {
template <bool is_axis_zero, bool is_batch_dims_zero>
struct LaunchGatherBatchKernelVectorized {
  template <int vec_size>
  struct Impl {
    template <typename T, typename Index>
    void operator()(const GPUDevice& d, const T* params, const Index* indices,
                    T* out, int64 outer_size, int64 gather_dim_size,
                    int64 indices_size, int64 slice_size, int64 out_size) {
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
        GatherBatchOpKernel<Tvec, Index, is_axis_zero, is_batch_dims_zero> task(
            params_vec, indices, out_vec, outer_size, gather_dim_size,
            indices_size, slice_size_vec, out_size_vec);
        cgh.parallel_for<
            GatherBatchOpKernel<Tvec, Index, is_axis_zero, is_batch_dims_zero>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    }
  };
};

}  // namespace detail

template <bool is_axis_zero, bool is_batch_dims_zero, typename T,
          typename Index>
void LaunchGatherKernel(const GPUDevice& d, const T* params,
                        const Index* indices, T* out, int64 outer_size,
                        int64 gather_dim_size, int64 indices_size,
                        int64 slice_size, int64 out_size) {
  // Note that the GPU memory allocator always returns aligned buffers, so the
  // alignment of data pointers is expected to be deterministic.
  // There will be performance cliffs when slice_size is not aligned, but there
  // is no easy way to handle the misalignment because each row will be aligned
  // differently.
  DispatchToVectorized<T, detail::LaunchGatherBatchKernelVectorized<
                              is_axis_zero, is_batch_dims_zero>::template Impl>(
      MinAlignmentOf(params, out, slice_size), d, params, indices, out,
      outer_size, gather_dim_size, indices_size, slice_size, out_size);
}

namespace functor {
template <typename Device, typename T, typename Index>
struct GatherFunctorBatched {
  int64_t operator()(OpKernelContext* ctx,
                     typename TTypes<T, 4>::ConstTensor params,
                     typename TTypes<Index>::ConstFlat indices,
                     typename TTypes<T, 4>::Tensor out);
};

template <typename T, typename Index>
struct GatherFunctorBatched<GPUDevice, T, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 4>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 4>::Tensor out) {
    const GPUDevice& d = ctx->eigen_gpu_device();
    const int64 out_size = out.size();
    if (out_size == 0) {
      // We need a check here since the CPU version does useful error checking
      // work if there are nonempty indices but empty slices, so the kernel is
      // executed in that case.  In the GPU case we don't know how to do error
      // checking, so we skip the loop entirely.
      return -1;
    }
    const bool is_batch_dims_zero = params.dimension(0) == 1;
    const bool is_axis_zero = params.dimension(1) == 1;
    const int64 outer_size = params.dimension(1);
    const int64 gather_dim_size = params.dimension(2);
    const int64 indices_size = indices.size() / params.dimension(0);
    const int64 slice_size = params.dimension(3);

    const auto function =
        is_axis_zero
            ? (is_batch_dims_zero ? LaunchGatherKernel<true, true, T, Index>
                                  : LaunchGatherKernel<true, false, T, Index>)
            : (is_batch_dims_zero ? LaunchGatherKernel<false, true, T, Index>
                                  : LaunchGatherKernel<false, false, T, Index>);
    function(d, params.data(), indices.data(), out.data(), outer_size,
             gather_dim_size, indices_size, slice_size, out_size);
    // TODO(fpmc): enable indices validation on GPU.
    // Right now checking for indices out of bound in the kernel would
    // require copying code between GPU/CPU, and thus slow.
    return -1;
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_GATHER_FUNCTOR_BATCHED_H_
