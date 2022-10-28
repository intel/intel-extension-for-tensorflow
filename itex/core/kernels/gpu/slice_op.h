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

#ifndef ITEX_CORE_KERNELS_GPU_SLICE_OP_H_
#define ITEX_CORE_KERNELS_GPU_SLICE_OP_H_

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;
namespace functor {

template <typename Device, typename T, int NDIMS>
struct Slice {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_sizes) {
    bool use_64bit = (input.size() > Eigen::NumTraits<int>::highest());
    if (!use_64bit) {
      Eigen::DSizes<int, NDIMS> indices;
      for (int i = 0; i < NDIMS; ++i) {
        indices[i] = slice_indices[i];
      }
      Eigen::DSizes<int, NDIMS> sizes;
      for (int i = 0; i < NDIMS; ++i) {
        sizes[i] = slice_sizes[i];
      }
      To32Bit(output).device(d) = To32Bit(input).slice(indices, sizes);
    } else {
      output.device(d) = input.slice(slice_indices, slice_sizes);
    }
  }
};

template <typename T, typename IntType, int vec_size>
struct SubSliceKernel {
  using Tvec = AlignedVector<T, vec_size>;
  SubSliceKernel(const T* input, T* out, IntType offset, IntType out_size_vec)
      : input_(input),
        out_(out),
        offset_(offset),
        out_size_vec_(out_size_vec) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= out_size_vec_) return;

    id = id * vec_size;
    Tvec in = *(reinterpret_cast<const Tvec*>(input_ + offset_ + id));
    *(reinterpret_cast<Tvec*>(out_ + id)) = in;
  }

 private:
  const T* input_;
  T* out_;
  IntType offset_, out_size_vec_;
};
namespace detail {
template <typename IntType>
struct LaunchSubSliceKernelVectorized {
  template <int vec_size>
  struct Impl {
    template <typename T>
    void operator()(const GPUDevice& d, const T* input, T* out, IntType begin,
                    IntType end, IntType slice_size) {
      IntType out_size_vec = (end - begin) * slice_size / vec_size;
      IntType offset = begin * slice_size;
      auto& stream = d.stream();
      auto workgroup_size =
          (*stream)
              .get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      auto num_workgroups =
          (out_size_vec + workgroup_size - 1) / workgroup_size;
      stream->submit([&](sycl::handler& cgh) {
        SubSliceKernel<T, IntType, vec_size> task(input, out, offset,
                                                  out_size_vec);
        cgh.parallel_for<SubSliceKernel<T, IntType, vec_size>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    }
  };
};
}  // namespace detail

template <typename T, typename IntType>
struct SubSliceFunctor {
  void operator()(const GPUDevice& d, const T* input, T* out, IntType begin,
                  IntType end, IntType slice_size) {
    DispatchToVectorized<
        T, detail::LaunchSubSliceKernelVectorized<IntType>::template Impl>(
        MinAlignmentOf(input), d, input, out, begin, end, slice_size);
  }
};

}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_SLICE_OP_H_
