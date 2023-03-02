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
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_shape.h"
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
  using Tvec = typename BaseTypeVectorize<T, vec_size>::type;
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

template <typename T, typename IntType, int NDIMS>
struct ScalarSliceKernel {
  using FastDivisor = Eigen::internal::TensorIntDivisor<IntType>;
  ScalarSliceKernel(const T* input, T* output,
                    const Eigen::array<IntType, NDIMS>& offset,
                    const Eigen::array<IntType, NDIMS>& input_stride,
                    const Eigen::array<IntType, NDIMS>& output_stride,
                    const Eigen::array<FastDivisor, NDIMS>& fast_output_stride,
                    const IntType element_count)
      : input_(input),
        output_(output),
        offset_(offset),
        input_stride_(input_stride),
        output_stride_(output_stride),
        fast_output_stride_(fast_output_stride),
        element_count_(element_count) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= element_count_) return;

    IntType input_index = 0;
    IntType output_index = id;
    for (int i = 0; i < NDIMS - 1; ++i) {
      const IntType idx = static_cast<IntType>(id) / fast_output_stride_[i];
      input_index += (idx + offset_[i]) * input_stride_[i];
      id -= idx * output_stride_[i];
    }
    input_index += (id + offset_[NDIMS - 1]);

    *(output_ + output_index) = *(input_ + input_index);
  }

 private:
  const T* input_;
  T* output_;
  const Eigen::array<IntType, NDIMS> offset_;
  const Eigen::array<IntType, NDIMS> input_stride_;
  const Eigen::array<IntType, NDIMS> output_stride_;
  const Eigen::array<FastDivisor, NDIMS> fast_output_stride_;
  const IntType element_count_;
};

template <typename T, typename IntType, int NDIMS>
struct PaddedScalarSliceKernel {
  using FastDivisor = Eigen::internal::TensorIntDivisor<IntType>;
  PaddedScalarSliceKernel(
      const T* input, T* output, const Eigen::array<IntType, NDIMS>& offset,
      const Eigen::array<IntType, NDIMS>& input_stride,
      const Eigen::array<IntType, NDIMS>& output_stride,
      const Eigen::array<IntType, NDIMS>& padded_output_stride,
      const Eigen::array<FastDivisor, NDIMS>& fast_padded_output_stride,
      const IntType element_count, const IntType last_dim)
      : input_(input),
        output_(output),
        offset_(offset),
        input_stride_(input_stride),
        output_stride_(output_stride),
        padded_output_stride_(padded_output_stride),
        fast_padded_output_stride_(fast_padded_output_stride),
        element_count_(element_count),
        last_dim_(last_dim) {}

  [[intel::reqd_sub_group_size(16)]] void operator()(
      sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= element_count_) return;

    IntType input_index = 0;
    IntType output_index = 0;
    for (int i = 0; i < NDIMS - 1; ++i) {
      const IntType idx =
          static_cast<IntType>(id) / fast_padded_output_stride_[i];
      input_index += (idx + offset_[i]) * input_stride_[i];
      output_index += idx * output_stride_[i];
      id -= idx * padded_output_stride_[i];
    }

    // return if workitem is for padded element
    if (id >= last_dim_) return;
    input_index += (id + offset_[NDIMS - 1]);
    output_index += id;

    *(output_ + output_index) = *(input_ + input_index);
  }

 private:
  const T* input_;
  T* output_;
  const Eigen::array<IntType, NDIMS> offset_;
  const Eigen::array<IntType, NDIMS> input_stride_;
  const Eigen::array<IntType, NDIMS> output_stride_;
  const Eigen::array<IntType, NDIMS> padded_output_stride_;
  const Eigen::array<FastDivisor, NDIMS> fast_padded_output_stride_;
  const IntType element_count_;
  const IntType last_dim_;
};

template <typename T, typename IntType, int NDIMS>
struct ScalarSlice {
  using FastDivisor = Eigen::internal::TensorIntDivisor<IntType>;
  void operator()(const GPUDevice& d, const T* input, T* output,
                  const TensorShape& input_shape,
                  const gtl::InlinedVector<int64, 4>& offset,
                  const gtl::InlinedVector<int64, 4>& size) {
    Eigen::array<IntType, NDIMS> input_shape_;
    Eigen::array<IntType, NDIMS> offset_;
    Eigen::array<IntType, NDIMS> size_;
    Eigen::array<IntType, NDIMS> padded_size_;
    Eigen::array<IntType, NDIMS> input_stride;
    Eigen::array<IntType, NDIMS> output_stride;
    Eigen::array<IntType, NDIMS> padded_output_stride;
    Eigen::array<FastDivisor, NDIMS> fast_padded_output_stride;
    for (int i = 0; i < NDIMS; ++i) {
      input_shape_[i] = input_shape.dim_size(i);
      offset_[i] = offset[i];
      size_[i] = size[i];
      padded_size_[i] = size[i];
    }

    // Pad last dimension to make each subgroup simd loading elements in
    // the same cacheline. Because padding may waste workitems, so
    // 1. if padded_shape < hardware_reside_work_item, use padding scalar to
    // improve cache hit
    // 2. if padded_shape >= hardware_reside_work_item, use pure scalar to fully
    // use hardware
    auto& stream = d.stream();
    auto dev = (*stream).get_device();
    const int simd_width =
        dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
    const int hardware_reside_work_item =
        dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>() *
        dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>() *
        dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
    IntType element_count = 1;
    IntType padded_element_count = 1;
    IntType last_dim = size_[NDIMS - 1];
    padded_size_[NDIMS - 1] = last_dim + (simd_width - last_dim % simd_width);
    for (int i = 0; i < NDIMS; ++i) {
      element_count *= size_[i];
      padded_element_count *= padded_size_[i];
    }
    const bool can_pad = padded_element_count < hardware_reside_work_item;

    // pure scalar is also a kind of padded scalar which padding 0 element
    // to the last dimension
    if (!can_pad) {
      padded_size_[NDIMS - 1] = size[NDIMS - 1];
      padded_element_count = element_count;
    }

    // caculate stride
    input_stride[NDIMS - 1] = 1;
    for (int i = NDIMS - 2; i >= 0; --i) {
      input_stride[i] = input_stride[i + 1] * input_shape_[i + 1];
    }
    output_stride[NDIMS - 1] = 1;
    padded_output_stride[NDIMS - 1] = 1;
    for (int i = NDIMS - 2; i >= 0; --i) {
      output_stride[i] = output_stride[i + 1] * size_[i + 1];
      padded_output_stride[i] =
          padded_output_stride[i + 1] * padded_size_[i + 1];
      fast_padded_output_stride[i] = FastDivisor(padded_output_stride[i]);
    }

    // submit sycl kernel.
    auto workgroup_size =
        dev.get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups =
        (padded_element_count + workgroup_size - 1) / workgroup_size;

    // use pure scalar.
    if (!can_pad) {
      stream->submit([&](sycl::handler& cgh) {
        ScalarSliceKernel<T, IntType, NDIMS> task(
            input, output, offset_, input_stride, output_stride,
            fast_padded_output_stride, padded_element_count);
        cgh.parallel_for<ScalarSliceKernel<T, IntType, NDIMS>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    } else {
      // use padded scalar.
      stream->submit([&](sycl::handler& cgh) {
        PaddedScalarSliceKernel<T, IntType, NDIMS> task(
            input, output, offset_, input_stride, output_stride,
            padded_output_stride, fast_padded_output_stride,
            padded_element_count, last_dim);
        cgh.parallel_for<PaddedScalarSliceKernel<T, IntType, NDIMS>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    }
  }
};
}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_SLICE_OP_H_
