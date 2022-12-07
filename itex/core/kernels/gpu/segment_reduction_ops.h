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

#ifndef ITEX_CORE_KERNELS_GPU_SEGMENT_REDUCTION_OPS_H_
#define ITEX_CORE_KERNELS_GPU_SEGMENT_REDUCTION_OPS_H_

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

using GPUDevice = Eigen::GpuDevice;

// Functor for SegmentSumGPUOp.
// output_rows: the number of output segments (unique segment ids in
//                'segment_ids').
// segment_ids_shape: shape of 'segment_ids' tensor.
// segment_ids: unsorted map from input to output segment ids at which to
//                perform segment sum operation.
// data_size: size of input data tensor.
// data: input data tensor.
// output: output reshaped to {output_rows, output.size/output_rows}
template <typename Device, typename T, typename Index>
struct SegmentSumFunctor {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output,
                  typename TTypes<float, 2>::Tensor output_fp32);
};

template <typename Device, typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor {
  void operator()(OpKernelContext* ctx, const Index num_segments,
                  const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output,
                  typename TTypes<float, 2>::Tensor output_fp32);
};

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF, typename AtomicReductionF>
struct SegmentReductionFunctor {
  void operator()(OpKernelContext* ctx, const Index output_rows,
                  const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output,
                  typename TTypes<float, 2>::Tensor output_fp32);
};

// Note: All the below reduction method should avoid race condition by yourself.

template <typename T>
struct SumOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    ItexAtomicAdd(dest, *value);
  }
  static constexpr bool is_associative = std::is_integral<T>::value;
};

template <typename T>
struct SumOpGpu<std::complex<T>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(
      std::complex<T>* dest, const std::complex<T>* value) {
    T* ptr = reinterpret_cast<T*>(dest);
    ItexAtomicAdd(ptr, value->real());
    ItexAtomicAdd(ptr + 1, value->imag());
  }
  static constexpr bool is_associative = std::is_integral<T>::value;
};

template <typename T>
struct ProdOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    ItexAtomicMul(dest, *value);
  }
  static constexpr bool is_associative = std::is_integral<T>::value;
};

template <typename T>
struct MaxOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    ItexAtomicMax(dest, *value);
  }
  static constexpr bool is_associative = true;
};

template <typename T>
struct MinOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    ItexAtomicMin(dest, *value);
  }
  static constexpr bool is_associative = true;
};

// Non-atomic reduction functors for the gpu.
template <typename T>
struct NonAtomicSumOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    *dest += *value;
  }
};

template <typename T>
struct NonAtomicProdOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    *dest *= *value;
  }
};

template <typename T>
struct NonAtomicMaxOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    *dest = *dest > *value ? *dest : *value;
  }
};

template <typename T>
struct NonAtomicMinOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    *dest = *dest = *dest < *value ? *dest : *value;
  }
};

// initial value functors
template <typename T>
struct Zero {
  EIGEN_STRONG_INLINE T operator()() const { return T(0); }
};

template <typename T>
struct One {
  EIGEN_STRONG_INLINE T operator()() const { return T(1); }
};

template <typename T>
struct Lowest {
  EIGEN_STRONG_INLINE T operator()() const {
    return Eigen::NumTraits<T>::lowest();
  }
};

template <typename T>
struct Highest {
  EIGEN_STRONG_INLINE T operator()() const {
    return Eigen::NumTraits<T>::highest();
  }
};

namespace impl {

template <typename T, typename Index, typename KernelReductionFunctor,
          typename = void>
struct UnsortedKernel {
  UnsortedKernel(Index input_total_size, Index inner_dim_size,
                 Index output_outer_dim_size, const Index* segment_ids,
                 const T* input, T* output, float* output_fp32)
      : input_total_size(input_total_size),
        inner_dim_size(inner_dim_size),
        output_outer_dim_size(output_outer_dim_size),
        segment_ids(segment_ids),
        input(input),
        output(output),
        output_fp32(output_fp32) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id(0);
    if (id >= input_total_size) return;
    auto input_row = id / inner_dim_size;
    auto col = id % inner_dim_size;
    auto output_row = segment_ids[input_row];
    if (output_row < 0 || output_row >= output_outer_dim_size) return;
    KernelReductionFunctor redux_op;
    redux_op(output + output_row * inner_dim_size + col,
             input + input_row * inner_dim_size + col);
  }

 private:
  Index input_total_size;
  Index inner_dim_size;
  Index output_outer_dim_size;
  const Index* segment_ids;
  const T* input;
  T* output;
  float* output_fp32;
};

template <typename T, typename Index, typename KernelReductionFunctor>
struct UnsortedKernel<
    T, Index, KernelReductionFunctor,
    typename std::enable_if_t<(std::is_same<T, Eigen::half>::value ||
                               std::is_same<T, Eigen::bfloat16>::value),
                              void>> {
  UnsortedKernel(Index input_total_size, Index inner_dim_size,
                 Index output_outer_dim_size, const Index* segment_ids,
                 const T* input, T* output, float* output_fp32)
      : input_total_size(input_total_size),
        inner_dim_size(inner_dim_size),
        output_outer_dim_size(output_outer_dim_size),
        segment_ids(segment_ids),
        input(input),
        output(output),
        output_fp32(output_fp32) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id(0);
    if (id >= input_total_size) return;
    auto input_row = id / inner_dim_size;
    auto col = id % inner_dim_size;
    auto output_row = segment_ids[input_row];
    if (output_row < 0 || output_row >= output_outer_dim_size) return;
    KernelReductionFunctor redux_op;
    float val = static_cast<float>(*(input + input_row * inner_dim_size + col));
    redux_op(output_fp32 + output_row * inner_dim_size + col, &val);
  }

 private:
  Index input_total_size;
  Index inner_dim_size;
  Index output_outer_dim_size;
  const Index* segment_ids;
  const T* input;
  T* output;
  float* output_fp32;
};

template <typename T, typename Index, typename KernelReductionFunctor>
struct UnsortedSegmentCustomKernel {
  Status operator()(const GPUDevice& device, const Index input_outer_dim_size,
                    const Index inner_dim_size,
                    const Index output_outer_dim_size, const Index* segment_ids,
                    const T* input, T* output, float* output_fp32) {
    auto stream = device.stream();
    const Index input_total_size = input_outer_dim_size * inner_dim_size;
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    auto num_work_group =
        (input_total_size + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      UnsortedKernel<T, Index, KernelReductionFunctor> task(
          input_total_size, inner_dim_size, output_outer_dim_size, segment_ids,
          input, output, output_fp32);
      cgh.parallel_for<UnsortedKernel<T, Index, KernelReductionFunctor>>(
          sycl::nd_range<1>(sycl::range<1>(work_group_size * num_work_group),
                            sycl::range<1>(work_group_size)),
          task);
    });
    return Status::OK();
  }
};

template <typename Index, typename KernelReductionFunctor>
struct UnsortedSegmentCustomKernel<Eigen::bfloat16, Index,
                                   KernelReductionFunctor> {
  Status operator()(const GPUDevice& device, const Index input_outer_dim_size,
                    const Index inner_dim_size,
                    const Index output_outer_dim_size, const Index* segment_ids,
                    const Eigen::bfloat16* input, Eigen::bfloat16* output,
                    float* output_fp32) {
    auto stream = device.stream();
    const Index input_total_size = input_outer_dim_size * inner_dim_size;
    const Index output_total_size = output_outer_dim_size * inner_dim_size;
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    auto num_work_group =
        (input_total_size + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      UnsortedKernel<Eigen::bfloat16, Index, KernelReductionFunctor> task(
          input_total_size, inner_dim_size, output_outer_dim_size, segment_ids,
          input, output, output_fp32);
      cgh.parallel_for<
          UnsortedKernel<Eigen::bfloat16, Index, KernelReductionFunctor>>(
          sycl::nd_range<1>(sycl::range<1>(work_group_size * num_work_group),
                            sycl::range<1>(work_group_size)),
          task);
    });
    ConvertFromFp32<GPUDevice, Eigen::bfloat16>(device, output_total_size,
                                                output_fp32, output);
    return Status::OK();
  }
};

template <typename Index, typename KernelReductionFunctor>
struct UnsortedSegmentCustomKernel<Eigen::half, Index, KernelReductionFunctor> {
  Status operator()(const GPUDevice& device, const Index input_outer_dim_size,
                    const Index inner_dim_size,
                    const Index output_outer_dim_size, const Index* segment_ids,
                    const Eigen::half* input, Eigen::half* output,
                    float* output_fp32) {
    auto stream = device.stream();
    const Index input_total_size = input_outer_dim_size * inner_dim_size;
    const Index output_total_size = output_outer_dim_size * inner_dim_size;
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    auto num_work_group =
        (input_total_size + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      UnsortedKernel<Eigen::half, Index, KernelReductionFunctor> task(
          input_total_size, inner_dim_size, output_outer_dim_size, segment_ids,
          input, output, output_fp32);
      cgh.parallel_for<
          UnsortedKernel<Eigen::half, Index, KernelReductionFunctor>>(
          sycl::nd_range<1>(sycl::range<1>(work_group_size * num_work_group),
                            sycl::range<1>(work_group_size)),
          task);
    });
    ConvertFromFp32<GPUDevice, Eigen::half>(device, output_total_size,
                                            output_fp32, output);
    return Status::OK();
  }
};

template <typename T, typename Index, typename ReductionF,
          typename AtomicReductionF, int OuterDimTileSize, typename = void>
struct SortedKernel {
  SortedKernel(Index input_outer_dim_size, Index inner_dim_size,
               Index output_outer_dim_size, const Index* segment_ids,
               const T* input, T* output, float* output_fp32,
               Index total_stripe_count, T initial_value)
      : input_outer_dim_size(input_outer_dim_size),
        inner_dim_size(inner_dim_size),
        output_outer_dim_size(output_outer_dim_size),
        segment_ids(segment_ids),
        input(input),
        output(output),
        output_fp32(output_fp32),
        total_stripe_count(total_stripe_count),
        initial_value(initial_value) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id(0);
    if (id >= total_stripe_count) return;
    auto segment_offset = id % inner_dim_size;
    auto input_outer_dim_index_base =
        id / inner_dim_size * Index(OuterDimTileSize);
    T reduce_res = initial_value;
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;
    auto actual_stripe_height =
        Index(OuterDimTileSize) <
                (input_outer_dim_size - input_outer_dim_index_base)
            ? Index(OuterDimTileSize)
            : (input_outer_dim_size - input_outer_dim_index_base);
    ReductionF reduction_op;
    AtomicReductionF atom_reduction_op;
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      if (current_output_segment_id > last_output_segment_id) {
        auto output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        if (last_output_segment_id == first_segment_id) {
          atom_reduction_op(output + output_index, &reduce_res);
        } else {
          reduction_op(output + output_index, &reduce_res);
        }
        reduce_res = initial_value;
      }
      reduction_op(&reduce_res,
                   input + (input_outer_dim_index_base + j) * inner_dim_size +
                       segment_offset);
      last_output_segment_id = current_output_segment_id;
    }
    auto output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    atom_reduction_op(output + output_index, &reduce_res);
  }

 private:
  Index input_outer_dim_size;
  Index inner_dim_size;
  Index output_outer_dim_size;
  const Index* segment_ids;
  const T* input;
  T* output;
  float* output_fp32;
  Index total_stripe_count;
  T initial_value;
};

template <typename T, typename Index, typename ReductionF,
          typename AtomicReductionF, int OuterDimTileSize>
struct SortedKernel<T, Index, ReductionF, AtomicReductionF, OuterDimTileSize,
                    std::enable_if_t<(std::is_same<T, Eigen::half>::value ||
                                      std::is_same<T, Eigen::bfloat16>::value ||
                                      std::is_same<T, double>::value),
                                     void>> {
  SortedKernel(Index input_outer_dim_size, Index inner_dim_size,
               Index output_outer_dim_size, const Index* segment_ids,
               const T* input, T* output, float* output_fp32,
               Index total_stripe_count, T initial_value)
      : input_outer_dim_size(input_outer_dim_size),
        inner_dim_size(inner_dim_size),
        output_outer_dim_size(output_outer_dim_size),
        segment_ids(segment_ids),
        input(input),
        output(output),
        output_fp32(output_fp32),
        total_stripe_count(total_stripe_count),
        initial_value(initial_value) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id(0);
    if (id >= total_stripe_count) return;
    auto segment_offset = id % inner_dim_size;
    auto input_outer_dim_index_base =
        id / inner_dim_size * Index(OuterDimTileSize);
    float reduce_res = static_cast<float>(initial_value);
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;
    auto actual_stripe_height =
        Index(OuterDimTileSize) <
                (input_outer_dim_size - input_outer_dim_index_base)
            ? Index(OuterDimTileSize)
            : (input_outer_dim_size - input_outer_dim_index_base);
    ReductionF reduction_op;
    AtomicReductionF atom_reduction_op;
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      if (current_output_segment_id > last_output_segment_id) {
        auto output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        if (last_output_segment_id == first_segment_id) {
          atom_reduction_op(output_fp32 + output_index, &reduce_res);
        } else {
          reduction_op(output_fp32 + output_index, &reduce_res);
        }
        reduce_res = static_cast<float>(initial_value);
      }
      float val = static_cast<float>(
          *(input + (input_outer_dim_index_base + j) * inner_dim_size +
            segment_offset));
      reduction_op(&reduce_res, &val);
      last_output_segment_id = current_output_segment_id;
    }
    auto output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    atom_reduction_op(output_fp32 + output_index, &reduce_res);
  }

 private:
  Index input_outer_dim_size;
  Index inner_dim_size;
  Index output_outer_dim_size;
  const Index* segment_ids;
  const T* input;
  T* output;
  float* output_fp32;
  Index total_stripe_count;
  T initial_value;
};

template <typename T, typename Index, int OuterDimTileSize, typename ReductionF,
          typename AtomicReductionF>
struct SortedSegmentCustomKernel {
  Status operator()(const GPUDevice& device, const Index input_outer_dim_size,
                    const Index inner_dim_size,
                    const Index output_outer_dim_size, const Index* segment_ids,
                    const T* input, T* output, float* output_fp32,
                    const Index total_stripe_count, const T initial_value) {
    auto stream = device.stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_group =
        (total_stripe_count + work_group_size - 1) / work_group_size;

    sycl::range<1> local_size(work_group_size);
    sycl::range<1> global_size(num_work_group * work_group_size);
    stream->submit([&](sycl::handler& cgh) {
      SortedKernel<T, Index, ReductionF, AtomicReductionF, OuterDimTileSize>
          task(input_outer_dim_size, inner_dim_size, output_outer_dim_size,
               segment_ids, input, output, output_fp32, total_stripe_count,
               initial_value);
      cgh.parallel_for<SortedKernel<T, Index, ReductionF, AtomicReductionF,
                                    OuterDimTileSize>>(
          sycl::nd_range<1>(global_size, local_size), task);
    });
    return Status::OK();
  }
};

template <typename Index, int OuterDimTileSize, typename ReductionF,
          typename AtomicReductionF>
struct SortedSegmentCustomKernel<Eigen::bfloat16, Index, OuterDimTileSize,
                                 ReductionF, AtomicReductionF> {
  Status operator()(const GPUDevice& device, const Index input_outer_dim_size,
                    const Index inner_dim_size,
                    const Index output_outer_dim_size, const Index* segment_ids,
                    const Eigen::bfloat16* input, Eigen::bfloat16* output,
                    float* output_fp32, const Index total_stripe_count,
                    const Eigen::bfloat16 initial_value) {
    auto stream = device.stream();
    const Index output_total_size = output_outer_dim_size * inner_dim_size;
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_group =
        (total_stripe_count + work_group_size - 1) / work_group_size;

    sycl::range<1> local_size(work_group_size);
    sycl::range<1> global_size(num_work_group * work_group_size);
    stream->submit([&](sycl::handler& cgh) {
      SortedKernel<Eigen::bfloat16, Index, ReductionF, AtomicReductionF,
                   OuterDimTileSize>
          task(input_outer_dim_size, inner_dim_size, output_outer_dim_size,
               segment_ids, input, output, output_fp32, total_stripe_count,
               initial_value);

      cgh.parallel_for<SortedKernel<Eigen::bfloat16, Index, ReductionF,
                                    AtomicReductionF, OuterDimTileSize>>(
          sycl::nd_range<1>(global_size, local_size), task);
    });
    ConvertFromFp32<GPUDevice, Eigen::bfloat16>(device, output_total_size,
                                                output_fp32, output);
    return Status::OK();
  }
};

template <typename Index, int OuterDimTileSize, typename ReductionF,
          typename AtomicReductionF>
struct SortedSegmentCustomKernel<Eigen::half, Index, OuterDimTileSize,
                                 ReductionF, AtomicReductionF> {
  Status operator()(const GPUDevice& device, const Index input_outer_dim_size,
                    const Index inner_dim_size,
                    const Index output_outer_dim_size, const Index* segment_ids,
                    const Eigen::half* input, Eigen::half* output,
                    float* output_fp32, const Index total_stripe_count,
                    const Eigen::half initial_value) {
    auto stream = device.stream();
    const Index output_total_size = output_outer_dim_size * inner_dim_size;
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_group =
        (total_stripe_count + work_group_size - 1) / work_group_size;

    sycl::range<1> local_size(work_group_size);
    sycl::range<1> global_size(num_work_group * work_group_size);
    stream->submit([&](sycl::handler& cgh) {
      SortedKernel<Eigen::half, Index, ReductionF, AtomicReductionF,
                   OuterDimTileSize>
          task(input_outer_dim_size, inner_dim_size, output_outer_dim_size,
               segment_ids, input, output, output_fp32, total_stripe_count,
               initial_value);
      cgh.parallel_for<SortedKernel<Eigen::half, Index, ReductionF,
                                    AtomicReductionF, OuterDimTileSize>>(
          sycl::nd_range<1>(global_size, local_size), task);
    });
    ConvertFromFp32<GPUDevice, Eigen::half>(device, output_total_size,
                                            output_fp32, output);
    return Status::OK();
  }
};

template <typename Index, int OuterDimTileSize, typename ReductionF,
          typename AtomicReductionF>
struct SortedSegmentCustomKernel<double, Index, OuterDimTileSize, ReductionF,
                                 AtomicReductionF> {
  Status operator()(const GPUDevice& device, const Index input_outer_dim_size,
                    const Index inner_dim_size,
                    const Index output_outer_dim_size, const Index* segment_ids,
                    const double* input, double* output, float* output_fp32,
                    const Index total_stripe_count,
                    const double initial_value) {
    auto stream = device.stream();
    const Index output_total_size = output_outer_dim_size * inner_dim_size;
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_group =
        (total_stripe_count + work_group_size - 1) / work_group_size;

    sycl::range<1> local_size(work_group_size);
    sycl::range<1> global_size(num_work_group * work_group_size);
    stream->submit([&](sycl::handler& cgh) {
      SortedKernel<double, Index, ReductionF, AtomicReductionF,
                   OuterDimTileSize>
          task(input_outer_dim_size, inner_dim_size, output_outer_dim_size,
               segment_ids, input, output, output_fp32, total_stripe_count,
               initial_value);
      cgh.parallel_for<SortedKernel<double, Index, ReductionF, AtomicReductionF,
                                    OuterDimTileSize>>(
          sycl::nd_range<1>(global_size, local_size), task);
    });
    ConvertFromFp32<GPUDevice, double>(device, output_total_size, output_fp32,
                                       output);
    return Status::OK();
  }
};

}  // end namespace impl

template <typename T, typename Index>
struct SegmentSumFunctor<GPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output,
                  typename TTypes<float, 2>::Tensor output_fp32) {
    if (output.size() == 0) {
      return;
    }
    // Set 'output' to initial value.
    if (std::is_same<T, Eigen::bfloat16>::value ||
        std::is_same<T, Eigen::half>::value || std::is_same<T, double>::value) {
      float init = 0.0f;
      output_fp32.device(d) = output_fp32.constant(init);
    } else {
      auto init = T(0);
      output.device(d) = output.constant(init);
    }
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }
    int num_segments = 1;
    for (int i = 1; i < segment_ids.dimension(0); i++) {
      if (segment_ids(i) != segment_ids(i - 1)) {
        num_segments += 1;
      }
    }

    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_outer_dim_size = segment_ids.dimension(0);
    const Index input_inner_dim_size = data_size / input_outer_dim_size;

    if (std::is_same<T, Eigen::bfloat16>::value ||
        std::is_same<T, Eigen::half>::value || std::is_same<T, double>::value)
      auto status =
          impl::UnsortedSegmentCustomKernel<T, Index,
                                            functor::SumOpGpu<float>>()(
              d, input_outer_dim_size, input_inner_dim_size, num_segments,
              segment_ids.data(), data, output.data(), output_fp32.data());
    else
      auto status =
          impl::UnsortedSegmentCustomKernel<T, Index, functor::SumOpGpu<T>>()(
              d, input_outer_dim_size, input_inner_dim_size, num_segments,
              segment_ids.data(), data, output.data(), output_fp32.data());
  }
};

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<GPUDevice, T, Index, InitialValueF, ReductionF> {
  void operator()(OpKernelContext* ctx, const Index num_segments,
                  const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output,
                  typename TTypes<float, 2>::Tensor output_fp32) {
    if (output.size() == 0) {
      return;
    }

    // Set 'output' to initial value.
    const GPUDevice& d = ctx->eigen_gpu_device();
    auto init = InitialValueF()();
    output.device(d) = output.constant(init);
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }

    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_outer_dim_size = segment_ids.dimension(0);
    const Index input_inner_dim_size = data_size / input_outer_dim_size;
    auto status = impl::UnsortedSegmentCustomKernel<T, Index, ReductionF>()(
        d, input_outer_dim_size, input_inner_dim_size, num_segments,
        segment_ids.data(), data, output.data(), output_fp32.data());
  }
};

template <typename Index, typename InitialValueF, typename ReductionF>
struct UnsortedSegmentFunctor<GPUDevice, Eigen::bfloat16, Index, InitialValueF,
                              ReductionF> {
  void operator()(OpKernelContext* ctx, const Index num_segments,
                  const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const Eigen::bfloat16* data,
                  typename TTypes<Eigen::bfloat16, 2>::Tensor output,
                  typename TTypes<float, 2>::Tensor output_fp32) {
    if (output.size() == 0) {
      return;
    }

    // Set 'output' to initial value.
    const GPUDevice& d = ctx->eigen_gpu_device();
    float init = InitialValueF()();
    output_fp32.device(d) = output_fp32.constant(init);
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }

    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_outer_dim_size = segment_ids.dimension(0);
    const Index input_inner_dim_size = data_size / input_outer_dim_size;
    auto status =
        impl::UnsortedSegmentCustomKernel<Eigen::bfloat16, Index, ReductionF>()(
            d, input_outer_dim_size, input_inner_dim_size, num_segments,
            segment_ids.data(), data, output.data(), output_fp32.data());
  }
};

template <typename Index, typename InitialValueF, typename ReductionF>
struct UnsortedSegmentFunctor<GPUDevice, Eigen::half, Index, InitialValueF,
                              ReductionF> {
  void operator()(OpKernelContext* ctx, const Index num_segments,
                  const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const Eigen::half* data,
                  typename TTypes<Eigen::half, 2>::Tensor output,
                  typename TTypes<float, 2>::Tensor output_fp32) {
    if (output.size() == 0) {
      return;
    }

    // Set 'output' to initial value.
    const GPUDevice& d = ctx->eigen_gpu_device();
    float init = static_cast<float>(InitialValueF()());
    output_fp32.device(d) = output_fp32.constant(init);
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }

    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_outer_dim_size = segment_ids.dimension(0);
    const Index input_inner_dim_size = data_size / input_outer_dim_size;
    auto status =
        impl::UnsortedSegmentCustomKernel<Eigen::half, Index, ReductionF>()(
            d, input_outer_dim_size, input_inner_dim_size, num_segments,
            segment_ids.data(), data, output.data(), output_fp32.data());
  }
};

// #define DEFINE_SORTED_GPU_SPECS_INDEX(T, Index) \
//   template struct SegmentSumFunctor<GPUDevice, T, Index>

// #define DEFINE_SORTED_GPU_SPECS(T)         \
//   DEFINE_SORTED_GPU_SPECS_INDEX(T, int32); \
//   DEFINE_SORTED_GPU_SPECS_INDEX(T, int64);

// TF_CALL_FLOAT_TYPES(DEFINE_SORTED_GPU_SPECS);
// TF_CALL_int32(DEFINE_SORTED_GPU_SPECS);

#define DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, T_Reduction, Index)       \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,             \
                                         functor::Lowest<T_Reduction>,    \
                                         functor::MaxOpGpu<T_Reduction>>; \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,             \
                                         functor::Highest<T_Reduction>,   \
                                         functor::MinOpGpu<T_Reduction>>; \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,             \
                                         functor::One<T_Reduction>,       \
                                         functor::ProdOpGpu<T_Reduction>>;

// sum is the only op that supports all input types currently
#define DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, T_Reduction, Index)   \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,        \
                                         functor::Zero<T_Reduction>, \
                                         functor::SumOpGpu<T_Reduction>>;

#define DEFINE_REAL_GPU_SPECS(T)                     \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, T, int32); \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, T, int64);

#define DEFINE_REAL_GPU_SPECS_BF16()                                   \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(Eigen::bfloat16, float, int32); \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(Eigen::bfloat16, float, int64);

#define DEFINE_REAL_GPU_SPECS_HALF()                               \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(Eigen::half, float, int32); \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(Eigen::half, float, int64);

#define DEFINE_SUM_GPU_SPECS(T)                     \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, T, int32); \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, T, int64);

#define DEFINE_SUM_GPU_SPECS_BF16()                                   \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(Eigen::bfloat16, float, int32); \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(Eigen::bfloat16, float, int64);

#define DEFINE_SUM_GPU_SPECS_HALF()                               \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(Eigen::half, float, int32); \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(Eigen::half, float, int64);

DEFINE_REAL_GPU_SPECS_BF16();
DEFINE_REAL_GPU_SPECS_HALF();
TF_CALL_float(DEFINE_REAL_GPU_SPECS);
TF_CALL_int32(DEFINE_REAL_GPU_SPECS);
DEFINE_SUM_GPU_SPECS_BF16();
DEFINE_SUM_GPU_SPECS_HALF();
TF_CALL_float(DEFINE_SUM_GPU_SPECS);
TF_CALL_int32(DEFINE_SUM_GPU_SPECS);

#undef DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_REAL_GPU_SPECS
#undef DEFINE_SUM_GPU_SPECS
#undef DEFINE_REAL_GPU_SPECS_BF16
#undef DEFINE_REAL_GPU_SPECS_HALF
#undef DEFINE_SUM_GPU_SPECS_BF16
#undef DEFINE_SUM_GPU_SPECS_HALF

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF, typename AtomicReductionF>
void SegmentReductionFunctor<T, Index, InitialValueF, ReductionF,
                             AtomicReductionF>::
operator()(OpKernelContext* ctx, const Index output_rows,
           const TensorShape& segment_ids_shape,
           typename TTypes<Index>::ConstFlat segment_ids, const Index data_size,
           const T* data, typename TTypes<T, 2>::Tensor output,
           typename TTypes<float, 2>::Tensor output_fp32) {
  if (output.size() == 0) {
    return;
  }

  // Set 'output' to initial value.
  const GPUDevice& d = ctx->eigen_gpu_device();
  auto init = InitialValueF()();
  if (std::is_same<T, Eigen::bfloat16>::value ||
      std::is_same<T, Eigen::half>::value || std::is_same<T, double>::value) {
    output_fp32.device(d) = output_fp32.constant(static_cast<float>(init));
  } else {
    output.device(d) = output.constant(static_cast<T>(init));
  }
  if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
    return;
  }

  // Launch kernel to compute sorted segment reduction.
  // Notes:
  // *) 'input_total_size' is the total number of elements to process.
  // *) 'segment_ids.shape' is a prefix of data's shape.
  // *) 'input_outer_dim_size' is the total number of segments to process.
  const Index input_total_size = data_size;
  const Index input_outer_dim_size = segment_ids.dimension(0);
  const Index input_inner_dim_size = input_total_size / input_outer_dim_size;

  const int OuterDimTileSize = 8;

  const Index input_outer_dim_num_stripe =
      Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));

  const Index total_stripe_count =
      input_inner_dim_size * input_outer_dim_num_stripe;

  auto status = impl::SortedSegmentCustomKernel<T, Index, OuterDimTileSize,
                                                ReductionF, AtomicReductionF>()(
      d, input_outer_dim_size, input_inner_dim_size, output_rows,
      segment_ids.data(), data, output.data(), output_fp32.data(),
      total_stripe_count, static_cast<T>(init));
}

#define DEFINE_SORTED_GPU_SPECS_INDEX(T, T_Reduction, Index) \
  template struct SegmentReductionFunctor<                   \
      T, Index, functor::Zero<T_Reduction>,                  \
      functor::NonAtomicSumOpGpu<T_Reduction>,               \
      functor::SumOpGpu<T_Reduction>>;                       \
  template struct SegmentReductionFunctor<                   \
      T, Index, functor::One<T_Reduction>,                   \
      functor::NonAtomicProdOpGpu<T_Reduction>,              \
      functor::ProdOpGpu<T_Reduction>>;                      \
  template struct SegmentReductionFunctor<                   \
      T, Index, functor::Highest<T_Reduction>,               \
      functor::NonAtomicMinOpGpu<T_Reduction>,               \
      functor::MinOpGpu<T_Reduction>>;                       \
  template struct SegmentReductionFunctor<                   \
      T, Index, functor::Lowest<T_Reduction>,                \
      functor::NonAtomicMaxOpGpu<T_Reduction>,               \
      functor::MaxOpGpu<T_Reduction>>;

#define DEFINE_SORTED_GPU_SPECS(T)            \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, T, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, T, int64);

#define DEFINE_SORTED_GPU_SPECS_BF16()                          \
  DEFINE_SORTED_GPU_SPECS_INDEX(Eigen::bfloat16, float, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(Eigen::bfloat16, float, int64);

#define DEFINE_SORTED_GPU_SPECS_HALF()                      \
  DEFINE_SORTED_GPU_SPECS_INDEX(Eigen::half, float, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(Eigen::half, float, int64);

TF_CALL_float(DEFINE_SORTED_GPU_SPECS);
TF_CALL_int32(DEFINE_SORTED_GPU_SPECS);
DEFINE_SORTED_GPU_SPECS_BF16();
DEFINE_SORTED_GPU_SPECS_HALF();
#ifdef ITEX_ENABLE_DOUBLE
DEFINE_SORTED_GPU_SPECS_INDEX(double, float, int32);
DEFINE_SORTED_GPU_SPECS_INDEX(double, float, int64);
#endif  // ITEX_ENABLE_DOUBLE
#undef DEFINE_SORTED_GPU_SPECS
#undef DEFINE_SORTED_GPU_SPECS_BF16
#undef DEFINE_SORTED_GPU_SPECS_HALF
#undef DEFINE_SORTED_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_SEGMENT_REDUCTION_OPS_H_
