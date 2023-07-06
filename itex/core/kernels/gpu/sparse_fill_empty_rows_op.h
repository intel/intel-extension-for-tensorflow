/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_SPARSE_FILL_EMPTY_ROWS_OP_H_
#define ITEX_CORE_KERNELS_GPU_SPARSE_FILL_EMPTY_ROWS_OP_H_

#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_types.h"

namespace itex {

namespace functor {

template <typename Device, typename T, typename Tindex>
struct SparseFillEmptyRows {
  // Note that the done callback is only used by the GPU implementation.
  Status operator()(OpKernelContext* context, const Tensor& default_value_t,
                    const Tensor& indices_t, const Tensor& values_t,
                    const Tensor& dense_shape_t);
};

}  // namespace functor

namespace impl {

template <typename Tindex>
struct CountElementsPerRowKernel {
  CountElementsPerRowKernel(const Tindex indices_num, const Tindex dense_rows,
                            const int rank, const Tindex* indices,
                            Tindex* elements_per_row,
                            Tindex* rows_are_not_ordered,
                            Tindex* first_invalid_index)
      : indices_num(indices_num),
        dense_rows(dense_rows),
        rank(rank),
        indices(indices),
        elements_per_row(elements_per_row),
        rows_are_not_ordered(rows_are_not_ordered),
        first_invalid_index(first_invalid_index) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= indices_num) return;
    Tindex row = indices[id * rank];
    if (row < 0 || row >= dense_rows) {
      ItexAtomicMin(first_invalid_index, id);
      return;
    }
    ItexAtomicAdd(&elements_per_row[row], 1);
    // Note that this only needs to compare rows, not columns, to satisfy the
    // row-major order invariant.
    if (id > 0 && row < indices[(id - 1) * rank]) {
      // TODO(itex): Replace this with atomic_ref::store when available.
      ItexAtomicMax(rows_are_not_ordered, 1);
    }
  }

 private:
  const Tindex indices_num;
  const Tindex dense_rows;
  const int rank;
  const Tindex* indices;
  Tindex* elements_per_row;
  Tindex* rows_are_not_ordered;
  Tindex* first_invalid_index;
};

template <typename Tindex>
struct ComputeEmptyRowIndicatorKernel {
  ComputeEmptyRowIndicatorKernel(const Tindex dense_rows,
                                 const Tindex* elements_per_row,
                                 bool* empty_row_indicator)
      : dense_rows(dense_rows),
        elements_per_row(elements_per_row),
        empty_row_indicator(empty_row_indicator) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= dense_rows) return;
    empty_row_indicator[id] = elements_per_row[id] == 0;
  }

 private:
  const Tindex dense_rows;
  const Tindex* elements_per_row;
  bool* empty_row_indicator;
};

template <typename T, typename Tindex>
struct ScatterInputElementsKernel {
  ScatterInputElementsKernel(const Tindex indices_num, const Tindex dense_rows,
                             const int rank, const Tindex* input_index_map,
                             const Tindex* indices, const T* values,
                             const Tindex* num_new_rows_before,
                             Tindex* output_indices, T* output_values,
                             Tindex* reverse_index_map)
      : indices_num(indices_num),
        dense_rows(dense_rows),
        rank(rank),
        input_index_map(input_index_map),
        indices(indices),
        values(values),
        num_new_rows_before(num_new_rows_before),
        output_indices(output_indices),
        output_values(output_values),
        reverse_index_map(reverse_index_map) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= indices_num) return;
    Tindex input_i = input_index_map ? input_index_map[id] : id;
    Tindex row = indices[input_i * rank];
    Tindex output_i = id + num_new_rows_before[row];
    for (int dim = 0; dim < rank; ++dim) {
      output_indices[output_i * rank + dim] = indices[input_i * rank + dim];
    }
    output_values[output_i] = values[input_i];
    if (reverse_index_map) {
      reverse_index_map[input_i] = output_i;
    }
  }

 private:
  const Tindex indices_num, dense_rows;
  const int rank;
  const Tindex* input_index_map;
  const Tindex* indices;
  const T* values;
  const Tindex* num_new_rows_before;
  Tindex* output_indices;
  T* output_values;
  Tindex* reverse_index_map;
};

template <typename T, typename Tindex>
struct ScatterNewElementsKernel {
  ScatterNewElementsKernel(const Tindex dense_rows, const int rank,
                           const T* default_value,
                           const Tindex* num_new_rows_through,
                           const Tindex* input_row_ends,
                           const bool* empty_row_indicator,
                           Tindex* output_indices, T* output_values)
      : dense_rows(dense_rows),
        rank(rank),
        default_value(default_value),
        num_new_rows_through(num_new_rows_through),
        input_row_ends(input_row_ends),
        empty_row_indicator(empty_row_indicator),
        output_indices(output_indices),
        output_values(output_values) {}

  void operator()(sycl::nd_item<1> item) const {
    auto row = item.get_global_linear_id();
    if (row >= dense_rows) return;
    if (!empty_row_indicator[row]) return;  // Only process empty rows
    Tindex input_i = (row == 0 ? 0 : input_row_ends[row - 1]);
    Tindex output_i = input_i + (row == 0 ? 0 : num_new_rows_through[row - 1]);
    for (int dim = 0; dim < rank; ++dim) {
      output_indices[output_i * rank + dim] = (dim == 0) ? row : 0;
    }
    output_values[output_i] = *default_value;
  }

 private:
  const Tindex dense_rows;
  const int rank;
  const T* default_value;
  const Tindex* num_new_rows_through;
  const Tindex* input_row_ends;
  const bool* empty_row_indicator;
  Tindex* output_indices;
  T* output_values;
};

template <typename Tindex>
struct CopyRowIndicesKernel {
  CopyRowIndicesKernel(const Tindex indices_num, const int rank,
                       const Tindex* indices, Tindex* row_indices)
      : indices_num(indices_num),
        rank(rank),
        indices(indices),
        row_indices(row_indices) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= indices_num) return;
    row_indices[id] = indices[id * rank];
  }

 private:
  const Tindex indices_num;
  const int rank;
  const Tindex* indices;
  Tindex* row_indices;
};

}  // namespace impl

template <typename Tindex>
Status LaunchCountElementsPerRowKernel(const Eigen::GpuDevice& device,
                                       Tindex indices_num, Tindex dense_rows,
                                       int rank, const Tindex* indices,
                                       Tindex* elements_per_row,
                                       Tindex* rows_are_not_ordered,
                                       Tindex* first_invalid_index) {
  auto stream = device.stream();
  auto work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_group = (indices_num + work_group_size - 1) / work_group_size;
  sycl::range<1> local_size(work_group_size);
  sycl::range<1> global_size(num_work_group * work_group_size);

  stream->submit([&](sycl::handler& cgh) {
    impl::CountElementsPerRowKernel<Tindex> task(
        indices_num, dense_rows, rank, indices, elements_per_row,
        rows_are_not_ordered, first_invalid_index);
    cgh.parallel_for<impl::CountElementsPerRowKernel<Tindex>>(
        sycl::nd_range<1>(global_size, local_size), task);
  });

  return Status::OK();
}

template <typename Tindex>
Status LaunchComputeEmptyRowIndicatorKernel(const Eigen::GpuDevice& device,
                                            Tindex dense_rows,
                                            Tindex* elements_per_row,
                                            bool* empty_row_indicator) {
  auto stream = device.stream();
  auto work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_group = (dense_rows + work_group_size - 1) / work_group_size;

  sycl::range<1> local_size(work_group_size);
  sycl::range<1> global_size(num_work_group * work_group_size);

  stream->submit([&](sycl::handler& cgh) {
    impl::ComputeEmptyRowIndicatorKernel<Tindex> task(
        dense_rows, elements_per_row, empty_row_indicator);
    cgh.parallel_for<impl::ComputeEmptyRowIndicatorKernel<Tindex>>(
        sycl::nd_range<1>(global_size, local_size), task);
  });

  return Status::OK();
}

template <typename T, typename Tindex>
Status LaunchScatterInputElementsKernel(
    const Eigen::GpuDevice& device, const Tindex indices_num,
    const Tindex dense_rows, const int rank, Tindex* input_index_map,
    const Tindex* indices, const T* values,
    const Tindex* num_empty_rows_through, Tindex* output_indices,
    T* output_values, Tindex* reverse_index_map) {
  auto stream = device.stream();
  auto work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_group = (indices_num + work_group_size - 1) / work_group_size;

  sycl::range<1> local_size(work_group_size);
  sycl::range<1> global_size(num_work_group * work_group_size);

  stream->submit([&](sycl::handler& cgh) {
    impl::ScatterInputElementsKernel<T, Tindex> task(
        indices_num, dense_rows, rank, input_index_map, indices, values,
        num_empty_rows_through, output_indices, output_values,
        reverse_index_map);
    cgh.parallel_for<impl::ScatterInputElementsKernel<T, Tindex>>(
        sycl::nd_range<1>(global_size, local_size), task);
  });
  return Status::OK();
}

template <typename T, typename Tindex>
Status LaunchScatterNewElementsKernel(
    const Eigen::GpuDevice& device, const Tindex dense_rows, const int rank,
    const T* default_value, Tindex* input_index_map,
    const Tindex* num_empty_rows_through, const Tindex* input_row_ends,
    const bool* empty_row_indicator, Tindex* output_indices, T* output_values) {
  auto stream = device.stream();
  auto work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_group = (dense_rows + work_group_size - 1) / work_group_size;

  sycl::range<1> local_size(work_group_size);
  sycl::range<1> global_size(num_work_group * work_group_size);

  stream->submit([&](sycl::handler& cgh) {
    impl::ScatterNewElementsKernel<T, Tindex> task(
        dense_rows, rank, default_value, num_empty_rows_through, input_row_ends,
        empty_row_indicator, output_indices, output_values);
    cgh.parallel_for<impl::ScatterNewElementsKernel<T, Tindex>>(
        sycl::nd_range<1>(global_size, local_size), task);
  });
  return Status::OK();
}

template <typename Tindex>
Status LaunchCopyRowIndicesKernel(const Eigen::GpuDevice& device,
                                  const Tindex indices_num, const int rank,
                                  const Tindex* indices, Tindex* row_indices) {
  auto stream = device.stream();
  auto work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_group = (indices_num + work_group_size - 1) / work_group_size;

  sycl::range<1> local_size(work_group_size);
  sycl::range<1> global_size(num_work_group * work_group_size);

  stream->submit([&](sycl::handler& cgh) {
    impl::CopyRowIndicesKernel<Tindex> task(indices_num, rank, indices,
                                            row_indices);
    cgh.parallel_for<impl::CopyRowIndicesKernel<Tindex>>(
        sycl::nd_range<1>(global_size, local_size), task);
  });
  return Status::OK();
}

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SPARSE_FILL_EMPTY_ROWS_OP_H_
