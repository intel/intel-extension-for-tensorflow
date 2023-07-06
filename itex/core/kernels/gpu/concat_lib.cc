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

#include "itex/core/kernels/gpu/concat_lib.h"

#include <limits>

#include "itex/core/kernels/gpu/gpu_device_array.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

template <typename T, typename IntType>
struct ConcatFixedKernel {
  ConcatFixedKernel(size_t num_work_items, size_t total_cols, IntType col_size,
                    const T* const* input_ptrs_ptr, T* output_ptr)
      : num_work_items(num_work_items),
        total_cols(total_cols),
        col_size(col_size),
        input_ptrs_ptr(input_ptrs_ptr),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> nd_item) const {
    const auto id = nd_item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }

    const int row_id = id / total_cols;
    const int col_id = id % total_cols;
    const int input_idx = col_id / col_size;
    const T* input_ptr = input_ptrs_ptr[input_idx];
    const int col_offset = col_id % col_size;
    output_ptr[row_id * total_cols + col_id] =
        input_ptr[row_id * col_size + col_offset];
  }

 private:
  size_t num_work_items;
  size_t total_cols;
  IntType col_size;
  const T* const* input_ptrs_ptr;
  T* output_ptr;
};

template <typename T, typename IntType>
void ConcatFixed(OpKernelContext* c, const IntType& col_size,
                 const size_t& total_rows, const size_t& total_cols,
                 GpuDeviceArrayStruct<const T*> input_ptr_data, T* output_ptr) {
  auto* stream = c->GetDeviceStream();
  const IntType wg_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();
  const IntType num_work_items = total_rows * total_cols;
  const IntType num_work_groups = DivUp(num_work_items, wg_size);
  stream->submit([&](sycl::handler& cgh) {
    const T* const* input_ptrs_ptr =
        GetGpuDeviceArrayOnDevice<const T*>(&input_ptr_data);
    sycl::range<1> global(num_work_groups * wg_size);
    sycl::range<1> local(wg_size);
    ConcatFixedKernel<T, IntType> task(num_work_items, total_cols, col_size,
                                       input_ptrs_ptr, output_ptr);
    cgh.parallel_for<ConcatFixedKernel<T, IntType>>(
        sycl::nd_range<1>(global, local), task);
  });
}

template <typename T, typename IntType>
struct ConcatVariableKernel {
  ConcatVariableKernel(size_t num_work_items, size_t total_cols,
                       size_t input_size, const IntType* col_scan,
                       const T* const* input_ptrs_ptr, T* output_ptr)
      : num_work_items(num_work_items),
        total_cols(total_cols),
        input_size(input_size),
        col_scan(col_scan),
        input_ptrs_ptr(input_ptrs_ptr),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> nd_item) const {
    const auto id = nd_item.get_global_linear_id();
    if (id >= num_work_items) return;

    const int col_id = id % total_cols;
    // do an initial binary search and then scan linearly from there
    // works well when there are many small segments and when the
    // segments are much longer
    const int segment =
        std::upper_bound(col_scan, col_scan + input_size, col_id) - col_scan -
        1;
    int curr_offset = col_scan[segment];
    int curr_segment = segment;
    int curr_col_offset;
    while ((curr_col_offset = col_scan[curr_segment + 1]) <= col_id) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    const int local_col = col_id - curr_offset;
    const int segment_width = curr_col_offset - curr_offset;
    const T* input_ptr = input_ptrs_ptr[curr_segment];
    const int row_id = id / total_cols;
    output_ptr[row_id * total_cols + col_id] =
        input_ptr[row_id * segment_width + local_col];
  }

 private:
  size_t num_work_items;
  size_t total_cols;
  size_t input_size;
  const IntType* col_scan;
  const T* const* input_ptrs_ptr;
  T* output_ptr;
};

template <typename T, typename IntType>
struct ConcatVariableKernelSLM {
  ConcatVariableKernelSLM(size_t num_work_items, size_t total_cols,
                          size_t input_size, size_t cache_count,
                          const IntType* col_scan,
                          const T* const* input_ptrs_ptr, T* output_ptr,
                          sycl::local_accessor<IntType, 1> scratch)
      : num_work_items(num_work_items),
        total_cols(total_cols),
        input_size(input_size),
        cache_count(cache_count),
        col_scan(col_scan),
        input_ptrs_ptr(input_ptrs_ptr),
        output_ptr(output_ptr),
        scratch(scratch) {}
  void operator()(sycl::nd_item<1> nd_item) const {
    auto local_id = nd_item.get_local_linear_id();
    for (int i = local_id; i < cache_count; i += nd_item.get_local_range(0)) {
      scratch[i] = col_scan[i];
    }
    sycl::group_barrier(nd_item.get_group(), sycl::memory_scope_work_group);

    const auto id = nd_item.get_global_linear_id();
    if (id >= num_work_items) return;

    const int col_id = id % total_cols;
    // do an initial binary search and then scan linearly from there
    // works well when there are many small segments and when the
    // segments are much longer
    IntType* scratch_ptr = ITEXGetLocalAccPointer<IntType>(scratch);
    const int segment =
        std::upper_bound(scratch_ptr, scratch_ptr + input_size, col_id) -
        scratch_ptr - 1;
    int curr_offset = scratch[segment];
    int curr_segment = segment;
    int curr_col_offset;
    while ((curr_col_offset = scratch[curr_segment + 1]) <= col_id) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    const int local_col = col_id - curr_offset;
    const int segment_width = curr_col_offset - curr_offset;
    const T* input_ptr = input_ptrs_ptr[curr_segment];
    const int row_id = id / total_cols;
    output_ptr[row_id * total_cols + col_id] =
        input_ptr[row_id * segment_width + local_col];
  }

 private:
  size_t num_work_items;
  size_t total_cols;
  size_t input_size;
  size_t cache_count;
  const IntType* col_scan;
  const T* const* input_ptrs_ptr;
  T* output_ptr;
  sycl::local_accessor<IntType, 1> scratch;
};

template <typename T, typename IntType>
void ConcatVariable(OpKernelContext* c, const size_t& total_rows,
                    const size_t& total_cols,
                    GpuDeviceArrayStruct<const T*> input_ptr_data,
                    GpuDeviceArrayStruct<IntType> output_scan, T* output_ptr) {
  auto* stream = c->GetDeviceStream();
  const IntType wg_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();
  const IntType num_work_items = total_rows * total_cols;
  const IntType num_work_groups = DivUp(num_work_items, wg_size);

  const T* const* input_ptrs_ptr =
      GetGpuDeviceArrayOnDevice<const T*>(&input_ptr_data);
  const IntType* col_scan = GetGpuDeviceArrayOnDevice(&output_scan);
  sycl::range<1> global(num_work_groups * wg_size);
  sycl::range<1> local(wg_size);

  auto slm_size = stream->get_device()
                      .template get_info<sycl::info::device::local_mem_size>();
  auto slm_used = output_scan.size * sizeof(IntType);
  // cache index to SLM
  if (slm_used < slm_size) {
    auto cache_count = output_scan.size;
    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<IntType, 1> scratch(cache_count, cgh);
      ConcatVariableKernelSLM<T, IntType> task(
          num_work_items, total_cols, input_ptr_data.size, cache_count,
          col_scan, input_ptrs_ptr, output_ptr, scratch);
      cgh.parallel_for<ConcatVariableKernelSLM<T, IntType>>(
          sycl::nd_range<1>(global, local), task);
    });
  } else {
    ConcatVariableKernel<T, IntType> task(num_work_items, total_cols,
                                          input_ptr_data.size, col_scan,
                                          input_ptrs_ptr, output_ptr);
    stream->parallel_for<ConcatVariableKernel<T, IntType>>(
        sycl::nd_range<1>(global, local), task);
  }
}

template <typename T, typename IntType, int vec_size>
struct InlinedConcatFixedKernel {
  using Tvec = typename BaseTypeVectorize<T, vec_size>::type;
  using Tscalar = typename BaseTypeVectorize<T, vec_size>::scalar;
  using FastDivisor = Eigen::internal::TensorIntDivisor<IntType>;
  InlinedConcatFixedKernel(size_t num_work_items, size_t total_cols,
                           IntType per_concat_elements, IntType col_size,
                           IntType per_vectorize_num, IntType vectorize_num,
                           int per_tail_num, int offset,
                           FastDivisor col_size_fast_divisor,
                           FastDivisor per_vectorize_num_fast_divisor,
                           FastDivisor per_tail_num_fast_divisor,
                           GpuDeviceArrayStruct<const T*> input_ptr_data,
                           T* output_ptr)
      : num_work_items_(num_work_items),
        total_cols_(total_cols),
        per_concat_elements_(per_concat_elements),
        col_size_(col_size),
        per_vectorize_num_(per_vectorize_num),
        vectorize_num_(vectorize_num),
        per_tail_num_(per_tail_num),
        offset_(offset),
        col_size_fast_divisor_(col_size_fast_divisor),
        per_vectorize_num_fast_divisor_(per_vectorize_num_fast_divisor),
        per_tail_num_fast_divisor_(per_tail_num_fast_divisor),
        input_ptr_data_(input_ptr_data),
        output_ptr_(output_ptr) {}
  void operator()(sycl::nd_item<1> nd_item) const {
    auto id = nd_item.get_global_linear_id();
    if (id >= num_work_items_) {
      return;
    }

    if (id < vectorize_num_) {
      IntType input_id =
          static_cast<IntType>(id) / per_vectorize_num_fast_divisor_;
      IntType input_offset =
          (static_cast<IntType>(id) - input_id * per_vectorize_num_) * vec_size;
      const T* input_ptr = input_ptr_data_.inline_values[input_id];
      Tvec in = *(reinterpret_cast<const Tvec*>(input_ptr + input_offset));
      IntType row_id = input_offset / col_size_fast_divisor_;
      IntType col_id = input_offset - row_id * col_size_;
      IntType output_offset =
          row_id * total_cols_ + (input_id + offset_) * col_size_ + col_id;
      if (output_offset % vec_size == 0 && col_id + vec_size <= col_size_) {
        *(reinterpret_cast<Tvec*>(output_ptr_ + output_offset)) = in;
      } else {
        for (int i = 0; i < vec_size; ++i) {
          *(reinterpret_cast<Tscalar*>(output_ptr_ + output_offset)) = in[i];
          if (col_id + 1 >= col_size_) {
            output_offset = output_offset - col_id + total_cols_;
            col_id = 0;
          } else {
            ++output_offset;
            ++col_id;
          }
        }
      }
    } else {
      IntType input_id = (static_cast<IntType>(id) - vectorize_num_) /
                         per_tail_num_fast_divisor_;
      IntType input_offset = per_concat_elements_ - per_tail_num_ +
                             (static_cast<IntType>(id) - vectorize_num_ -
                              input_id * per_tail_num_);
      IntType row_id = input_offset / col_size_fast_divisor_;
      IntType col_id = input_offset - row_id * col_size_;
      const T* input_ptr = input_ptr_data_.inline_values[input_id];
      IntType output_offset =
          row_id * total_cols_ + (input_id + offset_) * col_size_ + col_id;
      output_ptr_[output_offset] = input_ptr[input_offset];
    }
  }

 private:
  size_t num_work_items_;
  size_t total_cols_;
  IntType per_concat_elements_, col_size_;
  IntType per_vectorize_num_, vectorize_num_;
  int per_tail_num_;
  int offset_;
  FastDivisor col_size_fast_divisor_;
  FastDivisor per_vectorize_num_fast_divisor_;
  FastDivisor per_tail_num_fast_divisor_;
  GpuDeviceArrayStruct<const T*> input_ptr_data_;
  T* output_ptr_;
};
namespace detail {

template <typename T, typename IntType>
struct LaunchInlinedConcatFixedKernelVectorize {
  template <int vec_size>
  struct Impl {
    void operator()(OpKernelContext* c, const IntType& col_size,
                    const size_t& total_rows, const size_t& total_cols,
                    GpuDeviceArrayStruct<const T*> input_ptr_data,
                    T* output_ptr, int offset) {
      auto* stream = c->GetDeviceStream();
      const IntType wg_size =
          (*stream)
              .get_device()
              .get_info<sycl::info::device::max_work_group_size>();
      const IntType per_concat_elements = total_rows * col_size;
      const IntType per_vectorize_num = per_concat_elements / vec_size;
      const IntType vectorize_num = per_vectorize_num * input_ptr_data.size;
      const IntType per_tail_num =
          total_rows * col_size - per_vectorize_num * vec_size;
      const IntType tail_num = per_tail_num * input_ptr_data.size;
      const IntType num_work_items = vectorize_num + tail_num;
      const IntType num_work_groups = DivUp(num_work_items, wg_size);

      Eigen::internal::TensorIntDivisor<IntType> col_size_fast_divisor(
          col_size);

#define EigenFastDivisor(divisor, num)                         \
  Eigen::internal::TensorIntDivisor<IntType> divisor;          \
  if (num != 0) {                                              \
    divisor = Eigen::internal::TensorIntDivisor<IntType>(num); \
  }
      EigenFastDivisor(per_vectorize_num_fast_divisor, per_vectorize_num);
      EigenFastDivisor(per_tail_num_fast_divisor, per_tail_num);

#undef EigenFastDivisor

      stream->submit([&](sycl::handler& cgh) {
        sycl::range<1> global(num_work_groups * wg_size);
        sycl::range<1> local(wg_size);
        InlinedConcatFixedKernel<T, IntType, vec_size> task(
            num_work_items, total_cols, per_concat_elements, col_size,
            per_vectorize_num, vectorize_num, per_tail_num, offset,
            col_size_fast_divisor, per_vectorize_num_fast_divisor,
            per_tail_num_fast_divisor, input_ptr_data, output_ptr);
        cgh.parallel_for<InlinedConcatFixedKernel<T, IntType, vec_size>>(
            sycl::nd_range<1>(global, local), task);
      });
    }
  };
};
};  // namespace detail

template <typename T, typename IntType>
void ConcatFixedImpl(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output) {
  IntType num_inputs = inputs_flat.size();
  IntType col_size = inputs_flat[0]->dimension(1);
  T* output_ptr = output->data();
  size_t total_rows = output->dimension(0);
  size_t total_cols = output->dimension(1);
  int max_input_num_handle = 8;
  if (num_inputs >= max_input_num_handle * 16) {
    max_input_num_handle = num_inputs;
  }
  for (size_t i = 0; i < num_inputs; i += max_input_num_handle) {
    int input_num_handle = max_input_num_handle < num_inputs - i
                               ? max_input_num_handle
                               : num_inputs - i;
    GpuDeviceArrayOnHost<const T*> input_ptrs(c, input_num_handle);
    OP_REQUIRES_OK(c, input_ptrs.Init());
    for (size_t j = 0; j < input_num_handle; ++j) {
      input_ptrs.Set(j, inputs_flat[i + j]->data());
    }
    OP_REQUIRES_OK(c, input_ptrs.Finalize());
    GpuDeviceArrayStruct<const T*> input_ptr_data = input_ptrs.data();
    if (max_input_num_handle == 8) {
      DispatchToVectorized<T, detail::LaunchInlinedConcatFixedKernelVectorize<
                                  T, IntType>::template Impl>(
          MinAlignmentOf(output_ptr), c, col_size, total_rows, total_cols,
          input_ptr_data, output_ptr, i);
    } else {
      ConcatFixed<T, IntType>(c, col_size, total_rows, total_cols,
                              input_ptr_data, output_ptr);
    }
  }
}

template <typename T, typename IntType>
void ConcatVariableImpl(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output) {
  IntType num_inputs = inputs_flat.size();
  GpuDeviceArrayOnHost<const T*> input_ptrs(c, num_inputs);
  OP_REQUIRES_OK(c, input_ptrs.Init());
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ptrs.Set(i, inputs_flat[i]->data());
  }
  OP_REQUIRES_OK(c, input_ptrs.Finalize());

  GpuDeviceArrayOnHost<IntType> output_scan(c, num_inputs + 1);
  OP_REQUIRES_OK(c, output_scan.Init());
  IntType scan = 0;
  output_scan.Set(0, scan);
  for (int i = 0; i < num_inputs; ++i) {
    scan += inputs_flat[i]->dimension(1);
    output_scan.Set(i + 1, scan);
  }
  OP_REQUIRES_OK(c, output_scan.Finalize());

  size_t total_rows = output->dimension(0);
  size_t total_cols = output->dimension(1);
  GpuDeviceArrayStruct<const T*> input_ptr_data = input_ptrs.data();
  T* output_ptr = output->data();
  ConcatVariable<T, IntType>(c, total_rows, total_cols, input_ptr_data,
                             output_scan.data(), output_ptr);
}

template <typename T, typename IntType>
void ConcatSlice(
    const Eigen::GpuDevice& d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output) {
  Eigen::array<IntType, 2> offset{0, 0};
  for (int i = 0; i < inputs_flat.size(); ++i) {
    Eigen::array<IntType, 2> size;
    size[0] = inputs_flat[i]->dimension(0);
    size[1] = inputs_flat[i]->dimension(1);
    if (std::is_same<IntType, int32>::value) {
      To32Bit(*output).slice(offset, size).device(d) = To32Bit(*inputs_flat[i]);
    } else {
      output->slice(offset, size).device(d) = *inputs_flat[i];
    }

    offset[1] += size[1];
  }
}

template <typename T>
void Concat(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output) {
  bool one_size_input = true;
  for (int i = 1; i < inputs_flat.size(); ++i) {
    if (inputs_flat[i]->dimension(1) != inputs_flat[i - 1]->dimension(1)) {
      one_size_input = false;
      break;
    }
  }
  if (one_size_input) {
    if (output->size() < std::numeric_limits<int32>::max()) {
      ConcatFixedImpl<T, int32>(c, inputs_flat, output);
    } else {
      ConcatFixedImpl<T, int64>(c, inputs_flat, output);
    }
  } else {
    if (inputs_flat.size() < 16) {
      if (output->size() < std::numeric_limits<int32>::max()) {
        ConcatSlice<T, int32>(c->eigen_gpu_device(), inputs_flat, output);
      } else {
        ConcatSlice<T, int64>(c->eigen_gpu_device(), inputs_flat, output);
      }
    } else {
      if (output->size() < std::numeric_limits<int32>::max()) {
        ConcatVariableImpl<T, int32>(c, inputs_flat, output);
      } else {
        ConcatVariableImpl<T, int64>(c, inputs_flat, output);
      }
    }
  }
}

#define REGISTER_ITEX_GPU(T)                                                  \
  template void Concat<T>(                                                    \
      OpKernelContext * ctx,                                                  \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs,                                                             \
      typename TTypes<T, 2>::Matrix* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ITEX_GPU);
REGISTER_ITEX_GPU(int32);
REGISTER_ITEX_GPU(int64);
REGISTER_ITEX_GPU(bool);
REGISTER_ITEX_GPU(complex64);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_ITEX_GPU(double);
REGISTER_ITEX_GPU(complex128);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_ITEX_GPU
}  // namespace itex
