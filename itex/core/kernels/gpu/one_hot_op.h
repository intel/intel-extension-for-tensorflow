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

#ifndef ITEX_CORE_KERNELS_GPU_ONE_HOT_OP_H_
#define ITEX_CORE_KERNELS_GPU_ONE_HOT_OP_H_

#include <algorithm>
#include <limits>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace itex {
typedef Eigen::GpuDevice GPUDevice;

namespace impl {

template <typename T, typename TI, typename Index, bool readInCol>
struct OneHotEncodingKernel {
  using FastDivisor = Eigen::internal::TensorIntDivisor<Index>;
  OneHotEncodingKernel(Index elem_cnts, Index outElems_in_single_batch,
                       Index final_depth, Index indices_col,
                       const TI* indices_ptr, const T* on_value_ptr,
                       const T* off_value_ptr, const int64 total_out_elem_cnts,
                       const int32 split_num,
                       FastDivisor outElems_in_single_batch_fast_divisor,
                       FastDivisor final_depth_fast_divisor, T* output_ptr)
      : elem_cnts(elem_cnts),
        outElems_in_single_batch(outElems_in_single_batch),
        final_depth(final_depth),
        indices_col(indices_col),
        indices_ptr(indices_ptr),
        on_value_ptr(on_value_ptr),
        off_value_ptr(off_value_ptr),
        total_out_elem_cnts(total_out_elem_cnts),
        split_num(split_num),
        outElems_in_single_batch_fast_divisor_(
            outElems_in_single_batch_fast_divisor),
        final_depth_fast_divisor_(final_depth_fast_divisor),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id > elem_cnts) {
      return;
    }

    int32 cnt = 0;
    for (Index i = id; i < total_out_elem_cnts && cnt < split_num;
         i += elem_cnts) {
      cnt += 1;
      const Index batch_index =
          static_cast<Index>(i) /
          outElems_in_single_batch_fast_divisor_;  // batch id
      const Index flatten_index = i - outElems_in_single_batch * batch_index;

      const Index row = flatten_index / final_depth_fast_divisor_;
      const Index col = flatten_index - final_depth * row;

      if (readInCol) {
        const Index col_index =
            batch_index * indices_col + col;  // col index in indices
        const Index idx = indices_ptr[col_index];

        assert(idx >= 0 && idx < final_depth);
        output_ptr[i] = (idx == row) ? *on_value_ptr : *off_value_ptr;
      } else {
        const Index row_index =
            batch_index * indices_col + row;  // row index in indices
        const Index idx = indices_ptr[row_index];

        assert(idx >= 0 && idx < final_depth);
        output_ptr[i] = (idx == col) ? *on_value_ptr : *off_value_ptr;
      }
    }
  }

 private:
  Index elem_cnts;
  Index outElems_in_single_batch;
  Index final_depth;
  Index indices_col;
  const TI* indices_ptr;
  const T* on_value_ptr;
  const T* off_value_ptr;
  const int64 total_out_elem_cnts;
  const int32 split_num;
  FastDivisor outElems_in_single_batch_fast_divisor_;
  FastDivisor final_depth_fast_divisor_;
  T* output_ptr;
};

template <typename ValueOrVec, typename T, typename TI, typename Index,
          int vec_size, bool readInCol>
struct OneHotEncodingVectorizeKernel {
  using FastDivisor = Eigen::internal::TensorIntDivisor<Index>;
  OneHotEncodingVectorizeKernel(
      const Index num_work_items, Index outElems_in_single_batch,
      Index final_depth, Index indices_col, const TI* indices_ptr,
      const T* on_value_ptr, const T* off_value_ptr,
      const ValueOrVec off_value_vec, const Index vectorize_num,
      const Index total_vectorize_num, const Index tail_num,
      const int32 split_num, FastDivisor outElems_in_single_batch_fast_divisor,
      FastDivisor final_depth_fast_divisor, T* output_ptr)
      : num_work_items(num_work_items),
        outElems_in_single_batch(outElems_in_single_batch),
        final_depth(final_depth),
        indices_col(indices_col),
        indices_ptr(indices_ptr),
        on_value_ptr(on_value_ptr),
        off_value_ptr(off_value_ptr),
        off_value_vec(off_value_vec),
        vectorize_num(vectorize_num),
        total_vectorize_num(total_vectorize_num),
        tail_num(tail_num),
        split_num(split_num),
        outElems_in_single_batch_fast_divisor_(
            outElems_in_single_batch_fast_divisor),
        final_depth_fast_divisor_(final_depth_fast_divisor),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }

    if (id < vectorize_num) {
      int32 cnt = 0;
      for (Index i = id; i < total_vectorize_num && cnt < split_num;
           i += vectorize_num) {
        cnt += 1;
        const Index start_id = static_cast<Index>(i) * vec_size;

        const Index batch_index =
            start_id / outElems_in_single_batch_fast_divisor_;  // batch id
        const Index flatten_index =
            start_id - outElems_in_single_batch * batch_index;
        const Index row = flatten_index / final_depth_fast_divisor_;
        const Index col = flatten_index - final_depth * row;

        const Index row_index =
            batch_index * indices_col + row;  // row index in indices
        const Index col_index = batch_index * indices_col + col;
        Index indice_id = readInCol ? col_index : row_index;
        const Index idx = indices_ptr[indice_id];
        indice_id = readInCol ? row : col;  // renew index value
        assert(idx >= 0 && idx < final_depth);

        // compute on_pos
        Index on_pos = -1;
        if (indice_id + vec_size <= final_depth) {
          if (indice_id <= idx && idx < indice_id + vec_size)
            on_pos = idx - indice_id;
        } else if (indice_id <= idx && idx < final_depth) {
          on_pos = idx - indice_id;
        }
        if (on_pos == -1) {
          *(reinterpret_cast<ValueOrVec*>(output_ptr + start_id)) =
              off_value_vec;
        } else {
          assert(on_pos >= 0 && on_pos < vec_size);
          ValueOrVec value_vec_update = off_value_vec;
          *(reinterpret_cast<T*>(&value_vec_update) + on_pos) = *on_value_ptr;
          *(reinterpret_cast<ValueOrVec*>(output_ptr + start_id)) =
              value_vec_update;
        }

        if (indice_id + vec_size > final_depth) {  // diff batch_index
          const Index batch_index_next =
              (start_id + vec_size - 1) /
              outElems_in_single_batch_fast_divisor_;  // batch id
          const Index flatten_index_next =
              (start_id + vec_size - 1) -
              outElems_in_single_batch * batch_index_next;
          const Index row_next = flatten_index_next / final_depth_fast_divisor_;
          const Index col_next = flatten_index_next - final_depth * row_next;

          const Index row_index_next = batch_index_next * indices_col +
                                       row_next;  // row index in indices
          const Index col_index_next =
              batch_index_next * indices_col + col_next;
          Index indice_id_next = readInCol ? col_index_next : row_index_next;
          const Index idx_next = indices_ptr[indice_id_next];
          indice_id_next =
              readInCol ? row_next : col_next;  // renew index value
          assert(idx_next >= 0 && idx_next < final_depth);
          if (idx_next >= 0 && idx_next <= indice_id_next) {
            on_pos = idx_next + final_depth - indice_id;
            output_ptr[start_id + on_pos] = *on_value_ptr;
          }
        }
      }

    } else {  // tail
      const Index tail_id = total_vectorize_num * vec_size + id - vectorize_num;
      const Index batch_index =
          tail_id / outElems_in_single_batch_fast_divisor_;  // batch id
      const Index flatten_index =
          tail_id - outElems_in_single_batch * batch_index;

      const Index row = flatten_index / final_depth_fast_divisor_;
      const Index col = flatten_index - final_depth * row;

      if (readInCol) {
        const Index col_index =
            batch_index * indices_col + col;  // col index in indices
        const Index idx = indices_ptr[col_index];

        assert(idx >= 0 && idx < final_depth);
        output_ptr[tail_id] = (idx == row) ? *on_value_ptr : *off_value_ptr;
      } else {
        const Index row_index =
            batch_index * indices_col + row;  // row index in indices
        const Index idx = indices_ptr[row_index];

        assert(idx >= 0 && idx < final_depth);
        output_ptr[tail_id] = (idx == col) ? *on_value_ptr : *off_value_ptr;
      }
    }
  }

 private:
  const Index num_work_items;
  Index outElems_in_single_batch;
  Index final_depth;
  Index indices_col;
  const TI* indices_ptr;
  const T* on_value_ptr;
  const T* off_value_ptr;
  ValueOrVec off_value_vec;
  const Index vectorize_num;
  const Index total_vectorize_num;
  const Index tail_num;
  const int32 split_num;
  FastDivisor outElems_in_single_batch_fast_divisor_;
  FastDivisor final_depth_fast_divisor_;
  T* output_ptr;
};

/*
@param: indices
            batch>1:
                [batch, features]
            batch=1:
                [features]
@ret:   output
            batch>1:
                [batch, features, depth] if axis == -1
                [batch, depth, features] if axis == 1
            batch=1:
                [features, depth] if axis == -1
                [depth, features] if axis == 0
*/
template <typename T, typename TI, typename Index, bool readInCol>
struct OneHotDefaultKernel {
  Status operator()(const GPUDevice& device, const Tensor& indices,
                    const typename TTypes<T>::ConstScalar& on_value,
                    const typename TTypes<T>::ConstScalar& off_value,
                    const int32& elem_cnts, const Index& depth,
                    Index final_depth, Index indices_col,
                    const int64 total_out_elem_cnts, const int32 split_num,
                    const int& axis, Tensor* output) {
    const Index outElems_in_single_batch = indices_col * depth;
    auto stream = device.stream();
    auto max_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (elem_cnts + max_group_size - 1) / max_group_size;

    const TI* indices_ptr = indices.flat<TI>().data();
    const T* on_value_ptr = on_value.data();
    const T* off_value_ptr = off_value.data();
    T* output_ptr = output->flat<T>().data();

#define EigenFastDivisor(divisor, num)                       \
  Eigen::internal::TensorIntDivisor<Index> divisor;          \
  if (num != 0) {                                            \
    divisor = Eigen::internal::TensorIntDivisor<Index>(num); \
  }
    EigenFastDivisor(outElems_in_single_batch_fast_divisor,
                     outElems_in_single_batch);
    EigenFastDivisor(final_depth_fast_divisor, final_depth);

#undef EigenFastDivisor

    stream->submit([&](sycl::handler& cgh) {
      OneHotEncodingKernel<T, TI, Index, readInCol> task(
          elem_cnts, outElems_in_single_batch, final_depth, indices_col,
          indices_ptr, on_value_ptr, off_value_ptr, total_out_elem_cnts,
          split_num, outElems_in_single_batch_fast_divisor,
          final_depth_fast_divisor, output_ptr);
      cgh.parallel_for<OneHotEncodingKernel<T, TI, Index, readInCol>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * max_group_size),
                            sycl::range<1>(max_group_size)),
          task);
    });
    return Status::OK();
  }
};

template <typename T, typename TI, typename Index, bool readInCol>
struct LaunchOneHotVectorizeKernel {
  template <int vec_size>
  struct Impl {
    void operator()(const GPUDevice& device, const Tensor& indices,
                    const typename TTypes<T>::ConstScalar& on_value,
                    const typename TTypes<T>::ConstScalar& off_value,
                    const Index& depth, Index final_depth, Index indices_col,
                    const int& axis, const Index total_vectorize_num,
                    const Index vectorize_num, const Index tail_num,
                    const int32 split_num, Tensor* output) {
      const Index outElems_in_single_batch = indices_col * depth;
      auto stream = device.stream();
      auto max_group_size =
          (*stream)
              .get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      const Index num_work_items = vectorize_num + tail_num;
      const Index num_wg =
          (num_work_items + max_group_size - 1) / max_group_size;

      const TI* indices_ptr = indices.flat<TI>().data();
      const T* on_value_ptr = on_value.data();
      const T* off_value_ptr = off_value.data();
      T* output_ptr = output->flat<T>().data();

      using Tvec = typename BaseTypeVectorize<T, vec_size>::type;
      Tvec off_value_vec = static_cast<Tvec>(0);
      T off_value_single;
      stream->memcpy(&off_value_single, off_value_ptr, sizeof(T)).wait();
      for (int i = 0; i < vec_size; ++i) {
        *(reinterpret_cast<T*>(&off_value_vec) + i) = off_value_single;
      }

      Eigen::internal::TensorIntDivisor<Index>
          outElems_in_single_batch_fast_divisor(outElems_in_single_batch);
      Eigen::internal::TensorIntDivisor<Index> final_depth_fast_divisor(
          final_depth);

      stream->submit([&](sycl::handler& cgh) {
        OneHotEncodingVectorizeKernel<Tvec, T, TI, Index, vec_size, readInCol>
            task(num_work_items, outElems_in_single_batch, final_depth,
                 indices_col, indices_ptr, on_value_ptr, off_value_ptr,
                 off_value_vec, vectorize_num, total_vectorize_num, tail_num,
                 split_num, outElems_in_single_batch_fast_divisor,
                 final_depth_fast_divisor, output_ptr);
        cgh.parallel_for<OneHotEncodingVectorizeKernel<Tvec, T, TI, Index,
                                                       vec_size, readInCol>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * max_group_size),
                              sycl::range<1>(max_group_size)),
            task);
      });
    }
  };
};

}  // namespace impl

namespace functor {
template <typename T, typename TI>
struct OneHot {
  void Compute(OpKernelContext* context, const Tensor& indices,
               const typename TTypes<T>::ConstScalar& on_value,
               const typename TTypes<T>::ConstScalar& off_value,
               const int& axis, const int32& depth, Tensor* output) {
    const int64 total_out_elem_cnts = output->shape().num_elements();

    const int32 indices_col =
        indices.dim_size(indices.shape().dims() - 1);  // last dimention
    int32 final_depth =
        1;  // change depth value and output format according to axis
    if (axis != output->shape().dims() - 1) {
      final_depth = indices_col;
    } else {
      final_depth = depth;
    }
    int32 vec_size = std::min((int64_t)(15 / sizeof(T) + 1),
                              MinAlignmentOf(output->flat<T>().data()));

    // use OneHotVectorizeKernel
    if ((axis == output->shape().dims() - 1) && (vec_size > 1) &&
        (vec_size <= final_depth) && (total_out_elem_cnts >= 1024 * 1024)) {
      int32 split_num = 1;
      const int32 single_out_elem_cnts = 1 << 30;
      const int64 total_vectorize_num = total_out_elem_cnts / vec_size;
      const int32 tail_num =
          total_out_elem_cnts - total_vectorize_num * vec_size;

      // Note that the GPU memory allocator always returns aligned buffers, so
      // the alignment of data pointers is expected to be deterministic. There
      // will be performance cliffs when slice_size is not aligned, but there is
      // no easy way to handle the misalignment because each row will be aligned
      // differently.
      if (total_vectorize_num + tail_num > std::numeric_limits<int32>::max()) {
        split_num = static_cast<int32>(
            (total_vectorize_num + single_out_elem_cnts - 1) >> 30);
        const int32 vectorize_num = single_out_elem_cnts;

        DispatchToVectorized<T, impl::LaunchOneHotVectorizeKernel<
                                    T, TI, int64, false>::template Impl>(
            MinAlignmentOf(output->flat<T>().data()),
            context->eigen_gpu_device(), indices, on_value, off_value, depth,
            final_depth, indices_col, axis, total_vectorize_num, vectorize_num,
            tail_num, split_num, output);
      } else if (total_out_elem_cnts > std::numeric_limits<int32>::max()) {
        DispatchToVectorized<T, impl::LaunchOneHotVectorizeKernel<
                                    T, TI, int64, false>::template Impl>(
            MinAlignmentOf(output->flat<T>().data()),
            context->eigen_gpu_device(), indices, on_value, off_value, depth,
            final_depth, indices_col, axis, total_vectorize_num,
            total_vectorize_num, tail_num, split_num, output);
      } else {
        DispatchToVectorized<T, impl::LaunchOneHotVectorizeKernel<
                                    T, TI, int32, false>::template Impl>(
            MinAlignmentOf(output->flat<T>().data()),
            context->eigen_gpu_device(), indices, on_value, off_value, depth,
            final_depth, indices_col, axis, total_vectorize_num,
            total_vectorize_num, tail_num, split_num, output);
      }
    } else {  // use OneHotDefaultKernel
      int32 out_elem_cnts;
      int32 split_num = 1;

      if (total_out_elem_cnts <= std::numeric_limits<int32>::max()) {
        out_elem_cnts = static_cast<int32>(total_out_elem_cnts);

        if (axis == output->shape().dims() - 1) {
          auto status = impl::OneHotDefaultKernel<T, TI, int32, false>()(
              context->eigen_gpu_device(), indices, on_value, off_value,
              out_elem_cnts, depth, final_depth, indices_col,
              total_out_elem_cnts, split_num, axis, output);
        } else {
          auto status = impl::OneHotDefaultKernel<T, TI, int32, true>()(
              context->eigen_gpu_device(), indices, on_value, off_value,
              out_elem_cnts, depth, final_depth, indices_col,
              total_out_elem_cnts, split_num, axis, output);
        }
      } else {
        const int32 single_out_elem_cnts = 1 << 30;
        split_num = static_cast<int32>(
            (total_out_elem_cnts + single_out_elem_cnts - 1) >> 30);
        out_elem_cnts = single_out_elem_cnts;

        if (axis == output->shape().dims() - 1) {
          auto status = impl::OneHotDefaultKernel<T, TI, int64, false>()(
              context->eigen_gpu_device(), indices, on_value, off_value,
              out_elem_cnts, depth, final_depth, indices_col,
              total_out_elem_cnts, split_num, axis, output);
        } else {
          auto status = impl::OneHotDefaultKernel<T, TI, int64, true>()(
              context->eigen_gpu_device(), indices, on_value, off_value,
              out_elem_cnts, depth, final_depth, indices_col,
              total_out_elem_cnts, split_num, axis, output);
        }
      }
    }
  }
};
}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_ONE_HOT_OP_H_
