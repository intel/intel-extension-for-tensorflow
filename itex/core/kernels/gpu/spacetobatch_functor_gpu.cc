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

#include "itex/core/kernels/gpu/spacetobatch_functor.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

// Shape and padding parameters for space-to-batch and batch-to-space conversion
// GPU kernel.
template <int NUM_BLOCK_DIMS>
struct S2BParameters {
  typedef Eigen::internal::TensorIntDivisor<int32> FastDivisor;
  int32 space_tensor_batch;
  int32 batch_tensor_shape[NUM_BLOCK_DIMS + 2];
  int32 space_tensor_spatial_shape[NUM_BLOCK_DIMS];
  int32 pad_start[NUM_BLOCK_DIMS];
  int32 block_shape[NUM_BLOCK_DIMS];
  FastDivisor fastdiv_space_tensor_batch;
  FastDivisor fastdiv_batch_tensor_shape[NUM_BLOCK_DIMS + 2];
  FastDivisor fastdiv_block_shape[NUM_BLOCK_DIMS];
};

template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct S2B {
  S2B(const int32 nthreads, T* space_tensor_ptr,
      S2BParameters<NUM_BLOCK_DIMS> args, T* batch_tensor_ptr)
      : nthreads_(nthreads),
        space_tensor_ptr_(space_tensor_ptr),
        args_(args),
        batch_tensor_ptr_(batch_tensor_ptr) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= nthreads_) {
      return;
    }

    int32 remaining = id;

    int32 batch_tensor_pos[NUM_BLOCK_DIMS + 2];

    for (int dim = NUM_BLOCK_DIMS + 1; dim >= 1; --dim) {
      int32 n = remaining / args_.fastdiv_batch_tensor_shape[dim];
      batch_tensor_pos[dim] = remaining - n * args_.batch_tensor_shape[dim];
      remaining = n;
    }
    batch_tensor_pos[0] = remaining;

    int32 remaining_block_idx =
        batch_tensor_pos[0] / args_.fastdiv_space_tensor_batch;
    int32 space_tensor_idx = batch_tensor_pos[NUM_BLOCK_DIMS + 1];
    int32 space_tensor_stride = args_.batch_tensor_shape[NUM_BLOCK_DIMS + 1];
    const int32 space_tensor_batch_pos =
        batch_tensor_pos[0] - remaining_block_idx * args_.space_tensor_batch;
    for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
      int32 offset = remaining_block_idx;
      if (block_dim > 0) {
        offset = offset - (offset / args_.fastdiv_block_shape[block_dim]) *
                              args_.block_shape[block_dim];
      }
      int32 space_tensor_pos =
          batch_tensor_pos[block_dim + 1] * args_.block_shape[block_dim] +
          offset - args_.pad_start[block_dim];
      if (space_tensor_pos < 0 ||
          space_tensor_pos >= args_.space_tensor_spatial_shape[block_dim]) {
        if (B2S == false) {
          // In the space-to-batch case, write zero padding.
          batch_tensor_ptr_[id] = static_cast<T>(0);
        }
        break;
      }
      space_tensor_idx += space_tensor_stride * space_tensor_pos;
      space_tensor_stride *= args_.space_tensor_spatial_shape[block_dim];
      if (block_dim == 0) {
        space_tensor_idx += space_tensor_stride * space_tensor_batch_pos;
        if (B2S == false) {
          batch_tensor_ptr_[id] = *(space_tensor_ptr_ + space_tensor_idx);
        } else {
          space_tensor_ptr_[space_tensor_idx] = *(batch_tensor_ptr_ + id);
        }
      }
      remaining_block_idx =
          remaining_block_idx / args_.fastdiv_block_shape[block_dim];
    }
  }

 private:
  int32 nthreads_;
  T* space_tensor_ptr_;
  S2BParameters<NUM_BLOCK_DIMS> args_;
  T* batch_tensor_ptr_;
};

template <typename T, int NUM_BLOCK_DIMS, int VEC_SIZE, bool B2S>
struct VectorizedS2B {
  using Tvec = typename BaseTypeVectorize<T, VEC_SIZE>::type;
  using Tscalar = typename BaseTypeVectorize<T, VEC_SIZE>::scalar;
  VectorizedS2B(const int32 nthreads, T* space_tensor_ptr,
                S2BParameters<NUM_BLOCK_DIMS> args, T* batch_tensor_ptr)
      : nthreads_(nthreads),
        space_tensor_ptr_(space_tensor_ptr),
        args_(args),
        batch_tensor_ptr_(batch_tensor_ptr) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= nthreads_) {
      return;
    }

    // convert vec_id to element_id
    id = id * VEC_SIZE;

    int32 remaining = id;

    int32 batch_tensor_pos[NUM_BLOCK_DIMS + 2];

    for (int dim = NUM_BLOCK_DIMS + 1; dim >= 1; --dim) {
      int32 n = remaining / args_.fastdiv_batch_tensor_shape[dim];
      batch_tensor_pos[dim] = remaining - n * args_.batch_tensor_shape[dim];
      remaining = n;
    }
    batch_tensor_pos[0] = remaining;

    int32 remaining_block_idx =
        batch_tensor_pos[0] / args_.fastdiv_space_tensor_batch;
    int32 space_tensor_idx = batch_tensor_pos[NUM_BLOCK_DIMS + 1];
    int32 space_tensor_stride = args_.batch_tensor_shape[NUM_BLOCK_DIMS + 1];
    const int32 space_tensor_batch_pos =
        batch_tensor_pos[0] - remaining_block_idx * args_.space_tensor_batch;
    for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
      int32 offset = remaining_block_idx;
      if (block_dim > 0) {
        offset = offset - (offset / args_.fastdiv_block_shape[block_dim]) *
                              args_.block_shape[block_dim];
      }
      int32 space_tensor_pos =
          batch_tensor_pos[block_dim + 1] * args_.block_shape[block_dim] +
          offset - args_.pad_start[block_dim];
      if (space_tensor_pos < 0 ||
          space_tensor_pos >= args_.space_tensor_spatial_shape[block_dim]) {
        if (B2S == false) {
          // In the space-to-batch case, write zero padding.
          *(reinterpret_cast<Tvec*>(batch_tensor_ptr_ + id)) =
              Tvec(static_cast<Tscalar>(0));
        }
        break;
      }
      space_tensor_idx += space_tensor_stride * space_tensor_pos;
      space_tensor_stride *= args_.space_tensor_spatial_shape[block_dim];
      if (block_dim == 0) {
        space_tensor_idx += space_tensor_stride * space_tensor_batch_pos;
        if (B2S == false) {
          Tvec space_vec =
              *(reinterpret_cast<Tvec*>(space_tensor_ptr_ + space_tensor_idx));
          *(reinterpret_cast<Tvec*>(batch_tensor_ptr_ + id)) = space_vec;
        } else {
          Tvec batch_vec = *(reinterpret_cast<Tvec*>(batch_tensor_ptr_ + id));
          *(reinterpret_cast<Tvec*>(space_tensor_ptr_ + space_tensor_idx)) =
              batch_vec;
        }
      }
      remaining_block_idx =
          remaining_block_idx / args_.fastdiv_block_shape[block_dim];
    }
  }

 private:
  int32 nthreads_;
  T* space_tensor_ptr_;
  S2BParameters<NUM_BLOCK_DIMS> args_;
  T* batch_tensor_ptr_;
};

template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct LaunchS2BVectorized {
  template <int VEC_SIZE>
  struct Impl {
    void operator()(const GPUDevice& d, int32 total_count, T* space_tensor_ptr,
                    S2BParameters<NUM_BLOCK_DIMS> args, T* batch_tensor_ptr) {
      int32 vec_num = total_count / VEC_SIZE;
      auto& stream = d.stream();
      auto workgroup_size =
          (*stream)
              .get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      auto num_workgroups = (vec_num + workgroup_size - 1) / workgroup_size;
      stream->submit([&](sycl::handler& cgh) {
        VectorizedS2B<T, NUM_BLOCK_DIMS, VEC_SIZE, B2S> task(
            vec_num, space_tensor_ptr, args, batch_tensor_ptr);
        cgh.parallel_for<VectorizedS2B<T, NUM_BLOCK_DIMS, VEC_SIZE, B2S>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    }
  };
};

namespace functor {

template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct SpaceToBatchFunctor<GPUDevice, T, NUM_BLOCK_DIMS, B2S> {
  using SpaceT = typename std::conditional<B2S, T, const T>::type;
  using BatchT = typename std::conditional<B2S, const T, T>::type;
  Status operator()(
      const GPUDevice& d,
      typename TTypes<SpaceT, NUM_BLOCK_DIMS + 2>::Tensor space_tensor,
      const int64 block_shape[NUM_BLOCK_DIMS],
      const int64 paddings[NUM_BLOCK_DIMS * 2],
      typename TTypes<BatchT, NUM_BLOCK_DIMS + 2>::Tensor batch_tensor) {
    // Kernel execution fails if number of elements is zero.
    if (batch_tensor.size() == 0) {
      return Status::OK();
    }
    S2BParameters<NUM_BLOCK_DIMS> args;
    args.space_tensor_batch = space_tensor.dimension(0);
    args.fastdiv_space_tensor_batch = space_tensor.dimension(0);
    for (int block_dim = 0; block_dim < NUM_BLOCK_DIMS; ++block_dim) {
      if (block_shape[block_dim] > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("block_shape value exceeds 2^32-1");
      }
      args.block_shape[block_dim] = block_shape[block_dim];
      args.fastdiv_block_shape[block_dim] = block_shape[block_dim];
      if (space_tensor.dimension(block_dim + 1) >
          std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("space_tensor dimension exceeds 2^32-1");
      }
      args.space_tensor_spatial_shape[block_dim] =
          space_tensor.dimension(block_dim + 1);
      if (paddings[block_dim * 2] > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("paddings/crops value exceeds 2^32-1");
      }
      args.pad_start[block_dim] = paddings[block_dim * 2];
    }
    int64 total_count = 1;
    for (int dim = 0; dim < NUM_BLOCK_DIMS + 2; ++dim) {
      args.batch_tensor_shape[dim] = batch_tensor.dimension(dim);
      args.fastdiv_batch_tensor_shape[dim] = batch_tensor.dimension(dim);
      total_count *= args.batch_tensor_shape[dim];
    }
    if (total_count > std::numeric_limits<int32>::max()) {
      return errors::InvalidArgument(
          "number of batch_tensor elements exceeds 2^32-1");
    }

    // In S2B/B2S, only elements in last dimension are always adjacent in momory
    // and can be moved togather. If last dimension has 16N bytes, then we can
    // use vector. The 16byte vector size refer to Eigen's packet size.
    const int64 VecSizeBytes = 16;
    const int64 bytes_in_last_dim =
        batch_tensor.dimension(NUM_BLOCK_DIMS + 1) * sizeof(T);
    const bool is_vectorizable = bytes_in_last_dim % VecSizeBytes == 0;

    if (is_vectorizable) {
      DispatchToVectorized<
          T, LaunchS2BVectorized<T, NUM_BLOCK_DIMS, B2S>::template Impl>(
          VecSizeBytes, d, total_count, const_cast<T*>(space_tensor.data()),
          args, const_cast<T*>(batch_tensor.data()));
    } else {
      auto stream = d.stream();
      auto work_group_size =
          (*stream)
              .get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      auto num_work_items = total_count;
      auto num_work_groups =
          (num_work_items + work_group_size - 1) / work_group_size;

      stream->submit([&](sycl::handler& cgh) {
        S2B<T, NUM_BLOCK_DIMS, B2S> task(
            total_count, const_cast<T*>(space_tensor.data()), args,
            const_cast<T*>(batch_tensor.data()));
        cgh.parallel_for<S2B<T, NUM_BLOCK_DIMS, B2S>>(
            sycl::nd_range<1>(sycl::range<1>(num_work_groups * work_group_size),
                              sycl::range<1>(work_group_size)),
            task);
      });
    }

    return Status::OK();
  }
};

// Instantiate.
#define INSTANTIATE(NUM_BLOCK_DIMS, T)                                      \
  template struct SpaceToBatchFunctor<GPUDevice, T, NUM_BLOCK_DIMS, false>; \
  template struct SpaceToBatchFunctor<GPUDevice, T, NUM_BLOCK_DIMS, true>;  \
  /**/

#define INSTANTIATE_FOR_T(T) \
  TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS(INSTANTIATE, T)

TF_CALL_GPU_NUMBER_TYPES(INSTANTIATE_FOR_T);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(INSTANTIATE_FOR_T);
#endif  // ITEX_ENABLE_DOUBLE

#undef INSTANTIATE_FOR_T
#undef INSTANTIATE

}  // end namespace functor
}  // end namespace itex
