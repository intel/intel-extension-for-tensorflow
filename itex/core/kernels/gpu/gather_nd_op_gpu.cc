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

#include "itex/core/kernels/gpu/gather_nd_op.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index, int IXDIM>
struct GatherSliceOpKernel {
  GatherSliceOpKernel(const T* params, const Index* indices, T* out,
                      const Eigen::array<int64, IXDIM> batch_strides,
                      const Eigen::array<int64, IXDIM> batch_indices,
                      const int64 indices_size, const int64 slice_size,
                      const int64 out_size)
      : params_(params),
        indices_(indices),
        out_(out),
        batch_strides_(batch_strides),
        batch_indices_(batch_indices),
        indices_size_(indices_size),
        slice_size_(slice_size),
        out_size_(out_size) {}
  void operator()(sycl::nd_item<1> item) const {
    // TODO(itex): reduce inner loop into two loops:
    // one over the number of locs, and one over the offsets inside the locs.
    auto i = item.get_global_linear_id();
    if (i < out_size_) {
      const Index loc = i / slice_size_;
      const auto indices_i = indices_ + IXDIM * loc;
      bool out_of_bounds = false;
      Index offset = 0;
#pragma unroll
      for (int j = 0; j < IXDIM; ++j) {
        const Index index_j = indices_i[j];
        out_of_bounds |= !FastBoundsCheck(index_j, batch_indices_[j]);
        offset += batch_strides_[j] * index_j;
      }
      // TODO(itex):
      // This is the only part that depends on the offset.  The part
      // above does not need to be executed for every index i.
      // Is there a way to break the outer loop into two loops?  One
      // that determines how many slice_size-length locs are iterated
      // over, and another that iterates over slice_size iterations for
      // the correct indices?
      // NOTE(eriche):
      // You can consider one kernel where a warp or block is assigned
      // to one offset.  The calculation of offset can be shared within
      // the warp or block and then the warp / block can cooperate to
      // the copy.
      const Index loc_offset = i - loc * slice_size_;
      if (out_of_bounds) {
        out_[i] = static_cast<T>(0);
      } else {
        out_[i] = *(params_ + offset + loc_offset);
      }
    }
  }

 private:
  const T* params_;
  const Index* indices_;
  T* out_;
  const Eigen::array<int64, IXDIM> batch_strides_;
  const Eigen::array<int64, IXDIM> batch_indices_;
  const int64 indices_size_;
  const int64 slice_size_;
  const int64 out_size_;
};

template <typename T, typename Index, int IXDIM>
struct GatherSliceOpKernelNullIdx {
  GatherSliceOpKernelNullIdx(const T* params, T* out, const int64 slice_size,
                             const int64 out_size)
      : params_(params),
        out_(out),
        slice_size_(slice_size),
        out_size_(out_size) {}
  void operator()(sycl::nd_item<1> item) const {
    auto i = item.get_global_linear_id();
    if (i < out_size_) {
      const Index loc = i / slice_size_;
      const Index loc_offset = i - loc * slice_size_;
      out_[i] = *(params_ + loc_offset);
    }
  }

 private:
  const T* params_;
  T* out_;
  const int64 slice_size_;
  const int64 out_size_;
};

namespace functor {

template <typename T, typename Index, int IXDIM>
struct GatherNdSlice<GPUDevice, T, Index, IXDIM> {
  Index operator()(const GPUDevice& d, const Index unused_slice_size,
                   typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Matrix Tout) {
    const int64 indices_size = Tindices.dimension(1);
    const int64 out_size = Tout.size();
    int64 s_size = Tout.dimension(1);
    Eigen::array<int64, IXDIM> batch_strides;
    Eigen::array<int64, IXDIM> batch_indices;
    if (IXDIM > 0) {
      batch_strides[size_t(IXDIM - 1)] = s_size;
      batch_indices[size_t(IXDIM - 1)] = Tparams.dimension(IXDIM - 1);
    }
    for (int i = IXDIM - 1; i > 0; --i) {
      batch_indices[i - 1] = Tparams.dimension(i - 1);
      batch_strides[i - 1] = batch_strides[i] * Tparams.dimension(i);
    }
    auto& stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (out_size + group_size - 1) / group_size;
    if (Tindices.data() != nullptr && Tindices.size() != 0) {
      stream->submit([&](sycl::handler& cgh) {
        GatherSliceOpKernel<T, Index, IXDIM> task(
            Tparams.data(), Tindices.data(), Tout.data(), batch_strides,
            batch_indices, indices_size, s_size, out_size);

        cgh.parallel_for<GatherSliceOpKernel<T, Index, IXDIM>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
    } else {
      stream->submit([&](sycl::handler& cgh) {
        GatherSliceOpKernelNullIdx<T, Index, IXDIM> task(
            Tparams.data(), Tout.data(), s_size, out_size);
        cgh.parallel_for<GatherSliceOpKernelNullIdx<T, Index, IXDIM>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
    }

    // TODO(itex): enable indices validation on GPU.
    // Right now checking for indices out of bound in the kernel would
    // require copying code between GPU/CPU, and is too slow.
    return -1;
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, NDIM) \
  template struct functor::GatherNdSlice<GPUDevice, T, Index, NDIM>;

#define DEFINE_GPU_SPECS_INDEX(T, Index)    \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 0); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 1); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 2); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 3); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 4); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 5); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 6); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 7);

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

TF_CALL_int32(DEFINE_GPU_SPECS);
TF_CALL_int64(DEFINE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
TF_CALL_complex64(DEFINE_GPU_SPECS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_GPU_SPECS);
TF_CALL_complex128(DEFINE_GPU_SPECS);
#endif  // ITEX_ENABLE_DOUBLE

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace itex
