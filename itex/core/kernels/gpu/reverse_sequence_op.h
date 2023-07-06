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

#ifndef ITEX_CORE_KERNELS_GPU_REVERSE_SEQUENCE_OP_H_
#define ITEX_CORE_KERNELS_GPU_REVERSE_SEQUENCE_OP_H_

#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace generator {

template <typename T, typename Tlen, size_t Dims>
class ReverseGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ReverseGenerator(typename TTypes<T, Dims>::ConstTensor input, int32 batch_dim,
                   int32 seq_dim, typename TTypes<Tlen>::ConstVec seq_lengths)
      : input_(input),
        batch_dim_(batch_dim),
        seq_dim_(seq_dim),
        seq_lengths_(seq_lengths) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, Dims>& coords) const {
    Eigen::array<Eigen::DenseIndex, Dims> new_coords = coords;
    if (coords[seq_dim_] < seq_lengths_(coords[batch_dim_])) {
      new_coords[seq_dim_] =
          seq_lengths_(coords[batch_dim_]) - coords[seq_dim_] - 1;
    }

    return input_(new_coords);
  }

 private:
  typename TTypes<T, Dims>::ConstTensor input_;
  int32 batch_dim_;
  int32 seq_dim_;
  typename TTypes<Tlen>::ConstVec seq_lengths_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T, typename Tlen, size_t Dims>
struct ReverseSequence {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,
      int32 batch_dim, int32 seq_dim,
      typename TTypes<Tlen>::ConstVec seq_lengths,
      typename TTypes<T, Dims>::Tensor output) {
    generator::ReverseGenerator<T, Tlen, Dims> generator(input, batch_dim,
                                                         seq_dim, seq_lengths);
    output.device(d) = input.generate(generator);
  }
};

template <typename T, typename Tlen, size_t Dims>
struct ReverseSequenceKernelITEX_GPU {
  ReverseSequenceKernelITEX_GPU(
      const int32 batch_dim, const int32 seq_dim,
      const Eigen::DSizes<Eigen::DenseIndex, Dims> coord_dims,
      const Tlen* seq_lengths, const T* input, T* output, const int64 size)
      : batch_dim_(batch_dim),
        seq_dim_(seq_dim),
        coord_dims_(coord_dims),
        seq_lengths_(seq_lengths),
        input_(input),
        output_(output),
        size_(size) {}

  // Unflatten the indices and return the dimension dim
  inline int32 get_coord_dim(const int32 coord, const int32 dim) const {
    int32 mod = coord_dims_[Dims - 1];
    int32 div = 1;
    for (int32 i = dim; i < Dims - 1; ++i) {
      mod *= coord_dims_[i];
      div *= coord_dims_[i + 1];
    }
    return (coord % mod) / div;
  }

  inline void operator()(sycl::nd_item<1> item) const {
    const auto coord = item.get_global_id(0);

    if (coord >= size_) return;

    auto new_coord = coord;
    auto coord_seq_dim = get_coord_dim(coord, seq_dim_);
    auto coord_batch_dim = get_coord_dim(coord, batch_dim_);
    auto seq = seq_lengths_[coord_batch_dim];
    if (coord_seq_dim < seq) {
      new_coord = 1;
      for (int32 i = seq_dim_ + 1; i < Dims; ++i) new_coord *= coord_dims_[i];
      new_coord = coord + (seq - 2 * coord_seq_dim - 1) * new_coord;
    }
    output_[coord] = input_[new_coord];
  }

 private:
  const int32 batch_dim_;
  const int32 seq_dim_;
  const Eigen::DSizes<Eigen::DenseIndex, Dims> coord_dims_;
  const Tlen* seq_lengths_;
  const T* input_;
  T* output_;
  const int64 size_;
};

template <typename T, typename Tlen, size_t Dims>
struct ReverseSequence<GPUDevice, T, Tlen, Dims> {
  EIGEN_ALWAYS_INLINE static void Compute(
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input,
      int32 batch_dim, int32 seq_dim,
      typename TTypes<Tlen>::ConstVec seq_lengths,
      typename TTypes<T, Dims>::Tensor output) {
    auto stream = d.stream();
    const auto coord_dims = input.dimensions();

    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (input.size() + group_size - 1) / group_size;

    const int64 size = input.size();
    const int32 b_dim = batch_dim;
    const int32 s_dim = seq_dim;
    stream->submit([&](sycl::handler& cgh) {
      ReverseSequenceKernelITEX_GPU<T, Tlen, Dims> kernel(
          b_dim, s_dim, coord_dims, seq_lengths.data(), input.data(),
          output.data(), size);

      cgh.parallel_for<ReverseSequenceKernelITEX_GPU<T, Tlen, Dims> >(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          kernel);
    });
  }
};

}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_REVERSE_SEQUENCE_OP_H_
