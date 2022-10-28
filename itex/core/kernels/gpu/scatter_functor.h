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

#ifndef ITEX_CORE_KERNELS_GPU_SCATTER_FUNCTOR_H_
#define ITEX_CORE_KERNELS_GPU_SCATTER_FUNCTOR_H_

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

class OpKernelContext;
typedef Eigen::GpuDevice GPUDevice;
namespace scatter_op {

enum class UpdateOp { ASSIGN, ADD, SUB, MUL, DIV, MIN, MAX };

namespace internal {

template <typename T, scatter_op::UpdateOp op>
struct ScatterOpKernelDPCPPFunc;

template <typename T>
struct ScatterOpKernelDPCPPFunc<T, scatter_op::UpdateOp::ASSIGN> {
  void operator()(T* dest, T src) const { *dest = src; }
};

template <typename T>
struct ScatterOpKernelDPCPPFunc<T, scatter_op::UpdateOp::ADD> {
  void operator()(T* dest, T src) const { DpcppAtomicAdd(dest, src); }
};

template <typename T>
struct ScatterOpKernelDPCPPFunc<T, scatter_op::UpdateOp::SUB> {
  void operator()(T* dest, T src) const { DpcppAtomicSub(dest, src); }
};

template <typename T>
struct ScatterOpKernelDPCPPFunc<T, scatter_op::UpdateOp::MUL> {
  void operator()(T* dest, T src) const { DpcppAtomicMul(dest, src); }
};

template <typename T>
struct ScatterOpKernelDPCPPFunc<T, scatter_op::UpdateOp::DIV> {
  void operator()(T* dest, T src) const { DpcppAtomicDiv(dest, src); }
};

template <typename T>
struct ScatterOpKernelDPCPPFunc<T, scatter_op::UpdateOp::MIN> {
  void operator()(T* dest, T src) const { DpcppAtomicMin(dest, src); }
};

template <typename T>
struct ScatterOpKernelDPCPPFunc<T, scatter_op::UpdateOp::MAX> {
  void operator()(T* dest, T src) const { DpcppAtomicMax(dest, src); }
};

template <scatter_op::UpdateOp Op>
struct AssignDPCPP {};
template <>
struct AssignDPCPP<scatter_op::UpdateOp::ASSIGN> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = u;
  }
};

template <>
struct AssignDPCPP<scatter_op::UpdateOp::ADD> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = p + u;
  }

  template <typename Device, typename Params, typename Update>
  static void RunScalar(Device d, Params p, Update u) {
    p.device(d) = p + u;
  }
};

template <>
struct AssignDPCPP<scatter_op::UpdateOp::SUB> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = p - u;
  }
};

template <>
struct AssignDPCPP<scatter_op::UpdateOp::MUL> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = p * u;
  }
};

template <>
struct AssignDPCPP<scatter_op::UpdateOp::DIV> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = p / u;
  }
};

template <>
struct AssignDPCPP<scatter_op::UpdateOp::MIN> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = p.cwiseMin(u);
  }
};

template <>
struct AssignDPCPP<scatter_op::UpdateOp::MAX> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = p.cwiseMax(u);
  }
};
}  // namespace internal
}  // namespace scatter_op

namespace functor {
template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices);
};

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices);
};

// TODO(itex): Remove this specialization template when DPCPP atomic operators
// support bf16 datatype.
template <typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<GPUDevice, Eigen::bfloat16, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<Eigen::bfloat16>::Matrix params,
                   typename TTypes<Eigen::bfloat16>::ConstMatrix updates,
                   typename TTypes<Index>::Flat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      const Index index = internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      scatter_op::internal::AssignDPCPP<op>::Run(
          d, params.template chip<0>(index), updates.template chip<0>(i));
    }
    return -1;
  }
};

// TODO(itex): Remove this specialization template when DPCPP atomic operators
// support bf16 datatype.
template <typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<GPUDevice, Eigen::bfloat16, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<Eigen::bfloat16>::Matrix params,
                   const typename TTypes<Eigen::bfloat16>::ConstScalar update,
                   typename TTypes<Index>::Flat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    Eigen::Sizes<1> scalar_dim;
    const int dims = params.NumDimensions - 1;
    Eigen::array<int, dims> broadcast_dims;
    for (Index j = 0; j < dims; j++) {
      broadcast_dims[j] = params.dimension(j + 1);
    }
    // Broadcast update to broadcast_dims
    auto u = update.reshape(scalar_dim).broadcast(broadcast_dims);

    for (Index i = 0; i < N; i++) {
      const Index index = internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Broadcast update to params[index]
      scatter_op::internal::AssignDPCPP<op>::Run(
          d, params.template chip<0>(index), u);
    }
    return -1;
  }
};

// TODO(itex): Remove this specialization template when DPCPP atomic operators
// support fp16 datatype.
template <typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<GPUDevice, Eigen::half, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<Eigen::half>::Matrix params,
                   typename TTypes<Eigen::half>::ConstMatrix updates,
                   typename TTypes<Index>::Flat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      const Index index = internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      scatter_op::internal::AssignDPCPP<op>::Run(
          d, params.template chip<0>(index), updates.template chip<0>(i));
    }
    return -1;
  }
};

// TODO(itex): Remove this specialization template when DPCPP atomic operators
// support fp16 datatype.
template <typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<GPUDevice, Eigen::half, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<Eigen::half>::Matrix params,
                   const typename TTypes<Eigen::half>::ConstScalar update,
                   typename TTypes<Index>::Flat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    Eigen::Sizes<1> scalar_dim;
    const int dims = params.NumDimensions - 1;
    Eigen::array<int, dims> broadcast_dims;
    for (Index j = 0; j < dims; j++) {
      broadcast_dims[j] = params.dimension(j + 1);
    }
    // Broadcast update to broadcast_dims
    auto u = update.reshape(scalar_dim).broadcast(broadcast_dims);

    for (Index i = 0; i < N; i++) {
      const Index index = internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Broadcast update to params[index]
      scatter_op::internal::AssignDPCPP<op>::Run(
          d, params.template chip<0>(index), u);
    }
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarOpDPCPPKernel {
  ScatterScalarOpDPCPPKernel(T* params, const T* updates, const Index* indices,
                             Index first_dim_size, Index indices_size,
                             Index synthesized_updates_size)
      : params_(params),
        updates_(updates),
        indices_(indices),
        first_dim_size_(first_dim_size),
        indices_size_(indices_size),
        synthesized_updates_size_(synthesized_updates_size) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t i = item.get_global_linear_id();
    Index update_block = synthesized_updates_size_ / indices_size_;
    if (i < synthesized_updates_size_) {
      int indices_i = i / update_block;
      const T update_val = *updates_;
      int param_first_index = indices_[indices_i];
      if (!(param_first_index >= 0 && param_first_index < first_dim_size_)) {
        // Ignore indices that are out of range.
        return;
      }
      int params_i = param_first_index * update_block + (i % update_block);
      scatter_op::internal::ScatterOpKernelDPCPPFunc<T, op>()(
          &params_[params_i], update_val);
    }
  }

  T* params_;
  const T* updates_;
  const Index* indices_;
  Index first_dim_size_;
  Index indices_size_;
  Index synthesized_updates_size_;
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterOpDPCPPKernel {
  ScatterOpDPCPPKernel(T* params, const T* updates, const Index* indices,
                       Index first_dim_size, Index updates_size,
                       Index indices_size)
      : params_(params),
        updates_(updates),
        indices_(indices),
        first_dim_size_(first_dim_size),
        updates_size_(updates_size),
        indices_size_(indices_size) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t i = item.get_global_linear_id();
    Index update_block = updates_size_ / indices_size_;
    if (i < updates_size_) {
      int indices_i = i / update_block;
      int updates_i = i;
      int param_first_index = indices_[indices_i];
      if (!(param_first_index >= 0 && param_first_index < first_dim_size_)) {
        // Ignore indices that are out of range.
        return;
      }
      int params_i = param_first_index * update_block + (i % update_block);
      scatter_op::internal::ScatterOpKernelDPCPPFunc<T, op>()(
          &params_[params_i], updates_[updates_i]);
    }
  }

  T* params_;
  const T* updates_;
  const Index* indices_;
  Index first_dim_size_;
  Index updates_size_;
  Index indices_size_;
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<GPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstScalar updates,
                   typename TTypes<Index>::ConstFlat indices) {
    const Index first_dim_size = params.dimension(0);
    const Index indices_size = indices.size();
    const Index synthesized_updates_size = indices_size * params.dimension(1);

    auto stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (synthesized_updates_size + group_size - 1) / group_size;

    stream->submit([&](sycl::handler& cgh) {
      ScatterScalarOpDPCPPKernel<T, Index, op> task(
          params.data(), updates.data(), indices.data(), first_dim_size,
          indices_size, synthesized_updates_size);
      cgh.parallel_for<ScatterScalarOpDPCPPKernel<T, Index, op>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });

    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<GPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    const Index first_dim_size = params.dimension(0);
    const Index indices_size = indices.size();
    const Index updates_size = updates.size();

    auto stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (updates_size + group_size - 1) / group_size;

    stream->submit([&](sycl::handler& cgh) {
      ScatterOpDPCPPKernel<T, Index, op> task(params.data(), updates.data(),
                                              indices.data(), first_dim_size,
                                              updates_size, indices_size);
      cgh.parallel_for<ScatterOpDPCPPKernel<T, Index, op>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });

    return -1;
  }
};

}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_SCATTER_FUNCTOR_H_
