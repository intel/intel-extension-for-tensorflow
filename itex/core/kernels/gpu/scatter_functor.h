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

#include "itex/core/kernels/common/fill_functor.h"
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
struct ScatterOpKernelITEX_GPUFunc;

template <typename T>
struct ScatterOpKernelITEX_GPUFunc<T, scatter_op::UpdateOp::ASSIGN> {
  void operator()(T* dest, T src) const { *dest = src; }
};

template <typename T>
struct ScatterOpKernelITEX_GPUFunc<T, scatter_op::UpdateOp::ADD> {
  void operator()(T* dest, T src) const { ItexAtomicAdd(dest, src); }
};

template <typename T>
struct ScatterOpKernelITEX_GPUFunc<T, scatter_op::UpdateOp::SUB> {
  void operator()(T* dest, T src) const { ItexAtomicSub(dest, src); }
};

template <typename T>
struct ScatterOpKernelITEX_GPUFunc<T, scatter_op::UpdateOp::MUL> {
  void operator()(T* dest, T src) const { ItexAtomicMul(dest, src); }
};

template <typename T>
struct ScatterOpKernelITEX_GPUFunc<T, scatter_op::UpdateOp::DIV> {
  void operator()(T* dest, T src) const { ItexAtomicDiv(dest, src); }
};

template <typename T>
struct ScatterOpKernelITEX_GPUFunc<T, scatter_op::UpdateOp::MIN> {
  void operator()(T* dest, T src) const { ItexAtomicMin(dest, src); }
};

template <typename T>
struct ScatterOpKernelITEX_GPUFunc<T, scatter_op::UpdateOp::MAX> {
  void operator()(T* dest, T src) const { ItexAtomicMax(dest, src); }
};

}  // namespace internal
}  // namespace scatter_op

namespace functor {
template <typename Device, typename T, typename Index, scatter_op::UpdateOp op,
          typename Enable = void>
struct ScatterFunctor {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<float>::Matrix params_fp32);
};

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op,
          typename Enable = void>
struct ScatterScalarFunctor {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar updates,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<float>::Matrix params_fp32);
};

template <typename T, typename Index, scatter_op::UpdateOp op, typename = void>
struct ScatterScalarOpITEX_GPUKernel {
  ScatterScalarOpITEX_GPUKernel(T* params, const T* updates,
                                const Index* indices, Index first_dim_size,
                                Index indices_size,
                                Index synthesized_updates_size,
                                float* params_fp32)
      : params_(params),
        params_fp32_(params_fp32),
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
      scatter_op::internal::ScatterOpKernelITEX_GPUFunc<T, op>()(
          &params_[params_i], update_val);
    }
  }

  T* params_;
  float* params_fp32_;
  const T* updates_;
  const Index* indices_;
  Index first_dim_size_;
  Index indices_size_;
  Index synthesized_updates_size_;
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarOpITEX_GPUKernel<
    T, Index, op,
    typename std::enable_if_t<(std::is_same_v<T, Eigen::half> ||
                               std::is_same_v<T, Eigen::bfloat16>),
                              void>> {
  ScatterScalarOpITEX_GPUKernel(T* params, const T* updates,
                                const Index* indices, Index first_dim_size,
                                Index indices_size,
                                Index synthesized_updates_size,
                                float* params_fp32)
      : params_(params),
        params_fp32_(params_fp32),
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
      const float update_val = static_cast<float>(*updates_);
      int param_first_index = indices_[indices_i];
      if (!(param_first_index >= 0 && param_first_index < first_dim_size_)) {
        // Ignore indices that are out of range.
        return;
      }
      int params_i = param_first_index * update_block + (i % update_block);
      scatter_op::internal::ScatterOpKernelITEX_GPUFunc<float, op>()(
          &params_fp32_[params_i], update_val);
    }
  }

  T* params_;
  float* params_fp32_;
  const T* updates_;
  const Index* indices_;
  Index first_dim_size_;
  Index indices_size_;
  Index synthesized_updates_size_;
};

template <typename T, typename Index, scatter_op::UpdateOp op, typename = void>
struct ScatterOpITEX_GPUKernel {
  ScatterOpITEX_GPUKernel(T* params, const T* updates, const Index* indices,
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
      scatter_op::internal::ScatterOpKernelITEX_GPUFunc<T, op>()(
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
struct ScatterOpITEX_GPUKernel<
    T, Index, op,
    typename std::enable_if_t<(std::is_same_v<T, Eigen::half> ||
                               std::is_same_v<T, Eigen::bfloat16>),
                              void>> {
  ScatterOpITEX_GPUKernel(float* params_fp32, const T* updates,
                          const Index* indices, Index first_dim_size,
                          Index updates_size, Index indices_size)
      : params_fp32(params_fp32),
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
      scatter_op::internal::ScatterOpKernelITEX_GPUFunc<float, op>()(
          &params_fp32[params_i], static_cast<float>(updates_[updates_i]));
    }
  }

  float* params_fp32;
  const T* updates_;
  const Index* indices_;
  Index first_dim_size_;
  Index updates_size_;
  Index indices_size_;
};

// TODO(itex): Remove this specialization template when ITEX_GPU atomic
// operators support bf16/half datatype.
template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<
    GPUDevice, T, Index, op,
    typename std::enable_if<std::is_same<T, Eigen::bfloat16>::value ||
                            std::is_same<T, Eigen::half>::value>::type> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar updates,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<float>::Matrix params_fp32) {
    auto total_size = params.size();
    ConvertToFp32<GPUDevice, T>(d, total_size, params.data(),
                                params_fp32.data());

    const Index first_dim_size = params_fp32.dimension(0);
    const Index indices_size = indices.size();
    const Index synthesized_updates_size = indices_size * params.dimension(1);

    auto stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (synthesized_updates_size + group_size - 1) / group_size;

    stream->submit([&](sycl::handler& cgh) {
      ScatterScalarOpITEX_GPUKernel<T, Index, op> task(
          params.data(), updates.data(), indices.data(), first_dim_size,
          indices_size, synthesized_updates_size, params_fp32.data());
      cgh.parallel_for<ScatterScalarOpITEX_GPUKernel<T, Index, op>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
    const Index sizes = params_fp32.size();
    ConvertFromFp32<GPUDevice, T>(d, sizes, params_fp32.data(), params.data());
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<
    GPUDevice, T, Index, op,
    typename std::enable_if<!std::is_same<T, Eigen::bfloat16>::value &&
                            !std::is_same<T, Eigen::half>::value>::type> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar updates,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<float>::Matrix params_fp32) {
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
      ScatterScalarOpITEX_GPUKernel<T, Index, op> task(
          params.data(), updates.data(), indices.data(), first_dim_size,
          indices_size, synthesized_updates_size, params_fp32.data());
      cgh.parallel_for<ScatterScalarOpITEX_GPUKernel<T, Index, op>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });

    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<
    GPUDevice, T, Index, op,
    typename std::enable_if<std::is_same<T, Eigen::bfloat16>::value ||
                            std::is_same<T, Eigen::half>::value>::type> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<float>::Matrix params_fp32) {
    auto total_size = params.size();
    ConvertToFp32<GPUDevice, T>(d, total_size, params.data(),
                                params_fp32.data());

    const Index first_dim_size = params_fp32.dimension(0);
    const Index indices_size = indices.size();
    const Index updates_size = updates.size();

    auto stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (updates_size + group_size - 1) / group_size;

    stream->submit([&](sycl::handler& cgh) {
      ScatterOpITEX_GPUKernel<T, Index, op> task(
          params_fp32.data(), updates.data(), indices.data(), first_dim_size,
          updates_size, indices_size);
      cgh.parallel_for<ScatterOpITEX_GPUKernel<T, Index, op>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
    const Index sizes = params_fp32.size();
    ConvertFromFp32<GPUDevice, T>(d, sizes, params_fp32.data(), params.data());
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<
    GPUDevice, T, Index, op,
    typename std::enable_if<!std::is_same<T, Eigen::bfloat16>::value &&
                            !std::is_same<T, Eigen::half>::value>::type> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<float>::Matrix params_fp32) {
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
      ScatterOpITEX_GPUKernel<T, Index, op> task(params.data(), updates.data(),
                                                 indices.data(), first_dim_size,
                                                 updates_size, indices_size);
      cgh.parallel_for<ScatterOpITEX_GPUKernel<T, Index, op>>(
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
