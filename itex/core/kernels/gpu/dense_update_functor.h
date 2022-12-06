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

#ifndef ITEX_CORE_KERNELS_GPU_DENSE_UPDATE_FUNCTOR_H_
#define ITEX_CORE_KERNELS_GPU_DENSE_UPDATE_FUNCTOR_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

enum DenseUpdateType { ADD, SUB, ASSIGN };

namespace functor {

template <typename Device, typename T, DenseUpdateType OP>
struct DenseUpdate {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update);
};

template <typename T>
struct DenseUpdate<CPUDevice, T, ASSIGN> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};

template <typename T>
struct DenseUpdate<CPUDevice, T, ADD> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) += update;
  }
};

template <typename T>
struct DenseUpdate<CPUDevice, T, SUB> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) -= update;
  }
};

template <typename T>
struct DenseUpdate<GPUDevice, T, ASSIGN> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};

template <typename T>
struct DenseUpdate<GPUDevice, T, ADD> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) += update;
  }
};

template <typename T>
struct DenseUpdate<GPUDevice, T, SUB> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) -= update;
  }
};

}  // end namespace functor

#define DEFINE_GPU_KERNELS(T)                              \
  template struct functor::DenseUpdate<GPUDevice, T, ADD>; \
  template struct functor::DenseUpdate<GPUDevice, T, SUB>;
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_int32(DEFINE_GPU_KERNELS);
TF_CALL_int64(DEFINE_GPU_KERNELS);
TF_CALL_int8(DEFINE_GPU_KERNELS);
#undef DEFINE_GPU_KERNELS

#define DEFINE_GPU_KERNELS(T) \
  template struct functor::DenseUpdate<GPUDevice, T, ASSIGN>;
TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_int32(DEFINE_GPU_KERNELS);
TF_CALL_int64(DEFINE_GPU_KERNELS);
TF_CALL_int8(DEFINE_GPU_KERNELS);
TF_CALL_uint32(DEFINE_GPU_KERNELS);
TF_CALL_complex64(DEFINE_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_GPU_KERNELS);
TF_CALL_complex128(DEFINE_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE

#undef DEFINE_GPU_KERNELS

}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_DENSE_UPDATE_FUNCTOR_H_
