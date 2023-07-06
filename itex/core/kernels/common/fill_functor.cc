/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/kernels/common/fill_functor.h"

#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace functor {

// Explicit instantiations.
#define DEFINE_SETZERO_CPU(T) \
  template struct SetZeroFunctor<Eigen::ThreadPoolDevice, T>;
DEFINE_SETZERO_CPU(bool);
DEFINE_SETZERO_CPU(Eigen::half);
DEFINE_SETZERO_CPU(Eigen::bfloat16);
DEFINE_SETZERO_CPU(float);
#if defined(INTEL_CPU_ONLY) || defined(ITEX_ENABLE_DOUBLE)
DEFINE_SETZERO_CPU(double);
#endif
DEFINE_SETZERO_CPU(uint32);
DEFINE_SETZERO_CPU(uint64);
DEFINE_SETZERO_CPU(uint8);
DEFINE_SETZERO_CPU(int8);
DEFINE_SETZERO_CPU(uint16);
DEFINE_SETZERO_CPU(int16);
DEFINE_SETZERO_CPU(int32);
DEFINE_SETZERO_CPU(int64);
DEFINE_SETZERO_CPU(quint8);
DEFINE_SETZERO_CPU(qint8);
DEFINE_SETZERO_CPU(quint16);
DEFINE_SETZERO_CPU(qint16);
DEFINE_SETZERO_CPU(qint32);
DEFINE_SETZERO_CPU(complex64);
DEFINE_SETZERO_CPU(complex128);
// DEFINE_SETZERO_CPU(Variant);
#undef DEFINE_SETZERO_CPU

// Explicit instantiations.
#define DEFINE_SETONE_CPU(T) \
  template struct SetOneFunctor<Eigen::ThreadPoolDevice, T>;
DEFINE_SETONE_CPU(bool);
DEFINE_SETONE_CPU(Eigen::half);
DEFINE_SETONE_CPU(Eigen::bfloat16);
DEFINE_SETONE_CPU(float);
#if defined(INTEL_CPU_ONLY) || defined(ITEX_ENABLE_DOUBLE)
DEFINE_SETONE_CPU(double);
#endif
DEFINE_SETONE_CPU(uint32);
DEFINE_SETONE_CPU(uint64);
DEFINE_SETONE_CPU(uint8);
DEFINE_SETONE_CPU(int8);
DEFINE_SETONE_CPU(uint16);
DEFINE_SETONE_CPU(int16);
DEFINE_SETONE_CPU(int32);
DEFINE_SETONE_CPU(int64);
DEFINE_SETONE_CPU(complex64);
DEFINE_SETONE_CPU(complex128);
#undef DEFINE_SETONE_CPU

template <typename T>
struct FillFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in) {
    out.device(d) = out.constant(in());
  }
};

// Explicit instantiations.
#define DEFINE_FILL_CPU(T) \
  template struct FillFunctor<Eigen::ThreadPoolDevice, T>;

DEFINE_FILL_CPU(float);
DEFINE_FILL_CPU(Eigen::bfloat16);
DEFINE_FILL_CPU(Eigen::half);
DEFINE_FILL_CPU(bool);
TF_CALL_INTEGRAL_TYPES(DEFINE_FILL_CPU);
#undef DEFINE_FILL_CPU

#ifndef INTEL_CPU_ONLY
template <typename T>
struct FillFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& device, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in) {
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> rank1{1};
#else
    Eigen::IndexList<Eigen::type2index<1> > rank1;
#endif
    const int size = out.dimension(0);
    Eigen::array<int, 1> broadcast_dims{size};
    To32Bit(out).device(device) = in.reshape(rank1).broadcast(broadcast_dims);
  }
};

#define DEFINE_FILL_GPU(T) template struct FillFunctor<Eigen::GpuDevice, T>;

TF_CALL_GPU_ALL_TYPES(DEFINE_FILL_GPU);
TF_CALL_INTEGRAL_TYPES(DEFINE_FILL_GPU);
TF_CALL_complex64(DEFINE_FILL_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_FILL_GPU);
TF_CALL_complex128(DEFINE_FILL_GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef DEFINE_FILL_GPU

#endif  // NOT INTEL_CPU_ONLY

}  // namespace functor
}  // namespace itex
