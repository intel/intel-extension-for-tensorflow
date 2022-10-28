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

#ifndef ITEX_CORE_KERNELS_GPU_SCAN_OPS_H_
#define ITEX_CORE_KERNELS_GPU_SCAN_OPS_H_

#include "itex/core/kernels/gpu/scan_ops_gpu.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

typedef Eigen::Index Index;

template <typename Reducer, typename T>
struct Scan {
  void operator()(OpKernelContext* ctx, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out, const Reducer& reducer,
                  const bool reverse, const bool exclusive, const int num_outer,
                  const int num_scaned, const int num_inner) {
    // Perform the reverse ops directly with Eigen, which avoids copying the
    // tensor twice compared to using individual ops.
    const Eigen::GpuDevice& d = ctx->eigen_gpu_device();
    Eigen::array<bool, 3> dims;
    dims[0] = false;
    dims[1] = reverse;
    dims[2] = false;
    To32Bit(out).device(d) =
        To32Bit(in).reverse(dims).scan(1, reducer, exclusive).reverse(dims);
  }
};

template <typename T>
struct Scan<Eigen::internal::SumReducer<T>, T> {
  void operator()(OpKernelContext* ctx, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out,
                  const Eigen::internal::SumReducer<T>& reducer,
                  const bool reverse, const bool exclusive, const int num_outer,
                  const int num_scaned, const int num_inner) {
    typedef sycl::plus<T> BinaryOp;
    T init = T(0);
    const int elems = num_outer * num_scaned * num_inner;
    bool is_full_scan = (num_outer == 1) && (num_inner == 1);
    if (is_full_scan)
      launchFullScan<const T, T, T, BinaryOp>(ctx, in.data(), out.data(), init,
                                              BinaryOp(), exclusive, reverse,
                                              elems);
    else
      launchPartialScan<const T, T, T, BinaryOp>(
          ctx, in.data(), out.data(), init, BinaryOp(), exclusive, reverse,
          num_outer, num_scaned, num_inner);
  }
};

template <>
struct Scan<Eigen::internal::SumReducer<Eigen::bfloat16>, Eigen::bfloat16> {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<Eigen::bfloat16, 3>::ConstTensor in,
                  typename TTypes<Eigen::bfloat16, 3>::Tensor out,
                  const Eigen::internal::SumReducer<Eigen::bfloat16>& reducer,
                  const bool reverse, const bool exclusive, const int num_outer,
                  const int num_scaned, const int num_inner) {
    typedef float IntermediateType;
    typedef sycl::plus<IntermediateType> BinaryOp;
    IntermediateType init = IntermediateType(0);
    const int elems = num_outer * num_scaned * num_inner;
    bool is_full_scan = (num_outer == 1) && (num_inner == 1);
    if (is_full_scan)
      launchFullScan<const Eigen::bfloat16, Eigen::bfloat16, IntermediateType,
                     BinaryOp>(ctx, in.data(), out.data(), init, BinaryOp(),
                               exclusive, reverse, elems);
    else
      launchPartialScan<const Eigen::bfloat16, Eigen::bfloat16,
                        IntermediateType, BinaryOp>(
          ctx, in.data(), out.data(), init, BinaryOp(), exclusive, reverse,
          num_outer, num_scaned, num_inner);
  }
};

template <>
struct Scan<Eigen::internal::SumReducer<Eigen::half>, Eigen::half> {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<Eigen::half, 3>::ConstTensor in,
                  typename TTypes<Eigen::half, 3>::Tensor out,
                  const Eigen::internal::SumReducer<Eigen::half>& reducer,
                  const bool reverse, const bool exclusive, const int num_outer,
                  const int num_scaned, const int num_inner) {
    typedef float IntermediateType;
    typedef sycl::plus<IntermediateType> BinaryOp;
    IntermediateType init = IntermediateType(0);
    const int elems = num_outer * num_scaned * num_inner;
    bool is_full_scan = (num_outer == 1) && (num_inner == 1);
    if (is_full_scan)
      launchFullScan<const sycl::half, sycl::half, IntermediateType, BinaryOp>(
          ctx, reinterpret_cast<const sycl::half*>(in.data()),
          reinterpret_cast<sycl::half*>(out.data()), init, BinaryOp(),
          exclusive, reverse, elems);
    else
      launchPartialScan<const sycl::half, sycl::half, IntermediateType,
                        BinaryOp>(
          ctx, reinterpret_cast<const sycl::half*>(in.data()),
          reinterpret_cast<sycl::half*>(out.data()), init, BinaryOp(),
          exclusive, reverse, num_outer, num_scaned, num_inner);
  }
};

template <typename T>
struct Scan<Eigen::internal::ProdReducer<T>, T> {
  void operator()(OpKernelContext* ctx, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out,
                  const Eigen::internal::ProdReducer<T>& reducer,
                  const bool reverse, const bool exclusive, const int num_outer,
                  const int num_scaned, const int num_inner) {
    T init = T(1);
    typedef sycl::multiplies<T> BinaryOp;
    const int elems = num_outer * num_scaned * num_inner;
    bool is_full_scan = (num_outer == 1) && (num_inner == 1);
    if (is_full_scan)
      launchFullScan<const T, T, T, BinaryOp>(ctx, in.data(), out.data(), init,
                                              BinaryOp(), exclusive, reverse,
                                              elems);
    else
      launchPartialScan<const T, T, T, BinaryOp>(
          ctx, in.data(), out.data(), init, BinaryOp(), exclusive, reverse,
          num_outer, num_scaned, num_inner);
  }
};

template <>
struct Scan<Eigen::internal::ProdReducer<Eigen::bfloat16>, Eigen::bfloat16> {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<Eigen::bfloat16, 3>::ConstTensor in,
                  typename TTypes<Eigen::bfloat16, 3>::Tensor out,
                  const Eigen::internal::ProdReducer<Eigen::bfloat16>& reducer,
                  const bool reverse, const bool exclusive, const int num_outer,
                  const int num_scaned, const int num_inner) {
    typedef float IntermediateType;
    typedef sycl::multiplies<IntermediateType> BinaryOp;
    IntermediateType init = IntermediateType(1);
    const int elems = num_outer * num_scaned * num_inner;
    bool is_full_scan = (num_outer == 1) && (num_inner == 1);
    if (is_full_scan)
      launchFullScan<const Eigen::bfloat16, Eigen::bfloat16, IntermediateType,
                     BinaryOp>(ctx, in.data(), out.data(), init, BinaryOp(),
                               exclusive, reverse, elems);
    else
      launchPartialScan<const Eigen::bfloat16, Eigen::bfloat16,
                        IntermediateType, BinaryOp>(
          ctx, in.data(), out.data(), init, BinaryOp(), exclusive, reverse,
          num_outer, num_scaned, num_inner);
  }
};

template <>
struct Scan<Eigen::internal::ProdReducer<Eigen::half>, Eigen::half> {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<Eigen::half, 3>::ConstTensor in,
                  typename TTypes<Eigen::half, 3>::Tensor out,
                  const Eigen::internal::ProdReducer<Eigen::half>& reducer,
                  const bool reverse, const bool exclusive, const int num_outer,
                  const int num_scaned, const int num_inner) {
    typedef float IntermediateType;
    typedef sycl::multiplies<IntermediateType> BinaryOp;
    IntermediateType init = IntermediateType(1);
    const int elems = num_outer * num_scaned * num_inner;
    bool is_full_scan = (num_outer == 1) && (num_inner == 1);
    if (is_full_scan)
      launchFullScan<const sycl::half, sycl::half, IntermediateType, BinaryOp>(
          ctx, reinterpret_cast<const sycl::half*>(in.data()),
          reinterpret_cast<sycl::half*>(out.data()), init, BinaryOp(),
          exclusive, reverse, elems);
    else
      launchPartialScan<const sycl::half, sycl::half, IntermediateType,
                        BinaryOp>(
          ctx, reinterpret_cast<const sycl::half*>(in.data()),
          reinterpret_cast<sycl::half*>(out.data()), init, BinaryOp(),
          exclusive, reverse, num_outer, num_scaned, num_inner);
  }
};

template <typename T>
struct LogSumExp {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    auto mi = Eigen::internal::scalar_min_op<T>()(a, b);
    auto ma = Eigen::internal::scalar_max_op<T>()(a, b);

    auto sub = Eigen::internal::scalar_difference_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto log1p = Eigen::internal::scalar_log1p_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();

    auto logsumexp = add(log1p(exp(sub(mi, ma))), ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? ma : logsumexp;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T packetOp(const T& a,
                                                   const T& b) const {
    auto mi = Eigen::internal::pmin(a, b);
    auto ma = Eigen::internal::pmax(a, b);
    using Eigen::internal::padd;
    using Eigen::internal::pcmp_lt;
    using Eigen::internal::pexp;
    using Eigen::internal::plog1p;
    using Eigen::internal::pset1;
    using Eigen::internal::psub;

    auto logsumexp = padd(plog1p(pexp(psub(mi, ma))), ma);
    return pselect(pcmp_lt(ma, pset1(Eigen::NumTraits<T>::lowest())), ma,
                   logsumexp);
  }
};

template <typename T>
struct LogSumExpReducer {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp(*accum, t);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p,
                                                          Packet* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp.packetOp(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return -Eigen::NumTraits<T>::infinity();
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return Eigen::internal::pset1(initialize());
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet
  finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T
  finalizeBoth(const T saccum, const Packet& vaccum) const {
    // TODO(itex): auto max_reducer = Eigen::internal::MaxReducer<T,
    // Eigen::PropagateNaN>();
    auto max_reducer = Eigen::internal::MaxReducer<T>();
    auto sum_reducer = Eigen::internal::SumReducer<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();
    auto log = Eigen::internal::scalar_log_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();

    using Eigen::internal::pexp;
    using Eigen::internal::psub;

    // `ma = max(x1, ..., xn)`
    // If the max of all of the `xi` is `-infinity` then the result is
    // -infinity. If the max is larger than `-infinity` then it's safe to use
    // for normalization even if the other elements are `-infinity`.
    //
    // `logsumexp(x1, ..., xn) = ma + log (exp(x1 - ma) + ... + exp(xn - ma))`
    auto ma = max_reducer.finalizeBoth(saccum, vaccum);
    auto logsumexp = add(log(sum_reducer.finalizeBoth(
                             exp(saccum - ma), pexp(psub(vaccum, pset1(ma))))),
                         ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? initialize() : logsumexp;
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SCAN_OPS_H_
