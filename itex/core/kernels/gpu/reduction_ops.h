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

#ifndef ITEX_CORE_KERNELS_GPU_REDUCTION_OPS_H_
#define ITEX_CORE_KERNELS_GPU_REDUCTION_OPS_H_

#include "itex/core/kernels/gpu/col_reduction_kernels.h"
#include "itex/core/kernels/gpu/full_reduction_kernels.h"
#include "itex/core/kernels/gpu/row_reduction_kernels.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct SqrtOfReal {
  T operator()(const T& a) const {
    return T(Eigen::numext::sqrt(Eigen::numext::real(a)));
  }
};

template <typename T>
struct Square {
  T operator()(const T& a) const { return a * Eigen::numext::conj(a); }
};

template <typename T, typename OUT_T = T>
struct DividesBy {
  T divisor;

  explicit DividesBy(T divisor) : divisor(divisor) {}

  OUT_T operator()(const T& x) const { return x / divisor; }
};

template <>
struct DividesBy<float, Eigen::half> {
  float divisor;

  explicit DividesBy(float divisor) : divisor(divisor) {}

  Eigen::half operator()(const float& x) const {
    return Eigen::half(x / divisor);
  }
};

template <>
struct DividesBy<float, Eigen::bfloat16> {
  float divisor;

  explicit DividesBy(float divisor) : divisor(divisor) {}

  Eigen::bfloat16 operator()(const float& x) const {
    return Eigen::bfloat16(x / divisor);
  }
};

template <typename Reducer>
struct ReducerTraits {
  enum { IsScalarIdentity = true };
};

// Dummy class used for template specialization for mean reduction, which is
// accomplished by SumReducer and on-the-fly division by the reduction factor.
template <typename Scalar>
struct MeanReducer {
  Scalar initialize() const { return Scalar(0); }
};

// Dummy class used for template specialization for l2-norm reduction.
template <typename Scalar>
struct EuclideanNormReducer {
  Scalar initialize() const { return Scalar(0); }
};

template <typename Scalar>
struct ReducerTraits<EuclideanNormReducer<Scalar>> {
  enum { IsScalarIdentity = false };
};

template <typename OUT_T, typename IN_T, typename ReductionAxes,
          typename Reducer>
struct ReduceEigenImpl {
  void operator()(const GPUDevice& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes, const Reducer& reducer) {
    out.device(d) = in.reduce(reduction_axes, reducer);
  }
};

// For most reducers, the identity is Reducer::initialize()
template <typename Reducer>
struct Identity {
  static auto identity(const Reducer& reducer)
      -> decltype(reducer.initialize()) {
    return reducer.initialize();
  }
};

// MeanReducer is a special case, since it doesn't technically have an identity.
// Thus, ideally we'd return nan.  However, mean is instantiated for integer
// types as well, so we do the nan override only for floating point types.
#define FIX_MEAN_IDENTITY(T)                            \
  template <>                                           \
  struct Identity<functor::MeanReducer<T>> {            \
    static T identity(const functor::MeanReducer<T>&) { \
      return Eigen::NumTraits<T>::quiet_NaN();          \
    }                                                   \
  };
FIX_MEAN_IDENTITY(Eigen::bfloat16)
FIX_MEAN_IDENTITY(Eigen::half)
FIX_MEAN_IDENTITY(float)
#undef FIX_MEAN_IDENTITY

template <typename OUT_T, typename Reducer>
void FillIdentityEigenImpl(const GPUDevice& d, OUT_T out,
                           const Reducer& reducer) {
  out.device(d) = out.constant(Identity<Reducer>::identity(reducer));
}

// TODO(itex): Remove Eigen impl once compiler has support on prod, or && and.
// JIRA: https://jira.devtools.intel.com/browse/CMPLRLLVM-37440?src=confmacro
template <typename Reducer>
struct ReduceFunctor {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
    ReduceEigenImpl<OUT_T, IN_T, ReductionAxes, Reducer> reducer_impl;
    reducer_impl(ctx->eigen_gpu_device(), out, in, reduction_axes, reducer);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Reducer& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename IN_T, typename OUT_T, typename INIT_VALUE_T, typename Op,
          typename ReductionAxes,
          typename IN_FUNC = reduciton_helper::Identity<IN_T>,
          typename OUT_FUNC = reduciton_helper::Identity<INIT_VALUE_T>>
void ReduceGPUImpl(
    OpKernelContext* ctx, IN_T* in_data, OUT_T* out_data, int in_rank,
    int in_dim0, int in_dim1, int in_dim2, int out_rank, INIT_VALUE_T init,
    Op op, const ReductionAxes& reduction_axes,
    IN_FUNC in_func = reduciton_helper::Identity<IN_T>(),
    OUT_FUNC out_func = reduciton_helper::Identity<INIT_VALUE_T>()) {
  if (out_rank == 0) {
    const int in_size = in_dim0 * in_dim1 * in_dim2;
    LaunchFullReduction(ctx, in_data, out_data, init, in_size, op, in_func,
                        out_func);
  } else if (in_rank == 2 && out_rank == 1 &&
             reduction_axes[0] == 1) {  // row reduction
    LaunchRowReduction(ctx, in_data, out_data, init, in_dim0, in_dim1, op,
                       in_func, out_func);
  } else if (in_rank == 2 && out_rank == 1 &&
             reduction_axes[0] == 0) {  // column reduction
    if (in_dim1 == 1) {
      const int in_size = in_dim0;
      LaunchFullReduction(ctx, in_data, out_data, init, in_size, op, in_func,
                          out_func);
    } else {
      LaunchColReduction(ctx, in_data, out_data, init, 1, in_dim0, in_dim1, op,
                         in_func, out_func);
    }
  } else if (in_rank == 3 && out_rank == 2 &&
             reduction_axes[0] == 1) {  // column reduction
    LaunchColReduction(ctx, in_data, out_data, init, in_dim0, in_dim1, in_dim2,
                       op, in_func, out_func);
  } else {
    std::stringstream ss;
    ss << "Invalid reduction requested: in_rank, out_rank, axes " << in_rank
       << " " << out_rank;
    if (out_rank == 1) ss << " " << reduction_axes[0];
    if (out_rank == 2) ss << " " << reduction_axes[1];
    ITEX_LOG(FATAL) << ss.str();
  }
}

template <typename T>
struct ReduceFunctor<Eigen::internal::SumReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::SumReducer<T>& reducer) {
    //  For reduction op here, we use input datatype as internal computation
    //  type
    typedef sycl::plus<T> BinaryOp;
    T init = T(0);
    ReduceGPUImpl<const T, T, T, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::SumReducer<T>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename T>
struct ReduceFunctor<functor::MeanReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::MeanReducer<T>& reducer) {
    typedef sycl::plus<T> BinaryOp;

    int divisor = 1;
    if (out.rank() == 0)
      divisor = in.size();
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 0)
      divisor = in.dimension(0);
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    else if (out.rank() == 1 && in.rank() == 3 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2)
      divisor = in.dimension(0) * in.dimension(2);
    else if (out.rank() == 2 && in.rank() == 3 && reduction_axes[0] == 1)
      divisor = in.dimension(1);

    DividesBy<T> div_op(static_cast<T>(divisor));
    T init = T(0);
    ReduceGPUImpl<const T, T, T, BinaryOp, ReductionAxes,
                  reduciton_helper::Identity<T>, DividesBy<T>>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes, reduciton_helper::Identity<T>(), div_op);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const functor::MeanReducer<T>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename T>
struct ReduceFunctor<Eigen::internal::MinReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MinReducer<T>& reducer) {
    typedef sycl::minimum<T> BinaryOp;
    T init = Eigen::NumTraits<T>::highest();
    ReduceGPUImpl<const T, T, T, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MinReducer<T>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename T>
struct ReduceFunctor<Eigen::internal::MaxReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MaxReducer<T>& reducer) {
    typedef sycl::maximum<T> BinaryOp;
    T init = Eigen::NumTraits<T>::lowest();
    ReduceGPUImpl<const T, T, T, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MaxReducer<T>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename T>
struct ReduceFunctor<functor::EuclideanNormReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::EuclideanNormReducer<T>& reducer) {
    typedef sycl::plus<T> BinaryOp;
    T init = T(0);
    ReduceGPUImpl<const T, T, T, BinaryOp, ReductionAxes, Square<T>,
                  SqrtOfReal<T>>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes, Square<T>(), SqrtOfReal<T>());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const functor::EuclideanNormReducer<T>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename T>
struct ReduceFunctor<Eigen::internal::ProdReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::ProdReducer<T>& reducer) {
    typedef sycl::multiplies<T> BinaryOp;
    T init = T(1);
    ReduceGPUImpl<const T, T, T, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::ProdReducer<T>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<Eigen::internal::AndReducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::AndReducer& reducer) {
    //  as bool is not supported by compiler reduce api, use uint8_t
    typedef uint8_t InitValueT;
    typedef sycl::bit_and<InitValueT> BinaryOp;
    InitValueT init = InitValueT(1);
    ReduceGPUImpl<const InitValueT, InitValueT, InitValueT, BinaryOp,
                  ReductionAxes>(
        ctx, reinterpret_cast<const InitValueT*>(in.data()),
        reinterpret_cast<InitValueT*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::AndReducer& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<Eigen::internal::OrReducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::OrReducer& reducer) {
    //  as bool is not supported by compiler reduce api, use uint8_t
    typedef uint8_t InitValueT;
    typedef sycl::bit_or<InitValueT> BinaryOp;
    InitValueT init = InitValueT(0);
    ReduceGPUImpl<const InitValueT, InitValueT, InitValueT, BinaryOp,
                  ReductionAxes>(
        ctx, reinterpret_cast<const InitValueT*>(in.data()),
        reinterpret_cast<InitValueT*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::OrReducer& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

//  For Eigen::half, we use float as internal computation type
template <>
struct ReduceFunctor<Eigen::internal::SumReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::SumReducer<Eigen::half>& reducer) {
    typedef sycl::half BASE_T;
    typedef float InitValueT;
    typedef sycl::plus<InitValueT> BinaryOp;

    InitValueT init = InitValueT(0);
    ReduceGPUImpl<const BASE_T, BASE_T, InitValueT, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const BASE_T*>(in.data()),
        reinterpret_cast<BASE_T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::SumReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<functor::MeanReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::MeanReducer<Eigen::half>& reducer) {
    typedef sycl::half BASE_T;
    typedef float InitValueT;
    typedef sycl::plus<InitValueT> BinaryOp;

    InitValueT divisor = 1.f;
    if (out.rank() == 0)
      divisor = in.size();
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 0)
      divisor = in.dimension(0);
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    else if (out.rank() == 1 && in.rank() == 3 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2)
      divisor = in.dimension(0) * in.dimension(2);
    else if (out.rank() == 2 && in.rank() == 3 && reduction_axes[0] == 1)
      divisor = in.dimension(1);

    DividesBy<InitValueT, BASE_T> div_op(divisor);

    InitValueT init = InitValueT(0);
    ReduceGPUImpl<const BASE_T, BASE_T, InitValueT, BinaryOp, ReductionAxes,
                  reduciton_helper::Identity<BASE_T>,
                  DividesBy<InitValueT, BASE_T>>(
        ctx, reinterpret_cast<const BASE_T*>(in.data()),
        reinterpret_cast<BASE_T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes, reduciton_helper::Identity<BASE_T>(), div_op);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const functor::MeanReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<Eigen::internal::MinReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MinReducer<Eigen::half>& reducer) {
    typedef sycl::half BASE_T;
    typedef float InitValueT;
    typedef sycl::minimum<float> BinaryOp;

    InitValueT init = InitValueT(Eigen::NumTraits<Eigen::half>::highest());
    ReduceGPUImpl<const BASE_T, BASE_T, InitValueT, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const BASE_T*>(in.data()),
        reinterpret_cast<BASE_T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::MinReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<Eigen::internal::MaxReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MaxReducer<Eigen::half>& reducer) {
    typedef sycl::half BASE_T;
    typedef float InitValueT;
    typedef sycl::maximum<InitValueT> BinaryOp;

    InitValueT init = InitValueT(Eigen::NumTraits<Eigen::half>::lowest());
    ReduceGPUImpl<const BASE_T, BASE_T, InitValueT, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const BASE_T*>(in.data()),
        reinterpret_cast<BASE_T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::MaxReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<functor::EuclideanNormReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(
      OpKernelContext* ctx, OUT_T out, IN_T in,
      const ReductionAxes& reduction_axes,
      const functor::EuclideanNormReducer<Eigen::half>& reducer) {
    typedef sycl::half BASE_T;
    typedef float InitValueT;
    typedef sycl::plus<InitValueT> BinaryOp;

    InitValueT init = InitValueT(0);
    ReduceGPUImpl<const BASE_T, BASE_T, InitValueT, BinaryOp, ReductionAxes,
                  Square<BASE_T>, SqrtOfReal<InitValueT>>(
        ctx, reinterpret_cast<const BASE_T*>(in.data()),
        reinterpret_cast<BASE_T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes, Square<BASE_T>(), SqrtOfReal<InitValueT>());
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const functor::EuclideanNormReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<Eigen::internal::ProdReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::ProdReducer<Eigen::half>& reducer) {
    typedef sycl::half BASE_T;
    typedef float InitValueT;
    typedef sycl::multiplies<InitValueT> BinaryOp;

    InitValueT init = InitValueT(1);
    ReduceGPUImpl<const BASE_T, BASE_T, InitValueT, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const BASE_T*>(in.data()),
        reinterpret_cast<BASE_T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::ProdReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

//  For Eigen::bfloat16, we use float as internal computation type
template <>
struct ReduceFunctor<Eigen::internal::SumReducer<Eigen::bfloat16>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(
      OpKernelContext* ctx, OUT_T out, IN_T in,
      const ReductionAxes& reduction_axes,
      const Eigen::internal::SumReducer<Eigen::bfloat16>& reducer) {
    typedef Eigen::bfloat16 T;
    typedef float InitValueT;
    typedef sycl::plus<InitValueT> BinaryOp;

    InitValueT init = InitValueT(0);
    ReduceGPUImpl<const T, T, InitValueT, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::SumReducer<Eigen::bfloat16>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<functor::MeanReducer<Eigen::bfloat16>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::MeanReducer<Eigen::bfloat16>& reducer) {
    typedef Eigen::bfloat16 T;
    typedef float InitValueT;
    typedef sycl::plus<InitValueT> BinaryOp;

    InitValueT divisor = 1.f;
    if (out.rank() == 0)
      divisor = in.size();
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 0)
      divisor = in.dimension(0);
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    else if (out.rank() == 1 && in.rank() == 3 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2)
      divisor = in.dimension(0) * in.dimension(2);
    else if (out.rank() == 2 && in.rank() == 3 && reduction_axes[0] == 1)
      divisor = in.dimension(1);

    DividesBy<InitValueT, T> div_op(divisor);

    InitValueT init = InitValueT(0);
    ReduceGPUImpl<const T, T, InitValueT, BinaryOp, ReductionAxes,
                  reduciton_helper::Identity<T>, DividesBy<InitValueT, T>>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes, reduciton_helper::Identity<T>(), div_op);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const functor::MeanReducer<Eigen::bfloat16>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<Eigen::internal::MinReducer<Eigen::bfloat16>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(
      OpKernelContext* ctx, OUT_T out, IN_T in,
      const ReductionAxes& reduction_axes,
      const Eigen::internal::MinReducer<Eigen::bfloat16>& reducer) {
    typedef Eigen::bfloat16 T;
    typedef float InitValueT;
    typedef sycl::minimum<InitValueT> BinaryOp;

    T init = Eigen::NumTraits<T>::highest();
    ReduceGPUImpl<const T, T, InitValueT, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::MinReducer<Eigen::bfloat16>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<Eigen::internal::MaxReducer<Eigen::bfloat16>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(
      OpKernelContext* ctx, OUT_T out, IN_T in,
      const ReductionAxes& reduction_axes,
      const Eigen::internal::MaxReducer<Eigen::bfloat16>& reducer) {
    typedef Eigen::bfloat16 T;
    typedef float InitValueT;
    typedef sycl::maximum<InitValueT> BinaryOp;

    T init = Eigen::NumTraits<T>::lowest();
    ReduceGPUImpl<const T, T, InitValueT, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::MaxReducer<Eigen::bfloat16>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<Eigen::internal::ProdReducer<Eigen::bfloat16>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(
      OpKernelContext* ctx, OUT_T out, IN_T in,
      const ReductionAxes& reduction_axes,
      const Eigen::internal::ProdReducer<Eigen::bfloat16>& reducer) {
    typedef Eigen::bfloat16 T;
    typedef float InitValueT;
    typedef sycl::multiplies<InitValueT> BinaryOp;

    T init = T(1);
    ReduceGPUImpl<const T, T, InitValueT, BinaryOp, ReductionAxes>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::ProdReducer<Eigen::bfloat16>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <>
struct ReduceFunctor<functor::EuclideanNormReducer<Eigen::bfloat16>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(
      OpKernelContext* ctx, OUT_T out, IN_T in,
      const ReductionAxes& reduction_axes,
      const functor::EuclideanNormReducer<Eigen::bfloat16>& reducer) {
    typedef Eigen::bfloat16 T;
    typedef float InitValueT;
    typedef sycl::plus<InitValueT> BinaryOp;

    InitValueT init = InitValueT(0);
    ReduceGPUImpl<const T, T, InitValueT, BinaryOp, ReductionAxes, Square<T>,
                  SqrtOfReal<InitValueT>>(
        ctx, reinterpret_cast<const T*>(in.data()),
        reinterpret_cast<T*>(out.data()), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), init, BinaryOp(),
        reduction_axes, Square<T>(), SqrtOfReal<InitValueT>());
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const functor::EuclideanNormReducer<Eigen::bfloat16>& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_REDUCTION_OPS_H_
