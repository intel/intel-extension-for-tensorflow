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

#ifndef ITEX_CORE_KERNELS_COMMON_CAST_OP_H_
#define ITEX_CORE_KERNELS_COMMON_CAST_OP_H_

#include <limits>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define SPECIALIZE_CAST(DEVICE, OUT_TYPE, IN_OUT)                   \
  template <typename Device>                                        \
  struct CastFunctor<Device, OUT_TYPE, IN_OUT> {                    \
    void operator()(const Device& d,                                \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,     \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor,   \
                    bool truncate = false) {                        \
      if (truncate) {                                               \
        out_tensor.device(d) =                                      \
            in_tensor.unaryExpr(LSBZeroSetter<IN_OUT, OUT_TYPE>())  \
                .template cast<OUT_TYPE>();                         \
      } else {                                                      \
        out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>(); \
      }                                                             \
    }                                                               \
  };                                                                \
  template struct CastFunctor<DEVICE, OUT_TYPE, IN_OUT>;

#ifdef ITEX_ENABLE_DOUBLE
#define CAST_FUNCTORS(devname)                                        \
  SPECIALIZE_CAST(devname, Eigen::half, float)                        \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<float>)          \
  SPECIALIZE_CAST(devname, Eigen::bfloat16, float)                    \
  SPECIALIZE_CAST(devname, Eigen::bfloat16, std::complex<float>)      \
  SPECIALIZE_CAST(devname, float, double)                             \
  SPECIALIZE_CAST(devname, float, std::complex<double>)               \
  SPECIALIZE_CAST(devname, std::complex<float>, double)               \
  SPECIALIZE_CAST(devname, std::complex<float>, std::complex<double>) \
  SPECIALIZE_CAST(devname, Eigen::half, double)                       \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<double>)         \
  SPECIALIZE_CAST(devname, Eigen::bfloat16, double)                   \
  SPECIALIZE_CAST(devname, Eigen::bfloat16, std::complex<double>)     \
  template <typename OUT_TYPE, typename IN_OUT>                       \
  struct CastFunctor<devname, OUT_TYPE, IN_OUT> {                     \
    void operator()(const devname& d,                                 \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,       \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor,     \
                    bool truncate = false) {                          \
      out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>();     \
    }                                                                 \
  };

#else
#define CAST_FUNCTORS(devname)                                    \
  SPECIALIZE_CAST(devname, Eigen::half, float)                    \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<float>)      \
  SPECIALIZE_CAST(devname, Eigen::bfloat16, float)                \
  SPECIALIZE_CAST(devname, Eigen::bfloat16, std::complex<float>)  \
  template <typename OUT_TYPE, typename IN_OUT>                   \
  struct CastFunctor<devname, OUT_TYPE, IN_OUT> {                 \
    void operator()(const devname& d,                             \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,   \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor, \
                    bool truncate = false) {                      \
      out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>(); \
    }                                                             \
  };
#endif  // ITEX_ENABLE_DOUBLE

namespace itex {

typedef std::function<void(OpKernelContext&, const Tensor&, Tensor*,
                           bool trunc)>
    CastFunctorType;

namespace functor {

template <typename I>
constexpr int MantissaWidth() {
  return std::numeric_limits<I>::digits;
}

template <>
constexpr int MantissaWidth<Eigen::half>() {
  // Remember, there's 1 hidden bit
  return 10 + 1;
}

template <>
constexpr int MantissaWidth<Eigen::bfloat16>() {
  // Remember, there's 1 hidden bit
  return 7 + 1;
}

template <typename Device, typename Tout, typename Tin>
void Cast(const Device& d, typename TTypes<Tout>::Flat o,
          typename TTypes<Tin>::ConstFlat i) {
  o.device(d) = i.template cast<Tout>();
}

template <typename Device, typename Tout, typename Tin>
struct CastFunctor {
  void operator()(const Device& d, typename TTypes<Tout>::Flat o,
                  typename TTypes<Tin>::ConstFlat i, bool truncate = false);
};

// Only enable LSBZeroSetterHelper for 64 and 32 bit input data types.
// Specialize for others if needed in future.
template <typename I>
typename std::enable_if<sizeof(I) == 8, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(
        I& t, int n) {  // NOLINT(runtime/references)
  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!std::isnan(t)) {
    uint64_t* p = reinterpret_cast<uint64_t*>(&t);
    *p &= (0xFFFFFFFFFFFFFFFF << n);
  }
}

template <typename I>
typename std::enable_if<sizeof(I) == 4, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(
        I& t, int n) {  // NOLINT(runtime/references)
  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!std::isnan(t)) {
    uint32_t* p = reinterpret_cast<uint32_t*>(&t);
    *p &= (0xFFFFFFFF << n);
  }
}

// Set n least significant bits to 0
template <typename I, typename O>
struct LSBZeroSetter {
  EIGEN_EMPTY_STRUCT_CTOR(LSBZeroSetter)

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const I operator()(const I& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I t = a;
    LSBZeroSetterHelper(t, bits);
    return t;
  }
};

template <typename I, typename O>
struct LSBZeroSetter<std::complex<I>, std::complex<O>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const std::complex<I> operator()(
      const std::complex<I>& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I re = std::real(a);
    I img = std::imag(a);
    LSBZeroSetterHelper(re, bits);
    LSBZeroSetterHelper(img, bits);
    std::complex<I> toReturn(re, img);
    return toReturn;
  }
};

template <typename I, typename O>
struct LSBZeroSetter<std::complex<I>, O> {
  // Sets the 16 LSBits of the float to 0
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const std::complex<I> operator()(
      const std::complex<I>& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I re = std::real(a);
    I img = std::imag(a);
    LSBZeroSetterHelper(re, bits);
    LSBZeroSetterHelper(img, bits);
    std::complex<I> toReturn(re, img);
    return toReturn;
  }
};

}  // namespace functor

template <typename Device, typename SrcType, typename DstType>
struct CastDataType {
  void operator()(const Device& d, typename TTypes<SrcType>::ConstFlat input,
                  typename TTypes<DstType>::Flat output) {
    output.device(d) = input.template cast<DstType>();
  }
};
#ifndef INTEL_CPU_ONLY
typedef Eigen::GpuDevice GPUDevice;
template <typename SrcType, typename DstType>
struct CastDataType<GPUDevice, SrcType, DstType> {
  void operator()(const GPUDevice& d, typename TTypes<SrcType>::ConstFlat input,
                  typename TTypes<DstType>::Flat output) {
    // Use existing cast functor instead of directly casting Eigen tensor, as
    // otherwise we need to instantiate the cast function in a .cu.cc file
    functor::CastFunctor<GPUDevice, DstType, SrcType> cast;
    cast(d, output, input);
  }
};
#endif
}  // namespace itex

namespace Eigen {  // copy from official tensorflow
namespace internal {

// Eigen can't convert to/from complex numbers, because it is limited to cases
// that can be static_casted. But numpy is able to cast to/from complex, which
// we want to replicate. So we add specializations for complex here.()
template <typename From, typename To>
struct scalar_cast_op<std::complex<From>, To> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE To
  operator()(const std::complex<From>& a) const {
    // Replicate numpy behavior of returning just the real part
    return static_cast<To>(a.real());
  }
};

template <typename From, typename To>
struct scalar_cast_op<From, std::complex<To>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<To> operator()(
      const From& a) const {
    // Replicate numpy behavior of setting the imaginary part to 0
    return std::complex<To>(static_cast<To>(a), To(0));
  }
};

template <typename From, typename To>
struct scalar_cast_op<std::complex<From>, std::complex<To>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<To> operator()(
      const std::complex<From>& a) const {
    return std::complex<To>(static_cast<To>(a.real()),
                            static_cast<To>(a.imag()));
  }
};

template <typename From, typename To>
struct functor_traits_complex_impl {
  enum { Cost = NumTraits<To>::AddCost, PacketAccess = false };
};

template <typename From, typename To>
struct functor_traits<scalar_cast_op<std::complex<From>, To>>
    : functor_traits_complex_impl<std::complex<From>, To> {};
template <typename From, typename To>
struct functor_traits<scalar_cast_op<From, std::complex<To>>>
    : functor_traits_complex_impl<From, std::complex<To>> {};
// Needed to avoid ambiguous partial specialization
template <typename From, typename To>
struct functor_traits<scalar_cast_op<std::complex<From>, std::complex<To>>>
    : functor_traits_complex_impl<std::complex<From>, std::complex<To>> {};

}  // namespace internal
}  // namespace Eigen

#endif  // ITEX_CORE_KERNELS_COMMON_CAST_OP_H_
