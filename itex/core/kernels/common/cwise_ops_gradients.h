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

#ifndef ITEX_CORE_KERNELS_COMMON_CWISE_OPS_GRADIENTS_H_
#define ITEX_CORE_KERNELS_COMMON_CWISE_OPS_GRADIENTS_H_

#include "itex/core/kernels/common/cwise_ops.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
namespace internal {
// Gradient for the tanh function
template <typename T>
struct scalar_tanh_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    return output_gradient * (T(1) - output * output);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    return pmul(output_gradient,
                psub(pset1<Packet>(T(1)), pmul(output, output)));
  }
};

// Gradient for the sqrt function
template <typename T>
struct scalar_sqrt_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sqrt_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    if (output_gradient == T(0)) {
      return T(0);
    } else {
      const T out_conj = numext::conj(output);
      return (static_cast<T>(0.5) * output_gradient) / out_conj;
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    const Packet const_half = pset1<Packet>(static_cast<T>(0.5));
    const Packet out_conj = pconj(output);
    return mul_no_nan_op<T>().packetOp(pdiv(const_half, out_conj),
                                       output_gradient);
  }
};

template <typename T>
struct functor_traits<scalar_sqrt_gradient_op<T>> {
  enum {
    Cost = functor_traits<scalar_conjugate_op<T>>::Cost +
           functor_traits<scalar_product_op<T>>::Cost +
           functor_traits<scalar_quotient_op<T>>::Cost,
    PacketAccess = functor_traits<scalar_conjugate_op<T>>::PacketAccess &&
                   functor_traits<scalar_product_op<T>>::PacketAccess &&
                   functor_traits<scalar_quotient_op<T>>::PacketAccess
  };
};

// Gradient for the rsqrt function
template <typename T>
struct scalar_rsqrt_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_rsqrt_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    if (output_gradient == T(0)) {
      return T(0);
    } else {
      const T out_conj = numext::conj(output);
      return static_cast<T>(-0.5) * (output_gradient * out_conj) *
             (out_conj * out_conj);
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    const Packet const_half = pset1<Packet>(static_cast<T>(-0.5));
    const Packet out_conj = pconj(output);
    auto safe_pmul = [](const Packet& a, const Packet& b) {
      return mul_no_nan_op<T>().packetOp(a, b);
    };
    return safe_pmul(pmul(const_half, pmul(out_conj, out_conj)),
                     safe_pmul(out_conj, output_gradient));
  }
};

// Gradient for the sigmoid function
template <typename T>
struct scalar_sigmoid_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sigmoid_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    return output_gradient * output * (T(1) - output);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    return pmul(output_gradient,
                pmul(output, psub(pset1<Packet>(T(1)), output)));
  }
};

// Gradient for the inverse function
template <typename T>
struct scalar_inverse_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_inverse_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    if (output_gradient == T(0)) {
      return T(0);
    } else {
      const T out_conj = numext::conj(output);
      return -out_conj * out_conj * output_gradient;
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    const Packet out_conj = pconj(output);
    return mul_no_nan_op<T>().packetOp(pnegate(pmul(out_conj, out_conj)),
                                       output_gradient);
  }
};
template <typename T>
struct functor_traits<scalar_inverse_gradient_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost + 2 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasMul,
  };
};
}  // namespace internal
}  // namespace Eigen

namespace itex {
namespace functor {
template <typename Device, typename Functor>
struct SimpleBinaryFunctor {
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1);
};

// Partial specialization of BinaryFunctor for GPU devices
typedef Eigen::GpuDevice GPUDevice;
template <typename Functor>
struct SimpleBinaryFunctor<GPUDevice, Functor> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1) {
    out.device(d) = in0.binaryExpr(in1, typename Functor::func());
  }
};

template <typename T>
struct tanh_grad : base<T, Eigen::internal::scalar_tanh_gradient_op<T>> {};

template <typename T>
struct rsqrt_grad : base<T, Eigen::internal::scalar_rsqrt_gradient_op<T>> {};

template <typename T>
struct sqrt_grad : base<T, Eigen::internal::scalar_sqrt_gradient_op<T>> {};

template <typename T>
struct sigmoid_grad : base<T, Eigen::internal::scalar_sigmoid_gradient_op<T>> {
};

template <typename T>
struct igamma_grad_a : base<T, Eigen::internal::scalar_igamma_der_a_op<T>> {};

template <typename T>
struct inverse_grad : base<T, Eigen::internal::scalar_inverse_gradient_op<T>> {
};
}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_CWISE_OPS_GRADIENTS_H_
