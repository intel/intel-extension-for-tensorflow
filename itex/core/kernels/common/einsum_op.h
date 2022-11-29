#ifndef ITEX_CORE_KERNELS_COMMON_LINALG_EINSUM_OP_H_
#define ITEX_CORE_KERNELS_COMMON_LINALG_EINSUM_OP_H_

#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename Device, typename T, int N>
struct StrideFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.stride(strides);
  }
};

template <typename Device, typename T, int N>
struct InflateFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.inflate(strides);
  }
};
#ifdef INTEL_CPU_ONLY
#define DECLARE_SPECS_NDIM(T, NDIM)                                         \
  template struct functor::StrideFunctor<Eigen::ThreadPoolDevice, T, NDIM>; \
  template struct functor::InflateFunctor<Eigen::ThreadPoolDevice, T, NDIM>;
#else
#define DECLARE_SPECS_NDIM(T, NDIM)                                  \
  template struct functor::StrideFunctor<Eigen::GpuDevice, T, NDIM>; \
  template struct functor::InflateFunctor<Eigen::GpuDevice, T, NDIM>;
#endif
#define DECLARE_SPECS(T)    \
  DECLARE_SPECS_NDIM(T, 1); \
  DECLARE_SPECS_NDIM(T, 2); \
  DECLARE_SPECS_NDIM(T, 3); \
  DECLARE_SPECS_NDIM(T, 4); \
  DECLARE_SPECS_NDIM(T, 5); \
  DECLARE_SPECS_NDIM(T, 6);

#ifdef INTEL_CPU_ONLY
TF_CALL_CPU_NUMBER_TYPES(DECLARE_SPECS);
#else
TF_CALL_GPU_NUMBER_TYPES(DECLARE_SPECS);
#endif
#undef DECLARE_SPECS
#undef DECLARE_SPECS_NDIM

}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_LINALG_EINSUM_OP_H_