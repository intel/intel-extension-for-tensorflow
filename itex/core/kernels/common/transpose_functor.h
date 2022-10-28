/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_COMMON_TRANSPOSE_FUNCTOR_H_
#define ITEX_CORE_KERNELS_COMMON_TRANSPOSE_FUNCTOR_H_

#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_types.h"

namespace itex {

// Transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename Device>
Status DoTranspose(const Device& device, const Tensor& in,
                   const gtl::ArraySlice<int32> perm, Tensor* out);

// Conjugate and transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename Device>
Status DoConjugateTranspose(const Device& device, const Tensor& in,
                            const gtl::ArraySlice<int32> perm, Tensor* out);

// Convenience versions of DoTranspose that only swap the last (inner) two
// dimensions.
template <typename Device>
Status DoMatrixTranspose(const Device& device, const Tensor& in, Tensor* out);

// Convenience versions of DoConjugateTranspose that only swap the last (inner)
// two dimensions.
template <typename Device>
Status DoConjugateMatrixTranspose(const Device& device, const Tensor& in,
                                  Tensor* out);

// Primary device specific functor to be specialized for each device and type.
template <typename Device, typename T, bool conjugate = false>
struct Transpose {
  static void run(const Device& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out);
};

// Implementation details.
namespace internal {

typedef gtl::InlinedVector<int64, 8> TransposeDimsVec;
typedef gtl::InlinedVector<int32, 8> TransposePermsVec;

// If all non-singleton dimensions remain in ascending order, the shuffled
// singletons can be transposed by a reshape, saving a memory allocation & copy.
// |permutation| must be a permutation of {0, .., input_shape.dims() - 1}.
// That is, for all i, 0 <= perm[i] < input_shape.dims().
// In practice, this is checked in TransposeOp::Compute prior to calling this
// function, and the function sits here to facilitate unit testing.
inline bool NonSingletonDimensionsAlign(const TensorShape& input_shape,
                                        const std::vector<int32>& permutation) {
  int last_nonsingleton_perm_dim = -1;
  for (int perm_dim : permutation) {
    if (input_shape.dim_size(perm_dim) == 1) {
      continue;
    }
    if (perm_dim < last_nonsingleton_perm_dim) {
      return false;
    }
    last_nonsingleton_perm_dim = perm_dim;
  }
  return true;
}

// Uses Eigen to transpose.
template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const Tensor& in,
                         const gtl::ArraySlice<int32> perm, bool conjugate,
                         Tensor* out) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());
  if (conjugate) {
    y.device(d) = x.conjugate().shuffle(p);
  } else {
    y.device(d) = x.shuffle(p);
  }
}

template <typename Device>
Status DoTransposeImpl(const Device& d, const Tensor& in,
                       const gtl::ArraySlice<int32> perm, bool conjugate,
                       Tensor* out) {
  ITEX_CHECK_GE(in.dims(), 2);
  ITEX_CHECK_EQ(in.dims(), out->dims());
  ITEX_CHECK_EQ(in.dims(), perm.size());
  ITEX_CHECK_EQ(in.dtype(), out->dtype());
  switch (in.dtype()) {
    case DT_BOOL:
    case DT_INT8:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_UINT8:
      Transpose<Device, uint8>::run(d, in, perm, out);
      break;

    case DT_BFLOAT16:
    case DT_HALF:
    case DT_INT16:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_UINT16:
      Transpose<Device, uint16>::run(d, in, perm, out);
      break;

    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
      Transpose<Device, uint32>::run(d, in, perm, out);
      break;

    case DT_DOUBLE:
    case DT_INT64:
    case DT_UINT64:
      Transpose<Device, uint64>::run(d, in, perm, out);
      break;

    case DT_COMPLEX64:
      if (conjugate) {
        Transpose<Device, complex64, /*conjugate=*/true>::run(d, in, perm, out);
      } else {
        Transpose<Device, uint64>::run(d, in, perm, out);
      }
      break;

    case DT_COMPLEX128:
      if (conjugate) {
        Transpose<Device, complex128, /*conjugate=*/true>::run(d, in, perm,
                                                               out);
      } else {
        Transpose<Device, complex128, /*conjugate=*/false>::run(d, in, perm,
                                                                out);
      }
      break;

    default:
      return errors::Unimplemented("Unsupported dtype : ", in.dtype());
  }
  return Status::OK();
}

template <typename Device>
inline Status DoMatrixTransposeImpl(const Device& device, const Tensor& in,
                                    bool conjugate, Tensor* out) {
  const int ndims = in.dims();
  if (ndims == 0) return Status::OK();
  TransposePermsVec perm(ndims);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[ndims - 2], perm[ndims - 1]);
  return DoTransposeImpl(device, in, perm, conjugate, out);
}

template <typename Tperm>
inline Status PermutationHelper(const Tensor& perm, const int dims,
                                std::vector<int32>* permutation) {
  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // using volatile instead of SubtleMustCopy here so that the
  // asynchrony boundary is permutation.
  const volatile Tperm* perm_begin =
      reinterpret_cast<const volatile Tperm*>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return Status::OK();
}

inline dnnl::memory::dims ReorderStrides(const dnnl::memory::dims& strides,
                                         const gtl::ArraySlice<int32>& perm) {
  dnnl::memory::dims reordered_strides(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    reordered_strides[perm[i]] = strides[i];
  }
  return reordered_strides;
}

}  // namespace internal
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_TRANSPOSE_FUNCTOR_H_
