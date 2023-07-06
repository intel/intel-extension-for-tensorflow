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

#ifndef ITEX_CORE_KERNELS_GPU_SPARSE_TENSOR_DENSE_MAT_MUL_OP_H_
#define ITEX_CORE_KERNELS_GPU_SPARSE_TENSOR_DENSE_MAT_MUL_OP_H_

#include <map>

#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct SumType {
  using type = T;
};

template <>
struct SumType<Eigen::half> {
  using type = float;  // Use fp32 accumulator for fp16 input values
};

template <>
struct SumType<Eigen::bfloat16> {
  using type = float;  // Use fp32 accumulator for bf16 input values
};

template <typename Tindices, typename T>
struct SpdmLaunchParams {
  int wg_size_;
  int out_cols_;
  int out_rows_;
  int b_cols_;
  int n_;
  int nnz_;
  int total_size_;
  Tindices* a_idx_ptr_;
  T* a_val_ptr_;
  const T* b_ptr_;
  typename SumType<T>::type* out_ptr_;

  SpdmLaunchParams(int wg_size, int out_cols, int out_rows, int b_cols, int n,
                   int nnz, int total_size, Tindices* a_idx_ptr, T* a_val_ptr,
                   const T* b_ptr, typename SumType<T>::type* out_ptr)
      : wg_size_(wg_size),
        out_cols_(out_cols),
        out_rows_(out_rows),
        b_cols_(b_cols),
        n_(n),
        nnz_(nnz),
        total_size_(total_size),
        a_idx_ptr_(a_idx_ptr),
        a_val_ptr_(a_val_ptr),
        b_ptr_(b_ptr),
        out_ptr_(out_ptr) {}
};

template <typename Tindices, typename T>
using SpdmLaunchFunction =
    std::function<void(const gpuStream_t&, SpdmLaunchParams<Tindices, T>&)>;

template <typename Tindices, typename T>
using SpdmTunedRegistry = std::map<int64, SpdmLaunchFunction<Tindices, T>>;

#define DECLARE_SPDM_TUNED_FUNCS(Tindices, T) \
  SpdmTunedRegistry<Tindices, T> SpdmTunedFuncs_##Tindices##_##T;

#define DECLARE_INDICE(T)            \
  DECLARE_SPDM_TUNED_FUNCS(int64, T) \
  DECLARE_SPDM_TUNED_FUNCS(int32, T)

using half = Eigen::half;
using bf16 = Eigen::bfloat16;
DECLARE_INDICE(float)
DECLARE_INDICE(half)
DECLARE_INDICE(bf16)

#undef DECLARE_SPDM_TUNED_FUNCS
#undef DECLARE_INDICE

// We use 64bit to represent tuned key, the detail information is shown below:
// 34bit       33bit       32bit       0-31bit
// platform    adj_a       adj_b       out_col
constexpr int64 get_spdm_tuned_key(bool is_hpc, bool adj_a, bool adj_b,
                                   int out_cols) {
  int64 key =
      ((static_cast<int64>(adj_a) << 34) | (static_cast<int64>(adj_a) << 33) |
       (static_cast<int64>(adj_b) << 32) | (static_cast<int64>(out_cols)));
  return key;
}

// check if platform, adj_a and adj_b is same except out col.
bool matched_spdm_tuned_key(const int64 key, const int64 other_key) {
  return (key >> 32) == (other_key >> 32);
}

template <typename Tindices, typename T>
SpdmTunedRegistry<Tindices, T>& get_spdm_tuned_funcs() {
  static_assert(std::is_same<T, bf16>::value || std::is_same<T, half>::value ||
                    std::is_same<T, float>::value,
                "only support float, half and bfloat16");
  return nullptr;
}

#define TUNED_FUNCS(Tindices, T)                                        \
  template <>                                                           \
  SpdmTunedRegistry<Tindices, T>& get_spdm_tuned_funcs<Tindices, T>() { \
    return SpdmTunedFuncs_##Tindices##_##T;                             \
  }

#define TUNED_FUNCS_TYPE(T) \
  TUNED_FUNCS(int64, T);    \
  TUNED_FUNCS(int32, T);

TUNED_FUNCS_TYPE(float);
TUNED_FUNCS_TYPE(bf16);
TUNED_FUNCS_TYPE(half);

template <typename Tindices, typename T, bool IsHPC, bool ADJ_A, bool ADJ_B,
          int OutCols>
struct SpdmTunedRegistrar {
  explicit SpdmTunedRegistrar(SpdmLaunchFunction<Tindices, T> f) {
    int64 key = get_spdm_tuned_key(IsHPC, ADJ_A, ADJ_B, OutCols);
    get_spdm_tuned_funcs<Tindices, T>().insert({key, f});
  }
};

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
struct SparseTensorDenseMatMulFunctor {
  static EIGEN_ALWAYS_INLINE Status Compute(
      OpKernelContext* ctx, typename TTypes<T>::Matrix out,
      typename TTypes<Tindices>::ConstMatrix a_indices,
      typename TTypes<T>::ConstVec a_values, typename TTypes<T>::ConstMatrix b);
};

template <typename MATRIX, bool ADJ>
class MaybeAdjoint;

template <typename MATRIX>
class MaybeAdjoint<MATRIX, false> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaybeAdjoint(MATRIX m) : m_(m) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename MATRIX::Scalar operator()(
      const typename MATRIX::Index i, const typename MATRIX::Index j) const {
    return m_(i, j);
  }

 private:
  const MATRIX m_;
};

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T MaybeConj(T v) {
  return v;
}

template <typename MATRIX>
class MaybeAdjoint<MATRIX, true> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaybeAdjoint(MATRIX m) : m_(m) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename MATRIX::Scalar operator()(
      const typename MATRIX::Index i, const typename MATRIX::Index j) const {
    return Eigen::numext::conj(m_(j, i));
  }

 private:
  const MATRIX m_;
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SPARSE_TENSOR_DENSE_MAT_MUL_OP_H_
