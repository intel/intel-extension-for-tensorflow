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

#ifndef ITEX_CORE_UTILS_GPU_HELPER_H_
#define ITEX_CORE_UTILS_GPU_HELPER_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <algorithm>
#include <cstdint>
#include <utility>

#include "itex/core/utils/hw_info.h"
#include "itex/core/utils/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace reduciton_helper {
template <typename T>
using LocalAcc = sycl::accessor<T, 1, sycl::access::mode::read_write,
                                sycl::access::target::local>;

template <typename T>
struct Identity {
  inline T operator()(T x) const { return x; }
};
}  // namespace reduciton_helper

// return the ceil of log2, requiring x>0
inline unsigned int ceil_log2(unsigned int x) {
  int t = 32u - __builtin_clz(x);
  return x == (1 << (t - 1)) ? t - 1 : t;
}

template <typename T>
struct DefaultComputeType {
  using type = T;
};

template <>
struct DefaultComputeType<Eigen::half> {
  using type = float;
};

template <>
struct DefaultComputeType<Eigen::bfloat16> {
  using type = float;
};

template <typename T, int N>
union Pack {
  Pack() {
    // do nothing
  }
  Pack<T, N>& operator=(const Pack<T, N> other) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      elem[i] = other.elem[i];
    }
    return *this;
  }
  T elem[N];
};

template <typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int32 row_size) : src(src), row_size(row_size) {}
  template <int N>
  void Load(DST* dst, int32 row, int32 col) const {
    Pack<SRC, N> pack;
    const int32 offset = (row * row_size + col) / N;
    pack = *(reinterpret_cast<const Pack<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]);
    }
  }
  const SRC* src;
  int32 row_size;
};

struct SoftmaxInputShape {
  int32 batch_size;
  int32 num_heads;
  int32 seqlen_from;
  int32 seqlen_to;
};

template <typename SRC, typename DST>
struct AddMaskLoad {
  AddMaskLoad(const SRC* src, const SRC* mask, SoftmaxInputShape input_dims,
              int32 row_size)
      : src(src), mask(mask), row_size(row_size), input_dims(input_dims) {}
  template <int N>
  void Load(DST* dst, const int32& row, const int32& col) const {
    Pack<SRC, N> pack;
    Pack<SRC, N> mask_pack;
    const int32 offset = row * row_size + col;
    pack = *(reinterpret_cast<const Pack<SRC, N>*>(src) + offset / N);
    const int32 seqlen1_i = row % input_dims.seqlen_from;
    const int32 b_i = row / input_dims.seqlen_from / input_dims.num_heads;
    const int32 mask_offset =
        (b_i * input_dims.seqlen_from + seqlen1_i) * input_dims.seqlen_to + col;
    mask_pack =
        *(reinterpret_cast<const Pack<SRC, N>*>(mask) + mask_offset / N);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i] + mask_pack.elem[i]);
    }
  }
  const SRC* src;
  const SRC* mask;
  int32 row_size;
  SoftmaxInputShape input_dims;
};

template <typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int32 row_size) : dst(dst), row_size(row_size) {}
  template <int N>
  void Store(const SRC* src, const int32& row, const int32& col) const {
    Pack<DST, N> pack;
    const int32 offset = row * row_size + col;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      pack.elem[i] = static_cast<DST>(src[i]);
    }
    *(reinterpret_cast<Pack<DST, N>*>(dst) + offset / N) = pack;
  }
  DST* dst;
  int32 row_size;
};

inline void GetNumWorkGroups(sycl::device xpu_device, int32 workgroup_size,
                             int max_workgroups, int waves,
                             int* num_workgroups) {
  const int hw_concurrent_work_group = xpu_device.template get_info<
      sycl::ext::intel::info::device::gpu_subslices_per_slice>();

  int subslices_count = IsXeHPC(&xpu_device) ? hw_concurrent_work_group
                                             : hw_concurrent_work_group * 2;

  const int32_t hw_max_workgroup_size =
      xpu_device.template get_info<sycl::info::device::max_work_group_size>();

  *num_workgroups = std::max<int>(
      1, std::min<int32_t>(
             max_workgroups,
             subslices_count * hw_max_workgroup_size / workgroup_size * waves));
}

template <typename T>
inline T DivUp(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
inline T RoundUp(T val, T rounding) {
  return DivUp(val, rounding) * rounding;
}

// Allows for the treatment of an integral constant
// as a type at compile-time
template <int VAL>
struct Int2Type {
  enum { VALUE = VAL };
};

template <typename T>
inline void Swap(T& i, T& j) {
  T temp(i);
  i = j;
  j = temp;
}

constexpr int NumBits(const unsigned int n) {
  int count = 0;
  for (unsigned int m = n; m != 0; m >>= 1, ++count) {
  }
  return count;
}

#define UNROLL_ON_DEVICE _Pragma("unroll")

// Represents an aligned array of N elements of T. Data pointers can be
// reinterpreted as this type to generate vectorized loads/stores in a kernel.
template <typename T, uint32_t N, typename Func = sycl::plus<T>>
class alignas(alignof(T) * N) AlignedVector {
 public:
  typedef T value_type;
  static constexpr const uint32_t kSize = N;

  AlignedVector() = default;

  // Uniform initialization.
  explicit AlignedVector(value_type uniform) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < kSize; ++i) {
      values_[i] = uniform;
    }
  }
  // Uniform initialization with explicit conversion.
  // Note: This is required for T=Eigen::half because it only supports explicit
  // conversions from other types and its template constructor is too relaxed
  // to be able to use std::is_constructible.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  explicit AlignedVector(U uniform_u) {
    value_type uniform(uniform_u);
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < kSize; ++i) {
      values_[i] = uniform;
    }
  }
  // Implicit conversion.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value, int>::type = 0>
  AlignedVector(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < kSize; ++i) {
      values_[i] = other[i];
    }
  }
  // Explicit conversion.
  template <typename U,
            typename std::enable_if<!std::is_convertible<U, T>::value &&
                                        std::is_constructible<T, U>::value,
                                    int>::type = 0>
  explicit AlignedVector(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < kSize; ++i) {
      values_[i] = T(other[i]);
    }
  }

  template <typename U>
  void Load(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N; ++i) {
      values_[i] = static_cast<T>(other[i]);
    }
  }

  template <typename U>
  void Accumulate(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N; ++i) {
      values_[i] = Func()(values_[i], static_cast<T>(other[i]));
    }
  }

  template <typename U>
  void Store(AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N; ++i) {
      other[i] = static_cast<U>(values_[i]);
    }
  }

  template <typename U>
  void PartialStore(AlignedVector<U, N>& other, uint32_t num,
                    uint32_t offset = 0) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N && i < num; ++i) {
      other[i] = static_cast<U>(values_[i + offset]);
    }
  }

  value_type& operator[](uint32_t i) { return values_[i]; }
  const value_type& operator[](uint32_t i) const { return values_[i]; }

#define DEFINE_BINARY_UPDATE_OPERATOR(op)                   \
  AlignedVector& operator op(const AlignedVector& rhs) {    \
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < kSize; ++i) { \
      values_[i] op rhs[i];                                 \
    }                                                       \
    return *this;                                           \
  }
  DEFINE_BINARY_UPDATE_OPERATOR(+=)
  DEFINE_BINARY_UPDATE_OPERATOR(-=)
  DEFINE_BINARY_UPDATE_OPERATOR(*=)
  DEFINE_BINARY_UPDATE_OPERATOR(/=)
#undef DEFINE_BINARY_UPDATE_OPERATOR

#define DEFINE_BINARY_OPERATOR(op)                             \
  friend AlignedVector operator op(const AlignedVector& lhs,   \
                                   const AlignedVector& rhs) { \
    AlignedVector ret;                                         \
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < kSize; ++i) {    \
      ret[i] = lhs[i] op rhs[i];                               \
    }                                                          \
    return ret;                                                \
  }
  DEFINE_BINARY_OPERATOR(+)
  DEFINE_BINARY_OPERATOR(-)
  DEFINE_BINARY_OPERATOR(*)
  DEFINE_BINARY_OPERATOR(/)
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_FUNCTION(func)                        \
  friend AlignedVector func(const AlignedVector& lhs,       \
                            const AlignedVector& rhs) {     \
    AlignedVector ret;                                      \
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < kSize; ++i) { \
      ret[i] = func(lhs[i], rhs[i]);                        \
    }                                                       \
    return ret;                                             \
  }
  DEFINE_BINARY_FUNCTION(min)
  DEFINE_BINARY_FUNCTION(max)
#undef DEFINE_BINARY_FUNCTION

 private:
  value_type values_[N];
};

#undef UNROLL_ON_DEVICE

template <typename T, int vec_size>
struct BaseTypeVectorize {
  typedef AlignedVector<T, vec_size> type;
  typedef T scalar;
};

template <int vec_size>
struct BaseTypeVectorize<Eigen::half, vec_size> {
  typedef typename Eigen::internal::conditional<
      (vec_size >= 2), sycl::vec<sycl::half, vec_size>,
      AlignedVector<Eigen::half, vec_size>>::type type;
  typedef typename Eigen::internal::conditional<(vec_size >= 2), sycl::half,
                                                Eigen::half>::type scalar;
};

// Returns the maximum power-of-two alignment (in units of elements, not bytes)
// of a stride or pointer value.
inline int64_t alignment_of(int64_t element_stride) {
  return element_stride & -element_stride;
}

template <typename T>
inline int64_t alignment_of(T* ptr) {
  const intptr_t ptr_val = reinterpret_cast<std::uintptr_t>(ptr);
  // Pointers should always be aligned to sizeof(T) bytes.
  ITEX_DCHECK_EQ(ptr_val % sizeof(T), 0);
  // Note that we want the alignment in elements, not bytes.
  return alignment_of(ptr_val / sizeof(T));
}

template <typename... Args>
int64_t MinAlignmentOf(Args... args) {
  return std::min({alignment_of(args)...});
}

// Calls Functor<vec_size>()(args...) with vec_size set to the optimal GPU
// vector instruction size for type T that is <= max_vec_size. The max_vec_size
// argument should be set to the minimum alignment of all relevant parameters.
template <typename T, template <int VecSize> class Functor, typename... Args>
void DispatchToVectorized(int64_t max_vec_size, Args&&... args) {
  constexpr const int kOptimalVecSizeBytes = 16;
  // The optimal number of (aligned) elements of T to load/store in a
  // single instruction inside a kernel.
  constexpr const int optimal_vec_size =
      (kOptimalVecSizeBytes - 1) / sizeof(T) + 1;
  int64_t vec_size = std::min((int64_t)optimal_vec_size, max_vec_size);
  if (vec_size >= 16) {
    Functor<16>()(std::forward<Args>(args)...);
  } else if (vec_size >= 8) {
    Functor<8>()(std::forward<Args>(args)...);
  } else if (vec_size >= 4) {
    Functor<4>()(std::forward<Args>(args)...);
  } else if (vec_size >= 2) {
    Functor<2>()(std::forward<Args>(args)...);
  } else {
    Functor<1>()(std::forward<Args>(args)...);
  }
}

// This funciton is to back compatible with old DPCPP compiler
template <typename T, int Dims = 1>
inline T* ITEXGetLocalAccPointer(
    const sycl::local_accessor<T, Dims>& accessor) {
  return accessor.template get_multi_ptr<sycl::access::decorated::no>().get();
}
}  // namespace  itex

#endif  //  ITEX_CORE_UTILS_GPU_HELPER_H_
