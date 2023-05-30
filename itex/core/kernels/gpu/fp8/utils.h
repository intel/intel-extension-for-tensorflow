/* Copyright (c) 2023 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_GPU_FP8_UTILS_H_
#define ITEX_CORE_KERNELS_GPU_FP8_UTILS_H_

#include "itex/core/utils/float8.h"
#include "itex/core/utils/gpu_helper.h"

namespace itex {
namespace layernorm {

constexpr int THREADS_PER_SUBGROUP = 32;

using group = sycl::group<1>;
using sub_group = sycl::sub_group;

template <typename T>
struct TypeToVec2 {};

template <>
struct TypeToVec2<float> {
  using Type = sycl::float2;
};

template <typename T, typename Smem, int SUBGROUPS_M, int SUBGROUPS_N>
struct Reducer : public Reducer<T, Smem, SUBGROUPS_M, 1> {
  using Base = Reducer<T, Smem, SUBGROUPS_M, 1>;
  using Type = T;

  enum { SMEM_SIZE = SUBGROUPS_M * SUBGROUPS_N };

  inline Reducer(group& g, sub_group& sg,  // NOLINT
                 int sg_m, int sg_n, int lane, const Smem& scratch)
      : Base(g, sg, sg_m, sg_n, lane, scratch), g_(g), scratch_(scratch) {}

  template <typename Op>
  inline T allreduce(T data, const Op& op) {
    data = Base::reduce(data, op);
    if (this->lane_ == 0) {
      scratch_[this->sg_m_ * SUBGROUPS_N + this->sg_n_] = data;
    }
    sycl::group_barrier(this->g_);
    T out = T(0);
#pragma unroll
    for (int it = 0; it < SUBGROUPS_N; it++) {
      out = op(out, scratch_[this->sg_m_ * SUBGROUPS_N + it]);
    }
    return out;
  }

  template <typename Op>
  inline T reduce(T data, const Op& op) {
    data = Base::reduce(data, op);
    if (this->lane_ == 0) {
      scratch_[this->sg_m_ * SUBGROUPS_N + this->sg_n_] = data;
    }
    sycl::group_barrier(g_);
    T out = T(0);
    if (this->sg_n_ == 0 && this->lane_ == 0) {
#pragma unroll
      for (int it = 0; it < SUBGROUPS_N; it++) {
        out = op(out, scratch_[this->sg_m_ * SUBGROUPS_N + it]);
      }
    }
    return out;
  }

  group& g_;
  const Smem& scratch_;
};

template <typename T, typename Smem, int SUBGROUPS_M>
struct Reducer<T, Smem, SUBGROUPS_M, 1> {
  using Type = T;
  enum { SMEM_SIZE = 0 };

  inline Reducer(group& g, sub_group& sg,  // NOLINT
                 int sg_m, int sg_n, int lane, const Smem& scratch)
      : sg_(sg), sg_m_(sg_m), sg_n_(sg_n), lane_(lane) {}

  template <typename Op>
  inline T allreduce_(T data, const Op& op) {
#pragma unroll
    for (int it = 1; it < THREADS_PER_SUBGROUP; it *= 2) {
      data = op(data, sycl::permute_group_by_xor(sg_, data, it));
    }
    return data;
  }

  template <typename Op>
  inline T allreduce(T data, const Op& op) {
    return allreduce_(data, op);
  }

  template <typename Op>
  inline T reduce(T data, const Op& op) {
// only lane 0 holds the result!
#pragma unroll
    for (int it = THREADS_PER_SUBGROUP / 2; it > 0; it /= 2) {
      data = op(data, sycl::shift_group_left(sg_, data, it));
    }
    return data;
  }

  sub_group& sg_;
  int sg_m_;
  int sg_n_;
  int lane_;
};

}  // namespace layernorm

template <typename Elt_type, int NUM_ELTS>
struct Vec {
  using Vec_type = typename BaseTypeVectorize<Elt_type, NUM_ELTS>::type;
  using Scalar_type = typename BaseTypeVectorize<Elt_type, NUM_ELTS>::scalar;

  Vec_type data;

  template <typename S>
  inline void to(Vec<S, NUM_ELTS>& other) {  // NOLINT(*)
#pragma unroll
    for (int it = 0; it < NUM_ELTS; it++) {
      other.data[it] = typename Vec<S, NUM_ELTS>::Scalar_type(this->data[it]);
    }
  }

  inline void scale(Elt_type factor) {  // NOLINT(*)
#pragma unroll
    for (int it = 0; it < NUM_ELTS; it++) {
      this->data[it] *= factor;
    }
  }

  inline void load_from(const void* base_ptr, int idx = 0) {
    this->data = reinterpret_cast<const Vec_type*>(base_ptr)[idx];
  }

  // Pointer is cast to vector type
  inline void store_to(void* base_ptr, int idx = 0) const {
    reinterpret_cast<Vec_type*>(base_ptr)[idx] = this->data;
  }

  // Pointer is cast to element type. Loads min(count, NUM_ELT)
  // elements and any remaining elements are set to zero.
  inline void load_from_elts(const void* base_ptr, int idx = 0,
                             int count = NUM_ELTS) {
    const Scalar_type* elt_ptr =
        reinterpret_cast<const Scalar_type*>(base_ptr) + idx;
    if (count < NUM_ELTS || idx % NUM_ELTS != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELTS; it++) {
        this->data[it] = (it < count ? elt_ptr[it] : Scalar_type(0.f));
      }
    } else {
      this->load_from(elt_ptr);
    }
  }

  // Pointer is cast to element type. Stores min(count, NUM_ELT)
  // elements.
  inline void store_to_elts(void* base_ptr, int idx = 0,
                            int count = NUM_ELTS) const {
    Scalar_type* elt_ptr = static_cast<Scalar_type*>(base_ptr) + idx;
    if (count < NUM_ELTS || idx % NUM_ELTS != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELTS; it++) {
        if (it < count) {
          elt_ptr[it] = this->data[it];
        }
      }
    } else {
      this->store_to(elt_ptr);
    }
  }

  inline void clear() {
#pragma unroll
    for (int it = 0; it < NUM_ELTS; it++) {
      this->data[it] = Scalar_type(0);
    }
  }
};

template <typename T>
struct is_fp8 {
  static const bool value = std::is_same_v<T, float8_e4m3fn> ||
                            std::is_same_v<T, float8_e5m2> ||
                            std::is_same_v<T, float8_e4m3b11>;
};

#define FP8_TYPE_SWITCH(context, format, arithmatic_type, storage_type, ...) \
  if constexpr (std::is_same_v<storage_type, int8>) {                        \
    if (format == "E4M3") {                                                  \
      typedef float8_e4m3fn arithmatic_type;                                 \
      { __VA_ARGS__ }                                                        \
    } else if (format == "E5M2") {                                           \
      typedef float8_e5m2 arithmatic_type;                                   \
      { __VA_ARGS__ }                                                        \
    } else {                                                                 \
      context->SetStatus(errors::InvalidArgument("Invalid type"));           \
    }                                                                        \
  } else {                                                                   \
    typedef storage_type arithmatic_type;                                    \
    { __VA_ARGS__ }                                                          \
  }

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FP8_UTILS_H_
