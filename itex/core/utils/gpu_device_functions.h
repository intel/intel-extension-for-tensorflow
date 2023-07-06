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

#ifndef ITEX_CORE_UTILS_GPU_DEVICE_FUNCTIONS_H_
#define ITEX_CORE_UTILS_GPU_DEVICE_FUNCTIONS_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <algorithm>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace detail {
template <typename T, typename F>
void ItexAtomicCasHelper(T* ptr, F accumulate) {
  T assumed = *ptr;
  auto atm = sycl::atomic_ref<T, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>(*ptr);
  bool success;
  do {
    success = atm.compare_exchange_strong(assumed, accumulate(assumed));
  } while (!success);
}

// SYCL atomicCAS can only operate on 32 bit and 64 bit datatype.
// If the element type is smaller than 32 bits, 32 bit datatype can be used for
// the atomicCAS operation. In this case, we mask off the last two bits of the
// given address and use the result as an 32 bit aligned address to read the 32
// bit values from the memory. Then we can get out the 16 bit value, operate on
// it and store it into the 32 bit value. Finally, use 32 bit atomicCAS to reads
// and writes 32 bit values from the memory. In this implementation we use
// bit level arithmetic "<<" and ">>" to read/store 16 bit value from/to
// 32 bit value, and result of "<<" and ">>" is relative to byte order.
// Our implementation only works on Little-End byte order.

// Note: this method requires that the tensor buffers are 4 byte aligned and
// have a size of 4N, otherwise it may cause OOM error. Please check this
// condition before you use it.
template <typename F>
void ItexAtomicCasHelper(Eigen::half* ptr, F accumulate) {
  // get the 32-bit aligned address
  auto i_ptr = reinterpret_cast<std::uintptr_t>(ptr);
  std::uint32_t* addr =
      reinterpret_cast<std::uint32_t*>(i_ptr & 0xFFFFFFFFFFFFFFFCULL);
  // check if the operation is for the upper or lower 16-bit part in the aligned
  // 32-bit item
  bool upper = i_ptr & 2;
  auto atm = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>(*addr);
  std::uint32_t old_val = *addr;
  bool success;
  do {
    std::uint32_t val = old_val;
    Eigen::half newval = accumulate(Eigen::half_impl::raw_uint16_to_half(
        upper ? ((std::uint16_t)(val >> 16)) : ((std::uint16_t)(val))));
    std::uint16_t newval_s = *reinterpret_cast<std::uint16_t*>(&newval);
    std::uint32_t newval_u = val & (upper ? (0x0FFFFU) : (0xFFFF0000U));
    newval_u |= upper ? (((std::uint32_t)newval_s) << 16) : (newval_s);
    success = atm.compare_exchange_strong(old_val, newval_u);
  } while (!success);
}

template <typename F>
void ItexAtomicCasHelper(Eigen::bfloat16* ptr, F accumulate) {
  // get the 32-bit aligned address
  auto i_ptr = reinterpret_cast<std::uintptr_t>(ptr);
  std::uint32_t* addr =
      reinterpret_cast<std::uint32_t*>(i_ptr & 0xFFFFFFFFFFFFFFFCULL);
  // check if the operation is for the upper or lower 16-bit part in the aligned
  // 32-bit item
  bool upper = i_ptr & 2;
  auto atm = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>(*addr);
  std::uint32_t old_val = *addr;
  bool success;
  do {
    std::uint32_t val = old_val;
    Eigen::bfloat16 newval =
        accumulate(Eigen::bfloat16_impl::raw_uint16_to_bfloat16(
            upper ? ((std::uint16_t)(val >> 16)) : ((std::uint16_t)(val))));
    std::uint16_t newval_s = *reinterpret_cast<std::uint16_t*>(&newval);
    std::uint32_t newval_u = val & (upper ? (0x0FFFFU) : (0xFFFF0000U));
    newval_u |= upper ? (((std::uint32_t)newval_s) << 16) : (newval_s);
    success = atm.compare_exchange_strong(old_val, newval_u);
  } while (!success);
}
}  // namespace detail

template <typename T, typename U,
          sycl::memory_order DefaultOrder = sycl::memory_order::relaxed,
          sycl::memory_scope DefaultScope = sycl::memory_scope::device,
          sycl::access::address_space AddressSpace =
              sycl::access::address_space::global_space>
void ItexAtomicAdd(T* ptr, U value) {
  if constexpr (std::is_same_v<T, Eigen::half> ||
                std::is_same_v<T, Eigen::bfloat16>) {
    detail::ItexAtomicCasHelper(ptr, [value](T a) { return a + value; });
  } else {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, AddressSpace> atm_dest(
        *ptr);
    atm_dest.fetch_add(value);
  }
}

template <typename T, typename U>
void ItexAtomicSub(T* ptr, U value) {
  if constexpr (std::is_same_v<T, Eigen::half> ||
                std::is_same_v<T, Eigen::bfloat16>) {
    detail::ItexAtomicCasHelper(ptr, [value](T a) { return a - value; });
  } else {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atm_dest(*ptr);
    atm_dest.fetch_sub(value);
  }
}

template <typename T, typename U>
void ItexAtomicMax(T* ptr, U value) {
  if constexpr (std::is_same_v<T, Eigen::half> ||
                std::is_same_v<T, Eigen::bfloat16>) {
    detail::ItexAtomicCasHelper(ptr,
                                [value](T a) { return std::max(a, value); });
  } else {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atm_dest(*ptr);
    atm_dest.fetch_max(value);
  }
}

template <typename T, typename U>
void ItexAtomicMin(T* ptr, U value) {
  if constexpr (std::is_same_v<T, Eigen::half> ||
                std::is_same_v<T, Eigen::bfloat16>) {
    detail::ItexAtomicCasHelper(ptr,
                                [value](T a) { return std::min(a, value); });
  } else {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atm_dest(*ptr);
    atm_dest.fetch_min(value);
  }
}

template <typename T, typename U>
void ItexAtomicMul(T* ptr, U value) {
  detail::ItexAtomicCasHelper(ptr, [value](T a) { return a * value; });
}

template <typename T, typename U>
void ItexAtomicDiv(T* ptr, U value) {
  detail::ItexAtomicCasHelper(ptr, [value](T a) { return a / value; });
}

}  // namespace itex

#endif  // ITEX_CORE_UTILS_GPU_DEVICE_FUNCTIONS_H_
