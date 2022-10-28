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

namespace itex {

namespace detail {
// TODO(itex): add specializations for Eigen::half here
template <typename T, typename F>
void DpcppAtomicCasHelper(T* ptr, F accumulate) {
  T assumed = *ptr;
  auto atm = sycl::atomic_ref<T, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>(*ptr);
  bool success;
  do {
    success = atm.compare_exchange_strong(assumed, accumulate(assumed));
  } while (!success);
}
}  // namespace detail

template <typename T, typename U,
          sycl::memory_order DefaultOrder = sycl::memory_order::relaxed,
          sycl::memory_scope DefaultScope = sycl::memory_scope::device,
          sycl::access::address_space AddressSpace =
              sycl::access::address_space::global_space>
void DpcppAtomicAdd(T* ptr, U value) {
  sycl::atomic_ref<T, DefaultOrder, DefaultScope, AddressSpace> atm_dest(*ptr);
  atm_dest.fetch_add(value);
}

template <typename T, typename U>
void DpcppAtomicSub(T* ptr, U value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      atm_dest(*ptr);
  atm_dest.fetch_sub(value);
}

template <typename T, typename U>
void DpcppAtomicMax(T* ptr, U value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      atm_dest(*ptr);
  atm_dest.fetch_max(value);
}

template <typename T, typename U>
void DpcppAtomicMin(T* ptr, U value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      atm_dest(*ptr);
  atm_dest.fetch_min(value);
}

template <typename T, typename U>
void DpcppAtomicMul(T* ptr, U value) {
  detail::DpcppAtomicCasHelper(ptr, [value](T a) { return a * value; });
}

template <typename T, typename U>
void DpcppAtomicDiv(T* ptr, U value) {
  detail::DpcppAtomicCasHelper(ptr, [value](T a) { return a / value; });
}

}  // namespace itex

#endif  // ITEX_CORE_UTILS_GPU_DEVICE_FUNCTIONS_H_
