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

#ifndef ITEX_CORE_KERNELS_COMMON_FILL_FUNCTOR_H_
#define ITEX_CORE_KERNELS_COMMON_FILL_FUNCTOR_H_

#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace functor {

template <typename Device, typename T>
struct FillFunctor {
  // Computes on device "d": out = out.constant(in(0)),
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in);
};

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct SetZeroFunctor {
  // Computes on device "d": out = out.setZero(),
  void operator()(const Device& d, typename TTypes<T>::Flat out);
};

// Partial specialization of SetZeroFunctor<Device=Eigen::ThreadPoolDevice, T>.
template <typename T>
struct SetZeroFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<T>::Flat out) {
    out.device(d) = out.constant(T(0));
  }
};

// Partial specialization of FillFunctor<Device=GPUDevice, T>.
template <typename T>
struct SetZeroFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out) {
    if (out.size() != 0) {
      To32Bit(out).device(d) = To32Bit(out).constant(T(0));
    }
  }
};

template <typename Device, typename T>
struct SetOneFunctor {
  // Computes on device "d": out = out.setOne(),
  void operator()(const Device& d, typename TTypes<T>::Flat out);
};

// Partial specialization of SetOneFunctor<Device=Eigen::ThreadPoolDevice, T>.
template <typename T>
struct SetOneFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<T>::Flat out) {
    out.device(d) = out.constant(T(1));
  }
};

// Partial specialization of FillFunctor<Device=GPUDevice, T>.
template <typename T>
struct SetOneFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out) {
    To32Bit(out).device(d) = To32Bit(out).constant(T(1));
  }
};

template <typename Device, typename T>
struct SetNanFunctor {
  // Computes on device "d": out = out.setNan(),
  void operator()(const Device& d, typename TTypes<T>::Flat out);
};

template <typename T>
struct SetNanFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out) {
    To32Bit(out).device(d) =
        To32Bit(out).constant(Eigen::NumTraits<T>::quiet_NaN());
  }
};

}  // namespace functor

#ifndef INTEL_CPU_ONLY

template <typename T>
class ConvertFromFp32Kernel {
 public:
  ConvertFromFp32Kernel(int total_size, float* from, T* to)
      : total_size(total_size), from(from), to(to) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= total_size) {
      return;
    }
    to[index] = static_cast<T>(from[index]);
  }

 private:
  int total_size;
  float* from;
  T* to;
};

template <typename Device, typename T16bit>
void ConvertFromFp32(const Device& d, int total_size, float* from, T16bit* to) {
  auto work_group_size =
      d.stream()
          ->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_groups = (total_size + work_group_size - 1) / work_group_size;
  d.stream()->submit([&](sycl::handler& cgh) {
    sycl::nd_range<1> kernel_range(
        sycl::range<1>(work_group_size * num_work_groups),
        sycl::range<1>(work_group_size));
    ConvertFromFp32Kernel<T16bit> kernel(total_size, from, to);
    cgh.parallel_for<ConvertFromFp32Kernel<T16bit>>(kernel_range, kernel);
  });
}

template <typename T>
class ConvertToFp32Kernel {
 public:
  ConvertToFp32Kernel(int total_size, T* from, float* to)
      : total_size(total_size), from(from), to(to) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= total_size) {
      return;
    }
    to[index] = static_cast<float>(from[index]);
  }

 private:
  int total_size;
  T* from;
  float* to;
};

template <typename Device, typename T16bit>
void ConvertToFp32(const Device& d, int total_size, T16bit* from, float* to) {
  auto work_group_size =
      d.stream()
          ->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_groups = (total_size + work_group_size - 1) / work_group_size;
  d.stream()->submit([&](sycl::handler& cgh) {
    sycl::nd_range<1> kernel_range(
        sycl::range<1>(work_group_size * num_work_groups),
        sycl::range<1>(work_group_size));
    ConvertToFp32Kernel<T16bit> kernel(total_size, from, to);
    cgh.parallel_for<ConvertToFp32Kernel<T16bit>>(kernel_range, kernel);
  });
}
#endif  // NOT INTEL_CPU_ONLY

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_FILL_FUNCTOR_H_
