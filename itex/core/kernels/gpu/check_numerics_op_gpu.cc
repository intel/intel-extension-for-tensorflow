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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct CheckNumericsDpcppKernel {
  CheckNumericsDpcppKernel(size_t num_work_items, const T* data,
                           int* abnormal_detected)
      : num_work_items(num_work_items),
        data(data),
        abnormal_detected(abnormal_detected) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }

    int32 offset = id;
    if (sycl::isnan(static_cast<float>(data[offset]))) {
      abnormal_detected[0] = 1;
    }
    if (sycl::isinf(static_cast<float>(data[offset]))) {
      abnormal_detected[1] = 1;
    }
  }

 private:
  size_t num_work_items;
  const T* data;
  int* abnormal_detected;
};

template <typename T>
struct CheckNumericsLaunch {
  void Run(const GPUDevice& d, const T* data, int size,
           int* abnormal_detected) {
    auto stream = d.stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = size;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;

    stream->submit([&](sycl::handler& cgh) {
      CheckNumericsDpcppKernel<T> kernel_functor(num_work_items, data,
                                                 abnormal_detected);
      cgh.parallel_for<CheckNumericsDpcppKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          kernel_functor);
    });
  }
};

template struct CheckNumericsLaunch<Eigen::bfloat16>;
template struct CheckNumericsLaunch<Eigen::half>;
template struct CheckNumericsLaunch<float>;
template struct CheckNumericsLaunch<double>;

template <typename T>
struct CheckNumericsV2DpcppKernel {
  CheckNumericsV2DpcppKernel(size_t num_work_items, const T* data,
                             int* abnormal_detected)
      : num_work_items(num_work_items),
        data(data),
        abnormal_detected(abnormal_detected) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }

    int32 offset = id;

    if (sycl::isnan(static_cast<float>(data[offset]))) {
      abnormal_detected[0] = 1;
    }
    if (sycl::isinf(static_cast<float>(data[offset]))) {
      if (sycl::isless(static_cast<float>(data[offset]),
                       static_cast<float>(0.f))) {
        abnormal_detected[1] = 1;
      } else {
        abnormal_detected[2] = 1;
      }
    }
  }

 private:
  size_t num_work_items;
  const T* data;
  int* abnormal_detected;
};

template <typename T>
struct CheckNumericsV2Launch {
  void Run(const GPUDevice& d, const T* data, int size,
           int abnormal_detected[3]) {
    auto stream = d.stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = size;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;

    stream->submit([&](sycl::handler& cgh) {
      CheckNumericsV2DpcppKernel<T> kernel_functor(num_work_items, data,
                                                   abnormal_detected);
      cgh.parallel_for<CheckNumericsV2DpcppKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          kernel_functor);
    });
  }
};

template struct CheckNumericsV2Launch<Eigen::bfloat16>;
template struct CheckNumericsV2Launch<Eigen::half>;
template struct CheckNumericsV2Launch<float>;
template struct CheckNumericsV2Launch<double>;

}  // namespace itex
