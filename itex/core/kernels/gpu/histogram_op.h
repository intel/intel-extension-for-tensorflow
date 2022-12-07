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

#ifndef ITEX_CORE_KERNELS_GPU_HISTOGRAM_OP_H_
#define ITEX_CORE_KERNELS_GPU_HISTOGRAM_OP_H_

#include <algorithm>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Tout>
struct HistogramKernel {
  HistogramKernel(const T* input, Tout* out, float step, int32 nbins,
                  size_t input_size, T start, T end)
      : input_(input),
        out_(out),
        step_(step),
        nbins_(nbins),
        input_size_(input_size),
        start_(start),
        end_(end) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= input_size_) return;

    int32 idx = static_cast<float>(input_[id] - start_) / step_;
    idx = static_cast<int32>(idx >= 0) * sycl::min(idx, nbins_ - 1);
    // TODO(itex): replace atomic operation with other algo in the future
    ItexAtomicAdd<Tout, int>(out_ + idx, 1);
  }

 private:
  const T* input_;
  Tout* out_;
  float step_;
  int32 nbins_;
  size_t input_size_;
  T start_;
  T end_;
};

namespace functor {
template <typename Device, typename T, typename Tout>
struct HistogramFixedWidthFunctor {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        const typename TTypes<T, 1>::ConstTensor& value_range,
                        int32 nbins,
                        const typename TTypes<Tout, 1>::Tensor& out);
};

template <typename T, typename Tout>
struct HistogramFixedWidthFunctor<GPUDevice, T, Tout> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        const typename TTypes<T, 1>::ConstTensor& value_range,
                        int32 nbins,
                        const typename TTypes<Tout, 1>::Tensor& out) {
    auto& d = context->eigen_gpu_device();

    To32Bit(out).device(d) = To32Bit(out).constant(Tout(0));
    if (values.size() == 0 || value_range.size() == 0) return Status::OK();

    auto stream = d.stream();
    auto elems = values.size();
    auto wg_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_group = (elems + wg_size - 1) / wg_size;

    double step = static_cast<double>(value_range(1) - value_range(0)) / nbins;
    stream->submit([&](sycl::handler& cgh) {
      HistogramKernel<T, Tout> task(values.data(), out.data(), step, nbins,
                                    elems, value_range(0), value_range(1));
      cgh.parallel_for<HistogramKernel<T, Tout>>(
          sycl::nd_range<1>(sycl::range<1>(num_group * wg_size),
                            sycl::range<1>(wg_size)),
          task);
    });
    return Status::OK();
  }
};
}  // end namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_HISTOGRAM_OP_H_
