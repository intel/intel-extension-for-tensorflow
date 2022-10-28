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

#include "itex/core/kernels/gpu/compare_and_bitpack_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Maozhou: specialization for data types?
template <typename T>
struct CompareAndBitpackKernel {
  CompareAndBitpackKernel(const int size, const T* threshold, const T* input,
                          uint8* output)
      : size_(size), threshold_(threshold), input_(input), output_(output) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= size_) {
      return;
    }

    const T thresh = *threshold_;
    const T* block = input_ + 8 * id;
    output_[id] =
        ((*block > thresh) << 7) | ((*(block + 1) > thresh) << 6) |
        ((*(block + 2) > thresh) << 5) | ((*(block + 3) > thresh) << 4) |
        ((*(block + 4) > thresh) << 3) | ((*(block + 5) > thresh) << 2) |
        ((*(block + 6) > thresh) << 1) | ((*(block + 7) > thresh));
  }

 private:
  const int size_;
  const T* threshold_;
  const T* input_;
  uint8* output_;
};

template <typename T>
struct CompareAndBitpack<GPUDevice, T> {
  void operator()(OpKernelContext* ctx, typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::ConstScalar threshold,
                  TTypes<uint8>::Matrix output) {
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = output.size();
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      CompareAndBitpackKernel<T> task(num_work_items, threshold.data(),
                                      input.data(), output.data());
      cgh.parallel_for<CompareAndBitpackKernel<T> >(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          task);
    });
  }
};

#define DEFINE_GPU_SPECS(T) template struct CompareAndBitpack<GPUDevice, T>;

TF_CALL_half(DEFINE_GPU_SPECS);
TF_CALL_bfloat16(DEFINE_GPU_SPECS);
TF_CALL_float(DEFINE_GPU_SPECS);
TF_CALL_bool(DEFINE_GPU_SPECS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_GPU_SPECS);
#endif  // ITEX_ENABLE_DOUBLE
#undef DECLARE_GPU_SPECS

}  // namespace functor

}  // namespace itex
