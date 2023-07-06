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

#ifndef ITEX_CORE_KERNELS_GPU_FP8_FP8_QUANTIZE_GPU_H_
#define ITEX_CORE_KERNELS_GPU_FP8_FP8_QUANTIZE_GPU_H_

#include "itex/core/kernels/gpu/fp8/vectorized_pointwise.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {
namespace functor {
namespace detail {

struct Empty {};

struct Identity {
  float operator()(float value, const Empty&) { return value; }
};

}  // namespace detail

template <typename input_t, typename output_t>
void Fp8Quantize(OpKernelContext* context, const void* inp_ptr, void* out_ptr,
                 float* amax, const float* scale, int num_elements) {
  constexpr int nvec = 16 / sizeof(input_t);
  auto* stream = context->GetDeviceStream();
  int wg_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  int wg_num = DivUp(DivUp(num_elements, nvec), wg_size);
  stream->submit([&](sycl::handler& cgh) {
    PointWiseKernel<nvec, float, detail::Empty, detail::Identity, input_t,
                    output_t>
        task(inp_ptr, out_ptr, amax, scale, nullptr, {}, num_elements);
    cgh.parallel_for<PointWiseKernel<nvec, float, detail::Empty,
                                     detail::Identity, input_t, output_t>>(
        sycl::nd_range<1>(wg_num * wg_size, wg_size), task);
  });
}

template <typename input_t, typename output_t>
void Fp8Dequantize(OpKernelContext* context, const void* inp_ptr, void* out_ptr,
                   const float* scale_inv, int num_elements) {
  constexpr int nvec = 16 / sizeof(output_t);
  auto* stream = context->GetDeviceStream();
  int wg_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  int wg_num = DivUp(DivUp(num_elements, nvec), wg_size);
  stream->submit([&](sycl::handler& cgh) {
    PointWiseKernel<nvec, float, detail::Empty, detail::Identity, input_t,
                    output_t>
        task(inp_ptr, out_ptr, nullptr, nullptr, scale_inv, {}, num_elements);
    cgh.parallel_for<PointWiseKernel<nvec, float, detail::Empty,
                                     detail::Identity, input_t, output_t>>(
        sycl::nd_range<1>(wg_num * wg_size, wg_size), task);
  });
}
}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FP8_FP8_QUANTIZE_GPU_H_
