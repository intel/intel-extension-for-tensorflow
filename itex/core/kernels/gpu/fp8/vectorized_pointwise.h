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

#ifndef ITEX_CORE_KERNELS_GPU_FP8_VECTORIZED_POINTWISE_H_
#define ITEX_CORE_KERNELS_GPU_FP8_VECTORIZED_POINTWISE_H_

#include "itex/core/kernels/gpu/fp8/utils.h"
#include "itex/core/utils/float8.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/gpu_helper.h"

namespace itex {
template <int nvec, typename ComputeType, typename Param, typename OP,
          typename InputType, typename OutputType>
struct PointWiseKernel {
  using Ivec = Vec<InputType, nvec>;
  using Cvec = Vec<ComputeType, nvec>;
  using Ovec = Vec<OutputType, nvec>;
  using Oscalar = typename Ovec::Scalar_type;
  PointWiseKernel(const void* inp, void* out, ComputeType* amax,
                  const ComputeType* scale, const ComputeType* scale_inv,
                  Param p, int num_elements)
      : inp_(inp),
        out_(out),
        amax_(amax),
        scale_(scale),
        scale_inv_(scale_inv),
        p_(p),
        num_elements_(num_elements) {}

  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();

    Ivec x;
    Cvec xf;
    Ovec y;
    ComputeType max = 0.f;
    ComputeType scale = 0.f, scale_inv = 0.f;

    x.load_from_elts(inp_, nvec * id, num_elements_ - nvec * id);
    x.to(xf);
    if (is_fp8<InputType>::value) {
      scale_inv = *scale_inv_;
      xf.scale(scale_inv);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      ComputeType temp = OP()(xf.data[i], p_);
      if (is_fp8<OutputType>::value) {
        scale = *scale_;
        max = sycl::fmax(sycl::fabs(temp), max);
        temp = temp * scale;
      }
      y.data[i] = Oscalar(temp);
    }
    y.store_to_elts(out_, nvec * id, num_elements_ - nvec * id);

    if (is_fp8<OutputType>::value && amax_ != nullptr) {
      auto group_max =
          sycl::reduce_over_group(item.get_group(), max, sycl::maximum<>());
      if (item.get_local_linear_id() == 0) {
        ItexAtomicMax(amax_, group_max);
      }
    }
  }

 private:
  const void* inp_;
  void* out_;
  ComputeType* amax_;
  const ComputeType* scale_;
  const ComputeType* scale_inv_;
  Param p_;
  int num_elements_;
};
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FP8_VECTORIZED_POINTWISE_H_
