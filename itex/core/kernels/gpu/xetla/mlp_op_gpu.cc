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

#include "itex/core/kernels/gpu/xetla/mlp_op_gpu.h"

#include <algorithm>
#include <limits>

#include "itex/core/kernels/gpu/xetla/mlp_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T>
struct FusedDenseBiasAddGeluFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& feature,
                  const Tensor& weights, const Tensor& bias, Tensor* output,
                  Tensor* workspace) {
    uint32_t matrix_m = feature.dim_size(0);
    uint32_t matrix_k = feature.dim_size(1);
    uint32_t matrix_n = weights.dim_size(1);

    const T* feature_ptr = feature.flat<T>().data();
    const T* weights_ptr = weights.flat<T>().data();
    const T* bias_ptr = bias.flat<T>().data();
    T* output_ptr = output->flat<T>().data();
    T* workspace_ptr = workspace->flat<T>().data();

    constexpr uint32_t wg_tile_m = gpu::xetla::KernelAttr::wg_tile_m;
    constexpr uint32_t wg_tile_n = gpu::xetla::KernelAttr::wg_tile_n;
    constexpr uint32_t sg_tile_m = gpu::xetla::KernelAttr::sg_tile_m;
    constexpr uint32_t sg_tile_n = gpu::xetla::KernelAttr::sg_tile_n;
    uint32_t group_range_m = (matrix_m % wg_tile_m == 0)
                                 ? matrix_m / wg_tile_m
                                 : (matrix_m / wg_tile_m) + 1;
    uint32_t group_range_n = (matrix_n % wg_tile_n == 0)
                                 ? matrix_n / wg_tile_n
                                 : (matrix_n / wg_tile_n) + 1;
    uint32_t subgroup_range_m = (wg_tile_m % sg_tile_m == 0)
                                    ? wg_tile_m / sg_tile_m
                                    : (wg_tile_m / sg_tile_m) + 1;
    uint32_t subgroup_range_n = (wg_tile_n % sg_tile_n == 0)
                                    ? wg_tile_n / sg_tile_n
                                    : (wg_tile_n / sg_tile_n) + 1;
    cl::sycl::range<3> group_range{1, static_cast<size_t>(group_range_m),
                                   static_cast<size_t>(group_range_n)};
    cl::sycl::range<3> local_range{1, static_cast<size_t>(subgroup_range_m),
                                   static_cast<size_t>(subgroup_range_n)};
    cl::sycl::nd_range<3> thread_range(group_range * local_range, local_range);

    const auto& d = context->eigen_gpu_device();
    auto& stream = d.stream();

    using InT =
        typename std::conditional<std::is_same<T, Eigen::bfloat16>::value,
                                  gpu::xetla::bf16, sycl::half>::type;

#define CAST(ptr, src_t, dst_t) \
  reinterpret_cast<dst_t*>(const_cast<src_t*>(ptr))

    stream->submit([&](sycl::handler& cgh) {
      gpu::xetla::FusedDenseBiasAddGeluKernel<InT> task(
          CAST(feature_ptr, T, InT), CAST(weights_ptr, T, InT),
          CAST(bias_ptr, T, InT), CAST(output_ptr, T, InT),
          CAST(workspace_ptr, T, InT), matrix_m, matrix_n, matrix_k);
      cgh.parallel_for<class gpu::xetla::FusedDenseBiasAddGeluKernel<InT>>(
          thread_range, task);
#undef CAST
    });
  }
};

}  // namespace functor

#define DECLARE_GPU_SPEC(type) \
  template struct functor::FusedDenseBiasAddGeluFunctor<GPUDevice, type>;

DECLARE_GPU_SPEC(Eigen::bfloat16);
DECLARE_GPU_SPEC(Eigen::half);

#undef DECLARE_GPU_SPEC

}  // namespace itex
