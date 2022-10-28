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

#ifndef ITEX_CORE_KERNELS_ONEDNN_BLOCK_MATMUL_OP_H_
#define ITEX_CORE_KERNELS_ONEDNN_BLOCK_MATMUL_OP_H_

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::engine;
using dnnl::memory;

namespace itex {

template <typename Device, typename T>
class OneDnnMatMulBaseOp : public OpKernel {
 public:
  explicit OneDnnMatMulBaseOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  memory::desc CreateMemoryDescWithStrides(const memory::dims& dims) {
    memory::dims strides = CalculateTFStrides(dims);
    return memory::desc(dims, OneDnnType<T>(), strides);
  }

 protected:
  bool transpose_a_;
  bool transpose_b_;
  bool is_filter_const_ = false;
  bool inplace_sum_ = false;

  // Fusion util.
  PostOpUtil post_op_util_;

  // Weight cache manager
  WeightCacheManager<T> weight_cache_manager_;

  dnnl::fpmath_mode fp32_math_mode_ = dnnl::fpmath_mode::strict;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_ONEDNN_BLOCK_MATMUL_OP_H_
