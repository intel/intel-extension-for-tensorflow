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

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename Device, typename OutType>
class OneDnnShapeOp : public OpKernel {
 public:
  explicit OneDnnShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& src_tensor = ctx->input(0);
    // Get src_md
    OneDnnShape src_onednn_shape;
    GetOneDnnShape(ctx, 0, &src_onednn_shape);
    TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                   ? src_onednn_shape.GetTfShape()
                                   : src_tensor.shape();
    // Allocate output's data tensor and meta tensor
    const int rank = src_tf_shape.dims();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rank}), &out));
    auto vec = out->vec<OutType>();
    for (int i = 0; i < rank; ++i) {
      int64_t dim_size = src_tf_shape.dim_size(i);
      if (out->dtype() == DT_INT32) {
        OP_REQUIRES(
            ctx, FastBoundsCheck(dim_size, std::numeric_limits<int32>::max()),
            errors::InvalidArgument("Shape output type is 32-bit ", " but dim ",
                                    i, " is ", dim_size));
      }
      vec(i) = static_cast<OutType>(dim_size);
    }
  }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_GPU_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnShape")                     \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .HostMemory("input_meta")            \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          OneDnnShapeOp<GPUDevice, int32>);        \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnShape")                     \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .HostMemory("input_meta")            \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          OneDnnShapeOp<GPUDevice, int64_t>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_qint8(REGISTER_GPU_KERNEL);
TF_CALL_quint8(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#else
#define REGISTER_CPU_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnShape")                     \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          OneDnnShapeOp<CPUDevice, int32>);        \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnShape")                     \
                              .Device(DEVICE_CPU)                  \
                              .HostMemory("output")                \
                              .HostMemory("input_meta")            \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          OneDnnShapeOp<CPUDevice, int64_t>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_qint8(REGISTER_CPU_KERNEL);
TF_CALL_quint8(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#endif  // INTEL_CPU_ONLY

}  // namespace itex
