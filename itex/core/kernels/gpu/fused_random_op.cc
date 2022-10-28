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

#include "itex/core/kernels/gpu/fused_random_op_gpu.h"
#include "itex/core/kernels/gpu/random_op_gpu.h"

#include "itex/core/utils/lib/random/guarded_philox_random.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct GreaterEqual {
  inline T operator()(T x, T i) const { return T(x >= i); }
};

template <typename T>
class FusedRandomOp : public OpKernel {
 public:
  typedef random::UniformDistribution<random::PhiloxRandom, T> Uniform;
  explicit FusedRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &direction_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
  }

  void GetOutputShape(const TensorShape& shape0, const TensorShape& shape1,
                      TensorShape* out) {
    TensorShape l = shape0.dims() > shape1.dims() ? shape0 : shape1;
    TensorShape s = shape0.dims() > shape1.dims() ? shape1 : shape0;

    std::vector<int> vec(l.dims());
    int gap = l.dims() - s.dims();
    for (int i = 0; i < gap; ++i) {
      vec[i] = l.dim_size(i);
    }
    for (int i = 0; i < s.dims(); ++i) {
      vec[i + gap] = std::max(s.dim_size(i), l.dim_size(i + gap));
    }

    TF_ABORT_IF_ERROR(TensorShapeUtils::MakeShape(vec.data(), vec.size(), out));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    const Tensor& compare = ctx->input(1);
    TensorShape tensor_shape;
    TF_ABORT_IF_ERROR(MakeShape(shape, &tensor_shape));
    Tensor* output;
    TensorShape output_shape;
    GetOutputShape(tensor_shape, compare.shape(), &output_shape);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    auto output_flat = output->flat<T>();
    int dims = compare.dims() == 0 ? 0 : output_shape.dims();

    OP_REQUIRES(ctx, dims == 0,
                errors::InvalidArgument("Only support compare dim is 0 "));
    functor::FillPhiloxRandomFused<GPUDevice, Uniform, GreaterEqual<T> >()(
        ctx, ctx->eigen_device<GPUDevice>(),
        // Multiplier 256 is the same as in FillPhiloxRandomTask; do not
        // change it just here.
        generator_.ReserveRandomOutputs(output_flat.size(), 256),
        output_flat.data(), output_flat.size(), Uniform(),
        compare.flat<T>().data(), nullptr, nullptr, dims, nullptr, nullptr,
        GreaterEqual<T>());
  }

 private:
  GuardedPhiloxRandom generator_;
  int direction_ = 0;
  std::vector<string> fused_ops_;
};

#define REGISTER_FUSED_RANDOM_KERNEL(TYPE)                   \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedRandom")           \
                              .Device(DEVICE_GPU)            \
                              .HostMemory("shape")           \
                              .TypeConstraint<int32>("T")    \
                              .TypeConstraint<TYPE>("DstT"), \
                          FusedRandomOp<TYPE>);

REGISTER_FUSED_RANDOM_KERNEL(float);
REGISTER_FUSED_RANDOM_KERNEL(Eigen::bfloat16);
REGISTER_FUSED_RANDOM_KERNEL(Eigen::half);
}  // namespace itex
