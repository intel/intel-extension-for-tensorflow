/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/betainc_op.h"

#include "itex/core/utils/bcast.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class BetaincOp : public OpKernel {
 public:
  explicit BetaincOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    const Tensor& x = ctx->input(2);

    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& x_shape = x.shape();
    if (a_shape.dims() > 0 && b_shape.dims() > 0) {
      OP_REQUIRES(ctx, a_shape == b_shape,
                  errors::InvalidArgument(
                      "Shapes of a and b are inconsistent: ",
                      a_shape.DebugString(), " vs. ", b_shape.DebugString()));
    }
    if (a_shape.dims() > 0 && x_shape.dims() > 0) {
      OP_REQUIRES(ctx, a_shape == x_shape,
                  errors::InvalidArgument(
                      "Shapes of a and x are inconsistent: ",
                      a_shape.DebugString(), " vs. ", x_shape.DebugString()));
    }
    if (b_shape.dims() > 0 && x_shape.dims() > 0) {
      OP_REQUIRES(ctx, b_shape == x_shape,
                  errors::InvalidArgument(
                      "Shapes of b and x are inconsistent: ",
                      b_shape.DebugString(), " vs. ", x_shape.DebugString()));
    }

    TensorShape merged_shape(a_shape);
    if (b_shape.dims() > 0) merged_shape = b_shape;
    if (x_shape.dims() > 0) merged_shape = x_shape;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, merged_shape, &output));

    if (a_shape == b_shape && a_shape == x_shape) {
      functor::Betainc<Device, T, 1> functor;
      functor(ctx->eigen_device<Device>(), a.flat<T>(), b.flat<T>(),
              x.flat<T>(), output->flat<T>());
      return;
    }

    auto merged_shape_vec = BCast::FromShape(merged_shape);
    BCast a_shaper(BCast::FromShape(a_shape), merged_shape_vec);
    BCast b_shaper(BCast::FromShape(b_shape), merged_shape_vec);
    BCast x_shaper(BCast::FromShape(x_shape), merged_shape_vec);

    int ndims = static_cast<int>(a_shaper.x_reshape().size());

    switch (ndims) {
#define CASE(NDIM)                                                        \
  case NDIM: {                                                            \
    functor::Betainc<Device, T, NDIM> functor;                            \
    auto a_value = a.shaped<T, NDIM>(a_shaper.x_reshape());               \
    auto b_value = b.shaped<T, NDIM>(b_shaper.x_reshape());               \
    auto x_value = x.shaped<T, NDIM>(x_shaper.x_reshape());               \
    functor.BCast(ctx->eigen_device<Device>(), a_value,                   \
                  BCast::ToIndexArray<NDIM>(a_shaper.x_bcast()), b_value, \
                  BCast::ToIndexArray<NDIM>(b_shaper.x_bcast()), x_value, \
                  BCast::ToIndexArray<NDIM>(x_shaper.x_bcast()),          \
                  output->shaped<T, NDIM>(a_shaper.y_reshape()));         \
    return;                                                               \
  }

      CASE(1);
      CASE(2);
      default: {
        ctx->SetStatus(errors::InvalidArgument(
            "Broadcasting rank not supported: ", ndims));
        return;
      }
    }
  }
};

#define DEFINE_GPU_KERNELS_NDIM(T, NDIM) \
  template struct functor::Betainc<GPUDevice, T, NDIM>;

#define DEFINE_GPU_KERNELS(T)   \
  DEFINE_GPU_KERNELS_NDIM(T, 1) \
  DEFINE_GPU_KERNELS_NDIM(T, 2)

DEFINE_GPU_KERNELS(float);
#undef DEFINE_GPU_KERNELS
#undef DEFINE_GPU_KERNELS_NDIM

#define REGISTER_GPU_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Betainc").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BetaincOp<GPUDevice, type>);

REGISTER_GPU_KERNELS(float)
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_GPU_KERNELS(double)
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS
}  // namespace itex
