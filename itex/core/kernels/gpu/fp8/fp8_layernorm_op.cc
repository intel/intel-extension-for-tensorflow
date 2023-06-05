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

#include "itex/core/kernels/gpu/fp8/fp8_layernorm_gpu.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Tin, typename Tweight, typename Tout>
class Fp8LayerNormOp : public OpKernel {
 public:
  explicit Fp8LayerNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype", &fp8_dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index", &fp8_meta_index_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& gamma = context->input(1);
    const Tensor& beta = context->input(2);
    const Tensor& z_amax = context->input(3);
    const Tensor& z_scale = context->input(4);

    const float* z_scale_ptr = nullptr;
    float* z_amax_ptr = nullptr;
    if (std::is_same_v<Tout, int8>) {
      z_amax_ptr =
          const_cast<Tensor&>(z_amax).flat<float>().data() + fp8_meta_index_;
      z_scale_ptr = z_scale.flat<float>().data() + fp8_meta_index_;
    }

    TensorShape x_shape = x.shape();
    int row = x_shape.dim_size(0), col = x_shape.dim_size(1);

    Tensor *z = nullptr, *mu = nullptr, *rsigma = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &z));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({row}), &mu));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({row}), &rsigma));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_, output_t, Tout,
        functor::Fp8LayerNormFwd<GPUDevice, Tin, Tweight, output_t>()(
            context, x.flat<Tin>().data(), gamma.flat<Tweight>().data(),
            beta.flat<Tweight>().data(), mu->flat<float>().data(),
            rsigma->flat<float>().data(), z->flat<Tout>().data(), z_amax_ptr,
            z_scale_ptr, epsilon_, row, col););
  }

 private:
  float epsilon_;
  std::string fp8_dtype_;
  int fp8_meta_index_;
};

template <typename Tin, typename Tgrad, typename Tweight, typename Tout>
class Fp8LayerNormGradOp : public OpKernel {
 public:
  explicit Fp8LayerNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype", &fp8_dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index", &fp8_meta_index_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& dz = context->input(0);
    const Tensor& x = context->input(1);
    const Tensor& mu = context->input(2);
    const Tensor& rsigma = context->input(3);
    const Tensor& gamma = context->input(4);
    const Tensor& dz_scale_inv = context->input(5);

    const float* dz_scale_inv_ptr = nullptr;
    if constexpr (std::is_same_v<Tgrad, int8>) {
      dz_scale_inv_ptr = dz_scale_inv.flat<float>().data() + fp8_meta_index_;
    }

    TensorShape x_shape = x.shape();
    int row = x_shape.dim_size(0), col = x_shape.dim_size(1);

    Tensor *dx = nullptr, *dgamma = nullptr, *dbeta = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &dx));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({col}), &dgamma));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({col}), &dbeta));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_, grad_t, Tgrad,
        functor::Fp8LayerNormBwd<GPUDevice, Tin, grad_t, Tweight, Tout>()(
            context, dz.flat<Tgrad>().data(), x.flat<Tin>().data(),
            mu.flat<float>().data(), rsigma.flat<float>().data(),
            gamma.flat<Tweight>().data(), dx->flat<Tout>().data(),
            dgamma->flat<Tweight>().data(), dbeta->flat<Tweight>().data(),
            dz_scale_inv_ptr, row, col););
  }

 private:
  std::string fp8_dtype_;
  int fp8_meta_index_;
};

#define REGISTER_FP8_LAYERNORM(Tin, Tweight, Tout)                     \
  REGISTER_KERNEL_BUILDER(Name("Fp8LayerNorm")                         \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<Tin>("in_dtype")         \
                              .TypeConstraint<Tweight>("weight_dtype") \
                              .TypeConstraint<Tout>("out_dtype"),      \
                          Fp8LayerNormOp<Tin, Tweight, Tout>);

REGISTER_FP8_LAYERNORM(float, float, int8);
REGISTER_FP8_LAYERNORM(Eigen::bfloat16, Eigen::bfloat16, int8);
#undef REGISTER_FP8_LAYERNORM

#define REGISTER_FP8_LAYERNORM_GRAD(Tin, Tgrad, Tweight, Tout)         \
  REGISTER_KERNEL_BUILDER(Name("Fp8LayerNormGrad")                     \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<Tin>("in_dtype")         \
                              .TypeConstraint<Tgrad>("grad_dtype")     \
                              .TypeConstraint<Tweight>("weight_dtype") \
                              .TypeConstraint<Tout>("out_dtype"),      \
                          Fp8LayerNormGradOp<Tin, Tgrad, Tweight, Tout>);

REGISTER_FP8_LAYERNORM_GRAD(float, int8, float, float);
REGISTER_FP8_LAYERNORM_GRAD(Eigen::bfloat16, int8, Eigen::bfloat16,
                            Eigen::bfloat16);
#undef REGISTER_FP8_LAYERNORM_GRAD

}  // namespace itex
