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

#include "itex/core/kernels/gpu/fp8/fp8_quantize_fusion_gpu.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
class Fp8QuantizeDbiasOp : public OpKernel {
 public:
  explicit Fp8QuantizeDbiasOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype", &fp8_dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index", &fp8_meta_index_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grad = context->input(0);
    const Tensor& amax = context->input(1);
    const Tensor& scale = context->input(2);

    auto amax_ptr =
        const_cast<float*>(amax.flat<float>().data()) + fp8_meta_index_;
    auto scale_ptr = scale.flat<float>().data() + fp8_meta_index_;

    auto grad_shape = grad.shape();
    int row = grad_shape.dim_size(0), col = grad_shape.dim_size(1);

    Tensor *quantize_out, *dbias;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({row, col}),
                                                     &quantize_out));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({col}), &dbias));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_, output_t, int8,
        functor::Fp8QuantizeDbiasFused<T, output_t>(
            context, grad.flat<T>().data(), quantize_out->flat<int8>().data(),
            dbias->flat<T>().data(), amax_ptr, scale_ptr, row, col););
  }

 private:
  std::string fp8_dtype_;
  int fp8_meta_index_;
};

template <typename T>
class Fp8QuantizeDbiasDgeluOp : public OpKernel {
 public:
  explicit Fp8QuantizeDbiasDgeluOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype", &fp8_dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index", &fp8_meta_index_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grad = context->input(0);
    const Tensor& gelu_inp = context->input(1);
    const Tensor& amax = context->input(2);
    const Tensor& scale = context->input(3);

    auto amax_ptr =
        const_cast<float*>(amax.flat<float>().data()) + fp8_meta_index_;
    auto scale_ptr = scale.flat<float>().data() + fp8_meta_index_;

    auto grad_shape = grad.shape();
    int row = grad_shape.dim_size(0), col = grad_shape.dim_size(1);

    Tensor *dgelu, *dbias;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({row, col}), &dgelu));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({col}), &dbias));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_, output_t, int8,
        functor::Fp8QuantizeDbiasDgeluFused<T, output_t>(
            context, grad.flat<T>().data(), gelu_inp.flat<T>().data(),
            dbias->flat<T>().data(), dgelu->flat<int8>().data(), amax_ptr,
            scale_ptr, row, col););
  }

 private:
  std::string fp8_dtype_;
  int fp8_meta_index_;
};

#define REGISTER_FP8_QUANTIZE_FUSION(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Fp8QuantizeDbias")              \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<T>("grad_dtype"), \
                          Fp8QuantizeDbiasOp<T>);               \
  REGISTER_KERNEL_BUILDER(Name("Fp8QuantizeDbiasDgelu")         \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<T>("grad_dtype")  \
                              .TypeConstraint<T>("in_dtype"),   \
                          Fp8QuantizeDbiasDgeluOp<T>);

REGISTER_FP8_QUANTIZE_FUSION(float);
REGISTER_FP8_QUANTIZE_FUSION(Eigen::bfloat16);
#undef REGISTER_FP8_QUANTIZE_FUSION

}  // namespace itex
