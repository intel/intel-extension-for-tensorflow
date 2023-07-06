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

#include "itex/core/kernels/gpu/fp8/fp8_matmul_gpu.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Tsum, typename Tout>
class Fp8MatmulOp : public OpKernel {
 public:
  explicit Fp8MatmulOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype_a", &fp8_dtype_a_));
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype_b", &fp8_dtype_b_));
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype_c", &fp8_dtype_c_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_a", &fp8_meta_index_a_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_b", &fp8_meta_index_b_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_c", &fp8_meta_index_c_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(context, context->GetAttr("use_bias", &use_bias_));
    OP_REQUIRES_OK(context, context->GetAttr("has_post_add", &has_post_add_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& src = context->input(0);
    const Tensor& weight = context->input(1);
    const Tensor& bias = context->input(2);
    const Tensor& post_add = context->input(3);
    const Tensor& a_scale_inv = context->input(4);
    const Tensor& b_scale_inv = context->input(5);
    Tensor& c_amax = const_cast<Tensor&>(context->input(6));
    const Tensor& c_scale = context->input(7);

    // Inputs fp8 meta
    const float *src_scale_inv_ptr = nullptr, *weight_scale_inv_ptr = nullptr;
    src_scale_inv_ptr = a_scale_inv.flat<float>().data() + fp8_meta_index_a_;
    weight_scale_inv_ptr = b_scale_inv.flat<float>().data() + fp8_meta_index_b_;

    // Output fp8 meta
    const float* dst_scale_ptr = nullptr;
    float* dst_amax_ptr = nullptr;
    if (std::is_same_v<Tout, int8>) {
      dst_amax_ptr = c_amax.flat<float>().data() + fp8_meta_index_c_;
      dst_scale_ptr = c_scale.flat<float>().data() + fp8_meta_index_c_;
    }

    int batch = transpose_a_ ? src.dim_size(1) : src.dim_size(0);
    int feature = transpose_b_ ? weight.dim_size(0) : weight.dim_size(1);

    Tensor* dst = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch, feature}), &dst));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_a_, input_t, int8,
        FP8_TYPE_SWITCH(
            context, fp8_dtype_b_, weight_t, int8,
            FP8_TYPE_SWITCH(
                context, fp8_dtype_c_, output_t, Tout,
                functor::Fp8Matmul<input_t, weight_t, Tsum, output_t>(
                    context, src, src_scale_inv_ptr, weight,
                    weight_scale_inv_ptr, bias, post_add, dst, dst_amax_ptr,
                    dst_scale_ptr, use_bias_, has_post_add_, transpose_a_,
                    transpose_b_);)));
  }

 private:
  std::string fp8_dtype_a_;
  std::string fp8_dtype_b_;
  std::string fp8_dtype_c_;
  int fp8_meta_index_a_;
  int fp8_meta_index_b_;
  int fp8_meta_index_c_;
  bool transpose_a_;
  bool transpose_b_;
  bool use_bias_;
  bool has_post_add_;
};

#define REGISTER_FP8_MATMUL(Tsum, Tout)                           \
  REGISTER_KERNEL_BUILDER(Name("Fp8Matmul")                       \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<Tsum>("sum_dtype")  \
                              .TypeConstraint<Tout>("out_dtype"), \
                          Fp8MatmulOp<Tsum, Tout>);

REGISTER_FP8_MATMUL(float, int8);
REGISTER_FP8_MATMUL(float, float);
REGISTER_FP8_MATMUL(Eigen::bfloat16, int8);
REGISTER_FP8_MATMUL(Eigen::bfloat16, Eigen::bfloat16);
#undef REGISTER_FP8_MATMUL

}  // namespace itex
