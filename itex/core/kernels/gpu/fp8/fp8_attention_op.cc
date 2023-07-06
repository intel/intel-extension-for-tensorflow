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

#include "itex/core/kernels/gpu/fp8/fp8_attention_bwd_gpu.h"
#include "itex/core/kernels/gpu/fp8/fp8_attention_fwd_gpu.h"

namespace itex {

template <typename T>
class Fp8ScaledDotProductAttentionOp : public OpKernel {
 public:
  explicit Fp8ScaledDotProductAttentionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype", &fp8_dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_q", &fp8_meta_index_q_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_k", &fp8_meta_index_k_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_v", &fp8_meta_index_v_));
    OP_REQUIRES_OK(context, context->GetAttr("fp8_meta_index_attn",
                                             &fp8_meta_index_attn_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_z", &fp8_meta_index_z_));
    OP_REQUIRES_OK(context, context->GetAttr("dropout_prob", &dropout_prob_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& query = context->input(0);
    const Tensor& key = context->input(1);
    const Tensor& value = context->input(2);
    const Tensor& qk_scale = context->input(3);
    const Tensor& attention_mask = context->input(4);
    const Tensor& dropout_mask = context->input(5);
    const Tensor& q_scale_inv = context->input(6);
    const Tensor& k_scale_inv = context->input(7);
    const Tensor& v_scale_inv = context->input(8);
    Tensor& attn_amax = const_cast<Tensor&>(context->input(9));
    const Tensor& attn_scale = context->input(10);
    const Tensor& attn_scale_inv = context->input(11);
    Tensor& z_amax = const_cast<Tensor&>(context->input(12));
    const Tensor& z_scale = context->input(13);

    const float *q_scale_inv_ptr, *k_scale_inv_ptr, *v_scale_inv_ptr;
    q_scale_inv_ptr = q_scale_inv.flat<float>().data() + fp8_meta_index_q_;
    k_scale_inv_ptr = k_scale_inv.flat<float>().data() + fp8_meta_index_k_;
    v_scale_inv_ptr = v_scale_inv.flat<float>().data() + fp8_meta_index_v_;

    float* attn_amax_ptr;
    const float *attn_scale_ptr, *attn_scale_inv_ptr;
    attn_amax_ptr = attn_amax.flat<float>().data() + fp8_meta_index_attn_;
    attn_scale_ptr = attn_scale.flat<float>().data() + fp8_meta_index_attn_;
    attn_scale_inv_ptr =
        attn_scale_inv.flat<float>().data() + fp8_meta_index_attn_;

    float* z_amax_ptr;
    const float* z_scale_ptr;
    z_amax_ptr = z_amax.flat<float>().data() + fp8_meta_index_z_;
    z_scale_ptr = z_scale.flat<float>().data() + fp8_meta_index_z_;

    auto q_shape = query.shape();
    int batch = q_shape.dim_size(0), num_attention_heads = q_shape.dim_size(1),
        seq_len = q_shape.dim_size(2), head_size = q_shape.dim_size(3);

    Tensor *z, *softmax, *attn;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch, seq_len, num_attention_heads, head_size}),
            &z));
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            1, TensorShape({batch, num_attention_heads, seq_len, seq_len}),
            &softmax));
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            2, TensorShape({batch, num_attention_heads, seq_len, seq_len}),
            &attn));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_, input_t, int8,
        functor::Fp8ScaledDotProductAttentionFwd<input_t, T, input_t>(
            context, query, key, value, qk_scale, attention_mask, dropout_mask,
            softmax, attn, z, q_scale_inv_ptr, k_scale_inv_ptr, v_scale_inv_ptr,
            attn_amax_ptr, attn_scale_ptr, attn_scale_inv_ptr, z_amax_ptr,
            z_scale_ptr, dropout_prob_, batch, num_attention_heads, seq_len,
            head_size););
  }

 private:
  std::string fp8_dtype_;
  int fp8_meta_index_q_;
  int fp8_meta_index_k_;
  int fp8_meta_index_v_;
  int fp8_meta_index_attn_;
  int fp8_meta_index_z_;
  float dropout_prob_;
};

template <typename T>
class Fp8ScaledDotProductAttentionGradOp : public OpKernel {
 public:
  explicit Fp8ScaledDotProductAttentionGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_dtype_forward", &fp8_dtype_forward_));
    OP_REQUIRES_OK(
        context, context->GetAttr("fp8_dtype_backward", &fp8_dtype_backward_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_dz", &fp8_meta_index_dz_));
    OP_REQUIRES_OK(context, context->GetAttr("fp8_meta_index_attn",
                                             &fp8_meta_index_attn_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_q", &fp8_meta_index_q_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_k", &fp8_meta_index_k_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_v", &fp8_meta_index_v_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index_dp", &fp8_meta_index_dp_));
    OP_REQUIRES_OK(context, context->GetAttr("dropout_prob", &dropout_prob_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& dz = context->input(0);
    const Tensor& query = context->input(1);
    const Tensor& key = context->input(2);
    const Tensor& value = context->input(3);
    const Tensor& qk_scale = context->input(4);
    const Tensor& softmax = context->input(5);
    const Tensor& dropout_mask = context->input(6);
    const Tensor& attn = context->input(7);
    const Tensor& dz_scale_inv = context->input(8);
    const Tensor& attn_scale_inv = context->input(9);
    const Tensor& q_scale_inv = context->input(10);
    const Tensor& k_scale_inv = context->input(11);
    const Tensor& v_scale_inv = context->input(12);
    Tensor& dp_amax = const_cast<Tensor&>(context->input(13));
    const Tensor& dp_scale = context->input(14);
    const Tensor& dp_scale_inv = context->input(15);

    const float* dz_scale_inv_ptr;
    dz_scale_inv_ptr = dz_scale_inv.flat<float>().data() + fp8_meta_index_dz_;

    const float* attn_scale_inv_ptr;
    attn_scale_inv_ptr =
        attn_scale_inv.flat<float>().data() + fp8_meta_index_attn_;

    const float *q_scale_inv_ptr, *k_scale_inv_ptr, *v_scale_inv_ptr;
    q_scale_inv_ptr = q_scale_inv.flat<float>().data() + fp8_meta_index_q_;
    k_scale_inv_ptr = k_scale_inv.flat<float>().data() + fp8_meta_index_k_;
    v_scale_inv_ptr = v_scale_inv.flat<float>().data() + fp8_meta_index_v_;

    float* dp_amax_ptr;
    const float *dp_scale_ptr, *dp_scale_inv_ptr;
    dp_amax_ptr = dp_amax.flat<float>().data() + fp8_meta_index_dp_;
    dp_scale_ptr = dp_scale.flat<float>().data() + fp8_meta_index_dp_;
    dp_scale_inv_ptr = dp_scale_inv.flat<float>().data() + fp8_meta_index_dp_;

    auto q_shape = query.shape();
    int batch = q_shape.dim_size(0), num_attention_heads = q_shape.dim_size(1),
        seq_len = q_shape.dim_size(2), head_size = q_shape.dim_size(3);

    Tensor *dq, *dk, *dv;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch, num_attention_heads, seq_len, head_size}),
            &dq));
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            1, TensorShape({batch, num_attention_heads, seq_len, head_size}),
            &dk));
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            2, TensorShape({batch, num_attention_heads, seq_len, head_size}),
            &dv));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_forward_, fp8_fwd_t, int8,
        FP8_TYPE_SWITCH(
            context, fp8_dtype_backward_, fp8_bwd_t, int8,
            functor::Fp8ScaledDotProductAttentionBwd<fp8_fwd_t, T, fp8_bwd_t>(
                context, dz, query, key, value, qk_scale, softmax, dropout_mask,
                attn, dq, dk, dv, dz_scale_inv_ptr, attn_scale_inv_ptr,
                q_scale_inv_ptr, k_scale_inv_ptr, v_scale_inv_ptr, dp_amax_ptr,
                dp_scale_ptr, dp_scale_inv_ptr, dropout_prob_, batch,
                num_attention_heads, seq_len, head_size);));
  }

 private:
  std::string fp8_dtype_forward_;
  std::string fp8_dtype_backward_;
  int fp8_meta_index_dz_;
  int fp8_meta_index_attn_;
  int fp8_meta_index_q_;
  int fp8_meta_index_k_;
  int fp8_meta_index_v_;
  int fp8_meta_index_dp_;
  float dropout_prob_;
};

#define REGISTER_FP8_ATTENTION(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("Fp8ScaledDotProductAttention")     \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          Fp8ScaledDotProductAttentionOp<T>);      \
  REGISTER_KERNEL_BUILDER(Name("Fp8ScaledDotProductAttentionGrad") \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          Fp8ScaledDotProductAttentionGradOp<T>);

REGISTER_FP8_ATTENTION(float);
REGISTER_FP8_ATTENTION(Eigen::bfloat16);
#undef REGISTER_FP8_MATMUL

}  // namespace itex
