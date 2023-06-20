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

#ifndef ITEX_CORE_KERNELS_GPU_FP8_FP8_ATTENTION_FWD_GPU_H_
#define ITEX_CORE_KERNELS_GPU_FP8_FP8_ATTENTION_FWD_GPU_H_

#include <unordered_map>

#include "itex/core/kernels/common/cwise_ops_common.h"
#include "itex/core/kernels/gpu/fp8/fp8_quantize_gpu.h"
#include "itex/core/utils/onednn/onednn_util.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

using dnnl::memory;

namespace functor {

template <typename input_t, typename activate_t, typename output_t>
void Fp8ScaledDotProductAttentionFwd(
    OpKernelContext* ctx, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& qk_scale, const Tensor& attention_mask,
    const Tensor& dropout_mask, Tensor* softmax, Tensor* attn, Tensor* z,
    const float* q_scale_inv, const float* k_scale_inv,
    const float* v_scale_inv, float* attn_amax, const float* attn_scale,
    const float* attn_scale_inv, float* z_amax, const float* z_scale,
    float dropout_prob, int batch, int num_attention_heads, int seq_len,
    int head_size) {
  int hidden_size = num_attention_heads * head_size;
  int n = batch * seq_len * hidden_size;
  auto dnnl_engine = CreateDnnlEngine<GPUDevice>(*ctx);
  auto dnnl_stream = CreateDnnlStream(*ctx, dnnl_engine);
  Tensor query_temp, key_temp, value_temp;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DataTypeToEnum<activate_t>::v(),
               TensorShape({batch, num_attention_heads, seq_len, head_size}),
               &query_temp));
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DataTypeToEnum<activate_t>::v(),
               TensorShape({batch, num_attention_heads, seq_len, head_size}),
               &key_temp));
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DataTypeToEnum<activate_t>::v(),
               TensorShape({batch, num_attention_heads, seq_len, head_size}),
               &value_temp));

  // Quantize fp8 to activate_t
  Fp8Dequantize<input_t, activate_t>(ctx, query.flat<int8>().data(),
                                     query_temp.flat<activate_t>().data(),
                                     q_scale_inv, n);
  Fp8Dequantize<input_t, activate_t>(ctx, key.flat<int8>().data(),
                                     key_temp.flat<activate_t>().data(),
                                     k_scale_inv, n);

  // First BMM
  memory::dims qkbmm_src_dims = {batch, num_attention_heads, seq_len,
                                 head_size};
  memory::dims qkbmm_weight_dims = {batch, num_attention_heads, head_size,
                                    seq_len};
  memory::dims qkbmm_dst_dims = {batch, num_attention_heads, seq_len, seq_len};

  memory::dims qkbmm_src_strides = {hidden_size * seq_len, seq_len * head_size,
                                    head_size, 1};
  memory::dims qkbmm_weight_strides = {hidden_size * seq_len,
                                       seq_len * head_size, 1, head_size};
  memory::dims qkbmm_dst_strides = {num_attention_heads * seq_len * seq_len,
                                    seq_len * seq_len, seq_len, 1};

  auto qkbmm_src_md =
      memory::desc(qkbmm_src_dims, OneDnnType<activate_t>(), qkbmm_src_strides);
  auto qkbmm_weight_md = memory::desc(
      qkbmm_weight_dims, OneDnnType<activate_t>(), qkbmm_weight_strides);
  auto qkbmm_dst_md =
      memory::desc(qkbmm_dst_dims, OneDnnType<activate_t>(), qkbmm_dst_strides);

  dnnl::primitive_attr post_ops_attr;
  auto qkbmm_pd = dnnl::matmul::primitive_desc(dnnl_engine, qkbmm_src_md,
                                               qkbmm_weight_md, qkbmm_dst_md);

  auto qkbmm_src_mem = CreateDnnlMemory(
      qkbmm_src_md, dnnl_engine, GetTensorBuffer<activate_t>(&query_temp));
  auto qkbmm_weight_mem = CreateDnnlMemory(
      qkbmm_weight_md, dnnl_engine, GetTensorBuffer<activate_t>(&key_temp));
  auto qkbmm_dst_mem = CreateDnnlMemory(qkbmm_dst_md, dnnl_engine,
                                        GetTensorBuffer<activate_t>(softmax));
  std::unordered_map<int, dnnl::memory> qkbmm_args;
  qkbmm_args.emplace(DNNL_ARG_SRC, qkbmm_src_mem);
  qkbmm_args.emplace(DNNL_ARG_WEIGHTS, qkbmm_weight_mem);
  qkbmm_args.emplace(DNNL_ARG_DST, qkbmm_dst_mem);
  auto qkbmm_prim = dnnl::matmul(qkbmm_pd);
  qkbmm_prim.execute(dnnl_stream, qkbmm_args);

  functor::BinaryFunctor<GPUDevice, functor::mul<activate_t>, 1, false>().Right(
      ctx->eigen_device<GPUDevice>(), softmax->flat<activate_t>(),
      (const_cast<const Tensor*>(softmax))->flat<activate_t>(),
      qk_scale.scalar<activate_t>(), nullptr);

  // Attention mask
  memory::dims mask_src_0_dims =
      memory::dims({batch, num_attention_heads, seq_len, seq_len});
  memory::dims mask_src_1_dims = memory::dims({batch, 1, seq_len, seq_len});

  memory::dims mask_src_0_strides = {num_attention_heads * seq_len * seq_len,
                                     seq_len * seq_len, seq_len, 1};
  memory::dims mask_src_1_strides = {seq_len * seq_len, seq_len * seq_len,
                                     seq_len, 1};

  auto mask_src_0_md = memory::desc(mask_src_0_dims, OneDnnType<activate_t>(),
                                    mask_src_0_strides);
  auto mask_src_1_md = memory::desc(mask_src_1_dims, OneDnnType<activate_t>(),
                                    mask_src_1_strides);

  auto mask_src_0_mem = CreateDnnlMemory(mask_src_0_md, dnnl_engine,
                                         GetTensorBuffer<activate_t>(softmax));
  auto mask_src_1_mem = CreateDnnlMemory(
      mask_src_1_md, dnnl_engine, GetTensorBuffer<activate_t>(&attention_mask));
  auto mask_dst_mem = CreateDnnlMemory(mask_src_0_md, dnnl_engine,
                                       GetTensorBuffer<activate_t>(softmax));
  auto mask_pd =
      dnnl::binary::primitive_desc(dnnl_engine, dnnl::algorithm::binary_add,
                                   mask_src_0_md, mask_src_1_md, mask_src_0_md);
  auto mask_prim = dnnl::binary(mask_pd);
  mask_prim.execute(dnnl_stream, {{DNNL_ARG_SRC_0, mask_src_0_mem},
                                  {DNNL_ARG_SRC_1, mask_src_1_mem},
                                  {DNNL_ARG_DST, mask_dst_mem}});

  // Softmax input is mask output.
  dnnl::primitive_attr softmax_attr;
  softmax_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto softmax_pd = dnnl::softmax_forward::primitive_desc(
      dnnl_engine, dnnl::prop_kind::forward_training,
      dnnl::algorithm::softmax_accurate, mask_src_0_md, mask_src_0_md,
      3 /*axis*/, softmax_attr);
  Tensor scratchpad_tensor;
  int scratchpad_size =
      softmax_pd.scratchpad_desc().get_size() / sizeof(activate_t);
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                                         TensorShape({scratchpad_size}),
                                         &scratchpad_tensor));
  auto scratchpad_mem =
      CreateDnnlMemory(softmax_pd.scratchpad_desc(), dnnl_engine,
                       GetTensorBuffer<activate_t>(&scratchpad_tensor));
  auto softmax_prim = dnnl::softmax_forward(softmax_pd);
  softmax_prim.execute(dnnl_stream, {{DNNL_ARG_SRC, mask_dst_mem},
                                     {DNNL_ARG_DST, mask_dst_mem},
                                     {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  // Dropout
  int softmax_size = softmax->NumElements();
  Tensor dropout_scale, vbmm_src;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                                         TensorShape({1}), &dropout_scale));
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                                    TensorShape({softmax_size}), &vbmm_src));
  To32Bit(dropout_scale.flat<activate_t>())
      .device(ctx->eigen_device<GPUDevice>()) =
      To32Bit(dropout_scale.flat<activate_t>())
          .constant(static_cast<activate_t>(1.0f / (1.0f - dropout_prob)));
  Eigen::array<Eigen::DenseIndex, 1> bcast0, bcast1;
  bcast0[0] = 1;
  bcast1[1] = softmax_size;
  functor::BinaryFunctor<GPUDevice, functor::mul<activate_t>, 1, false>().BCast(
      ctx->eigen_device<GPUDevice>(), vbmm_src.flat<activate_t>(),
      (const_cast<const Tensor*>(softmax))->flat<activate_t>(), bcast0,
      (const_cast<const Tensor&>(dropout_scale)).flat<activate_t>(), bcast1,
      nullptr);
  functor::BinaryFunctor<GPUDevice, functor::mul<activate_t>, 1, false>()(
      ctx->eigen_device<GPUDevice>(), vbmm_src.flat<activate_t>(),
      (const_cast<const Tensor&>(vbmm_src)).flat<activate_t>(),
      dropout_mask.flat<activate_t>(), nullptr);

  Fp8Quantize<activate_t, output_t>(ctx, vbmm_src.flat<activate_t>().data(),
                                    attn->flat<int8>().data(), attn_amax,
                                    attn_scale, softmax_size);
  Fp8Dequantize<output_t, activate_t>(ctx, attn->flat<int8>().data(),
                                      vbmm_src.flat<activate_t>().data(),
                                      attn_scale_inv, softmax_size);
  Fp8Dequantize<input_t, activate_t>(ctx, value.flat<int8>().data(),
                                     value_temp.flat<activate_t>().data(),
                                     v_scale_inv, n);

  Tensor z_temp;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                                    TensorShape({z->NumElements()}), &z_temp));

  // Second BMM
  memory::dims vbmm_weight_dims = {batch, num_attention_heads, seq_len,
                                   head_size};
  memory::dims vbmm_dst_dims = {batch, num_attention_heads, seq_len, head_size};

  memory::dims vbmm_weight_strides = {hidden_size * seq_len,
                                      seq_len * head_size, head_size, 1};
  memory::dims vbmm_dst_strides = {seq_len * hidden_size, head_size,
                                   hidden_size, 1};

  auto vbmm_weight_md = memory::desc(vbmm_weight_dims, OneDnnType<activate_t>(),
                                     vbmm_weight_strides);
  auto vbmm_dst_md =
      memory::desc(vbmm_dst_dims, OneDnnType<activate_t>(), vbmm_dst_strides);

  auto vbmm_weight_mem = CreateDnnlMemory(
      vbmm_weight_md, dnnl_engine, GetTensorBuffer<activate_t>(&value_temp));
  auto vbmm_src_mem = CreateDnnlMemory(qkbmm_dst_md, dnnl_engine,
                                       GetTensorBuffer<activate_t>(&vbmm_src));
  auto vbmm_dst_mem = CreateDnnlMemory(vbmm_dst_md, dnnl_engine,
                                       GetTensorBuffer<activate_t>(&z_temp));

  auto vbmm_pd = dnnl::matmul::primitive_desc(dnnl_engine, qkbmm_dst_md,
                                              vbmm_weight_md, vbmm_dst_md);
  auto vbmm_prim = dnnl::matmul(vbmm_pd);
  vbmm_prim.execute(dnnl_stream, {{DNNL_ARG_SRC, vbmm_src_mem},
                                  {DNNL_ARG_WEIGHTS, vbmm_weight_mem},
                                  {DNNL_ARG_DST, vbmm_dst_mem}});

  Fp8Quantize<activate_t, output_t>(ctx, z_temp.flat<activate_t>().data(),
                                    z->flat<int8>().data(), z_amax, z_scale,
                                    z->NumElements());
}

}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FP8_FP8_ATTENTION_FWD_GPU_H_
