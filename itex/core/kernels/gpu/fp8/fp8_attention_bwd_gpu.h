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

#ifndef ITEX_CORE_KERNELS_GPU_FP8_FP8_ATTENTION_BWD_GPU_H_
#define ITEX_CORE_KERNELS_GPU_FP8_FP8_ATTENTION_BWD_GPU_H_

#include "itex/core/kernels/common/cwise_ops_common.h"
#include "itex/core/kernels/gpu/fp8/fp8_quantize_gpu.h"
#include "itex/core/kernels/gpu/reduction_ops_common.h"
#include "itex/core/utils/onednn/onednn_util.h"

namespace itex {

using dnnl::memory;

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename fp8_fwd_t, typename activate_t, typename fp8_bwd_t>
void Fp8ScaledDotProductAttentionBwd(
    OpKernelContext* ctx, const Tensor& dz, const Tensor& query,
    const Tensor& key, const Tensor& value, const Tensor& qk_scale,
    const Tensor& softmax, const Tensor& dropout_mask, const Tensor& attn,
    Tensor* dq, Tensor* dk, Tensor* dv, const float* dz_scale_inv,
    const float* attn_scale_inv, const float* q_scale_inv,
    const float* k_scale_inv, const float* v_scale_inv, float* dp_amax,
    const float* dp_scale, const float* dp_scale_inv, float dropout_prob,
    int batch, int num_attention_heads, int seq_len, int head_size) {
  int hidden_size = num_attention_heads * head_size;
  auto dnnl_engine = CreateDnnlEngine<GPUDevice>(*ctx);
  auto dnnl_stream = CreateDnnlStream(*ctx, dnnl_engine);

  Tensor dz_temp, attn_temp;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                                    TensorShape({dz.NumElements()}), &dz_temp));
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                              TensorShape({attn.NumElements()}), &attn_temp));
  Fp8Dequantize<fp8_bwd_t, activate_t>(ctx, dz.flat<int8>().data(),
                                       dz_temp.flat<activate_t>().data(),
                                       dz_scale_inv, dz.NumElements());
  Fp8Dequantize<fp8_fwd_t, activate_t>(ctx, attn.flat<int8>().data(),
                                       attn_temp.flat<activate_t>().data(),
                                       attn_scale_inv, attn.NumElements());

  // Backward for second BMM
  memory::dims dv_src_dims = {batch, num_attention_heads, seq_len, seq_len};
  memory::dims dv_src_strides = {num_attention_heads * seq_len * seq_len,
                                 seq_len * seq_len, 1, seq_len};
  memory::dims dv_weight_dims = {batch, num_attention_heads, seq_len,
                                 head_size};
  memory::dims dv_weight_strides = {seq_len * hidden_size, head_size,
                                    hidden_size, 1};
  memory::dims dv_dst_dims = {batch, num_attention_heads, seq_len, head_size};
  memory::dims dv_dst_strides = {hidden_size * seq_len, head_size * seq_len,
                                 head_size, 1};

  auto dv_src_md =
      memory::desc(dv_src_dims, OneDnnType<activate_t>(), dv_src_strides);
  auto dv_src_mem = CreateDnnlMemory(dv_src_md, dnnl_engine,
                                     GetTensorBuffer<activate_t>(&attn_temp));
  auto dv_weight_md =
      memory::desc(dv_weight_dims, OneDnnType<activate_t>(), dv_weight_strides);
  auto dv_weight_mem = CreateDnnlMemory(dv_weight_md, dnnl_engine,
                                        GetTensorBuffer<activate_t>(&dz_temp));
  auto dv_dst_md =
      memory::desc(dv_dst_dims, OneDnnType<activate_t>(), dv_dst_strides);
  auto dv_dst_mem =
      CreateDnnlMemory(dv_dst_md, dnnl_engine, GetTensorBuffer<activate_t>(dv));

  auto dv_matmul_pd = dnnl::matmul::primitive_desc(dnnl_engine, dv_src_md,
                                                   dv_weight_md, dv_dst_md);
  auto dv_matmul_prim = dnnl::matmul(dv_matmul_pd);
  dv_matmul_prim.execute(dnnl_stream, {{DNNL_ARG_SRC, dv_src_mem},
                                       {DNNL_ARG_WEIGHTS, dv_weight_mem},
                                       {DNNL_ARG_DST, dv_dst_mem}});

  memory::dims ds_src_dims = {batch, num_attention_heads, seq_len, head_size};
  memory::dims ds_src_strides = {seq_len * hidden_size, head_size, hidden_size,
                                 1};
  memory::dims ds_weight_dims = {batch, num_attention_heads, head_size,
                                 seq_len};
  memory::dims ds_weight_strides = {seq_len * hidden_size, head_size * seq_len,
                                    1, head_size};
  memory::dims ds_dst_dims = {batch, num_attention_heads, seq_len, seq_len};
  memory::dims ds_dst_strides = {num_attention_heads * seq_len * seq_len,
                                 seq_len * seq_len, seq_len, 1};

  auto ds_src_md =
      memory::desc(ds_src_dims, OneDnnType<activate_t>(), ds_src_strides);
  auto ds_src_mem = CreateDnnlMemory(ds_src_md, dnnl_engine,
                                     GetTensorBuffer<activate_t>(&dz_temp));

  Tensor v_temp;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                              TensorShape({value.NumElements()}), &v_temp));
  Fp8Dequantize<fp8_fwd_t, activate_t>(ctx, value.flat<int8>().data(),
                                       v_temp.flat<activate_t>().data(),
                                       v_scale_inv, value.NumElements());

  auto ds_weight_md =
      memory::desc(ds_weight_dims, OneDnnType<activate_t>(), ds_weight_strides);
  auto ds_weight_mem = CreateDnnlMemory(ds_weight_md, dnnl_engine,
                                        GetTensorBuffer<activate_t>(&v_temp));

  Tensor dp;
  int softmax_size = softmax.NumElements();
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                                         TensorShape({softmax_size}), &dp));
  auto ds_dst_md =
      memory::desc(ds_dst_dims, OneDnnType<activate_t>(), ds_dst_strides);
  auto ds_dst_mem = CreateDnnlMemory(ds_dst_md, dnnl_engine,
                                     GetTensorBuffer<activate_t>(&dp));

  auto ds_matmul_pd = dnnl::matmul::primitive_desc(dnnl_engine, ds_src_md,
                                                   ds_weight_md, ds_dst_md);
  auto ds_matmul_prim = dnnl::matmul(ds_matmul_pd);
  ds_matmul_prim.execute(dnnl_stream, {{DNNL_ARG_SRC, ds_src_mem},
                                       {DNNL_ARG_WEIGHTS, ds_weight_mem},
                                       {DNNL_ARG_DST, ds_dst_mem}});

  // Backward for dropout
  functor::BinaryFunctor<GPUDevice, functor::mul<activate_t>, 1, false>()(
      ctx->eigen_device<GPUDevice>(), dp.flat<activate_t>(),
      (const_cast<const Tensor&>(dp)).flat<activate_t>(),
      dropout_mask.flat<activate_t>(), nullptr);
  Tensor dropout_scale;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                                         TensorShape({1}), &dropout_scale));
  To32Bit(dropout_scale.flat<activate_t>())
      .device(ctx->eigen_device<GPUDevice>()) =
      To32Bit(dropout_scale.flat<activate_t>())
          .constant(static_cast<activate_t>(1.0f / (1.0f - dropout_prob)));
  Eigen::array<Eigen::DenseIndex, 1> dropout_bcast0, dropout_bcast1;
  dropout_bcast0[0] = 1;
  dropout_bcast1[1] = softmax_size;
  functor::BinaryFunctor<GPUDevice, functor::mul<activate_t>, 1, false>().BCast(
      ctx->eigen_device<GPUDevice>(), dp.flat<activate_t>(),
      (const_cast<const Tensor&>(dp)).flat<activate_t>(), dropout_bcast0,
      (const_cast<const Tensor&>(dropout_scale)).flat<activate_t>(),
      dropout_bcast1, nullptr);

  // Backward for softmax
  Tensor softmax_workspace0;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DataTypeToEnum<activate_t>::v(),
               TensorShape({batch * num_attention_heads * seq_len, seq_len}),
               &softmax_workspace0));
  functor::BinaryFunctor<GPUDevice, functor::mul<activate_t>, 1, false>()(
      ctx->eigen_device<GPUDevice>(), softmax_workspace0.flat<activate_t>(),
      (const_cast<const Tensor&>(dp)).flat<activate_t>(),
      softmax.flat<activate_t>(), nullptr);
  Tensor softmax_workspace1;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                          DataTypeToEnum<activate_t>::v(),
                          TensorShape({batch * num_attention_heads * seq_len}),
                          &softmax_workspace1));
  functor::ReduceFunctor<Eigen::internal::SumReducer<activate_t>>::Reduce(
      ctx, softmax_workspace1.tensor<activate_t, 1>(),
      softmax_workspace0.tensor<activate_t, 2>(), ReduceAxies<1, false>().value,
      Eigen::internal::SumReducer<activate_t>());
  softmax_workspace1.set_shape(
      TensorShape({batch * num_attention_heads * seq_len, 1}));
  dp.set_shape(TensorShape({batch * num_attention_heads * seq_len, seq_len}));
  Eigen::array<Eigen::DenseIndex, 2> softmax_bcast0, softmax_bcast1;
  softmax_bcast0[0] = softmax_bcast0[1] = softmax_bcast1[0] = 1;
  softmax_bcast1[1] = seq_len;
  functor::BinaryFunctor<GPUDevice, functor::sub<activate_t>, 2, false>().BCast(
      ctx->eigen_device<GPUDevice>(), dp.tensor<activate_t, 2>(),
      (const_cast<const Tensor&>(dp)).tensor<activate_t, 2>(), softmax_bcast0,
      (const_cast<const Tensor&>(softmax_workspace1)).tensor<activate_t, 2>(),
      softmax_bcast1, nullptr);
  functor::BinaryFunctor<GPUDevice, functor::mul<activate_t>, 1, false>()(
      ctx->eigen_device<GPUDevice>(), dp.flat<activate_t>(),
      (const_cast<const Tensor&>(dp)).flat<activate_t>(),
      softmax.flat<activate_t>(), nullptr);

  functor::BinaryFunctor<GPUDevice, functor::mul<activate_t>, 1, false>().Right(
      ctx->eigen_device<GPUDevice>(), dp.flat<activate_t>(),
      (const_cast<const Tensor&>(dp)).flat<activate_t>(),
      qk_scale.scalar<activate_t>(), nullptr);

  Tensor dp_fp8;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<int8>::v(),
                                         TensorShape({softmax_size}), &dp_fp8));
  Fp8Quantize<activate_t, fp8_bwd_t>(ctx, dp.flat<activate_t>().data(),
                                     dp_fp8.flat<int8>().data(), dp_amax,
                                     dp_scale, dp.NumElements());
  Fp8Dequantize<fp8_bwd_t, activate_t>(ctx, dp_fp8.flat<int8>().data(),
                                       dp.flat<activate_t>().data(),
                                       dp_scale_inv, dp.NumElements());

  // Backward for first BMM
  memory::dims dq_src_dims = {batch, num_attention_heads, seq_len, seq_len};
  memory::dims dq_src_strides = {num_attention_heads * seq_len * seq_len,
                                 seq_len * seq_len, seq_len, 1};
  memory::dims dq_weight_dims = {batch, num_attention_heads, seq_len,
                                 head_size};
  memory::dims dq_weight_strides = {hidden_size * seq_len, seq_len * head_size,
                                    head_size, 1};
  memory::dims dq_dst_dims = {batch, num_attention_heads, seq_len, head_size};
  memory::dims dq_dst_strides = {hidden_size * seq_len, head_size * seq_len,
                                 head_size, 1};

  auto dq_src_md =
      memory::desc(dq_src_dims, OneDnnType<activate_t>(), dq_src_strides);
  auto dq_src_mem = CreateDnnlMemory(dq_src_md, dnnl_engine,
                                     GetTensorBuffer<activate_t>(&dp));
  auto dq_weight_md =
      memory::desc(dq_weight_dims, OneDnnType<activate_t>(), dq_weight_strides);
  Tensor k_temp;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                                    TensorShape({key.NumElements()}), &k_temp));
  Fp8Dequantize<fp8_fwd_t, activate_t>(ctx, key.flat<int8>().data(),
                                       k_temp.flat<activate_t>().data(),
                                       k_scale_inv, key.NumElements());
  auto dq_weight_mem = CreateDnnlMemory(dq_weight_md, dnnl_engine,
                                        GetTensorBuffer<activate_t>(&k_temp));
  auto dq_dst_md =
      memory::desc(dq_dst_dims, OneDnnType<activate_t>(), dq_dst_strides);
  auto dq_dst_mem =
      CreateDnnlMemory(dq_dst_md, dnnl_engine, GetTensorBuffer<activate_t>(dq));

  auto dq_matmul_pd = dnnl::matmul::primitive_desc(dnnl_engine, dq_src_md,
                                                   dq_weight_md, dq_dst_md);
  auto dq_matmul_prim = dnnl::matmul(dq_matmul_pd);
  dq_matmul_prim.execute(dnnl_stream, {{DNNL_ARG_SRC, dq_src_mem},
                                       {DNNL_ARG_WEIGHTS, dq_weight_mem},
                                       {DNNL_ARG_DST, dq_dst_mem}});

  memory::dims dk_src_dims = {batch, num_attention_heads, seq_len, seq_len};
  memory::dims dk_src_strides = {num_attention_heads * seq_len * seq_len,
                                 seq_len * seq_len, 1, seq_len};
  memory::dims dk_weight_dims = {batch, num_attention_heads, seq_len,
                                 head_size};
  memory::dims dk_weight_strides = {hidden_size * seq_len, seq_len * head_size,
                                    head_size, 1};
  memory::dims dk_dst_dims = {batch, num_attention_heads, seq_len, head_size};
  memory::dims dk_dst_strides = {hidden_size * seq_len, head_size * seq_len,
                                 head_size, 1};

  auto dk_src_md =
      memory::desc(dk_src_dims, OneDnnType<activate_t>(), dk_src_strides);
  auto dk_src_mem = CreateDnnlMemory(dk_src_md, dnnl_engine,
                                     GetTensorBuffer<activate_t>(&dp));
  Tensor q_temp;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DataTypeToEnum<activate_t>::v(),
                              TensorShape({query.NumElements()}), &q_temp));
  Fp8Dequantize<fp8_fwd_t, activate_t>(ctx, query.flat<int8>().data(),
                                       q_temp.flat<activate_t>().data(),
                                       q_scale_inv, query.NumElements());
  auto dk_weight_md =
      memory::desc(dk_weight_dims, OneDnnType<activate_t>(), dk_weight_strides);
  auto dk_weight_mem = CreateDnnlMemory(dk_weight_md, dnnl_engine,
                                        GetTensorBuffer<activate_t>(&q_temp));
  auto dk_dst_md =
      memory::desc(dk_dst_dims, OneDnnType<activate_t>(), dk_dst_strides);

  auto dk_dst_mem =
      CreateDnnlMemory(dk_dst_md, dnnl_engine, GetTensorBuffer<activate_t>(dk));

  auto dk_matmul_pd = dnnl::matmul::primitive_desc(dnnl_engine, dk_src_md,
                                                   dk_weight_md, dk_dst_md);
  auto dk_matmul_prim = dnnl::matmul(dk_matmul_pd);
  dk_matmul_prim.execute(dnnl_stream, {{DNNL_ARG_SRC, dk_src_mem},
                                       {DNNL_ARG_WEIGHTS, dk_weight_mem},
                                       {DNNL_ARG_DST, dk_dst_mem}});
}

}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FP8_FP8_ATTENTION_BWD_GPU_H_
