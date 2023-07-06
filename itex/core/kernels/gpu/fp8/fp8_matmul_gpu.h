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

#ifndef ITEX_CORE_KERNELS_GPU_FP8_FP8_MATMUL_GPU_H_
#define ITEX_CORE_KERNELS_GPU_FP8_FP8_MATMUL_GPU_H_

#include <string>
#include <unordered_map>
#include <utility>

#include "itex/core/kernels/gpu/fp8/fp8_quantize_gpu.h"
#include "itex/core/utils/onednn/onednn_util.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename input_t, typename weight_t, typename sum_t,
          typename output_t>
void Fp8Matmul(OpKernelContext* ctx, const Tensor& src,
               const float* src_scale_inv, const Tensor& weight,
               const float* weight_scale_inv, const Tensor& bias,
               const Tensor& post_add, Tensor* dst, float* dst_amax,
               const float* dst_scale, bool use_bias, bool has_post_add,
               bool transpose_a, bool transpose_b) {
  // Quantize fp8 to bias type
  Tensor src_temp, weight_temp;
  sum_t* dst_temp_ptr = nullptr;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DataTypeToEnum<sum_t>::v(),
                              TensorShape({src.NumElements()}), &src_temp));
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<sum_t>::v(),
                                         TensorShape({weight.NumElements()}),
                                         &weight_temp));

  Fp8Dequantize<input_t, sum_t>(ctx, src.flat<int8>().data(),
                                src_temp.flat<sum_t>().data(), src_scale_inv,
                                src.NumElements());
  Fp8Dequantize<weight_t, sum_t>(ctx, weight.flat<int8>().data(),
                                 weight_temp.flat<sum_t>().data(),
                                 weight_scale_inv, weight.NumElements());

  auto dnnl_engine = CreateDnnlEngine<GPUDevice>(*ctx);
  auto dnnl_stream = CreateDnnlStream(*ctx, dnnl_engine);

  try {
    auto src_dims = TFShapeToOneDnnDims(src.shape());
    auto src_strides = CalculateTFStrides(src_dims);
    int idx_last = src.dims() - 1;
    int idx_2nd_last = src.dims() - 2;
    if (transpose_a) {
      std::swap(src_dims[idx_last], src_dims[idx_2nd_last]);
      std::swap(src_strides[idx_last], src_strides[idx_2nd_last]);
    }
    dnnl::memory::dims weight_dims = TFShapeToOneDnnDims(weight.shape());
    dnnl::memory::dims weight_strides = CalculateTFStrides(weight_dims);
    if (transpose_b) {
      std::swap(weight_dims[idx_last], weight_dims[idx_2nd_last]);
      std::swap(weight_strides[idx_last], weight_strides[idx_2nd_last]);
    }
    dnnl::memory::dims dst_dims = {src_dims[0], weight_dims[1]};
    dnnl::memory::dims dst_strides = CalculateTFStrides(dst_dims);
    auto src_md =
        dnnl::memory::desc(src_dims, OneDnnType<sum_t>(), src_strides);
    auto weight_md =
        dnnl::memory::desc(weight_dims, OneDnnType<sum_t>(), weight_strides);
    auto dst_md =
        dnnl::memory::desc(dst_dims, OneDnnType<sum_t>(), dst_strides);
    dnnl::matmul::primitive_desc matmul_pd;
    dnnl::primitive_attr post_ops_attr;

    // Support matmul + add fusion, it occurs in LayerNormMLP backward pass
    dnnl::memory post_add_mem;
    if (has_post_add) {
      dnnl::post_ops post_ops = dnnl::post_ops();
      auto post_add_dims = TFShapeToOneDnnDims(post_add.shape());
      auto post_add_strides = CalculateTFStrides(post_add_dims);
      auto post_add_md = dnnl::memory::desc(post_add_dims, OneDnnType<sum_t>(),
                                            post_add_strides);
      post_ops.append_binary(dnnl::algorithm::binary_add, post_add_md);
      post_ops_attr.set_post_ops(post_ops);
      post_add_mem = CreateDnnlMemory(post_add_md, dnnl_engine,
                                      GetTensorBuffer<sum_t>(&post_add));
    }
    matmul_pd = dnnl::matmul::primitive_desc(dnnl_engine, src_md, weight_md,
                                             dst_md, post_ops_attr);

    dnnl::memory bias_mem;
    if (use_bias) {
      dnnl::memory::dims bias_dims(dst_dims.size(), 1);
      bias_dims[dst_dims.size() - 1] = dst_dims[dst_dims.size() - 1];
      auto bias_strides = CalculateTFStrides(bias_dims);
      auto bias_md =
          dnnl::memory::desc(bias_dims, OneDnnType<sum_t>(), bias_strides);
      bias_mem =
          CreateDnnlMemory(bias_md, dnnl_engine, GetTensorBuffer<sum_t>(&bias));
      matmul_pd = dnnl::matmul::primitive_desc(dnnl_engine, src_md, weight_md,
                                               bias_md, dst_md, post_ops_attr);
    }

    auto src_mem = CreateDnnlMemory(src_md, dnnl_engine,
                                    GetTensorBuffer<sum_t>(&src_temp));
    auto weight_mem = CreateDnnlMemory(weight_md, dnnl_engine,
                                       GetTensorBuffer<sum_t>(&weight_temp));
    if constexpr (is_fp8<output_t>::value) {
      Tensor dst_temp;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<sum_t>::v(),
                                             TensorShape({dst->NumElements()}),
                                             &dst_temp));
      dst_temp_ptr =
          reinterpret_cast<sum_t*>(GetTensorBuffer<sum_t>(&dst_temp));
    } else {
      dst_temp_ptr = reinterpret_cast<sum_t*>(GetTensorBuffer<sum_t>(dst));
    }
    auto dst_mem = CreateDnnlMemory(dst_md, dnnl_engine,
                                    reinterpret_cast<void*>(dst_temp_ptr));
    auto matmul_primitive = dnnl::matmul(matmul_pd);

    std::unordered_map<int, dnnl::memory> matmul_args;
    matmul_args.emplace(DNNL_ARG_SRC, src_mem);
    matmul_args.emplace(DNNL_ARG_WEIGHTS, weight_mem);
    matmul_args.emplace(DNNL_ARG_DST, dst_mem);
    if (use_bias) {
      matmul_args.emplace(DNNL_ARG_BIAS, bias_mem);
    }
    if (has_post_add) {
      int binary_post_op_position = 0;
      matmul_args.emplace(
          DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) |
              DNNL_ARG_SRC_1,
          post_add_mem);
    }
    matmul_primitive.execute(dnnl_stream, matmul_args);
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(
        ctx, errors::Aborted("Operation received an exception:", error_msg));
  }

  // Quantize to fp8 if needed
  if (is_fp8<output_t>::value) {
    Fp8Quantize<sum_t, output_t>(ctx, dst_temp_ptr, dst->flat<int8>().data(),
                                 dst_amax, dst_scale, dst->NumElements());
  }
}

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FP8_FP8_MATMUL_GPU_H_
