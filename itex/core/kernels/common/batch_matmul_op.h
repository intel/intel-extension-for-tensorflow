/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_COMMON_BATCH_MATMUL_OP_H_
#define ITEX_CORE_KERNELS_COMMON_BATCH_MATMUL_OP_H_

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "itex/core/kernels/common/host_data_cache.h"
#include "itex/core/kernels/common/matmul_op.h"

namespace itex {

using dnnl::matmul;
using dnnl::memory;

template <typename Device, typename Tlhs, typename Trhs, typename Toutput>
class BatchMatMulOp : public OpKernel {
 public:
  explicit BatchMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_x", &this->transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_y", &this->transpose_b_));
    if (ctx->HasAttr("is_filter_const")) {
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("is_filter_const", &this->is_filter_const_));
    }

    if (std::is_same<Trhs, qint8>::value)
      return;  // INT8 kernel have own contstruction code.

    if (ctx->HasAttr("fused_ops")) {
      std::vector<string> fused_ops;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops));

      // TODO(itex): Replace Add(Sum)/Mul(output_scale) fusion to Binary post
      //             op fusion manually. Will refine related fusion to binary
      //             fusion in future.
      for (int i = 0; i < fused_ops.size(); ++i) {
        if (fused_ops[i] == "Add") fused_ops[i] = "BinaryAdd";
        if (fused_ops[i] == "Mul") fused_ops[i] = "BinaryMul";
      }
      OP_REQUIRES(ctx, post_op_util_.AddOps(fused_ops),
                  errors::InvalidArgument(
                      "Found unsupported fusion in Fused BatchMatMul."));

      if (post_op_util_.HasBinary()) {
        const int binary_num = post_op_util_.GetBinaryNum();
        OP_REQUIRES(
            ctx, binary_num <= kMaxBinaryNum_,
            errors::Unimplemented(
                "The number of Binary op fusion in BatchMatMul is: ",
                binary_num, ", which is greater than ", kMaxBinaryNum_));
      }

      // Set alpha if get `LeakyRelu` after adding ops.
      if (post_op_util_.HasLeakyRelu()) {
        float alpha;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("leakyrelu_alpha", &alpha));
        post_op_util_.SetLeakyReluAlpha(alpha);
      }
    }

    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));

    fp32_math_mode_ = GetFP32MathMode<Device>();
  }

  void Init(OpKernelContext* ctx) {
    const Tensor& src_tensor = ctx->input(kSrcIndex_);
    const Tensor& wei_tensor = ctx->input(kWeightIndex_);

    TensorShape src_shape = src_tensor.shape();
    TensorShape wei_shape = wei_tensor.shape();

    // Reset cached args.
    fwd_primitive_args_.clear();
    input_dims_.clear();
    for (int i = 0; i < src_shape.dims(); ++i) {
      input_dims_.push_back(src_shape.dim_size(i));
    }
    weights_dims_.clear();
    for (int i = 0; i < wei_shape.dims(); ++i) {
      weights_dims_.push_back(wei_shape.dim_size(i));
    }

    MatMulBCast bcast(src_shape.dim_sizes(), wei_shape.dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Input tensors must have compatible batch dimensions: ",
                    src_shape.DebugString(), " vs. ", wei_shape.DebugString()));

    int64 d0 = src_shape.dim_size(src_shape.dims() - 2);
    int64 d1 = src_shape.dim_size(src_shape.dims() - 1);

    int64 d2 = wei_shape.dim_size(wei_shape.dims() - 2);
    int64 d3 = wei_shape.dim_size(wei_shape.dims() - 1);

    if (this->transpose_a_) std::swap(d0, d1);
    if (this->transpose_b_) std::swap(d2, d3);

    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "Input[0] mismatch Input[1] shape :", d1, " vs. ", d2, ": ",
                    src_shape.DebugString(), " ", wei_shape.DebugString(), " ",
                    this->transpose_a_, " ", this->transpose_b_));

    // Follow below steps to construct valid oneDNN primitive params if
    // broadcast is required:
    //   1. Figure out the real output tf shape
    //   2. Expand input tf shapes, and use them to prepare input md
    dst_shape_ = bcast.output_batch_shape();
    dst_shape_.AddDim(d0);
    dst_shape_.AddDim(d3);

    if (!post_op_util_.HasBias() && dst_shape_.num_elements() == 0) {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(kDstIndex_, dst_shape_, &dst_tensor_));
      is_init_ = true;
      is_input_zero_ = true;
      return;
    }

    try {
      // src_md and wei_md: plain md for BatchMatMul primitive execution,
      // which are broadcasted and expressed by dims/strides
      auto params = MatMulBaseUtil::CreateMatMulParams(
          src_shape, wei_shape, dst_shape_, this->transpose_a_,
          this->transpose_b_);
      auto src_md =
          memory::desc(params->a_dims, OneDnnType<Tlhs>(), params->a_strides);
      auto wei_md =
          memory::desc(params->b_dims, OneDnnType<Trhs>(), params->b_strides);
      auto dst_md = memory::desc(params->c_dims, OneDnnType<Toutput>(),
                                 params->c_strides);

      // Get a more appropriate md by oneDNN selection if filter is constant
      auto wei_md_prefer =
          is_filter_const_ ? memory::desc(params->b_dims, OneDnnType<Trhs>(),
                                          memory::format_tag::any)
                           : wei_md;
      memory::desc bias_md;
      if (post_op_util_.HasBias()) {
        const Tensor& bias_tensor = ctx->input(kBiasIndex_);

        // bias use same dims as dst
        bias_md = memory::desc(params->bias_dims, OneDnnType<Toutput>(),
                               params->bias_strides);
        // create bias memory
        bias_mem_ = CreateDnnlMemory(bias_md, onednn_engine_,
                                     GetTensorBuffer<Toutput>(&bias_tensor));
      }

      // Create matmul forward primitive
      auto fwd_pd =
          GetPrimitiveDesc(ctx, src_md, wei_md_prefer, bias_md, dst_md);
      matmul_primitive_ = matmul(fwd_pd);

      // Create src memory, check if src needs to be reordered
      src_mem_ = CreateDnnlMemory(src_md, onednn_engine_,
                                  GetTensorBuffer<Tlhs>(&src_tensor));

      auto weights_mem_input = CreateDnnlMemory(
          wei_md, onednn_engine_, GetTensorBuffer<Trhs>(&wei_tensor));

      Tensor tmp_weight;
      Trhs* wei_cached_data = nullptr;
      wei_md_prefer = fwd_pd.weights_desc();
      // Reorder only happens once weight is Const, see the initializer of
      // `wei_md_prefer` for more details.
      is_weight_reorder_ = (wei_md_prefer != wei_md);
      if (is_weight_reorder_) {
        if (this->weight_cache_manager_.IsEmpty()) {
          // Cache weight in first time executing this node
          this->weight_cache_manager_.SetCache(
              ctx, wei_md, wei_md_prefer, GetTensorBuffer<Trhs>(&wei_tensor),
              onednn_engine_);
        }

        wei_cached_data =
            this->weight_cache_manager_.GetCache(ctx, wei_md_prefer);

        // Weight cache may be failed, need to check it here.
        if (wei_cached_data != nullptr) {
          weights_mem_ =
              CreateDnnlMemory(wei_md_prefer, onednn_engine_, wei_cached_data);
        } else {
          // During training, reorder weight in each iteration
          int64_t reorder_size = wei_md_prefer.get_size() / sizeof(Trhs);
          OP_REQUIRES_OK(
              ctx, ctx->allocate_temp(DataTypeToEnum<Trhs>::v(),
                                      TensorShape{reorder_size}, &tmp_weight));
          weights_mem_ = CreateDnnlMemory(wei_md_prefer, onednn_engine_,
                                          GetTensorBuffer<Trhs>(&tmp_weight));
          ReorderMemory(*ctx, &weights_mem_input, &weights_mem_,
                        onednn_engine_);
        }
      } else {
        weights_mem_ = CreateDnnlMemory(wei_md, onednn_engine_,
                                        GetTensorBuffer<Trhs>(&wei_tensor));
      }

      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(kDstIndex_, dst_shape_, &dst_tensor_));

      // Create dst memory
      dst_mem_ = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine_,
                                  GetTensorBuffer<Toutput>(dst_tensor_));

      scratchpad_size_ = fwd_pd.scratchpad_desc().get_size() / sizeof(Tlhs);
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<Tlhs>::v(),
                                             TensorShape({scratchpad_size_}),
                                             scratchpad_tensor_.get()));
      scratchpad_mem_ =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine_,
                       GetTensorBuffer<Tlhs>(scratchpad_tensor_.get()));

      // Execute BatchMatMul
      fwd_primitive_args_.emplace(DNNL_ARG_SRC, src_mem_);
      fwd_primitive_args_.emplace(DNNL_ARG_WEIGHTS, weights_mem_);
      fwd_primitive_args_.emplace(DNNL_ARG_DST, dst_mem_);
      fwd_primitive_args_.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem_);
      if (post_op_util_.HasBias()) {
        fwd_primitive_args_.emplace(DNNL_ARG_BIAS, bias_mem_);
      }
#ifdef ITEX_ONEDNN_3_0
      if (post_op_util_.HasOutputScales()) {
        float alpha = post_op_util_.GetOutputScale()[0];
        float* output_scale_ptr =
            output_scale_cache_.GetCachedPtr(ctx, &alpha, 1);
        dnnl::memory scale_mem(
            {{1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x},
            onednn_engine_, output_scale_ptr);
        fwd_primitive_args_.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                                    scale_mem);
      }
#endif
      is_init_ = true;
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void InitOrSetMemory(OpKernelContext* context) {
    if (!(enable_cache_ && is_init_ &&
          context->is_input_same(kSrcIndex_, input_dims_) &&
          context->is_input_same(kWeightIndex_, weights_dims_))) {
      Init(context);
      return;
    }

    if (is_input_zero_) {
      OP_REQUIRES_OK(context, context->allocate_output(kDstIndex_, dst_shape_,
                                                       &dst_tensor_));
      return;
    }

    src_mem_.set_data_handle(context->tensor_data(kSrcIndex_));

    // Reorder only happens once weight is Const.
    // No need to reassign Const handle since it's already cached.
    if (!is_weight_reorder_) {
      weights_mem_.set_data_handle(context->tensor_data(kWeightIndex_));
    }

    if (post_op_util_.HasBias()) {
      bias_mem_.set_data_handle(context->tensor_data(kBiasIndex_));
    }

    for (int i = 0; i < post_op_util_.GetBinaryNum(); ++i) {
      binary_mem_[i].set_data_handle(
          context->tensor_data(binary_start_index_ + i));
    }

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tlhs>::v(),
                                          TensorShape({scratchpad_size_}),
                                          scratchpad_tensor_.get()));
    scratchpad_mem_.set_data_handle(
        GetTensorBuffer<Tlhs>(scratchpad_tensor_.get()));

    OP_REQUIRES_OK(context, context->allocate_output(kDstIndex_, dst_shape_,
                                                     &dst_tensor_));
    dst_mem_.set_data_handle(GetTensorBuffer<Toutput>(dst_tensor_));
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);
    dst_tensor_ = nullptr;
    onednn_engine_ = CreateDnnlEngine<Device>(*context);
    onednn_stream_ = CreateDnnlStream(*context, onednn_engine_);
    scratchpad_tensor_ = std::make_shared<Tensor>();
    InitOrSetMemory(context);

    // Skip primitive execution if the calculation is meaningless.
    if (!is_input_zero_) {
      matmul_primitive_.execute(onednn_stream_, fwd_primitive_args_);
    }

    scratchpad_tensor_.reset();
  }

  virtual void AccumulateMulAndInt8Scale(OpKernelContext* ctx,
                                         float* mul_value) {
    return;
  }

  matmul::primitive_desc GetPrimitiveDesc(OpKernelContext* ctx,
                                          const memory::desc& src_desc,
                                          const memory::desc& weights_desc,
                                          const memory::desc& bias_desc,
                                          const memory::desc& dst_desc) {
    dnnl::primitive_attr post_ops_attr;
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (std::is_same<Tlhs, float>::value) {
      post_ops_attr.set_fpmath_mode(fp32_math_mode_);
    }

    if (post_op_util_.HasOutputScales()) {
      // mul_value = INT8 scale
      float mul_value = 1.0;
      AccumulateMulAndInt8Scale(ctx, &mul_value);

      std::vector<float> scales = {mul_value};
      post_op_util_.SetOutputScale(scales);
    }

    std::vector<memory::desc> md_list;
    binary_start_index_ =
        1 + (post_op_util_.HasBias() ? kBiasIndex_ : kWeightIndex_);
    for (int i = 0; i < post_op_util_.GetBinaryNum(); ++i) {
      // FusedBatchMatMul need to set binary input md in node execution.
      const Tensor& binary_tensor = ctx->input(binary_start_index_ + i);

      // Same as input and weight of BatchMatMul, binary tensor also needs to:
      //   1. Get original block/plain md
      //   2. Figure out the extended md for primitive execution
      //   3. Reorder original md to extended md if needed
      TensorShape tf_shape = binary_tensor.shape();

      ITEX_CHECK(binary_tensor.NumElements() == 1 || tf_shape.dims() >= 3)
          << "Binary input of FusedBatchMatMul must be scalar or have 3 dims "
          << "at least, but got " << tf_shape.dims();

      auto binary_dims = TFShapeToOneDnnDims(tf_shape);
      auto binary_strides = CalculateTFStrides(binary_dims);
      auto binary_md =
          memory::desc(binary_dims, OneDnnType<Toutput>(), binary_strides);

      // FIXME(itex): Simply ingnore reorder this time, will fix it soon.
      md_list.push_back(binary_md);
      binary_mem_[i] = CreateDnnlMemory(
          binary_md, onednn_engine_, GetTensorBuffer<Toutput>(&binary_tensor));
      fwd_primitive_args_.insert(
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, binary_mem_[i]});
    }

    post_op_util_.SetPostOpAttr(&post_ops_attr, md_list);
#ifdef ITEX_ONEDNN_3_0
    if (post_op_util_.HasBias()) {
      return matmul::primitive_desc(onednn_engine_, src_desc, weights_desc,
                                    bias_desc, dst_desc, post_ops_attr);
    } else {
      return matmul::primitive_desc(onednn_engine_, src_desc, weights_desc,
                                    dst_desc, post_ops_attr);
    }
#else
    if (post_op_util_.HasBias()) {
      auto fwd_desc = matmul::desc(src_desc, weights_desc, bias_desc, dst_desc);
      return matmul::primitive_desc(fwd_desc, post_ops_attr, onednn_engine_);
    } else {
      auto fwd_desc = matmul::desc(src_desc, weights_desc, dst_desc);
      return matmul::primitive_desc(fwd_desc, post_ops_attr, onednn_engine_);
    }
#endif  // ITEX_ONEDNN_3_0
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool is_filter_const_ = false;
  bool inplace_sum_ = false;

 protected:
  // Fusion util.
  PostOpUtil post_op_util_;
  // Weight cache manager
  WeightCacheManager<Trhs> weight_cache_manager_;
  bool enable_cache_ = false;
  bool is_init_ = false;
  bool is_input_zero_ = false;
  bool is_weight_reorder_;
  static const int kSrcIndex_ = 0;
  static const int kWeightIndex_ = 1;
  static const int kBiasIndex_ = 2;
  static const int kDstIndex_ = 0;
  // Hard code the max number of supported binary post op fusion.
  static const int kMaxBinaryNum_ = 2;

  dnnl::fpmath_mode fp32_math_mode_ = dnnl::fpmath_mode::strict;

 private:
  mutex mu_compute_;

  std::unordered_map<int, memory> fwd_primitive_args_;
  memory src_mem_, weights_mem_, bias_mem_, dst_mem_,
      binary_mem_[kMaxBinaryNum_], scratchpad_mem_;
  dnnl::matmul matmul_primitive_;
  Tensor* dst_tensor_;
  std::shared_ptr<Tensor> scratchpad_tensor_;
  int64_t scratchpad_size_, binary_start_index_;
  std::vector<int64> input_dims_, weights_dims_;
  TensorShape dst_shape_;
  dnnl::stream onednn_stream_;
  dnnl::engine onednn_engine_;
#ifdef ITEX_ONEDNN_3_0
  HostDataCache<Device, float> output_scale_cache_;
#endif
};

// V2 is for latest Intel TF BatchMatMul INT8 new API.
// V1 is for previous BatchMatMul INT8 V1 new API, it seems the V1 API is not
// used by Intel-TF.
// Currently, the V1 and V2 kernel differences only lie on construction
// function

template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          bool is_v2 = true>
class QuantizedBatchMatMulV2Op
    : public BatchMatMulOp<Device, Tlhs, Trhs, Toutput> {
 public:
  explicit QuantizedBatchMatMulV2Op(OpKernelConstruction* context)
      : BatchMatMulOp<Device, Tlhs, Trhs, Toutput>(context) {
    int num_args = 0;
    std::vector<string> fused_ops;

    if (context->HasAttr("fused_ops")) {
      OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));

      // TODO(itex): Replace Add(Sum)/Mul(output_scale) fusion to Binary post
      //             op fusion manually. Will refine related fusion to binary
      //             fusion in future.
      for (int i = 0; i < fused_ops.size(); ++i) {
        if (fused_ops[i] == "Add") fused_ops[i] = "BinaryAdd";
        if (fused_ops[i] == "Mul") fused_ops[i] = "BinaryMul";
      }
    }

    if (is_v2) {
      kInputIndexMinLhs = fused_ops.size() + 1;
      kInputIndexMaxLhs = fused_ops.size() + 2;
      kInputIndexMinRhs = fused_ops.size() + 3;
      kInputIndexMaxRhs = fused_ops.size() + 4;
    } else {
      // Need to add Quantized flag for legacy API manually.
      this->post_op_util_.AddOps({"Quantized"});

      if (context->HasAttr("num_args")) {
        OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));

        if (context->HasAttr("fused_ops")) {
          OP_REQUIRES(context, num_args == fused_ops.size(),
                      errors::InvalidArgument(
                          "_QuantizedFusedBatchMatMulV2AndDequantize should "
                          "have same number of additional "
                          "inputs as the number of fusions"));
        }
      }

      kInputIndexMinLhs = num_args + 2;
      kInputIndexMaxLhs = num_args + 3;
      kInputIndexMinRhs = num_args + 4;
      kInputIndexMaxRhs = num_args + 5;
    }

    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops),
                errors::Unimplemented(
                    "Fusion is not implemented for QuantizedBatchMatMul: [",
                    absl::StrJoin(fused_ops, ","), "]"));

    ITEX_CHECK_OK(ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false,
                                     &this->enable_cache_));
  }

  void AccumulateMulAndInt8Scale(OpKernelContext* context,
                                 float* mul_value) override {
    const float min_lhs =
        context->input(kInputIndexMinLhs).template flat<float>()(0);
    const float max_lhs =
        context->input(kInputIndexMaxLhs).template flat<float>()(0);
    const float min_rhs =
        context->input(kInputIndexMinRhs).template flat<float>()(0);
    const float max_rhs =
        context->input(kInputIndexMaxRhs).template flat<float>()(0);
    const float range_lhs = std::max(std::abs(min_lhs), std::abs(max_lhs));
    const float range_rhs = std::max(std::abs(min_rhs), std::abs(max_rhs));
    const float max_int8_lhs =
        (std::is_same<Tlhs, quint8>::value) ? 255.0f : 127.0f;
    const float max_int8_rhs =
        (std::is_same<Trhs, quint8>::value) ? 255.0f : 127.0f;
    float scale_output = range_lhs * range_rhs / (max_int8_lhs * max_int8_rhs);
    *mul_value = *mul_value * scale_output;
  }

 private:
  int kInputIndexMinLhs;
  int kInputIndexMaxLhs;
  int kInputIndexMinRhs;
  int kInputIndexMaxRhs;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_BATCH_MATMUL_OP_H_
