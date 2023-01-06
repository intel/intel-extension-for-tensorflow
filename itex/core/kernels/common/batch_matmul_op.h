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
      // TODO(itex): Replace Add(Sum) fusion to binary::add fusion manually.
      //             Will refine all Add fusion to binary:add in future.
      for (int i = 0; i < fused_ops.size(); ++i) {
        if (fused_ops[i] == "Add") fused_ops[i] = "BinaryAdd";
      }
      OP_REQUIRES(ctx, this->post_op_util_.AddOps(fused_ops),
                  errors::InvalidArgument(
                      "Found unsupported fusion in Fused BatchMatMul."));
    }

    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));
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

    if (dst_shape_.num_elements() == 0) {
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

      // Create matmul forward primitive
      auto fwd_desc = matmul::desc(src_md, wei_md_prefer, dst_md);
      auto fwd_pd = GetPrimitiveDesc(ctx, fwd_desc);
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

    if (this->post_op_util_.HasBinary()) {
      add_mem_.set_data_handle(context->tensor_data(add_index_));
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
                                          const matmul::desc& fwd_desc) {
    int post_op_input_index = kWeightIndex_ + 1;
    dnnl::primitive_attr post_ops_attr;
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // TODO(itex): Since ITEX currently combine mul post op and scale of
    // INT8 together. Here maybe slight accuracy difference with Intel-TF with
    // INT8-BF16 BatchMatMul Intel-TF: INT8 scale (fp32) * mul (bf16) ITEX: INT8
    // scale (fp32) * mul (fp32)
    if (this->post_op_util_.HasOutputScales()) {
      // mul_value = INT8 scale * mul
      float mul_value = 1.0;
      if (this->post_op_util_.HasMul()) {
        // BatchMatMul + Mul needs to set scale in node execution
        const Tensor& scale_tensor = ctx->input(post_op_input_index);

        if (scale_tensor.NumElements() != 1) {
          ITEX_LOG(FATAL) << "Mul tensor must be a scalar.";
        }

#ifndef INTEL_CPU_ONLY
        if (IsMulCacheEmpty()) {
          // Cache weight
          const Toutput* mul_device_data = scale_tensor.flat<Toutput>().data();
          CacheMul(ctx, mul_device_data);
        }
        Toutput* mul_host_data = GetCachedMul(ctx);
        mul_value = static_cast<float>(mul_host_data[0]);
#else
        mul_value = static_cast<float>(scale_tensor.flat<Toutput>()(0));
#endif  // INTEL_CPU_ONLY
      }

      AccumulateMulAndInt8Scale(ctx, &mul_value);

      std::vector<float> scales = {mul_value};
      this->post_op_util_.SetOutputScale(scales);
      post_op_input_index++;
    }

    std::vector<memory::desc> md_list;
    if (this->post_op_util_.HasBinary()) {
      // BatchMatMul + Add needs to set add input md in node execution.
      const Tensor& add_tensor = ctx->input(post_op_input_index);
      add_index_ = post_op_input_index;

      // Same as input and weight of BatchMatMul, add tensor also needs to:
      //   1. Get original block/plain md
      //   2. Figure out the extended md for primitive execution
      //   3. Reorder original md to extended md if needed
      TensorShape tf_shape = add_tensor.shape();

      ITEX_CHECK(tf_shape.dims() >= 3)
          << "Add input of FusedBatchMatMul must have 3 dims at least";

      auto add_dims = TFShapeToOneDnnDims(tf_shape);
      auto add_strides = CalculateTFStrides(add_dims);
      auto add_md = memory::desc(add_dims, OneDnnType<Toutput>(), add_strides);

      // FIXME(itex): Simply ingnore reorder this time, will fix it soon.
      md_list.push_back(add_md);
      add_mem_ = CreateDnnlMemory(add_md, onednn_engine_,
                                  GetTensorBuffer<Toutput>(&add_tensor));
      fwd_primitive_args_.insert(
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, add_mem_});
      post_op_input_index++;
    }

    this->post_op_util_.SetPostOpAttr(&post_ops_attr, md_list);
    return matmul::primitive_desc(fwd_desc, post_ops_attr, onednn_engine_);
  }

 private:
#ifndef INTEL_CPU_ONLY
  // TODO(itex): Wrap all cache related code to a module, reuse this module
  inline bool IsMulCacheEmpty() TF_LOCKS_EXCLUDED(mul_cache_mu_) {
    tf_shared_lock lock(&mul_cache_mu_);
    return (!mul_cached_tensor_.IsInitialized());
  }

  void AllocatePersistentTensor(OpKernelContext* ctx, Tensor** mul_tensor) {
    ITEX_DCHECK(mul_tensor);
    TensorShape mul_tf_shape;
    // Only one number is stored
    mul_tf_shape.AddDim(1);
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_persistent(
                            DataTypeToEnum<Toutput>::value, mul_tf_shape,
                            &mul_cached_tensor_, mul_tensor, alloc_attr));
  }

  void CacheMul(OpKernelContext* ctx, const Toutput* mul_device_data)
      TF_LOCKS_EXCLUDED(mul_cache_mu_) {
    mutex_lock lock(&mul_cache_mu_);

    // If mul is already cached, there's nothing to do.
    if (mul_cached_tensor_.IsInitialized()) {
      return;
    }

    // Create cached mul buffer
    Tensor* mul_tensor_ptr = nullptr;
    AllocatePersistentTensor(ctx, &mul_tensor_ptr);
    Toutput* mul_host_data =
        const_cast<Toutput*>(mul_tensor_ptr->flat<Toutput>().data());

    // TODO(itex): refactor the memcpy code
    auto* ITEX_GPU_stream = ctx->GetDeviceStream();
    auto event = ITEX_GPU_stream->memcpy(mul_host_data, mul_device_data,
                                         1 * sizeof(Toutput));
    event.wait();
  }

  Toutput* GetCachedMul(OpKernelContext* ctx) TF_LOCKS_EXCLUDED(mul_cache_mu_) {
    tf_shared_lock lock(&mul_cache_mu_);
    const Tensor& mul_cached_data = *mul_cached_tensor_.AccessTensor(ctx);

    return static_cast<Toutput*>(
        const_cast<Toutput*>(mul_cached_data.flat<Toutput>().data()));
  }

#endif  // INTEL_CPU_ONLY

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
  const int kSrcIndex_ = 0;
  const int kWeightIndex_ = 1;
  const int kDstIndex_ = 0;

 private:
  mutex mul_cache_mu_, mu_compute_;
  PersistentTensor mul_cached_tensor_ TF_GUARDED_BY(mul_cache_mu_);

  std::unordered_map<int, memory> fwd_primitive_args_;
  memory src_mem_, weights_mem_, dst_mem_, add_mem_, scratchpad_mem_;
  dnnl::matmul matmul_primitive_;
  Tensor* dst_tensor_;
  std::shared_ptr<Tensor> scratchpad_tensor_;
  int64_t scratchpad_size_, add_index_;
  std::vector<int64> input_dims_, weights_dims_;
  TensorShape dst_shape_;
  dnnl::stream onednn_stream_;
  dnnl::engine onednn_engine_;
};

// V2 is for latest Intel TF BatchMatMul INT8 new API.
// V1 is for previous BatchMatMul INT8 V1 new API, it seems the V1 API is not
// used by Intel-TF.
// Currently, the V1 and V2 kernel differences only lie on construction function

template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          bool is_v2 = true>
class QuantizedBatchMatMulV2Op
    : public BatchMatMulOp<Device, Tlhs, Trhs, Toutput> {
 public:
  explicit QuantizedBatchMatMulV2Op(OpKernelConstruction* context)
      : BatchMatMulOp<Device, Tlhs, Trhs, Toutput>(context) {
    if (is_v2) {
      this->post_op_util_.AddOps({"Quantized"});

      std::vector<string> fused_ops;
      if (context->HasAttr("fused_ops")) {
        OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
      }

      std::set<std::vector<string>> supported_fusions = {
          {"Dequantize"},
          {"Mul", "Dequantize"},
          {"Add", "Dequantize"},
          {"Mul", "Add", "Dequantize"}};

      OP_REQUIRES(context,
                  supported_fusions.find(fused_ops) != supported_fusions.end(),
                  errors::Unimplemented(
                      "Fusion is not implemented for BatchMatMul INT8: [",
                      absl::StrJoin(fused_ops, ","), "]"));

      for (int i = 0; i < fused_ops.size(); ++i) {
        string op = fused_ops[i];
        if (op == "Dequantize") {
          continue;
        } else if (op == "Mul") {
          this->fused_ops_.push_back(op);
        } else if (op == "Add") {
          this->fused_ops_.push_back("BinaryAdd");
        } else {
          OP_REQUIRES(context, false,
                      errors::Unimplemented(
                          "BatchMatMul INT8 doesn't support post op: ", op));
        }
      }

      kInputIndexMinLhs = this->fused_ops_.size() + 2;
      kInputIndexMaxLhs = this->fused_ops_.size() + 3;
      kInputIndexMinRhs = this->fused_ops_.size() + 4;
      kInputIndexMaxRhs = this->fused_ops_.size() + 5;

      this->post_op_util_.AddOps(this->fused_ops_);
    } else {
      this->post_op_util_.AddOps({"Quantized"});

      if (context->HasAttr("fused_ops")) {
        OP_REQUIRES_OK(context,
                       context->GetAttr("fused_ops", &this->fused_ops_));
      }
      for (int i = 0; i < this->fused_ops_.size(); ++i) {
        if (this->fused_ops_[i] == "Add") this->fused_ops_[i] = "BinaryAdd";
      }
      if (context->HasAttr("num_args")) {
        OP_REQUIRES_OK(context, context->GetAttr("num_args", &this->num_args_));
      } else {
        this->num_args_ = 0;
      }

      if (context->HasAttr("fused_ops") && context->HasAttr("num_args")) {
        if (this->fused_ops_ == std::vector<string>{"Mul"} ||
            this->fused_ops_ == std::vector<string>{"Mul", "BinaryAdd"}) {
          OP_REQUIRES(context, this->num_args_ == this->fused_ops_.size(),
                      errors::InvalidArgument(
                          "_QuantizedFusedBatchMatMulV2AndDequantize should "
                          "have same number of additional "
                          "inputs as the number of fusions"));
        } else {
          OP_REQUIRES(
              context, false,
              errors::Unimplemented("Fusion is not implemented: [",
                                    absl::StrJoin(this->fused_ops_, ","), "]"));
        }
      }

      kInputIndexMinLhs = this->num_args_ + 2;
      kInputIndexMaxLhs = this->num_args_ + 3;
      kInputIndexMinRhs = this->num_args_ + 4;
      kInputIndexMaxRhs = this->num_args_ + 5;

      this->post_op_util_.AddOps(this->fused_ops_);
    }

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
  // INT8 kernel may need some post op information during runtime.
  std::vector<string> fused_ops_;
  int num_args_;

  int kInputIndexMinLhs;
  int kInputIndexMaxLhs;
  int kInputIndexMinRhs;
  int kInputIndexMaxRhs;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_BATCH_MATMUL_OP_H_
