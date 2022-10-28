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
  }

  void Compute(OpKernelContext* ctx) override {
    try {
      const int src_index = 0;
      const int wei_index = 1;
      const int dst_index = 0;

      const Tensor& src_tensor = ctx->input(src_index);
      const Tensor& wei_tensor = ctx->input(wei_index);

      TensorShape src_tf_shape = src_tensor.shape();
      TensorShape wei_tf_shape = wei_tensor.shape();

      MatMulBCast bcast(src_tf_shape.dim_sizes(), wei_tf_shape.dim_sizes());

      OP_REQUIRES(
          ctx, bcast.IsValid(),
          errors::InvalidArgument(
              "Input tensors must have compatible batch dimensions: ",
              src_tf_shape.DebugString(), " vs. ", wei_tf_shape.DebugString()));

      int64 d0 = src_tf_shape.dim_size(src_tf_shape.dims() - 2);
      int64 d1 = src_tf_shape.dim_size(src_tf_shape.dims() - 1);

      int64 d2 = wei_tf_shape.dim_size(wei_tf_shape.dims() - 2);
      int64 d3 = wei_tf_shape.dim_size(wei_tf_shape.dims() - 1);

      if (this->transpose_a_) std::swap(d0, d1);
      if (this->transpose_b_) std::swap(d2, d3);

      OP_REQUIRES(
          ctx, d1 == d2,
          errors::InvalidArgument("Input[0] mismatch Input[1] shape :", d1,
                                  " vs. ", d2, ": ", src_tf_shape.DebugString(),
                                  " ", wei_tf_shape.DebugString(), " ",
                                  this->transpose_a_, " ", this->transpose_b_));

      // Follow below steps to construct valid oneDNN primitive params if
      // broadcast is required:
      //   1. Figure out the real output tf shape
      //   2. Expand input tf shapes, and use them to prepare input md
      TensorShape dst_tf_shape = bcast.output_batch_shape();
      dst_tf_shape.AddDim(d0);
      dst_tf_shape.AddDim(d3);

      if (dst_tf_shape.num_elements() == 0) {
        Tensor* dst_tensor = nullptr;
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output(dst_index, dst_tf_shape, &dst_tensor));
        return;
      }

      // src_md and wei_md: plain md for BatchMatMul primitive execution,
      // which are broadcasted and expressed by dims/strides
      auto params = MatMulBaseUtil::CreateMatMulParams(
          src_tf_shape, wei_tf_shape, dst_tf_shape, this->transpose_a_,
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

      auto onednn_engine = CreateDnnlEngine<Device>(*ctx);

      // Create matmul forward primitive
      std::unordered_map<int, memory> fwd_primitive_args;
      auto fwd_desc = matmul::desc(src_md, wei_md_prefer, dst_md);
      auto fwd_pd =
          GetPrimitiveDesc(ctx, fwd_desc, &fwd_primitive_args, onednn_engine);
      auto fwd_primitive = matmul(fwd_pd);

      // Create src memory, check if src needs to be reordered
      memory src_mem = CreateDnnlMemory(src_md, onednn_engine,
                                        GetTensorBuffer<Tlhs>(&src_tensor));

      memory wei_mem = CreateDnnlMemory(wei_md, onednn_engine,
                                        GetTensorBuffer<Trhs>(&wei_tensor));

      Tensor tmp_weight;
      Trhs* wei_cached_data = nullptr;
      wei_md_prefer = fwd_pd.weights_desc();
      if (wei_md_prefer != wei_md && this->is_filter_const_) {
        if (this->weight_cache_manager_.IsEmpty()) {
          // Cache weight in first time executing this node
          this->weight_cache_manager_.SetCache(
              ctx, wei_md, wei_md_prefer, GetTensorBuffer<Trhs>(&wei_tensor),
              onednn_engine);
        }

        wei_cached_data =
            this->weight_cache_manager_.GetCache(ctx, wei_md_prefer);

        // Weight cache may be failed, need to check it here.
        if (wei_cached_data != nullptr) {
          wei_mem =
              CreateDnnlMemory(wei_md_prefer, onednn_engine, wei_cached_data);
        } else {
          // During training, reorder weight in each iteration
          int64_t reorder_size = wei_md_prefer.get_size() / sizeof(Trhs);
          OP_REQUIRES_OK(
              ctx, ctx->allocate_temp(DataTypeToEnum<Trhs>::v(),
                                      TensorShape{reorder_size}, &tmp_weight));
          void* data_handle = GetTensorBuffer<Trhs>(&tmp_weight);
          auto mem_reordered =
              CreateDnnlMemory(wei_md_prefer, onednn_engine, data_handle);
          ReorderMemory(*ctx, &wei_mem, &mem_reordered, onednn_engine);
          wei_mem = mem_reordered;
        }
      }

      Tensor* dst_tensor = nullptr;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(dst_index, dst_tf_shape, &dst_tensor));

      // Create dst memory
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      GetTensorBuffer<Toutput>(dst_tensor));

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          fwd_pd.scratchpad_desc().get_size() / sizeof(Tlhs);
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<Tlhs>::v(),
                                             TensorShape({scratchpad_size}),
                                             &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<Tlhs>(&scratchpad_tensor));

      // Execute BatchMatMul
      auto onednn_stream = CreateDnnlStream(*ctx, onednn_engine);
      fwd_primitive_args.emplace(DNNL_ARG_SRC, src_mem);
      fwd_primitive_args.emplace(DNNL_ARG_WEIGHTS, wei_mem);
      fwd_primitive_args.emplace(DNNL_ARG_DST, dst_mem);
      fwd_primitive_args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem);
      fwd_primitive.execute(onednn_stream, fwd_primitive_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  virtual void AccumulateMulAndInt8Scale(OpKernelContext* ctx,
                                         float* mul_value) {
    return;
  }

  matmul::primitive_desc GetPrimitiveDesc(
      OpKernelContext* ctx, const matmul::desc& fwd_desc,
      std::unordered_map<int, memory>* fwd_args,
      const dnnl::engine& onednn_engine) {
    const int kPostOpStartIdx = 2;
    int post_op_input_index = kPostOpStartIdx;
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

    if (this->post_op_util_.HasBinary()) {
      // BatchMatMul + Add needs to set add input md in node execution.
      const Tensor& add_tensor = ctx->input(post_op_input_index);

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

      this->post_op_util_.SetBinaryInput(add_md);
      auto add_mem = CreateDnnlMemory(add_md, onednn_engine,
                                      GetTensorBuffer<Toutput>(&add_tensor));
      fwd_args->insert(
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, add_mem});
      post_op_input_index++;
    }

    this->post_op_util_.SetPostOpAttr(&post_ops_attr);
    return matmul::primitive_desc(fwd_desc, post_ops_attr, onednn_engine);
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
    auto* dpcpp_stream = ctx->GetDeviceStream();
    auto event = dpcpp_stream->memcpy(mul_host_data, mul_device_data,
                                      1 * sizeof(Toutput));
    event.wait();
  }

  Toutput* GetCachedMul(OpKernelContext* ctx) TF_LOCKS_EXCLUDED(mul_cache_mu_) {
    tf_shared_lock lock(&mul_cache_mu_);
    const Tensor& mul_cached_data = *mul_cached_tensor_.AccessTensor(ctx);

    return static_cast<Toutput*>(
        const_cast<Toutput*>(mul_cached_data.flat<Toutput>().data()));
  }

  mutex mul_cache_mu_;
  PersistentTensor mul_cached_tensor_ TF_GUARDED_BY(mul_cache_mu_);
#endif  // INTEL_CPU_ONLY

  bool transpose_a_;
  bool transpose_b_;
  bool is_filter_const_ = false;
  bool inplace_sum_ = false;

  // Weight cache manager
  WeightCacheManager<Trhs> weight_cache_manager_;

 protected:
  // Fusion util.
  PostOpUtil post_op_util_;
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
        } else if (op == "Mul" || op == "Add") {
          this->fused_ops_.push_back(op);
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
      if (context->HasAttr("num_args")) {
        OP_REQUIRES_OK(context, context->GetAttr("num_args", &this->num_args_));
      } else {
        this->num_args_ = 0;
      }

      if (context->HasAttr("fused_ops") && context->HasAttr("num_args")) {
        if (this->fused_ops_ == std::vector<string>{"Mul"} ||
            this->fused_ops_ == std::vector<string>{"Mul", "Add"}) {
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
