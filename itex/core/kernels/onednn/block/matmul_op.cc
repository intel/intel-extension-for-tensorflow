/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/kernels/onednn/block/matmul_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

using dnnl::matmul;
using dnnl::memory;

namespace itex {

template <typename Device, typename T>
class OneDnnMatMulOp : public OneDnnMatMulBaseOp<Device, T> {
 public:
  explicit OneDnnMatMulOp(OpKernelConstruction* context)
      : OneDnnMatMulBaseOp<Device, T>(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_a", &this->transpose_a_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_b", &this->transpose_b_));
    OP_REQUIRES_OK(
        context, context->GetAttr("is_filter_const", &this->is_filter_const_));
    if (context->HasAttr("fused_ops")) {
      std::vector<string> fused_ops;
      OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
      OP_REQUIRES(
          context, this->post_op_util_.AddOps(fused_ops),
          errors::InvalidArgument("Found unsupported fusion in Fused MatMul."));

      // Set alpha if get `LeakyRelu` after adding ops.
      if (this->post_op_util_.HasLeakyRelu()) {
        float alpha;
        OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha", &alpha));
        this->post_op_util_.SetLeakyReluAlpha(alpha);
      }
    }
    if (context->HasAttr("inplace_sum")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("inplace_sum", &this->inplace_sum_));
    }
    this->fp32_math_mode_ = GetFP32MathMode<Device>();

    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));
  }

  memory::desc CreateMatMulMemoryDesc(memory::dims md, bool is_adjoint) {
    if (is_adjoint)
      return memory::desc(md, OneDnnType<T>(), memory::format_tag::ba);
    return memory::desc(md, OneDnnType<T>(), memory::format_tag::ab);
  }

  void InitOrSetMemory(OpKernelContext* context) {
    if (!(enable_cache_ && is_init_ &&
          IsInputSame(context, 0, input_dims_, src_onednn_shape_) &&
          IsInputSame(context, 1, weight_dims_, weight_onednn_shape_))) {
      Init(context);
      return;
    }

    if (is_input_zero_) {
      OneDnnShape dst_onednn_shape;
      dst_onednn_shape.SetOneDnnTensor(false);
      AllocateOutputSetOneDnnShape(context, kDstIndex_, &dst_tensor_,
                                   dst_tf_shape_, dst_onednn_shape_);
      return;
    }

    if (is_src_reordered_) {
      src_mem_.set_data_handle(context->tensor_data(kSrcIndex_));
      src_reorder_mem_.set_data_handle(
          GetTensorBuffer<T>(&src_reorder_tensor_));
      ReorderMemory(*context, &src_mem_, &src_reorder_mem_, onednn_engine_);
    } else {
      src_mem_.set_data_handle(context->tensor_data(kSrcIndex_));
    }

    if (is_weight_reordered_) {
      if (!this->is_filter_const_) {
        weight_mem_.set_data_handle(context->tensor_data(kWeightIndex_));
        weight_reorder_mem_.set_data_handle(
            GetTensorBuffer<T>(&weight_reorder_tensor_));
        ReorderMemory(*context, &weight_mem_, &weight_reorder_mem_,
                      onednn_engine_);
      }
    } else {
      // No reorder needed
      weight_mem_.set_data_handle(context->tensor_data(kWeightIndex_));
    }

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          TensorShape({scratchpad_size_}),
                                          scratchpad_tensor_.get()));
    scratchpad_mem_.set_data_handle(
        GetTensorBuffer<T>(scratchpad_tensor_.get()));

    // Allocate output data tensor, whether the output layout is blocked or
    // not relies on the result of IsBlockedMd
    SetOutputTensorShape(fwd_pd_.dst_desc(), OneDnnTensorFormat::FORMAT_NC,
                         &dst_tf_shape_, &dst_onednn_shape_,
                         IsBlockedMd(fwd_pd_.dst_desc()));

    // Handle Add fusion.
    if (this->post_op_util_.HasAdd()) {
      int is_forward_success = kUnsuccess_;
      add_tensor_ = &context->input(kAddIndex_);

      if (add_onednn_shape_ == dst_onednn_shape_) {
        // Try to do in-place.
        if (this->inplace_sum_) {
          context->set_output(kDstIndex_, *add_tensor_);
          ForwardMetaData(context, kAddIndex_, kDstIndex_, dst_onednn_shape_);
          dst_tensor_ = context->mutable_output(kDstIndex_);
          is_forward_success = kAddIndex_;
        } else {
          ForwardOrAllocateOutputSetOneDnnShape(
              context, kAddIndex_, kDstIndex_, &dst_tensor_, dst_tf_shape_,
              dst_onednn_shape_, &is_forward_success);
        }
      }
      // Reorder is needed. Check `dst_tensor_` first:
      //   1) nullptr, add shape is different with dst shape;
      //   2) not nullptr, forward is failed but dst has been allocated;
      if (dst_tensor_ == nullptr) {
        AllocateOutputSetOneDnnShape(context, kDstIndex_, &dst_tensor_,
                                     dst_tf_shape_, dst_onednn_shape_);
      }
      if (is_forward_success == kUnsuccess_) {
        fuse_add_src_mem_.set_data_handle(GetTensorBuffer<T>(add_tensor_));
        fuse_add_dst_mem_.set_data_handle(GetTensorBuffer<T>(dst_tensor_));
        ReorderMemory(*context, &fuse_add_src_mem_, &fuse_add_dst_mem_,
                      onednn_engine_);
      }
    } else {
      AllocateOutputSetOneDnnShape(context, kDstIndex_, &dst_tensor_,
                                   dst_tf_shape_, dst_onednn_shape_);
    }

    if (this->post_op_util_.HasBias()) {
      bias_mem_.set_data_handle(context->tensor_data(kBiasIndex_));
    }
    dst_mem_.set_data_handle(GetTensorBuffer<T>(dst_tensor_));
  }

  void Init(OpKernelContext* context) {
    try {
      const Tensor& src_tensor = context->input(kSrcIndex_);
      const Tensor& weight_tensor = context->input(kWeightIndex_);

      GetOneDnnShape(context, kSrcIndex_, &src_onednn_shape_);
      GetOneDnnShape(context, kWeightIndex_, &weight_onednn_shape_);
      fwd_primitive_args_.clear();

      auto input_shape = src_tensor.shape();
      input_dims_.clear();
      for (int i = 0; i < input_shape.dims(); ++i) {
        input_dims_.push_back(input_shape.dim_size(i));
      }
      auto weight_tensor_shape = weight_tensor.shape();
      weight_dims_.clear();
      for (int i = 0; i < weight_tensor_shape.dims(); ++i) {
        weight_dims_.push_back(weight_tensor_shape.dim_size(i));
      }

      OP_REQUIRES(context,
                  (Eigen::internal::is_same<Device, CPUDevice>::value ||
                   !(this->transpose_a_ && src_onednn_shape_.IsOneDnnTensor())),
                  errors::InvalidArgument(
                      "OneDnnMatMul with block layout input and "
                      "transpose_a = true is only supported on CPU"));
      OP_REQUIRES(
          context,
          (Eigen::internal::is_same<Device, CPUDevice>::value ||
           !(this->transpose_b_ && weight_onednn_shape_.IsOneDnnTensor())),
          errors::InvalidArgument(
              "OneDnnMatMul with block layout weight and transpose_b = true "
              "is only supported on CPU"));

      TensorShape src_tf_shape = src_onednn_shape_.IsOneDnnTensor()
                                     ? src_onednn_shape_.GetTfShape()
                                     : src_tensor.shape();
      TensorShape weight_tf_shape = weight_onednn_shape_.IsOneDnnTensor()
                                        ? weight_onednn_shape_.GetTfShape()
                                        : weight_tensor.shape();
      const int batch = this->transpose_a_ ? src_tf_shape.dim_size(1)
                                           : src_tf_shape.dim_size(0);
      const int k = this->transpose_a_ ? src_tf_shape.dim_size(0)
                                       : src_tf_shape.dim_size(1);
      const int channel = this->transpose_b_ ? weight_tf_shape.dim_size(0)
                                             : weight_tf_shape.dim_size(1);

      memory::dims src_dims = {batch, k};
      memory::dims weight_dims = {k, channel};

      dst_tf_shape_ = {src_dims[0], weight_dims[1]};

      if (dst_tf_shape_.num_elements() == 0) {
        is_input_zero_ = true;
        OneDnnShape dst_onednn_shape;
        dst_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kDstIndex_, &dst_tensor_,
                                     dst_tf_shape_, dst_onednn_shape_);
        is_init_ = true;
        return;
      }

      auto src_md =
          src_onednn_shape_.IsOneDnnTensor()
              ? src_onednn_shape_.GetOneDnnLayout()
              : memory::desc(src_dims, OneDnnType<T>(),
                             this->transpose_a_ ? memory::format_tag::ba
                                                : memory::format_tag::ab);
      auto weight_md =
          weight_onednn_shape_.IsOneDnnTensor()
              ? weight_onednn_shape_.GetOneDnnLayout()
              : memory::desc(weight_dims, OneDnnType<T>(),
                             this->transpose_b_ ? memory::format_tag::ba
                                                : memory::format_tag::ab);
      memory::dims dst_dims = {src_dims[0], weight_dims[1]};

      // Use any format if:
      // 1. Input tensor is oneDNN tensor and it's not transposed
      // 2. Input tensor is not oneDNN tensor and Weight is const
      bool is_src_any =
          src_onednn_shape_.IsOneDnnTensor() && !this->transpose_a_;
      bool is_wei_any = weight_onednn_shape_.IsOneDnnTensor()
                            ? !this->transpose_b_
                            : this->is_filter_const_;

      auto src_exec_md =
          is_src_any
              ? memory::desc(src_dims, OneDnnType<T>(), memory::format_tag::any)
              : CreateMatMulMemoryDesc(src_dims, this->transpose_a_);
      auto weight_exec_md =
          is_wei_any ? memory::desc(weight_dims, OneDnnType<T>(),
                                    memory::format_tag::any)
                     : CreateMatMulMemoryDesc(weight_dims, this->transpose_b_);
      auto dst_exec_md = this->CreateMemoryDescWithStrides(dst_dims);

      // Check post ops.
      dnnl::primitive_attr post_op_attr;
      this->post_op_util_.SetPostOpAttr(&post_op_attr);
      post_op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<T, float>::value) {
        post_op_attr.set_fpmath_mode(this->fp32_math_mode_);
      }

      if (this->post_op_util_.HasBias()) {
        memory::desc bias_any_md = memory::desc(
            {1, weight_dims[1]}, OneDnnType<T>(), memory::format_tag::ab);
        fwd_pd_ =
            matmul::primitive_desc(onednn_engine_, src_exec_md, weight_exec_md,
                                   bias_any_md, dst_exec_md, post_op_attr);
      } else {
        fwd_pd_ =
            matmul::primitive_desc(onednn_engine_, src_exec_md, weight_exec_md,
                                   dst_exec_md, post_op_attr);
      }
      fwd_primitive_ = matmul(fwd_pd_);
      // Create src memory, check if src needs to be reordered
      src_mem_ = CreateDnnlMemory(src_md, onednn_engine_,
                                  GetTensorBuffer<T>(&src_tensor));

      is_src_reordered_ = (src_md != fwd_pd_.src_desc());
      if (is_src_reordered_) {
        int64 src_reorder_size = fwd_pd_.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor_));
        src_reorder_mem_ =
            CreateDnnlMemory(fwd_pd_.src_desc(), onednn_engine_,
                             GetTensorBuffer<T>(&src_reorder_tensor_));
        ReorderMemory(*context, &src_mem_, &src_reorder_mem_, onednn_engine_);
      }

      memory::desc expected_md = fwd_pd_.weights_desc();

      is_weight_reordered_ = (weight_md != expected_md);
      if (is_weight_reordered_) {
        T* weight_cached_data = nullptr;
        if (this->is_filter_const_) {
          if (this->weight_cache_manager_.IsEmpty()) {
            // Cache weight in first time executing this node
            this->weight_cache_manager_.SetCache(
                context, weight_md, expected_md,
                GetTensorBuffer<T>(&weight_tensor), onednn_engine_);
          }

          weight_cached_data =
              this->weight_cache_manager_.GetCache(context, expected_md);
        }
        if (weight_cached_data != nullptr) {
          weight_reorder_mem_ =
              CreateDnnlMemory(expected_md, onednn_engine_, weight_cached_data);
        } else {
          // During training, reorder weight in each iteration
          weight_mem_ = CreateDnnlMemory(weight_md, onednn_engine_,
                                         GetTensorBuffer<T>(&weight_tensor));

          int64 weight_reorder_size =
              fwd_pd_.weights_desc().get_size() / sizeof(T);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DataTypeToEnum<T>::v(),
                                      TensorShape({weight_reorder_size}),
                                      &weight_reorder_tensor_));
          weight_reorder_mem_ =
              CreateDnnlMemory(expected_md, onednn_engine_,
                               GetTensorBuffer<T>(&weight_reorder_tensor_));
          ReorderMemory(*context, &weight_mem_, &weight_reorder_mem_,
                        onednn_engine_);
        }
      } else {
        // No reorder needed
        weight_mem_ = CreateDnnlMemory(weight_md, onednn_engine_,
                                       GetTensorBuffer<T>(&weight_tensor));
      }

      // Allocate output data tensor, whether the output layout is blocked or
      // not relies on the result of IsBlockedMd
      SetOutputTensorShape(fwd_pd_.dst_desc(), OneDnnTensorFormat::FORMAT_NC,
                           &dst_tf_shape_, &dst_onednn_shape_,
                           IsBlockedMd(fwd_pd_.dst_desc()));

      // Handle Add fusion.
      if (this->post_op_util_.HasAdd()) {
        add_tensor_ = &context->input(kAddIndex_);
        GetOneDnnShape(context, kAddIndex_, &add_onednn_shape_);
        int is_forward_success = kUnsuccess_;
        // Try to do in-place.
        if (add_onednn_shape_ == dst_onednn_shape_) {
          // TODO(itex): Remove this workaround when inplace works.
          if (this->inplace_sum_) {
            context->set_output(kDstIndex_, *add_tensor_);
            ForwardMetaData(context, kAddIndex_, kDstIndex_, dst_onednn_shape_);
            dst_tensor_ = context->mutable_output(kDstIndex_);
            is_forward_success = kAddIndex_;
          } else {
            ForwardOrAllocateOutputSetOneDnnShape(
                context, kAddIndex_, kDstIndex_, &dst_tensor_, dst_tf_shape_,
                dst_onednn_shape_, &is_forward_success);
          }
        }

        // Reorder is needed. Check `dst_tensor_` first:
        //   1) nullptr, add shape is different with dst shape;
        //   2) not nullptr, forward is failed but dst has been allocated;
        if (dst_tensor_ == nullptr) {
          AllocateOutputSetOneDnnShape(context, kDstIndex_, &dst_tensor_,
                                       dst_tf_shape_, dst_onednn_shape_);
        }

        if (is_forward_success == kUnsuccess_) {
          // In-place do not success, need reorder.
          auto add_md = add_onednn_shape_.IsOneDnnTensor()
                            ? add_onednn_shape_.GetOneDnnLayout()
                            : memory::desc(dst_dims, OneDnnType<T>(),
                                           memory::format_tag::ab);
          fuse_add_src_mem_ = CreateDnnlMemory(add_md, onednn_engine_,
                                               GetTensorBuffer<T>(add_tensor_));
          fuse_add_dst_mem_ =
              CreateDnnlMemory(fwd_pd_.dst_desc(), onednn_engine_,
                               GetTensorBuffer<T>(dst_tensor_));
          ReorderMemory(*context, &fuse_add_src_mem_, &fuse_add_dst_mem_,
                        onednn_engine_);
        }
      } else {
        AllocateOutputSetOneDnnShape(context, kDstIndex_, &dst_tensor_,
                                     dst_tf_shape_, dst_onednn_shape_);
      }

      // Create dst memory
      T* dst_data = dst_tensor_->flat<T>().data();
      dst_mem_ = CreateDnnlMemory(fwd_pd_.dst_desc(), onednn_engine_,
                                  static_cast<void*>(dst_data));

      scratchpad_size_ = fwd_pd_.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size_}),
                                            scratchpad_tensor_.get()));
      scratchpad_mem_ =
          dnnl::memory(fwd_pd_.scratchpad_desc(), onednn_engine_,
                       GetTensorBuffer<T>(scratchpad_tensor_.get()));

      // Execute MatMul
      onednn_stream_ = CreateDnnlStream(*context, onednn_engine_);
      fwd_primitive_args_ = {
          {DNNL_ARG_SRC, is_src_reordered_ ? src_reorder_mem_ : src_mem_},
          {DNNL_ARG_WEIGHTS,
           is_weight_reordered_ ? weight_reorder_mem_ : weight_mem_},
          {DNNL_ARG_DST, dst_mem_},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem_}};

      if (this->post_op_util_.HasBias()) {
        const Tensor& bias_tensor = context->input(kBiasIndex_);
        // Create bias memory, since it is 1-dimension, no reordered needed
        bias_mem_ = CreateDnnlMemory(fwd_pd_.bias_desc(), onednn_engine_,
                                     GetTensorBuffer<T>(&bias_tensor));

        fwd_primitive_args_.emplace(DNNL_ARG_BIAS, bias_mem_);
      }
      is_init_ = true;
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);
    dst_tensor_ = nullptr;
    onednn_engine_ = CreateDnnlEngine<Device>(*context);
    // onednn_stream has thread safety issue, need create a new one in
    // every compute.
    onednn_stream_ = CreateDnnlStream(*context, onednn_engine_);

    scratchpad_tensor_ = std::make_shared<Tensor>();
    InitOrSetMemory(context);

    // Skip primitive execution if the calculation is meaningless.
    if (is_input_zero_) {
      scratchpad_tensor_.reset();
      return;
    }

    fwd_primitive_.execute(onednn_stream_, fwd_primitive_args_);
    scratchpad_tensor_.reset();
  }

 private:
  const int kSrcIndex_ = 0, kWeightIndex_ = 1, kDstIndex_ = 0;
  // For Handling Add fusion.
  const int kAddIndex_ = 3, kUnsuccess_ = -1, kBiasIndex_ = 2;

  OneDnnShape src_onednn_shape_, weight_onednn_shape_, dst_onednn_shape_;
  std::vector<int64> input_dims_, weight_dims_;
  memory src_mem_, dst_mem_, src_reorder_mem_, weight_mem_, weight_reorder_mem_,
      scratchpad_mem_, bias_mem_, fuse_add_src_mem_, fuse_add_dst_mem_;
  OneDnnShape add_onednn_shape_;
  TensorShape dst_tf_shape_;
  dnnl::stream onednn_stream_;
  dnnl::engine onednn_engine_;
  matmul::primitive_desc fwd_pd_;
  matmul fwd_primitive_;
  std::unordered_map<int, memory> fwd_primitive_args_;
  Tensor* dst_tensor_;
  Tensor src_reorder_tensor_, weight_reorder_tensor_;
  std::shared_ptr<Tensor> scratchpad_tensor_;
  int64_t scratchpad_size_ = 0;
  const Tensor* add_tensor_;
  bool is_input_zero_ = false;
  bool is_init_ = false;
  bool is_src_reordered_ = false, is_weight_reordered_ = false;
  bool enable_cache_ = false;
  mutex mu_compute_;
};

template <typename Device, typename T>
class OneDnnFusedMatMulGradOp : public OpKernel {
 public:
  explicit OneDnnFusedMatMulGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    bool transpose_b;
    std::vector<string> fused_ops;

    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b));
    OP_REQUIRES(context, !transpose_b,
                errors::InvalidArgument(
                    "_OneDnnFusedMatMulGrad only supports transpose_b = "
                    "false."));

    OP_REQUIRES(context, fused_ops.size() == 1,
                errors::InvalidArgument(
                    "_OneDnnFusedMatMulGrad must have 1 post-arguments at "
                    "most."));
    OP_REQUIRES(context, fused_ops[0] == "BiasAddGrad",
                errors::InvalidArgument(
                    "The 1st post-argument of _OneDnnFusedMatMulGrad must be "
                    "BiasAddGrad."));
    fp32_math_mode_ = GetFP32MathMode<Device>();
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const int kSrcIndex = 0;
      const int kDiffDstIndex = 1;
      const int kDiffWeightIndex = 0;
      const int kDiffBiasIndex = 1;
      const Tensor& src_tensor = context->input(kSrcIndex);
      const Tensor& diff_dst_tensor = context->input(kDiffDstIndex);

      OneDnnShape src_onednn_shape;
      OneDnnShape diff_dst_onednn_shape;
      GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
      GetOneDnnShape(context, kDiffDstIndex, &diff_dst_onednn_shape);
      auto src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                              ? src_onednn_shape.GetTfShape()
                              : src_tensor.shape();
      auto diff_dst_tf_shape = diff_dst_onednn_shape.IsOneDnnTensor()
                                   ? diff_dst_onednn_shape.GetTfShape()
                                   : diff_dst_tensor.shape();

      OP_REQUIRES(context, !(transpose_a_ && src_onednn_shape.IsOneDnnTensor()),
                  errors::InvalidArgument(
                      "_OneDnnFusedMatMulGrad only support transpose_a = "
                      "false, when has block layout input"));

      const int batch = src_tf_shape.dim_size(0);
      const int k = src_tf_shape.dim_size(1);
      const int channel = diff_dst_tf_shape.dim_size(1);

      OP_REQUIRES(
          context, batch == diff_dst_tf_shape.dim_size(0),
          errors::InvalidArgument(
              "Matrix size-incompatible: In[0]: ", src_tf_shape.DebugString(),
              ", In[1]: ", diff_dst_tf_shape.DebugString()));

      if (batch == 0 || channel == 0) {
        return;
      }

      // Create primitive.
      memory::dims src_dims = memory::dims({batch, k});
      memory::dims diff_dst_dims = memory::dims({batch, channel});
      memory::dims diff_weight_dims = memory::dims({channel, k});
      memory::dims diff_bias_dims = memory::dims({channel});
      dnnl::memory::format_tag src_format = dnnl::memory::format_tag::nc;
      dnnl::memory::format_tag diff_dst_format = dnnl::memory::format_tag::nc;
      dnnl::memory::format_tag diff_weight_format =
          dnnl::memory::format_tag::io;
      auto src_md = src_onednn_shape.IsOneDnnTensor()
                        ? src_onednn_shape.GetOneDnnLayout()
                        : memory::desc(src_dims, OneDnnType<T>(), src_format);
      auto diff_dst_md =
          diff_dst_onednn_shape.IsOneDnnTensor()
              ? diff_dst_onednn_shape.GetOneDnnLayout()
              : memory::desc(diff_dst_dims, OneDnnType<T>(), diff_dst_format);
      auto diff_weight_md =
          memory::desc({diff_weight_dims}, OneDnnType<T>(), diff_weight_format);

      auto src_md_any = memory::desc({src_dims}, OneDnnType<T>(),
                                     dnnl::memory::format_tag::any);
      auto diff_dst_md_any = memory::desc({diff_dst_dims}, OneDnnType<T>(),
                                          dnnl::memory::format_tag::any);
      auto diff_weight_md_any = memory::desc(
          {diff_weight_dims}, OneDnnType<T>(), dnnl::memory::format_tag::any);
      auto diff_bias_md = memory::desc({diff_bias_dims}, OneDnnType<T>(),
                                       dnnl::memory::format_tag::x);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<T, float>::value) {
        attr.set_fpmath_mode(fp32_math_mode_);
      }
      auto fwd_pd = dnnl::inner_product_forward::primitive_desc(
          onednn_engine, dnnl::prop_kind::forward, src_md_any,
          diff_weight_md_any, diff_bias_md, diff_dst_md_any, attr);
      auto matmul_bwd_pd = dnnl::inner_product_backward_weights::primitive_desc(
          onednn_engine, src_md_any, diff_weight_md_any, diff_bias_md,
          diff_dst_md_any, fwd_pd, attr);
      auto matmul_bwd_primitive =
          dnnl::inner_product_backward_weights(matmul_bwd_pd);

      // Allocate output tensors.
      Tensor* diff_weight_tensor = nullptr;
      Tensor* diff_bias_tensor = nullptr;
      TensorShape diff_weight_tensor_shape({k, channel});
      // We should always reorder output diff_weight into IO plain format. Only
      // one blocked format(NC) is allowed for 2-dimensional tensor, since
      // OneDnn cannot handle multiple formats in one model.
      OneDnnShape diff_weight_onednn_shape;
      diff_weight_onednn_shape.SetOneDnnTensor(false);
      AllocateOutputSetOneDnnShape(
          context, kDiffWeightIndex, &diff_weight_tensor,
          diff_weight_tensor_shape, diff_weight_onednn_shape);

      TensorShape bias_tensor_shape({channel});
      OneDnnShape bias_onednn_shape;
      bias_onednn_shape.SetOneDnnTensor(false);
      AllocateOutputSetOneDnnShape(context, kDiffBiasIndex, &diff_bias_tensor,
                                   bias_tensor_shape, bias_onednn_shape);

      // Create memory primitive.
      void* src_data = GetTensorBuffer<T>(&src_tensor);
      void* diff_dst_data = GetTensorBuffer<T>(&diff_dst_tensor);
      void* diff_bias_data = GetTensorBuffer<T>(diff_bias_tensor);
      void* diff_weight_data = GetTensorBuffer<T>(diff_weight_tensor);
      dnnl::memory src_mem = CreateDnnlMemory(src_md, onednn_engine, src_data);
      dnnl::memory diff_dst_mem =
          CreateDnnlMemory(diff_dst_md, onednn_engine, diff_dst_data);
      dnnl::memory diff_bias_mem =
          CreateDnnlMemory(diff_bias_md, onednn_engine, diff_bias_data);
      dnnl::memory diff_weight_mem =
          CreateDnnlMemory(diff_weight_md, onednn_engine, diff_weight_data);

      // Reorder input memory.
      dnnl::memory src_reorder_mem;
      Tensor src_reorder_tensor_;
      bool is_src_reordered = (src_md != matmul_bwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size =
            matmul_bwd_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor_));

        src_reorder_mem =
            CreateDnnlMemory(matmul_bwd_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<T>(&src_reorder_tensor_));
        ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
      }

      dnnl::memory diff_dst_reorder_mem;
      Tensor diff_dst_reorder_tensor;
      bool is_diff_dst_reordered =
          (diff_dst_md != matmul_bwd_pd.diff_dst_desc());
      if (is_diff_dst_reordered) {
        int64 diff_dst_reorder_size =
            matmul_bwd_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::v(),
                                    TensorShape({diff_dst_reorder_size}),
                                    &diff_dst_reorder_tensor));

        diff_dst_reorder_mem =
            CreateDnnlMemory(matmul_bwd_pd.diff_dst_desc(), onednn_engine,
                             GetTensorBuffer<T>(&diff_dst_reorder_tensor));
        ReorderMemory(*context, &diff_dst_mem, &diff_dst_reorder_mem,
                      onednn_engine);
      }

      // Prepare tmp buffer for diff_weight.
      Tensor diff_weight_tmp;
      dnnl::memory diff_weight_mem_tmp;
      bool is_diff_weight_reordered =
          (diff_weight_md != matmul_bwd_pd.diff_weights_desc());
      if (is_diff_weight_reordered) {
        int64 diff_weight_size_tmp =
            matmul_bwd_pd.diff_weights_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(
            context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({diff_weight_size_tmp}),
                                            &diff_weight_tmp));
        diff_weight_mem_tmp =
            CreateDnnlMemory(matmul_bwd_pd.diff_weights_desc(), onednn_engine,
                             GetTensorBuffer<T>(&diff_weight_tmp));
      }

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          matmul_bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(matmul_bwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      // Execute.
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> bwd_primitive_args = {
          {DNNL_ARG_SRC, is_src_reordered ? src_reorder_mem : src_mem},
          {DNNL_ARG_DIFF_DST,
           is_diff_dst_reordered ? diff_dst_reorder_mem : diff_dst_mem},
          {DNNL_ARG_DIFF_WEIGHTS,
           is_diff_weight_reordered ? diff_weight_mem_tmp : diff_weight_mem},
          {DNNL_ARG_DIFF_BIAS, diff_bias_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
      matmul_bwd_primitive.execute(onednn_stream, bwd_primitive_args);

      if (is_diff_weight_reordered) {
        ReorderMemory(*context, &diff_weight_mem_tmp, &diff_weight_mem,
                      onednn_engine);
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  bool transpose_a_;
  dnnl::fpmath_mode fp32_math_mode_ = dnnl::fpmath_mode::strict;
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                              \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnMatMul")            \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<TYPE>("T")   \
                              .HostMemory("a_meta")        \
                              .HostMemory("b_meta")        \
                              .HostMemory("product_meta"), \
                          OneDnnMatMulOp<GPUDevice, TYPE>) \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedMatMul")       \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<TYPE>("T")   \
                              .HostMemory("a_meta")        \
                              .HostMemory("b_meta")        \
                              .HostMemory("args_meta")     \
                              .HostMemory("product_meta"), \
                          OneDnnMatMulOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#define REGISTER_FUSEDMATMUL_GRAD_TYPES(type)                \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedMatMulGrad")     \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("T")     \
                              .HostMemory("a_meta")          \
                              .HostMemory("b_meta")          \
                              .HostMemory("product_meta")    \
                              .HostMemory("bias_grad_meta"), \
                          OneDnnFusedMatMulGradOp<GPUDevice, type>);
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_FUSEDMATMUL_GRAD_TYPES);

#else
#define REGISTER_KERNEL(TYPE)                                                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),      \
      OneDnnMatMulOp<CPUDevice, TYPE>)                                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnFusedMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnMatMulOp<CPUDevice, TYPE>)                                         \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedMatMulGrad")                       \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<TYPE>("T"),                      \
                          OneDnnFusedMatMulGradOp<CPUDevice, TYPE>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#endif  // INTEL_CPU_ONLY

}  // namespace itex
