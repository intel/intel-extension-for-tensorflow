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

#ifndef ITEX_CORE_KERNELS_ONEDNN_BLOCK_CONV_OPS_IMPL_H_
#define ITEX_CORE_KERNELS_ONEDNN_BLOCK_CONV_OPS_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/kernels/common/cast_op.h"
#include "itex/core/kernels/common/conv_ops.h"
#include "itex/core/kernels/common/host_data_cache.h"
#include "itex/core/kernels/onednn/block/quantized_ops.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/quantization_util.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;

namespace itex {
template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Tsummand, bool pad_enabled = false,
          bool quantized_bias_enabled = false, bool is_depthwise = false>
class OneDnnConvOp : public OpKernel {
 public:
  explicit OneDnnConvOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));

    is_conv2d_ = (strides_.size() == 4);
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, (strides_.size() == 4 || strides_.size() == 5),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 or 5 dimensions"));

    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    if (context->HasAttr("explicit_paddings")) {
      OP_REQUIRES_OK(
          context, context->GetAttr("explicit_paddings", &explicit_paddings_));
      // TODO(itex): Fix this debug check
      ITEX_DCHECK_OK(CheckValidPadding(padding_, explicit_paddings_,
                                       is_conv2d_ ? 4 : 5, data_format_));
    }

    if (context->HasAttr("is_filter_const")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_filter_const", &is_filter_const_));
    }

    if (is_conv2d_) {
      OP_REQUIRES(context, dilations_.size() == 4,
                  errors::InvalidArgument("Sliding window dilations field must "
                                          "specify 4 dimensions"));
      const int64 dilation_n = GetTensorDim(dilations_, data_format_, 'N');
      const int64 dilation_c = GetTensorDim(dilations_, data_format_, 'C');
      const int64 dilation_h = GetTensorDim(dilations_, data_format_, 'H');
      const int64 dilation_w = GetTensorDim(dilations_, data_format_, 'W');
      OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "dilations in the batch and depth dimensions."));
      OP_REQUIRES(
          context, dilation_h > 0 && dilation_w > 0,
          errors::InvalidArgument("Dilated rates should be larger than 0."));
    } else {
      OP_REQUIRES(context, dilations_.size() == 5,
                  errors::InvalidArgument("Dilation rates field must "
                                          "specify 5 dimensions"));
      OP_REQUIRES(context,
                  (GetTensorDim(dilations_, data_format_, 'N') == 1 &&
                   GetTensorDim(dilations_, data_format_, 'C') == 1),
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "dilations rates in the batch and depth dimensions."));
      OP_REQUIRES(
          context,
          (GetTensorDim(dilations_, data_format_, '0') > 0 &&
           GetTensorDim(dilations_, data_format_, '1') > 0 &&
           GetTensorDim(dilations_, data_format_, '2') > 0),
          errors::InvalidArgument("Dilated rates should be larger than 0."));
    }

    // Padding fusion (pad_enabled).
    if (pad_enabled) {
      OP_REQUIRES(
          context, padding_ == Padding::VALID,
          errors::InvalidArgument("Pad can only be fused with `VALID` Conv."));
    }

    if (context->HasAttr("inplace_sum")) {
      OP_REQUIRES_OK(context, context->GetAttr("inplace_sum", &inplace_sum_));
    }

    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));

    fp32_math_mode_ = GetFP32MathMode<Device>();
  }

  void InitOrSetMemory(OpKernelContext* context) {
    if (!(enable_cache_ && is_init_ &&
          IsInputSame(context, 0, input_dims_, src_onednn_shape_) &&
          IsInputSame(context, 1, filter_dims_, filter_onednn_shape_))) {
      Init(context);
      return;
    }

    if (is_input_zero_) {
      AllocateOutputSetOneDnnShape(context, this->kDstIndex_,
                                   &this->dst_tensor_, this->dst_tf_shape_,
                                   this->dst_onednn_shape_);
      return;
    }

    if (is_src_reordered_) {
      int64 src_out_size = fwd_pd_.src_desc().get_size() / sizeof(Tinput);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                            TensorShape({src_out_size}),
                                            this->src_data_output_.get()));
      src_mem_input_.set_data_handle(context->tensor_data(kSrcIndex_));
      src_mem_.set_data_handle(
          GetTensorBuffer<Tinput>(this->src_data_output_.get()));
      src_reorder_.execute(onednn_stream_, src_reorder_args_);
    } else {
      src_mem_.set_data_handle(context->tensor_data(kSrcIndex_));
    }

    if (is_filter_reordered_) {
      if (!is_filter_const_) {
        filter_mem_input_.set_data_handle(context->tensor_data(kFilterIndex_));
        filter_mem_.set_data_handle(GetTensorBuffer<Tfilter>(&tmp_weight_));

        weight_reorder_.execute(onednn_stream_, weight_reorder_args_);
      }
    } else {
      filter_mem_.set_data_handle(context->tensor_data(kFilterIndex_));
    }

    if (post_op_util_.HasBias()) {
      const Tensor& bias_tensor = context->input(kBiasIndex_);
      Tbias* bias_data = this->GetBiasHandle(context, bias_tensor);
      bias_mem_.set_data_handle(bias_data);
    }

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                          TensorShape({scratchpad_size_}),
                                          scratchpad_tensor_.get()));
    scratchpad_mem_.set_data_handle(
        GetTensorBuffer<Tinput>(scratchpad_tensor_.get()));

    AllocateOutputTensor(context, fwd_pd_, dst_dims_onednn_, data_fmt_onednn_,
                         &dst_onednn_shape_, dst_shape_, &dst_tensor_);
    dst_mem_.set_data_handle(
        reinterpret_cast<Tsummand*>(GetTensorBuffer<Toutput>(dst_tensor_)));
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);
    dst_tensor_ = nullptr;
    onednn_engine_ = CreateDnnlEngine<Device>(*context);
    // onednn_stream has thread safety issue, need create a new one in
    // every compute.
    onednn_stream_ = CreateDnnlStream(*context, onednn_engine_);
    src_data_output_ = std::make_shared<Tensor>();
    scratchpad_tensor_ = std::make_shared<Tensor>();
    InitOrSetMemory(context);

    // Skip primitive execution if the calculation is meaningless.
    if (is_input_zero_) {
      src_data_output_.reset();
      scratchpad_tensor_.reset();
      return;
    }
    if (this->post_op_util_.HasOutputScales()) {
      float* output_scale_ptr = output_scale_cache_.GetCachedPtr(
          context, this->post_op_util_.GetOutputScale().data(),
          this->post_op_util_.GetOutputScale().size());
      dnnl::memory scales_mem(
          {{static_cast<dnnl_dim_t>(
               this->post_op_util_.GetOutputScale().size())},
           memory::data_type::f32,
           memory::format_tag::x},
          this->onednn_engine_, reinterpret_cast<void*>(output_scale_ptr));
      this->fwd_primitives_args_.emplace(
          DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_mem);
    }
    fwd_primitive_.execute(onednn_stream_, fwd_primitives_args_);
    src_data_output_.reset();
    scratchpad_tensor_.reset();
  }

  void Init(OpKernelContext* context) {
    try {
      fwd_primitives_args_.clear();

      // Input tensors
      const Tensor& src_tensor = context->input(kSrcIndex_);
      const Tensor& filter_tensor = context->input(kFilterIndex_);

      auto input_shape = src_tensor.shape();
      input_dims_.clear();
      for (int i = 0; i < input_shape.dims(); ++i) {
        input_dims_.push_back(input_shape.dim_size(i));
      }
      auto filter_tensor_shape = filter_tensor.shape();
      filter_dims_.clear();
      for (int i = 0; i < filter_tensor_shape.dims(); ++i) {
        filter_dims_.push_back(filter_tensor_shape.dim_size(i));
      }

      // Get shapes of input & filter tensors
      GetOneDnnShape(context, kSrcIndex_, &src_onednn_shape_);
      GetOneDnnShape(context, kFilterIndex_, &filter_onednn_shape_);
      TensorShape src_tf_shape = src_onednn_shape_.IsOneDnnTensor()
                                     ? src_onednn_shape_.GetTfShape()
                                     : src_tensor.shape();
      TensorShape filter_tf_shape = filter_tensor.shape();

      // Memory dimensions
      memory::dims src_dims, filter_dims, pad_left_dims, pad_right_dims,
          dilation_dims, stride_dims, bias_dims;
      memory::dims dst_dims_tf;
      OneDnnConvUtil conv_util(context, data_format_, strides_, dilations_,
                               padding_, explicit_paddings_, is_conv2d_,
                               is_depthwise);

      if (pad_enabled) {
        const int kPadIndex =
            post_op_util_.HasBias() ? kBiasIndex_ + 1 : kBiasIndex_;

        conv_util.InitPadWithFusion(kPadIndex, true);
      }
      bool is_grouped_convolution;
      conv_util.InitFwdDimensions(
          src_tf_shape, filter_tf_shape, &src_dims, &filter_dims, &stride_dims,
          &dilation_dims, &dst_dims_tf, &dst_dims_onednn_, &pad_left_dims,
          &pad_right_dims, &is_grouped_convolution);

      // OneDNN dilations start from 0.
      for (int i = 0; i < dilation_dims.size(); ++i) {
        --dilation_dims[i];
      }

      dst_tf_shape_ = OneDnnDimsToTFShape(dst_dims_tf);
      // Corner cases: output with 0 elements and 0 batch size.
      if (dst_tf_shape_.num_elements() == 0 || dst_dims_tf[0] == 0) {
        is_input_zero_ = true;
        AllocateOutputSetOneDnnShape(context, this->kDstIndex_,
                                     &this->dst_tensor_, this->dst_tf_shape_,
                                     this->dst_onednn_shape_);
        is_init_ = true;
        return;
      }

      if (is_depthwise) {
        OP_REQUIRES(context, is_conv2d_,
                    errors::InvalidArgument(
                        "Only 2D convolution is supported for depthwise."));
      }

      // Get OneDnn layout for data
      data_fmt_onednn_ =
          TFDataFormatToOneDnnDataFormat(data_format_, is_conv2d_);
      memory::format_tag data_layout =
          OneDnnTensorFormatToTag(data_fmt_onednn_);
      OP_REQUIRES(context, data_layout != memory::format_tag::undef,
                  errors::InvalidArgument("Invalid data format"));

      // Although filter shape (filter_dims) required is in OneDnn order,
      // the layout is Tensorflow's layout (HWIO) and (HWIGO) for
      // depthwise/group convolutions.
      auto filter_layout = is_conv2d_ ? (is_depthwise || is_grouped_convolution
                                             ? memory::format_tag::hwigo
                                             : memory::format_tag::hwio)
                                      : memory::format_tag::dhwio;
      memory::desc src_md =
          src_onednn_shape_.IsOneDnnTensor()
              ? src_onednn_shape_.GetOneDnnLayout()
              : memory::desc(src_dims, OneDnnType<Tinput>(), data_layout);
      memory::desc src_md_prefer =
          memory::desc(src_dims, OneDnnType<Tinput>(), memory::format_tag::any);
      if (src_dims[1] == 3 && std::is_same<Device, GPUDevice>::value) {
        src_md_prefer = src_md;
      }

      memory::desc filter_md =
          memory::desc(filter_dims, OneDnnType<Tfilter>(), filter_layout);
      // block format filter is allowed with plain src. preferred for both
      // layout disabled or enabled
      memory::desc filter_md_prefer = memory::desc(
          filter_dims, OneDnnType<Tfilter>(), memory::format_tag::any);

      // TODO(itex): redesign the code for Tsummand
      // The reason for using Tsummand is to deal with the situation for int8
      // fusion conv + bias + add + relu fusion. Two inputs for add op may be
      // respectively quint8 and qint8.
      memory::desc dst_md;
      if (std::is_same<Toutput, Tsummand>::value) {
        dst_md = memory::desc({dst_dims_onednn_}, OneDnnType<Toutput>(),
                              memory::format_tag::any);
        add_dst_md_ = dst_md;
      } else if ((std::is_same<Toutput, float>::value ||
                  std::is_same<Toutput, Eigen::half>::value) &&
                 std::is_same<Tfilter, qint8>::value) {
        auto dst_format_tag = src_onednn_shape_.IsOneDnnTensor()
                                  ? src_onednn_shape_.GetFormatTag()
                                  : data_layout;
        dst_md = memory::desc({dst_dims_onednn_}, OneDnnType<Toutput>(),
                              dst_format_tag);
        add_dst_md_ = dst_md;
        src_md_prefer = src_md;
      } else {
        dst_md = memory::desc({dst_dims_onednn_}, OneDnnType<Tsummand>(),
                              memory::format_tag::any);
        add_dst_md_ = memory::desc({dst_dims_onednn_}, OneDnnType<Toutput>(),
                                   memory::format_tag::any);
      }
      // TODO(itex): Currently, we have 2 separate code in dealing with post
      // op.  Try to combine these two codes together
      // 1. For fp32 ops, to read information from "fused_ops" attr during op
      // construction. We use "post_op_util_.AddOps" in this situation
      // 2. For int8 ops, set post op information during op compute, since there
      // is no "fused_ops" attr for these ops. We use "ExtendInt8PostOps" in
      // this situation
      this->ExtendInt8PostOps(context);
      // Set post op attribution.
      dnnl::primitive_attr post_ops_attr;
      post_op_util_.SetPostOpAttr(&post_ops_attr);
      post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<Tinput, float>::value) {
        post_ops_attr.set_fpmath_mode(fp32_math_mode_);
      }

      if (this->post_op_util_.HasOutputScales() &&
          post_op_util_.GetOutputScale().size() > 1 && is_depthwise) {
        // For depthwise convolution mask should be 1<<0 + 1<<1 in onednn3.0
        post_ops_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 3);
      }

      fwd_pd_ = ConvFwdPd(onednn_engine_, prop_kind::forward,
                          dnnl::algorithm::convolution_direct, src_md_prefer,
                          filter_md_prefer, dst_md, stride_dims, dilation_dims,
                          pad_left_dims, pad_right_dims, post_ops_attr);

      if (post_op_util_.HasBias()) {
        const Tensor& bias_tensor = context->input(kBiasIndex_);
        TensorShape bias_tensor_shape = bias_tensor.shape();
        conv_util.GetBiasDimension(bias_tensor_shape, &bias_dims);

        // TODO(itex): use format_tag::any for bias
        auto bias_md =
            memory::desc(bias_dims, OneDnnType<Tbias>(), memory::format_tag::x);
        Tbias* bias_data = this->GetBiasHandle(context, bias_tensor);

        // OneDNN 3.0 requires float Bias for bias in INT8 model, will have an
        // internal conversion when bias is INT8.
        if (std::is_same<Tbias, qint32>::value &&
            !std::is_same<Toutput, qint32>::value) {
          bias_md = memory::desc(bias_dims, OneDnnType<float>(),
                                 memory::format_tag::x);
          bias_mem_ = CreateDnnlMemory(bias_md, onednn_engine_,
                                       static_cast<void*>(bias_data));
        } else {
          bias_mem_ = CreateDnnlMemory(bias_md, onednn_engine_, bias_data);
        }
        fwd_primitives_args_.insert({DNNL_ARG_BIAS, bias_mem_});

        fwd_pd_ = ConvFwdPd(onednn_engine_, prop_kind::forward,
                            dnnl::algorithm::convolution_direct, src_md_prefer,
                            filter_md_prefer, bias_md, dst_md, stride_dims,
                            dilation_dims, pad_left_dims, pad_right_dims,
                            post_ops_attr);
      }
      fwd_primitive_ = dnnl::convolution_forward(fwd_pd_);

      // Create a temp conv primitve desc to get real add md.
      auto add_fwd_pd =
          ConvFwdPd(onednn_engine_, prop_kind::forward,
                    dnnl::algorithm::convolution_direct, src_md_prefer,
                    filter_md_prefer, add_dst_md_, stride_dims, dilation_dims,
                    pad_left_dims, pad_right_dims);
      add_dst_md_ = add_fwd_pd.dst_desc();

      int64 dst_data_size = fwd_pd_.dst_desc().get_size() / sizeof(Toutput);
      dst_shape_ = TensorShape({dst_data_size});

      AllocateOutputTensor(context, fwd_pd_, dst_dims_onednn_, data_fmt_onednn_,
                           &dst_onednn_shape_, dst_shape_, &dst_tensor_);

      // Check whether src and filter need to be reordered.
      is_src_reordered_ = (src_md != fwd_pd_.src_desc());
      src_mem_input_ = CreateDnnlMemory(src_md, onednn_engine_,
                                        GetTensorBuffer<Tinput>(&src_tensor));

      if (is_src_reordered_) {
        int64 src_out_size = fwd_pd_.src_desc().get_size() / sizeof(Tinput);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                              TensorShape({src_out_size}),
                                              src_data_output_.get()));
        src_mem_ =
            CreateDnnlMemory(fwd_pd_.src_desc(), onednn_engine_,
                             GetTensorBuffer<Tinput>(src_data_output_.get()));
        src_reorder_args_.clear();
        src_reorder_args_.insert({DNNL_ARG_SRC, src_mem_input_});
        src_reorder_args_.insert({DNNL_ARG_DST, src_mem_});
        src_reorder_ = dnnl::reorder(src_mem_input_, src_mem_);

        src_reorder_.execute(onednn_stream_, src_reorder_args_);
      } else {
        src_mem_ = src_mem_input_;
      }

      is_filter_reordered_ = (filter_md != fwd_pd_.weights_desc());
      filter_mem_input_ = CreateDnnlMemory(
          filter_md, onednn_engine_, GetTensorBuffer<Tfilter>(&filter_tensor));

      if (is_filter_reordered_) {
        Tfilter* filter_cached_data = nullptr;
        if (is_filter_const_) {
          if (weight_cache_manager_.IsEmpty()) {
            // Cache weight
            weight_cache_manager_.SetCache(
                context, filter_md, fwd_pd_.weights_desc(),
                GetTensorBuffer<Tfilter>(&filter_tensor), onednn_engine_);
          }
          filter_cached_data =
              weight_cache_manager_.GetCache(context, fwd_pd_.weights_desc());
          if (filter_cached_data != nullptr) {
            filter_mem_ = CreateDnnlMemory(fwd_pd_.weights_desc(),
                                           onednn_engine_, filter_cached_data);
          }
        }
        if (filter_cached_data == nullptr) {
          int64 reorder_filter_data_size =
              fwd_pd_.weights_desc().get_size() / sizeof(Tfilter);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DataTypeToEnum<Tfilter>::v(),
                                      TensorShape({reorder_filter_data_size}),
                                      &tmp_weight_));
          void* filter_data_handle = GetTensorBuffer<Tfilter>(&tmp_weight_);
          filter_mem_ = CreateDnnlMemory(fwd_pd_.weights_desc(), onednn_engine_,
                                         filter_data_handle);
          weight_reorder_args_.clear();
          weight_reorder_args_.insert({DNNL_ARG_SRC, filter_mem_input_});
          weight_reorder_args_.insert({DNNL_ARG_DST, filter_mem_});
          weight_reorder_ = dnnl::reorder(filter_mem_input_, filter_mem_);
          weight_reorder_.execute(onednn_stream_, weight_reorder_args_);
        }
      } else {
        filter_mem_ = filter_mem_input_;
      }

      // TODO(itex): redesign the code for Tsummand
      dst_mem_ = CreateDnnlMemory(
          fwd_pd_.dst_desc(), onednn_engine_,
          reinterpret_cast<Tsummand*>(GetTensorBuffer<Toutput>(dst_tensor_)));

      scratchpad_size_ = fwd_pd_.scratchpad_desc().get_size() / sizeof(Tinput);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                            TensorShape({scratchpad_size_}),
                                            scratchpad_tensor_.get()));
      scratchpad_mem_ =
          dnnl::memory(fwd_pd_.scratchpad_desc(), onednn_engine_,
                       GetTensorBuffer<Tinput>(scratchpad_tensor_.get()));

      // Execute convolution
      fwd_primitives_args_.insert({DNNL_ARG_SRC, src_mem_});
      fwd_primitives_args_.insert({DNNL_ARG_WEIGHTS, filter_mem_});
      fwd_primitives_args_.insert({DNNL_ARG_DST, dst_mem_});
      fwd_primitives_args_.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem_});

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

 protected:
  std::vector<int64_t> explicit_paddings_;
  const int kSrcIndex_ = 0, kFilterIndex_ = 1, kBiasIndex_ = 2, kAddIndex_ = 3;
  const int kDstIndex_ = 0;

  /* Fused Conv */
  PostOpUtil post_op_util_;

  // Cache oneDNN object and TF memory
  bool is_init_ = false;
  bool is_input_zero_ = false;
  bool is_src_reordered_ = false;
  bool is_filter_reordered_ = false;

  // This one for dnnl primitive input
  dnnl::memory src_mem_;
  // This one for TF input when input need reorder.
  dnnl::memory src_mem_input_;
  // This one for dnnl primitive weight
  dnnl::memory filter_mem_;
  // This one for TF weight when weight need reorder.
  dnnl::memory filter_mem_input_;
  dnnl::memory dst_mem_;
  dnnl::memory scratchpad_mem_;
  dnnl::memory bias_mem_;
  dnnl::memory::dims dst_dims_onednn_;

  // This one for dnnl sum fusion.
  dnnl::memory::desc add_dst_md_;

  dnnl::stream onednn_stream_;
  dnnl::engine onednn_engine_;

  dnnl::reorder src_reorder_;
  dnnl::reorder weight_reorder_;
  primitive fwd_primitive_;
  ConvFwdPd fwd_pd_;

  std::unordered_map<int, memory> fwd_primitives_args_;
  std::unordered_map<int, memory> src_reorder_args_;
  std::unordered_map<int, memory> weight_reorder_args_;

  OneDnnShape dst_onednn_shape_;
  TensorShape dst_tf_shape_;
  OneDnnTensorFormat data_fmt_onednn_;
  TensorShape dst_shape_;
  std::vector<int64> input_dims_, filter_dims_;
  OneDnnShape src_onednn_shape_, filter_onednn_shape_;

  std::shared_ptr<Tensor> src_data_output_;
  Tensor* dst_tensor_ = nullptr;
  // This one for dnnl primitive weight when weight need reorder.
  Tensor tmp_weight_;
  std::shared_ptr<Tensor> scratchpad_tensor_;
  int64_t scratchpad_size_ = 0;

  bool enable_cache_ = false;
  dnnl::fpmath_mode fp32_math_mode_ = dnnl::fpmath_mode::strict;

 private:
  bool is_conv2d_;
  bool is_filter_const_ = false;
  bool inplace_sum_ = false;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  // Weight cache manager
  WeightCacheManager<Tfilter> weight_cache_manager_;

  mutex mu_compute_;

  HostDataCache<Device, float> output_scale_cache_;

 protected:
  // ExtendInt8PostOps is only used in Int8 ops.
  virtual void ExtendInt8PostOps(OpKernelContext* context) {}

  virtual void AllocateOutputTensor(
      OpKernelContext* context,
      const dnnl::convolution_forward::primitive_desc& conv_pd,
      const memory::dims& dst_dims_onednn, OneDnnTensorFormat dst_tf_format,
      OneDnnShape* dst_onednn_shape, TensorShape tensor_shape,
      Tensor** dst_tensor) {
    ITEX_DCHECK(dst_tensor);

    // NHWC Conv may prefer NCHW (also plain format) as its primitive format.
    // Need to record this info in meta data to reorder the data correctly.
    SetOutputTensorShape(add_dst_md_, dst_tf_format, &tensor_shape,
                         dst_onednn_shape, true /*is_onednn*/);

    // TODO(itex): Try to apply the code below to Int8 situation
    if (post_op_util_.HasAdd() &&
        (std::is_same<Toutput, float>::value ||
         std::is_same<Toutput, Eigen::half>::value ||
         std::is_same<Toutput, Eigen::bfloat16>::value)) {
      const Tensor& add_tensor = context->input(kAddIndex_);
      OneDnnShape add_onednn_shape;
      GetOneDnnShape(context, kAddIndex_, &add_onednn_shape);

      // Check if reorder is needed.
      if (add_onednn_shape == *dst_onednn_shape) {
        // TODO(itex): Remove this workaround when inplace works.
        if (inplace_sum_) {
          context->set_output(kDstIndex_, add_tensor);
          ForwardMetaData(context, kAddIndex_, kDstIndex_, *dst_onednn_shape);
          *dst_tensor = context->mutable_output(kDstIndex_);
          return;
        }
        const int kUnsuccess = -1;
        int is_forward_success = kUnsuccess;
        ForwardOrAllocateOutputSetOneDnnShape(
            context, kAddIndex_, kDstIndex_, dst_tensor, tensor_shape,
            *dst_onednn_shape, &is_forward_success);

        // Everything is done if forward succeed.
        if (is_forward_success != kUnsuccess) return;
      }

      // Reorder is needed. Check `*dst_tensor` first:
      //   1) nullptr, add shape is different with dst shape;
      //   2) not nullptr, forward is failed but dst has been allocated;
      if (*dst_tensor == nullptr) {
        AllocateOutputSetOneDnnShape(context, kDstIndex_, dst_tensor,
                                     tensor_shape, *dst_onednn_shape);
      }

      auto dst_layout =
          OneDnnTensorFormatToTag(dst_onednn_shape->GetTfDataFormat());
      OP_REQUIRES(context, dst_layout != memory::format_tag::undef,
                  errors::InvalidArgument(
                      "OneDnnConvOp: Invalid data format in AddN fusion."));
      auto add_md = add_onednn_shape.IsOneDnnTensor()
                        ? add_onednn_shape.GetOneDnnLayout()
                        : memory::desc(dst_dims_onednn, OneDnnType<Tsummand>(),
                                       dst_layout);
      memory fuse_add_src = memory(add_md, this->onednn_engine_,
                                   GetTensorBuffer<Toutput>(&add_tensor));
      memory fuse_add_dst = memory(add_dst_md_, this->onednn_engine_,
                                   GetTensorBuffer<Toutput>(*dst_tensor));
      ReorderMemory(*context, &fuse_add_src, &fuse_add_dst,
                    this->onednn_engine_);
    } else {
      OP_REQUIRES(
          context,
          (!post_op_util_.HasAdd() ||
           (post_op_util_.HasAdd() && (std::is_same<Toutput, qint8>::value ||
                                       std::is_same<Toutput, quint8>::value ||
                                       std::is_same<Toutput, qint32>::value))),
          errors::InvalidArgument(
              "OneDnnConvOp: Invalid data type in AddN fusion."));
      AllocateOutputSetOneDnnShape(context, kDstIndex_, dst_tensor,
                                   tensor_shape, *dst_onednn_shape);
    }
    return;
  }

  virtual Tbias* GetBiasHandle(OpKernelContext* context,
                               const Tensor& bias_tensor) {
    return static_cast<Tbias*>(
        const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
  }
};

template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Tsummand, bool pad_enabled = false,
          bool quantized_bias_enabled = false, bool is_depthwise = false>
class OneDnnFusedConvOp
    : public OneDnnConvOp<Device, Tinput, Tfilter, Tbias, Toutput, Tsummand,
                          pad_enabled, quantized_bias_enabled, is_depthwise> {
 public:
  explicit OneDnnFusedConvOp(OpKernelConstruction* context)
      : OneDnnConvOp<Device, Tinput, Tfilter, Tbias, Toutput, Tsummand,
                     pad_enabled, quantized_bias_enabled, is_depthwise>(
            context) {
    int num_args;
    std::vector<string> fused_ops;

    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    OP_REQUIRES(context, !(fused_ops.empty()),
                errors::InvalidArgument(
                    "Fused Conv2D must have at least one fused op."));
    OP_REQUIRES(
        context, this->post_op_util_.AddOps(fused_ops),
        errors::InvalidArgument("Found unsupported fusion in Fused Conv2D."));

    // Set alpha if get `LeakyRelu` after adding ops.
    if (this->post_op_util_.HasLeakyRelu()) {
      float alpha;
      OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha", &alpha));
      this->post_op_util_.SetLeakyReluAlpha(alpha);
    }

    // FIXME(itex): Check this when `num_inputs()` is implemented.
    // const int num_args_expected = context->num_inputs() / 2 - 2;
    // OP_REQUIRES(
    //     context, num_args_expected == num_args,
    //     errors::InvalidArgument("Expect num_args to be ", num_args_expected,
    //                             " but received ", num_args));
  }
  virtual ~OneDnnFusedConvOp() {}
};

template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Tsummand, bool quantized_bias_enabled, bool is_depthwise>
class OneDnnQuantizedConvOp
    // Currently, Tfilter only supports qint8
    : public OneDnnConvOp<Device, Tinput, qint8, Tbias, Toutput, Tsummand,
                          false, quantized_bias_enabled, is_depthwise> {
 public:
  explicit OneDnnQuantizedConvOp(OpKernelConstruction* context)
      : OneDnnConvOp<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
                     quantized_bias_enabled, is_depthwise>(context) {
    bool is_filter_const;
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_filter_const", &is_filter_const));

    if (quantized_bias_enabled) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_bias_const", &is_bias_const_));
    }

    OP_REQUIRES(context, is_filter_const,
                errors::InvalidArgument("Filter must be a constant"));

    // Code to deal with some legacy int8 pb
    if (context->HasAttr("padding_list")) {
      OP_REQUIRES_OK(
          context, context->GetAttr("padding_list", &this->explicit_paddings_));
    }

    std::vector<string> fused_ops;
    fused_ops.push_back("Quantized");
    if (quantized_bias_enabled) {
      fused_ops.push_back("BiasAdd");
    }
    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops),
                errors::InvalidArgument(
                    "Found unsupported fusion in QuantizedConv2D."));

    // Set input/output tensor index
    int bias_index_offset = quantized_bias_enabled ? 1 : 0;
    kSrcMinRangeIndex = 2 + bias_index_offset;
    kSrcMaxRangeIndex = 3 + bias_index_offset;
    kFilterMinRangeIndex = 4 + bias_index_offset;
    kFilterMaxRangeIndex = 5 + bias_index_offset;
    kMinFreezedIndex = 6 + bias_index_offset;
    kMaxFreezedIndex = 7 + bias_index_offset;
  }

  void Compute(OpKernelContext* context) override {
    // Compute int32 output tensor
    OneDnnConvOp<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
                 quantized_bias_enabled, is_depthwise>::Compute(context);

    const float min_input = context->input(kSrcMinRangeIndex).flat<float>()(0);
    const float max_input = context->input(kSrcMaxRangeIndex).flat<float>()(0);

    AllocateBlockOutputMinMax<Tinput, qint8, Toutput>(
        context, min_input, max_input, kFilterMinRangeIndex,
        kFilterMaxRangeIndex, kMinFreezedIndex, kMaxFreezedIndex,
        kDstMinRangeIndex, kDstMaxRangeIndex);
  }

 protected:
  void ExtendInt8PostOps(OpKernelContext* context) override {
    // When the output type is quint8, the output data is requantized
    // into quint8. A post_op "output_scale" is added to do the conversion.
    // Otherwise the output_scale will be 1.f
    const Tensor& min_filter_vector = context->input(kFilterMinRangeIndex);
    const Tensor& max_filter_vector = context->input(kFilterMaxRangeIndex);
    size_t depth = min_filter_vector.NumElements();
    std::vector<float> scales(depth, 1.f);

    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value) {
      const float min_input =
          context->input(kSrcMinRangeIndex).flat<float>()(0);
      const float max_input =
          context->input(kSrcMaxRangeIndex).flat<float>()(0);

      // min_freezed_output and max_freezed_output are the actual range
      // for the output.
      const float min_freezed_output =
          context->input(kMinFreezedIndex).flat<float>()(0);
      const float max_freezed_output =
          context->input(kMaxFreezedIndex).flat<float>()(0);

      float int_output_limit =
          std::is_same<Toutput, quint8>::value ? 255.0f : 127.0f;

      const float* min_filter = min_filter_vector.flat<float>().data();
      const float* max_filter = max_filter_vector.flat<float>().data();

      float float_input_range =
          std::max(std::abs(min_input), std::abs(max_input));
      float float_output_range =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      const float int_const_scale_limit =
          (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
      for (size_t i = 0; i < depth; ++i) {
        // For simplicity and symmetry, we set filter range to be outer
        // bounds of min_filter and max_filter.
        float float_filter_range =
            std::max(std::abs(min_filter[i]), std::abs(max_filter[i]));
        scales[i] = int_output_limit * float_input_range * float_filter_range /
                    (int_const_scale_limit * float_output_range);
      }
    }
    this->post_op_util_.SetOutputScale(scales);
  }

  Tbias* GetBiasHandle(OpKernelContext* context,
                       const Tensor& bias_tensor) override {
    if (std::is_same<Tbias, qint32>::value) {
      if (std::is_same<Toutput, qint32>::value) {
        return static_cast<Tbias*>(
            const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
      }
      if (is_bias_const_ && !bias_cache_manager.IsEmpty()) {
        return static_cast<Tbias*>(bias_cache_manager.GetCache(context));
      }
      Tensor scaled_bias;
      TF_ABORT_IF_ERROR(context->allocate_temp(
          DataTypeToEnum<float>::v(), bias_tensor.shape(), &scaled_bias));
      const Device& d = context->eigen_device<Device>();

      Tensor bias_tensor_int32;
      ITEX_CHECK_OK(bias_tensor_int32.BitcastFrom(bias_tensor, DT_INT32,
                                                  bias_tensor.shape()));
      CastDataType<Device, int32, float>{}(
          d, const_cast<const Tensor&>(bias_tensor_int32).flat<int32>(),
          scaled_bias.flat<float>());

      const std::vector<float>& scale = this->post_op_util_.GetOutputScale();
      float* bias_scales_ptr;
      if (std::is_same<Toutput, float>::value ||
          std::is_same<Toutput, Eigen::bfloat16>::value ||
          std::is_same<Toutput, Eigen::half>::value) {
        const float min_input =
            context->input(kSrcMinRangeIndex).flat<float>()(0);
        const float max_input =
            context->input(kSrcMaxRangeIndex).flat<float>()(0);
        const Tensor& min_filter_vector = context->input(kFilterMinRangeIndex);
        const Tensor& max_filter_vector = context->input(kFilterMaxRangeIndex);
        const float* min_filter = min_filter_vector.flat<float>().data();
        const float* max_filter = max_filter_vector.flat<float>().data();

        const float int_const_scale_limit =
            (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0
                                                  : 127.0 * 127.0;
        // Re-scale bias if either of following 2 conditions are met:
        // 1. Bias is not const;
        // 2. Bias is const, but bias cache is empty (first iteration).

        // TODO(itex): avoid to use new memory
        size_t depth = min_filter_vector.NumElements();
        scales_.resize(depth);

        for (size_t i = 0; i < depth; ++i) {
          float tmp_scale =
              (std::max(std::abs(max_input), std::abs(min_input)) *
               std::max(std::abs(max_filter[i]), std::abs(min_filter[i]))) /
              int_const_scale_limit;
          // TODO(itex): Check whether delete some instuctions about
          // scales_are_valid is correct
          scales_[i] = tmp_scale;
        }
        if (bias_cache_manager.IsEmpty()) {
          bias_scales_ptr = bias_scale_cache_.GetCachedPtr(
              context, scales_.data(), scales_.size());
        }

      } else {
        if (bias_cache_manager.IsEmpty()) {
          bias_scales_ptr = bias_scale_cache_.GetCachedPtr(
              context, scale.data(), scale.size());
        }
      }
      if (bias_cache_manager.IsEmpty()) {
        dnnl::primitive_attr bias_attr;
        memory bias_scales_mem({{static_cast<dnnl_dim_t>(scale.size())},
                                memory::data_type::f32,
                                memory::format_tag::x},
                               this->onednn_engine_,
                               reinterpret_cast<void*>(bias_scales_ptr));
        if (scale.size() == 1) {
          bias_attr.set_scales_mask(DNNL_ARG_SRC, 0);
        } else {
          bias_attr.set_scales_mask(DNNL_ARG_SRC, 1);
        }

        auto bias_md =
            memory::desc({static_cast<int>(bias_tensor.NumElements())},
                         OneDnnType<float>(), memory::format_tag::x);
        void* bias_data = static_cast<void*>(
            const_cast<float*>(scaled_bias.flat<float>().data()));

        // TODO(itex): Check whether the bias_md is always equals to
        // conv_pd.bias_desc()
        bias_cache_manager.SetCache(context, bias_md, bias_attr, bias_data,
                                    this->onednn_engine_, bias_scales_mem);
      }
      return static_cast<Tbias*>(bias_cache_manager.GetCache(context));
    }

    const float min_input = context->input(kSrcMinRangeIndex).flat<float>()(0);
    const float max_input = context->input(kSrcMaxRangeIndex).flat<float>()(0);
    const Tensor& min_filter_vector = context->input(kFilterMinRangeIndex);
    const Tensor& max_filter_vector = context->input(kFilterMaxRangeIndex);
    const float* min_filter = min_filter_vector.flat<float>().data();
    const float* max_filter = max_filter_vector.flat<float>().data();

    const float int_const_scale_limit =
        (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
    // Re-scale bias if either of following 2 conditions are met:
    // 1. Bias is not const;
    // 2. Bias is const, but bias cache is empty (first iteration).

    // TODO(itex): avoid to use new memory
    size_t depth = min_filter_vector.NumElements();
    scales_.resize(depth);
    const std::vector<float>& scale = this->post_op_util_.GetOutputScale();
    for (size_t i = 0; i < depth; ++i) {
      float tmp_scale =
          int_const_scale_limit /
          (std::max(std::abs(max_input), std::abs(min_input)) *
           std::max(std::abs(max_filter[i]), std::abs(min_filter[i])));
      // TODO(itex): Check whether delete some instuctions about
      // scales_are_valid is correct
      scales_[i] = tmp_scale * scale[i];
    }
    // TODO(itex): is_bias_const_ is useless, delete it
    if (!is_bias_const_ || bias_cache_manager.IsEmpty()) {
      dnnl::primitive_attr bias_attr;
      float* bias_scales_ptr =
          bias_scale_cache_.GetCachedPtr(context, scales_.data(), depth);
      memory bias_scales_mem({{static_cast<dnnl_dim_t>(depth)},
                              memory::data_type::f32,
                              memory::format_tag::x},
                             this->onednn_engine_,
                             reinterpret_cast<void*>(bias_scales_ptr));
      if (depth == 1) {
        bias_attr.set_scales_mask(DNNL_ARG_SRC, 0);
      } else {
        bias_attr.set_scales_mask(DNNL_ARG_SRC, 1);
      }

      auto bias_md = memory::desc({static_cast<int>(bias_tensor.NumElements())},
                                  OneDnnType<Tbias>(), memory::format_tag::x);
      void* bias_data = static_cast<void*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));

      // TODO(itex): Check whether the bias_md is always equals to
      // conv_pd.bias_desc()
      bias_cache_manager.SetCache(context, bias_md, bias_attr, bias_data,
                                  this->onednn_engine_, bias_scales_mem);
    }
    return bias_cache_manager.GetCache(context);
  }

 protected:
  bool is_bias_const_;

  // input and output tensor index
  int kSrcMinRangeIndex;
  int kSrcMaxRangeIndex;
  int kFilterMinRangeIndex;
  int kFilterMaxRangeIndex;
  int kMinFreezedIndex;
  int kMaxFreezedIndex;
  const int kDstMinRangeIndex = 1;
  const int kDstMaxRangeIndex = 2;

 private:
  std::vector<float> scales_;
  // Bias cache manager
  BiasCacheManager<Tbias> bias_cache_manager;
  HostDataCache<Device, float> bias_scale_cache_;
};

template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Tsummand, bool quantized_bias_enabled, bool is_depthwise>
class OneDnnQuantizedConvReluOp
    : public OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                                   quantized_bias_enabled, is_depthwise> {
 public:
  explicit OneDnnQuantizedConvReluOp(OpKernelConstruction* context)
      : OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                              quantized_bias_enabled, is_depthwise>(context) {
    std::vector<string> fused_ops;
    fused_ops.push_back("Relu");
    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops),
                errors::InvalidArgument(
                    "Found unsupported fusion in QuantizedConvRelu."));
  }

 protected:
  void ExtendInt8PostOps(OpKernelContext* context) override {
    OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                          quantized_bias_enabled,
                          is_depthwise>::ExtendInt8PostOps(context);
    this->post_op_util_.SetPostOpScale("Relu", 1.0);
  }
};

template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Tsummand, bool quantized_bias_enabled, bool is_depthwise>
class OneDnnQuantizedConvSumReluOp
    : public OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                                   quantized_bias_enabled, is_depthwise> {
 public:
  explicit OneDnnQuantizedConvSumReluOp(OpKernelConstruction* context)
      : OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                              quantized_bias_enabled, is_depthwise>(context) {
    std::vector<string> fused_ops;
    fused_ops.push_back("Add");
    fused_ops.push_back("Relu");
    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops),
                errors::InvalidArgument(
                    "Found unsupported fusion in QuantizedConvSumRelu."));

    int bias_index_offset = quantized_bias_enabled ? 1 : 0;
    int min_max_freezed_offset = (std::is_same<Toutput, quint8>::value ||
                                  std::is_same<Toutput, qint8>::value)
                                     ? 2
                                     : 0;
    kSummandDataIndex = 6 + bias_index_offset + min_max_freezed_offset;
    kSummandMinRangeIndex = 7 + bias_index_offset + min_max_freezed_offset;
    kSummandMaxRangeIndex = 8 + bias_index_offset + min_max_freezed_offset;
  }

 protected:
  void ExtendInt8PostOps(OpKernelContext* context) override {
    OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                          quantized_bias_enabled,
                          is_depthwise>::ExtendInt8PostOps(context);
    // Calculate the scale (beta in OneDnn api term) for sum
    float sum_post_op_scale;
    if (!std::is_same<Toutput, qint32>::value) {
      const Tensor& summand = context->input(kSummandDataIndex);
      // TODO(itex): investigate OpKernel::input_type
      DataType summand_type = summand.dtype();
      ITEX_CHECK((summand_type == DT_QINT8) || (summand_type == DT_QUINT8));

      const float min_freezed_output =
          context->input(this->kMinFreezedIndex).template flat<float>()(0);
      const float max_freezed_output =
          context->input(this->kMaxFreezedIndex).template flat<float>()(0);
      const float min_freezed_summand =
          context->input(kSummandMinRangeIndex).flat<float>()(0);
      const float max_freezed_summand =
          context->input(kSummandMaxRangeIndex).flat<float>()(0);

      float scale_output =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      float scale_summand = std::max(std::abs(min_freezed_summand),
                                     std::abs(max_freezed_summand));
      // if summand_type is also DT_QUINT8 as the scale_output,
      // the scaling factor of 255.0f cancels each other and thus is avoided.
      // If it is not then  it is DT_INT8 and is scaled appropriately.

      if (std::is_same<Toutput, quint8>::value && summand_type == DT_QINT8) {
        sum_post_op_scale = 255.0f * scale_summand / (scale_output * 127.0f);
      } else {
        sum_post_op_scale = scale_summand / scale_output;
      }
    } else {
      sum_post_op_scale = 1.0;
    }

    this->post_op_util_.SetPostOpScale("Add", sum_post_op_scale);
    this->post_op_util_.SetPostOpScale("Relu", 1.0);
  }

  void AllocateOutputTensor(OpKernelContext* context,
                            const ConvFwdPd& conv_prim_desc,
                            const memory::dims& output_dims_onednn_order,
                            OneDnnTensorFormat output_tf_format,
                            OneDnnShape* output_onednn_shape,
                            TensorShape tensor_shape,
                            Tensor** dst_tensor) override {
    if (!std::is_same<Toutput, qint32>::value) {
      Tensor& summand = const_cast<Tensor&>(context->input(kSummandDataIndex));

      // TODO(itex): We could try to use Tsummand here
      DataType summand_type = summand.dtype();
      ITEX_CHECK((summand_type == DT_QINT8) || (summand_type == DT_QUINT8));

      // TODO(itex): Handle both block and plain layout tensors
      if (std::is_same<Toutput, quint8>::value && summand_type == DT_QINT8) {
        // TODO(itex): TF proper uses bitcastfrom, check whether there is
        // problem here.
        OP_REQUIRES_OK(
            context, summand.BitcastFrom(summand, DT_QUINT8, summand.shape()));
      }

      // Here is workaround to always forward add tensor in conv + bias + add +
      // relu int8 fusion
      // FIXME(itex): Implement code for "inplace_sum = False" and discuss with
      // LPOT about new design.
      // JIRA: https://jira.devtools.intel.com/browse/TFDO-5059
      if (std::is_same<Toutput, qint8>::value &&
          std::is_same<Tsummand, qint8>::value &&
          context->input(kSummandDataIndex).dtype() == DT_QUINT8) {
        // To bypass the INC pb generation bug. INC may wrongly set Tsummand
        // attr qint8 when the actual input is quint8. Intel-TF can avoid the
        // issue by internal type check in forward_input_to_output_with_shape.
        // Since ITEX have to use set_output here, it will always inplace, and
        // cause crash.
        // TODO(itex): Discuss with INC to fix incorrect pb.
        OP_REQUIRES_OK(context,
                       context->allocate_output(this->kDstIndex_, tensor_shape,
                                                dst_tensor));
      } else {
        context->set_output(this->kDstIndex_,
                            context->input(kSummandDataIndex));
      }

      SetOutputTensorShape(this->add_dst_md_, output_tf_format, &tensor_shape,
                           output_onednn_shape, true /*is_onednn*/);
      AllocateMetaData(context, this->kDstIndex_, *output_onednn_shape);
      *dst_tensor = context->mutable_output(this->kDstIndex_);
      return;
    }
    // TODO(itex): investigate the influence of additional attr tensor_shape
    OneDnnConvOp<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
                 quantized_bias_enabled,
                 is_depthwise>::AllocateOutputTensor(context, conv_prim_desc,
                                                     output_dims_onednn_order,
                                                     output_tf_format,
                                                     output_onednn_shape,
                                                     tensor_shape, dst_tensor);
    const Tensor& summand = context->input(kSummandDataIndex);
    if (summand.dtype() != DT_FLOAT) {
      ITEX_LOG(FATAL) << "Current fusion requires summand to be float";
    }
    OneDnnShape summand_onednn_shape;
    GetOneDnnShape(context, kSummandDataIndex, &summand_onednn_shape);
    // We need to compute scale for the summand
    const float min_input =
        context->input(this->kSrcMinRangeIndex).template flat<float>()(0);
    const float max_input =
        context->input(this->kSrcMaxRangeIndex).template flat<float>()(0);
    const Tensor& min_filter_vector =
        context->input(this->kFilterMinRangeIndex);
    const Tensor& max_filter_vector =
        context->input(this->kFilterMaxRangeIndex);
    const float* min_filter = min_filter_vector.flat<float>().data();
    const float* max_filter = max_filter_vector.flat<float>().data();

    const float int_const_scale_limit =
        (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
    size_t depth = min_filter_vector.NumElements();
    std::vector<float> scales(depth);
    for (size_t i = 0; i < depth; ++i) {
      // TODO(itex): scale factors for UINT8(inputs) & INT8(weights) are
      // done regularly. A Cleaner design to address all mapping in one
      // function needs to be implemented in future which also supports other
      // quantized type mapping in future.
      scales[i] = int_const_scale_limit /
                  (std::max(std::abs(max_input), std::abs(min_input)) *
                   std::max(std::abs(max_filter[i]), std::abs(min_filter[i])));
    }
    dnnl::primitive_attr reorder_attr;
    dnnl::memory reorder_scales_mem({{static_cast<dnnl_dim_t>(depth)},
                                     dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::x},
                                    this->onednn_engine_,
                                    reinterpret_cast<void*>(scales.data()));
    if (depth == 1) {
      reorder_attr.set_scales_mask(DNNL_ARG_SRC, 0);
    } else {
      reorder_attr.set_scales_mask(DNNL_ARG_SRC, 2);
    }

    auto summand_md =
        summand_onednn_shape.IsOneDnnTensor()
            ? summand_onednn_shape.GetOneDnnLayout()
            // TODO(itex): is hard code nhwc correct?
            : memory::desc(output_dims_onednn_order, OneDnnType<Tbias>(),
                           memory::format_tag::nhwc);

    // Reasons for using Tbias: if code here is executed before the requantize
    // op is fused with int8 conv op. At that time, both bias tensor and summand
    // tensor are fp32.
    void* summand_buf =
        static_cast<void*>(const_cast<Tbias*>(summand.flat<Tbias>().data()));
    void* dst_buf = static_cast<void*>((*dst_tensor)->flat<Tsummand>().data());

    memory summand_mem =
        CreateDnnlMemory(summand_md, this->onednn_engine_, summand_buf);
    memory dst_mem = CreateDnnlMemory(conv_prim_desc.dst_desc(),
                                      this->onednn_engine_, dst_buf);

    dnnl::reorder summand_scaled_primitive =
        dnnl::reorder(summand_mem, dst_mem, reorder_attr);
    std::unordered_map<int, dnnl::memory> reorder_args = {
        {DNNL_ARG_SRC, summand_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, reorder_scales_mem},
    };
    auto onednn_stream = CreateDnnlStream(*context, this->onednn_engine_);
    summand_scaled_primitive.execute(onednn_stream, reorder_args);
  }

 protected:
  int kSummandDataIndex;
  int kSummandMinRangeIndex;
  int kSummandMaxRangeIndex;
};

template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Tsummand, bool quantized_bias_enabled, bool is_depthwise>
class OneDnnQuantizeV2WithQuantizedConv2DOp
    : public OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                                   quantized_bias_enabled, is_depthwise> {
 public:
  explicit OneDnnQuantizeV2WithQuantizedConv2DOp(OpKernelConstruction* context)
      : OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                              quantized_bias_enabled, is_depthwise>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));

    is_conv2d_ = (strides_.size() == 4);
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, (strides_.size() == 4 || strides_.size() == 5),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 or 5 dimensions"));

    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

    // Code to deal with some legacy int8 pb
    if (context->HasAttr("padding_list")) {
      OP_REQUIRES_OK(
          context, context->GetAttr("padding_list", &this->explicit_paddings_));
    }

    if (context->HasAttr("is_filter_const")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_filter_const", &is_filter_const_));
    }

    if (is_conv2d_) {
      OP_REQUIRES(context, dilations_.size() == 4,
                  errors::InvalidArgument("Sliding window dilations field must "
                                          "specify 4 dimensions"));
      const int64 dilation_n = GetTensorDim(dilations_, data_format_, 'N');
      const int64 dilation_c = GetTensorDim(dilations_, data_format_, 'C');
      const int64 dilation_h = GetTensorDim(dilations_, data_format_, 'H');
      const int64 dilation_w = GetTensorDim(dilations_, data_format_, 'W');
      OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "dilations in the batch and depth dimensions."));
      OP_REQUIRES(
          context, dilation_h > 0 && dilation_w > 0,
          errors::InvalidArgument("Dilated rates should be larger than 0."));
    } else {
      OP_REQUIRES(context, dilations_.size() == 5,
                  errors::InvalidArgument("Dilation rates field must "
                                          "specify 5 dimensions"));
      OP_REQUIRES(context,
                  (GetTensorDim(dilations_, data_format_, 'N') == 1 &&
                   GetTensorDim(dilations_, data_format_, 'C') == 1),
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "dilations rates in the batch and depth dimensions."));
      OP_REQUIRES(
          context,
          (GetTensorDim(dilations_, data_format_, '0') > 0 &&
           GetTensorDim(dilations_, data_format_, '1') > 0 &&
           GetTensorDim(dilations_, data_format_, '2') > 0),
          errors::InvalidArgument("Dilated rates should be larger than 0."));
    }

    // Padding fusion (pad_enabled).
    if (pad_enabled) {
      OP_REQUIRES(
          context, padding_ == Padding::VALID,
          errors::InvalidArgument("Pad can only be fused with `VALID` Conv."));
    }

    if (context->HasAttr("inplace_sum")) {
      OP_REQUIRES_OK(context, context->GetAttr("inplace_sum", &inplace_sum_));
    }

    string mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_string));
    OP_REQUIRES(context,
                (mode_string == "MIN_COMBINED" || mode_string == "MIN_FIRST" ||
                 mode_string == "SCALED"),
                errors::InvalidArgument("Mode string must be 'MIN_COMBINED',"
                                        " 'MIN_FIRST', or 'SCALED', is '" +
                                        mode_string + "'"));
    if (mode_string == "MIN_COMBINED") {
      mode_ = QuantizeMode::MIN_COMBINED;
    } else if (mode_string == "MIN_FIRST") {
      mode_ = QuantizeMode::MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QuantizeMode::SCALED;
    }
    OP_REQUIRES(context,
                (mode_string == "SCALED" || mode_string == "MIN_FIRST"),
                errors::InvalidArgument(
                    "_ITEXQuantizeV2 only supports SCALED and MIN_FIRST MODE"));

    string round_mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("round_mode", &round_mode_string));
    OP_REQUIRES(context,
                (round_mode_string == "HALF_AWAY_FROM_ZERO" ||
                 round_mode_string == "HALF_TO_EVEN"),
                errors::InvalidArgument("Round mode string must be "
                                        "'HALF_AWAY_FROM_ZERO' or "
                                        "'HALF_TO_EVEN', is '" +
                                        round_mode_string + "'"));
    if (round_mode_string == "HALF_AWAY_FROM_ZERO") {
      round_mode_ = QuantizeRoundMode::ROUND_HALF_AWAY_FROM_ZERO;
    } else if (round_mode_string == "HALF_TO_EVEN") {
      OP_REQUIRES(context, mode_string == "SCALED",
                  errors::InvalidArgument("Round mode 'HALF_TO_EVEN' "
                                          "only supported for mode 'SCALED', "
                                          "but mode is '" +
                                          mode_string + "'."));
      round_mode_ = QuantizeRoundMode::ROUND_HALF_TO_EVEN;
    }
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("ensure_minimum_range",
                                             &ensure_minimum_range_));
    std::vector<string> fused_ops;
    fused_ops.push_back("Relu");
    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops),
                errors::InvalidArgument(
                    "Found unsupported fusion in QuantizedConvRelu."));
  }

  void InitOrSetMemory(OpKernelContext* context) {
    if (!(this->enable_cache_ && this->is_init_ &&
          IsInputSame(context, 0, this->input_dims_,
                      this->src_onednn_shape_))) {
      Init(context);
      return;
    }

    if (this->is_input_zero_) {
      AllocateOutputSetOneDnnShape(context, this->kDstIndex_,
                                   &this->dst_tensor_, this->dst_tf_shape_,
                                   this->dst_onednn_shape_);
      return;
    }

    if (this->is_src_reordered_) {
      int64 src_out_size = this->fwd_pd_.src_desc().get_size() / sizeof(qint8);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<qint8>::v(),
                                            TensorShape({src_out_size}),
                                            this->src_data_output_.get()));
      this->src_mem_input_.set_data_handle(
          context->tensor_data(this->kSrcIndex_));
      this->src_mem_.set_data_handle(
          GetTensorBuffer<qint8>(this->src_data_output_.get()));

      this->src_reorder_.execute(this->onednn_stream_, this->src_reorder_args_);
    } else {
      this->src_mem_.set_data_handle(context->tensor_data(this->kSrcIndex_));
    }

    if (this->is_filter_reordered_) {
      if (!is_filter_const_) {
        this->filter_mem_input_.set_data_handle(
            context->tensor_data(this->kFilterIndex_));
        this->filter_mem_.set_data_handle(
            GetTensorBuffer<qint8>(&this->tmp_weight_));

        this->weight_reorder_.execute(this->onednn_stream_,
                                      this->weight_reorder_args_);
      }
    } else {
      this->filter_mem_.set_data_handle(
          context->tensor_data(this->kFilterIndex_));
    }

    if (this->post_op_util_.HasBias()) {
      const Tensor& bias_tensor = context->input(this->kBiasIndex_);
      Tbias* bias_data = this->GetBiasHandle(context, bias_tensor);
      this->bias_mem_.set_data_handle(bias_data);
    }

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                          TensorShape({this->scratchpad_size_}),
                                          this->scratchpad_tensor_.get()));
    this->scratchpad_mem_.set_data_handle(
        GetTensorBuffer<Tinput>(this->scratchpad_tensor_.get()));

    AllocateOutputTensor(context, this->fwd_pd_, this->dst_dims_onednn_,
                         this->data_fmt_onednn_, &this->dst_onednn_shape_,
                         this->dst_shape_, &this->dst_tensor_);
    this->dst_mem_.set_data_handle(reinterpret_cast<Tsummand*>(
        GetTensorBuffer<Toutput>(this->dst_tensor_)));
  }

  void Init(OpKernelContext* context) {
    try {
      this->fwd_primitives_args_.clear();

      // Input tensors
      const Tensor& src_tensor = context->input(this->kSrcIndex_);
      const Tensor& filter_tensor = context->input(this->kFilterIndex_);
      // in remapper, we replace min_input with the min_range of quantizeV2 node
      const Tensor& q_min_range = context->input(this->kSrcMinRangeIndex);
      const Tensor& q_max_range = context->input(this->kSrcMaxRangeIndex);

      auto input_shape = src_tensor.shape();
      this->input_dims_.clear();
      for (int i = 0; i < input_shape.dims(); ++i) {
        this->input_dims_.push_back(input_shape.dim_size(i));
      }

      // Get shapes of input & filter tensors
      OneDnnShape filter_onednn_shape;
      GetOneDnnShape(context, this->kSrcIndex_, &this->src_onednn_shape_);
      TensorShape src_tf_shape = this->src_onednn_shape_.IsOneDnnTensor()
                                     ? this->src_onednn_shape_.GetTfShape()
                                     : src_tensor.shape();
      TensorShape filter_tf_shape = filter_tensor.shape();

      // Memory dimensions
      memory::dims src_dims, filter_dims, pad_left_dims, pad_right_dims,
          dilation_dims, stride_dims, bias_dims;
      memory::dims dst_dims_tf;

      OneDnnConvUtil conv_util(context, data_format_, strides_, dilations_,
                               padding_, this->explicit_paddings_, is_conv2d_,
                               is_depthwise);

      if (pad_enabled) {
        const int kPadIndex = this->post_op_util_.HasBias()
                                  ? this->kBiasIndex_ + 1
                                  : this->kBiasIndex_;

        conv_util.InitPadWithFusion(kPadIndex, true);
      }
      bool is_grouped_convolution;
      conv_util.InitFwdDimensions(
          src_tf_shape, filter_tf_shape, &src_dims, &filter_dims, &stride_dims,
          &dilation_dims, &dst_dims_tf, &this->dst_dims_onednn_, &pad_left_dims,
          &pad_right_dims, &is_grouped_convolution);

      // OneDNN dilations start from 0.
      for (int i = 0; i < dilation_dims.size(); ++i) {
        --dilation_dims[i];
      }

      this->dst_tf_shape_ = OneDnnDimsToTFShape(dst_dims_tf);
      // Corner cases: output with 0 elements and 0 batch size.
      if (this->dst_tf_shape_.num_elements() == 0 || dst_dims_tf[0] == 0) {
        this->is_input_zero_ = true;
        AllocateOutputSetOneDnnShape(context, this->kDstIndex_,
                                     &this->dst_tensor_, this->dst_tf_shape_,
                                     this->dst_onednn_shape_);
        return;
      }

      if (is_depthwise) {
        OP_REQUIRES(context, is_conv2d_,
                    errors::InvalidArgument(
                        "Only 2D convolution is supported for depthwise."));
      }

      // Get OneDnn layout for data
      this->data_fmt_onednn_ =
          TFDataFormatToOneDnnDataFormat(data_format_, is_conv2d_);
      memory::format_tag data_layout =
          OneDnnTensorFormatToTag(this->data_fmt_onednn_);
      OP_REQUIRES(context, data_layout != memory::format_tag::undef,
                  errors::InvalidArgument("Invalid data format"));

      // Although filter shape (filter_dims) required is in OneDnn order,
      // the layout is Tensorflow's layout (HWIO) and (HWIGO) for
      // depthwise/group convolutions.
      auto filter_layout = is_conv2d_
                               ? (is_depthwise ? memory::format_tag::hwigo
                                               : memory::format_tag::hwio)
                               : memory::format_tag::dhwio;
      memory::desc src_md =
          this->src_onednn_shape_.IsOneDnnTensor()
              ? this->src_onednn_shape_.GetOneDnnLayout()
              : memory::desc(src_dims, OneDnnType<Tinput>(), data_layout);
      memory::desc src_md_prefer =
          memory::desc(src_dims, OneDnnType<qint8>(), memory::format_tag::any);
      if (src_dims[1] == 3 && std::is_same<Device, GPUDevice>::value) {
        // This is fusion from FP32 to INT8, so need change the data type.
        src_md_prefer = memory::desc(src_dims, OneDnnType<qint8>(),
                                     memory::format_tag::any);
      }

      memory::desc filter_md =
          memory::desc(filter_dims, OneDnnType<qint8>(), filter_layout);
      // block format filter is allowed with plain src. preferred for both
      // layout disabled or enabled
      memory::desc filter_md_prefer = memory::desc(
          filter_dims, OneDnnType<qint8>(), memory::format_tag::any);

      // TODO(itex): redesign the code for Tsummand
      // The reason for using Tsummand is to deal with the situation for int8
      // fusion conv + bias + add + relu fusion. Two inputs for add op may be
      // respectively quint8 and qint8.
      memory::desc dst_md;
      if (std::is_same<Toutput, Tsummand>::value) {
        dst_md = memory::desc({this->dst_dims_onednn_}, OneDnnType<Toutput>(),
                              memory::format_tag::any);
        this->add_dst_md_ = dst_md;
      } else {
        dst_md = memory::desc({this->dst_dims_onednn_}, OneDnnType<Tsummand>(),
                              memory::format_tag::any);
        this->add_dst_md_ =
            memory::desc({this->dst_dims_onednn_}, OneDnnType<Toutput>(),
                         memory::format_tag::any);
      }

      // TODO(itex): Currently, we have 2 separate code in dealing with post
      // op.  Try to combine these two codes together
      // 1. For fp32 ops, to read information from "fused_ops" attr during op
      // construction. We use "post_op_util_.AddOps" in this situation
      // 2. For int8 ops, set post op information during op compute, since there
      // is no "fused_ops" attr for these ops. We use "ExtendInt8PostOps" in
      // this situation
      this->ExtendInt8PostOps(context);
      // Set post op attribution.
      dnnl::primitive_attr post_ops_attr;
      this->post_op_util_.SetPostOpAttr(&post_ops_attr);
      post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      // Create a convolution descriptor
      this->fwd_pd_ =
          ConvFwdPd(this->onednn_engine_, prop_kind::forward,
                    dnnl::algorithm::convolution_direct, src_md_prefer,
                    filter_md_prefer, dst_md, stride_dims, dilation_dims,
                    pad_left_dims, pad_right_dims, post_ops_attr);

      if (this->post_op_util_.HasBias()) {
        const Tensor& bias_tensor = context->input(this->kBiasIndex_);
        TensorShape bias_tensor_shape = bias_tensor.shape();
        conv_util.GetBiasDimension(bias_tensor_shape, &bias_dims);

        // TODO(itex): use format_tag::any for bias
        auto bias_md =
            memory::desc(bias_dims, OneDnnType<Tbias>(), memory::format_tag::x);

        Tbias* bias_data = this->GetBiasHandle(context, bias_tensor);
        this->bias_mem_ =
            CreateDnnlMemory(bias_md, this->onednn_engine_, bias_data);
        this->fwd_primitives_args_.insert({DNNL_ARG_BIAS, this->bias_mem_});
        this->fwd_pd_ = ConvFwdPd(this->onednn_engine_, prop_kind::forward,
                                  dnnl::algorithm::convolution_direct,
                                  src_md_prefer, filter_md_prefer, bias_md,
                                  dst_md, stride_dims, dilation_dims,
                                  pad_left_dims, pad_right_dims, post_ops_attr);
      }

      this->fwd_primitive_ = dnnl::convolution_forward(this->fwd_pd_);

      // Create a temp conv primitve desc to get real add dst md.
      auto add_fwd_pd =
          ConvFwdPd(this->onednn_engine_, prop_kind::forward,
                    dnnl::algorithm::convolution_direct, src_md_prefer,
                    filter_md_prefer, this->add_dst_md_, stride_dims,
                    dilation_dims, pad_left_dims, pad_right_dims);
      this->add_dst_md_ = add_fwd_pd.dst_desc();

      int64 dst_data_size =
          this->fwd_pd_.dst_desc().get_size() / sizeof(Toutput);
      this->dst_shape_ = TensorShape({dst_data_size});

      AllocateOutputTensor(context, this->fwd_pd_, this->dst_dims_onednn_,
                           this->data_fmt_onednn_, &this->dst_onednn_shape_,
                           this->dst_shape_, &this->dst_tensor_);

      // add quantizeV2 logic
      int num_slices = 1;
      if (axis_ > -1) {
        num_slices = q_min_range.NumElements();
      }
      min_range.reserve(num_slices);
      max_range.reserve(num_slices);
      if (num_slices == 1) {
        const float min_range_before_adjust =
            q_min_range.template flat<float>()(0);
        const float max_range_before_adjust =
            q_max_range.template flat<float>()(0);
        AdjustInputMinMaxRange(context, min_range_before_adjust,
                               max_range_before_adjust, &min_range[0],
                               &max_range[0]);
      } else {
        auto min_ranges_before_adjust = q_min_range.template vec<float>();
        auto max_ranges_before_adjust = q_max_range.template vec<float>();
        for (int i = 0; i < num_slices; ++i) {
          AdjustInputMinMaxRange(context, min_ranges_before_adjust(i),
                                 max_ranges_before_adjust(i), &min_range[i],
                                 &max_range[i]);
        }
      }
      // Calculating scales and zeropoints for quantization.
      std::vector<float> scale_factor(num_slices, 0);
      // Zeropoint not used currently. Because we use legacy MIN_FIRST
      // implemenation.
      std::vector<int32> zero_points(num_slices, 0);
      // only scaled mode
      if (mode_ == QuantizeMode::SCALED) {
        GetScaleAndZeropointAndAlignMinMax<qint8>(
            min_range.data(), max_range.data(), mode_,
            QuantDequantFlag::Quantize, num_slices, scale_factor.data(),
            zero_points.data());
      } else {
        ITEX_LOG(FATAL) << "This fusion does not such quantize mode";
      }

      // Check whether src and filter need to be reordered.
      this->is_src_reordered_ = (src_md != this->fwd_pd_.src_desc());
      this->src_mem_input_ = CreateDnnlMemory(
          src_md, this->onednn_engine_, GetTensorBuffer<Tinput>(&src_tensor));

      // Set the scale factor for quantize
      dnnl::primitive_attr reorder_post_ops_attr;
      if (mode_ == QuantizeMode::SCALED) {
        if (num_slices == 1) {
          reorder_post_ops_attr.set_scales_mask(DNNL_ARG_SRC, 0);
        } else {
          int mask = static_cast<int>(std::pow(2, axis_));
          reorder_post_ops_attr.set_scales_mask(DNNL_ARG_SRC, mask);
        }
      } else {
        ITEX_LOG(FATAL) << "This fusion does not such quantize mode";
      }
      if (this->is_src_reordered_) {
        int64 src_out_size =
            this->fwd_pd_.src_desc().get_size() / sizeof(qint8);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<qint8>::v(),
                                              TensorShape({src_out_size}),
                                              this->src_data_output_.get()));
        this->src_mem_ = CreateDnnlMemory(
            this->fwd_pd_.src_desc(), this->onednn_engine_,
            GetTensorBuffer<qint8>(this->src_data_output_.get()));
        this->src_reorder_args_.clear();
        this->src_reorder_args_.insert({DNNL_ARG_SRC, this->src_mem_input_});
        this->src_reorder_args_.insert({DNNL_ARG_DST, this->src_mem_});
        float* src_reorder_scale_ptr = src_reorder_scale_cache_.GetCachedPtr(
            context, scale_factor.data(), num_slices);
        memory src_reorder_scales_mem(
            {{num_slices}, memory::data_type::f32, memory::format_tag::x},
            this->onednn_engine_,
            reinterpret_cast<void*>(src_reorder_scale_ptr));
        this->src_reorder_args_.insert(
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_reorder_scales_mem});
        dnnl::reorder::primitive_desc reorder_pd =
            dnnl::reorder::primitive_desc(
                this->onednn_engine_, src_md, this->onednn_engine_,
                this->fwd_pd_.src_desc(), reorder_post_ops_attr);
        this->src_reorder_ = dnnl::reorder(reorder_pd);

        this->src_reorder_.execute(this->onednn_stream_,
                                   this->src_reorder_args_);
      } else {
        this->src_mem_ = this->src_mem_input_;
      }

      this->is_filter_reordered_ = (filter_md != this->fwd_pd_.weights_desc());
      this->filter_mem_input_ =
          CreateDnnlMemory(filter_md, this->onednn_engine_,
                           GetTensorBuffer<qint8>(&filter_tensor));

      if (this->is_filter_reordered_) {
        qint8* filter_cached_data = nullptr;
        if (is_filter_const_) {
          if (weight_cache_manager_.IsEmpty()) {
            // Cache weight
            weight_cache_manager_.SetCache(
                context, filter_md, this->fwd_pd_.weights_desc(),
                GetTensorBuffer<qint8>(&filter_tensor), this->onednn_engine_);
          }
          filter_cached_data = weight_cache_manager_.GetCache(
              context, this->fwd_pd_.weights_desc());
          if (filter_cached_data != nullptr) {
            this->filter_mem_ =
                CreateDnnlMemory(this->fwd_pd_.weights_desc(),
                                 this->onednn_engine_, filter_cached_data);
          }
        }
        if (filter_cached_data == nullptr) {
          int64 reorder_filter_data_size =
              this->fwd_pd_.weights_desc().get_size() / sizeof(qint8);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DataTypeToEnum<qint8>::v(),
                                      TensorShape({reorder_filter_data_size}),
                                      &this->tmp_weight_));
          void* filter_data_handle = GetTensorBuffer<qint8>(&this->tmp_weight_);
          this->filter_mem_ =
              CreateDnnlMemory(this->fwd_pd_.weights_desc(),
                               this->onednn_engine_, filter_data_handle);
          this->weight_reorder_args_.clear();
          this->weight_reorder_args_.insert(
              {DNNL_ARG_SRC, this->filter_mem_input_});
          this->weight_reorder_args_.insert({DNNL_ARG_DST, this->filter_mem_});
          this->weight_reorder_ =
              dnnl::reorder(this->filter_mem_input_, this->filter_mem_);

          this->weight_reorder_.execute(this->onednn_stream_,
                                        this->weight_reorder_args_);
        }
      } else {
        this->filter_mem_ = this->filter_mem_input_;
      }

      // TODO(itex): redesign the code for Tsummand
      this->dst_mem_ =
          CreateDnnlMemory(this->fwd_pd_.dst_desc(), this->onednn_engine_,
                           reinterpret_cast<Tsummand*>(
                               GetTensorBuffer<Toutput>(this->dst_tensor_)));

      this->scratchpad_size_ =
          this->fwd_pd_.scratchpad_desc().get_size() / sizeof(Tinput);
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                          TensorShape({this->scratchpad_size_}),
                                          this->scratchpad_tensor_.get()));
      this->scratchpad_mem_ =
          dnnl::memory(this->fwd_pd_.scratchpad_desc(), this->onednn_engine_,
                       GetTensorBuffer<Tinput>(this->scratchpad_tensor_.get()));

      // Execute convolution
      this->fwd_primitives_args_.insert({DNNL_ARG_SRC, this->src_mem_});
      this->fwd_primitives_args_.insert({DNNL_ARG_WEIGHTS, this->filter_mem_});
      this->fwd_primitives_args_.insert({DNNL_ARG_DST, this->dst_mem_});
      this->fwd_primitives_args_.insert(
          {DNNL_ARG_SCRATCHPAD, this->scratchpad_mem_});

      this->is_init_ = true;
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
    this->onednn_engine_ = CreateDnnlEngine<Device>(*context);
    // onednn_stream has thread safety issue, need create a new one in
    // every compute.
    this->onednn_stream_ = CreateDnnlStream(*context, this->onednn_engine_);
    this->src_data_output_ = std::make_shared<Tensor>();
    this->scratchpad_tensor_ = std::make_shared<Tensor>();
    InitOrSetMemory(context);

    // Skip primitive execution if the calculation is meaningless.
    if (this->is_input_zero_) {
      this->src_data_output_.reset();
      this->scratchpad_tensor_.reset();
      return;
    }
    if (this->post_op_util_.HasOutputScales()) {
      float* output_scale_ptr = output_scale_cache_.GetCachedPtr(
          context, this->post_op_util_.GetOutputScale().data(),
          this->post_op_util_.GetOutputScale().size());
      dnnl::memory scales_mem(
          {{static_cast<dnnl_dim_t>(
               this->post_op_util_.GetOutputScale().size())},
           memory::data_type::f32,
           memory::format_tag::x},
          this->onednn_engine_, reinterpret_cast<void*>(output_scale_ptr));
      this->fwd_primitives_args_.emplace(
          DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_mem);
    }

    this->fwd_primitive_.execute(this->onednn_stream_,
                                 this->fwd_primitives_args_);

    this->scratchpad_tensor_.reset();
    float min_input = min_range[0];
    float max_input = max_range[0];

    AllocateBlockOutputMinMax<Tinput, qint8, Toutput>(
        context, min_input, max_input, this->kFilterMinRangeIndex,
        this->kFilterMaxRangeIndex, this->kMinFreezedIndex,
        this->kMaxFreezedIndex, this->kDstMinRangeIndex,
        this->kDstMaxRangeIndex);
    this->src_data_output_.reset();
  }

 protected:
  void ExtendInt8PostOps(OpKernelContext* context) override {
    OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                          quantized_bias_enabled,
                          is_depthwise>::ExtendInt8PostOps(context);
    this->post_op_util_.SetPostOpScale("Relu", 1.0);
  }

  void AdjustInputMinMaxRange(OpKernelContext* context, float input_min_range,
                              float input_max_range, float* adjust_min_range,
                              float* adjust_max_range) {
    OP_REQUIRES(context, (input_max_range >= input_min_range),
                errors::InvalidArgument(
                    "input_max_range must be larger than input_min_range."));

    *adjust_min_range = std::min(0.0f, input_min_range);
    // When the minimum and maximum ranges are too close together, nudge them
    // apart by a small value so that they are slightly different. This helps
    // us avoid creating ill-formed buffers where all quantized values map to
    // the same float number. These kinds of buffers cause problems for
    // downstream ops when they need to do calculations on them.
    // We pick the value by making sure that zero is not more than 100x the
    // overall range from the maximum, so that the value can be easily
    // represented when we promote the quantized value to a higher
    // intermediate bit depth, since that's a common requirement.
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                  fabsf(input_max_range))) *
                          ensure_minimum_range_;
    *adjust_max_range =
        std::max(0.0f, std::max(input_max_range, *adjust_min_range + epsilon));
  }

  void AllocateOutputTensor(
      OpKernelContext* context,
      const dnnl::convolution_forward::primitive_desc& conv_pd,
      const memory::dims& dst_dims_onednn, OneDnnTensorFormat dst_tf_format,
      OneDnnShape* dst_onednn_shape, TensorShape tensor_shape,
      Tensor** dst_tensor) override {
    ITEX_DCHECK(dst_tensor);
    // NHWC Conv may prefer NCHW (also plain format) as its primitive format.
    // Need to record this info in meta data to reorder the data correctly.
    SetOutputTensorShape(conv_pd.dst_desc(), dst_tf_format, &tensor_shape,
                         dst_onednn_shape, true /*is_onednn*/);
    {
      OP_REQUIRES(
          context, !this->post_op_util_.HasAdd(),
          errors::InvalidArgument("OneDnnConvOp: Don't support AddN fusion."));
      AllocateOutputSetOneDnnShape(context, this->kDstIndex_, dst_tensor,
                                   tensor_shape, *dst_onednn_shape);
    }
    return;
  }

 private:
  std::vector<int64_t> explicit_paddings_;
  bool is_conv2d_;
  bool is_filter_const_ = false;
  bool inplace_sum_ = false;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  // Weight cache manager
  WeightCacheManager<qint8> weight_cache_manager_;

  mutex mu_compute_;

 private:
  bool pad_enabled = false;

  std::vector<float> min_range;
  std::vector<float> max_range;
  // quantizeV2 variables
  float ensure_minimum_range_;
  QuantizeMode mode_;
  QuantizeRoundMode round_mode_;
  int axis_;
  bool narrow_range_;
  HostDataCache<Device, float> output_scale_cache_;
  HostDataCache<Device, float> src_reorder_scale_cache_;
};

template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Tsummand, bool quantized_bias_enabled, bool is_depthwise>
class OneDnnQuantizedConv2DWithDequantizeOp
    : public OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                                   quantized_bias_enabled, is_depthwise> {
 public:
  explicit OneDnnQuantizedConv2DWithDequantizeOp(OpKernelConstruction* context)
      : OneDnnQuantizedConvOp<Device, Tinput, Tbias, Toutput, Tsummand,
                              quantized_bias_enabled, is_depthwise>(context) {}

  void Compute(OpKernelContext* context) override {
    // Compute int32 output tensor
    OneDnnConvOp<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
                 quantized_bias_enabled, is_depthwise>::Compute(context);
  }

 protected:
  void ExtendInt8PostOps(OpKernelContext* context) override {
    // When the output type is quint8, the output data is requantized
    // into quint8. A post_op "output_scale" is added to do the conversion.
    // Otherwise the output_scale will be 1.f
    const Tensor& min_filter_vector =
        context->input(this->kFilterMinRangeIndex);
    const Tensor& max_filter_vector =
        context->input(this->kFilterMaxRangeIndex);
    size_t depth = min_filter_vector.NumElements();
    std::vector<float> scales(depth, 1.f);

    if (std::is_same<Toutput, float>::value ||
        std::is_same<Toutput, Eigen::half>::value) {
      const float min_input =
          context->input(this->kSrcMinRangeIndex).template flat<float>()(0);
      const float max_input =
          context->input(this->kSrcMaxRangeIndex).template flat<float>()(0);

      const float* min_filter = min_filter_vector.flat<float>().data();
      const float* max_filter = max_filter_vector.flat<float>().data();

      float float_input_range =
          std::max(std::abs(min_input), std::abs(max_input));
      const float int_const_scale_limit =
          (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
      for (size_t i = 0; i < depth; ++i) {
        // For simplicity and symmetry, we set filter range to be outer
        // bounds of min_filter and max_filter.
        float float_filter_range =
            std::max(std::abs(min_filter[i]), std::abs(max_filter[i]));
        scales[i] =
            float_input_range * float_filter_range / (int_const_scale_limit);
      }
    }
    this->post_op_util_.SetOutputScale(scales);
  }
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_ONEDNN_BLOCK_CONV_OPS_IMPL_H_
