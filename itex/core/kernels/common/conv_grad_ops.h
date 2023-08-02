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

#ifndef ITEX_CORE_KERNELS_COMMON_CONV_GRAD_OPS_H_
#define ITEX_CORE_KERNELS_COMMON_CONV_GRAD_OPS_H_

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/conv_ops.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

using ConvBwdInputPd = dnnl::convolution_backward_data::primitive_desc;
using ConvBwdInputPrimitive = dnnl::convolution_backward_data;
using ConvBwdFilterPd = dnnl::convolution_backward_weights::primitive_desc;
using ConvBwdFilterPrimitive = dnnl::convolution_backward_weights;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;

template <typename Device, class T, bool is_depthwise = false>
class ConvBackpropCommonOp : public OpKernel {
 public:
  ~ConvBackpropCommonOp() {}
  explicit ConvBackpropCommonOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format_string;
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_format", &data_format_string));
    OP_REQUIRES(context,
                FormatFromString(data_format_string, &this->data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &this->strides_));
    const int64 stride_n =
        GetTensorDim(this->strides_, this->data_format_, 'N');
    const int64 stride_c =
        GetTensorDim(this->strides_, this->data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, (strides_.size() == 4 || strides_.size() == 5),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 or 5 dimensions"));

    is_conv2d_ = (strides_.size() == 4);

    // Depthwise Convolution doesn't have dilation parameter
    if (!is_depthwise) {
      OP_REQUIRES_OK(context, context->GetAttr("dilations", &this->dilations_));
      if (is_conv2d_) {
        // Check Conv2D dilations
        OP_REQUIRES(
            context, this->dilations_.size() == 4,
            errors::InvalidArgument("Sliding window dilations field must "
                                    "specify 4 dimensions"));
        int dilation_n =
            GetTensorDim(this->dilations_, this->data_format_, 'N');
        int dilation_c =
            GetTensorDim(this->dilations_, this->data_format_, 'C');
        int dilation_h =
            GetTensorDim(this->dilations_, this->data_format_, 'H');
        int dilation_w =
            GetTensorDim(this->dilations_, this->data_format_, 'W');
        OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
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
    } else {
      // Set dilations as 1 for depthwise conv
      // for future support to align with Tensorflow
      this->dilations_ = {1, 1, 1, 1};
    }

    OP_REQUIRES_OK(context, context->GetAttr("padding", &(this->padding_)));
    if (context->HasAttr("explicit_paddings")) {
      OP_REQUIRES_OK(context, context->GetAttr("explicit_paddings",
                                               &this->explicit_paddings_));
    }
    OP_REQUIRES_OK(context,
                   CheckValidPadding(padding_, explicit_paddings_,
                                     is_conv2d_ ? 4 : 5, data_format_));

    fp32_math_mode_ = GetFP32MathMode<Device>();
  }

 protected:
  bool is_conv2d_;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  TensorFormat data_format_;
  dnnl::fpmath_mode fp32_math_mode_ = dnnl::fpmath_mode::strict;
};

// Get TensorFlow shape of filter tensor.
inline TensorShape GetTensorShape(const Tensor& tensor) {
  TensorShape shape;
  if (tensor.dtype() == DT_INT32) {
    ITEX_CHECK_EQ(TensorShapeUtils::MakeShape(tensor.vec<int32>(), &shape).ok(),
                  true);
  } else {
    ITEX_CHECK_EQ(TensorShapeUtils::MakeShape(tensor.vec<int64>(), &shape).ok(),
                  true);
  }

  return shape;
}

template <typename Device, typename T, bool is_depthwise = false,
          bool bias_enabled = false, bool pad_enabled = false>
class ConvBackpropFilterOp
    : public ConvBackpropCommonOp<Device, T, is_depthwise> {
 public:
  explicit ConvBackpropFilterOp(OpKernelConstruction* context)
      : ConvBackpropCommonOp<Device, T, is_depthwise>(context) {
    if (bias_enabled) {
      std::vector<string> fused_ops_;
      OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops_));
      OP_REQUIRES(context, fused_ops_.size() == 1,
                  errors::InvalidArgument(
                      "OneDnnFusedMatMul must have 1 post-arguments at most."));
      OP_REQUIRES(context, fused_ops_[0] == "BiasAddGrad",
                  errors::InvalidArgument(
                      "The 1st post-argument of OneDnnConvBackpropFilter must "
                      "be BiasAddGrad."));
    }
    // Pad fusion check.
    if (pad_enabled) {
      OP_REQUIRES(
          context, this->padding_ == Padding::VALID,
          errors::InvalidArgument("Pad can only be fused with `VALID` Conv."));
    }
  }

  void Compute(OpKernelContext* context) override {
    const int kInputIdx = 0, kFilterIdx = 1, kOutbpropIdx = 2;
    const int kOutputIdx = 0, kDiffBiasIndex = 1;
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      // Input tensors
      const Tensor& src_tensor = context->input(kInputIdx);
      const Tensor& filter_sizes_tensor = context->input(kFilterIdx);
      const Tensor& diff_dst_tensor = context->input(kOutbpropIdx);

      const TensorShape& src_shape = src_tensor.shape();
      const TensorShape& diff_dst_shape = diff_dst_tensor.shape();
      // V2 op has shape tensor, otherwise need to resolve shape from filter.
      TensorShape filter_shape;
      if (filter_sizes_tensor.dtype() == DT_INT32 ||
          filter_sizes_tensor.dtype() == DT_INT64) {
        // TODO(itex): rename GetTensorShape to CreateTensorShape
        filter_shape = GetTensorShape(filter_sizes_tensor);
      } else {
        filter_shape = filter_sizes_tensor.shape();
      }

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* diff_filter_tensor = nullptr;
      Tensor* diff_bias_tensor = nullptr;
      if (src_shape.num_elements() == 0 || filter_shape.num_elements() == 0 ||
          diff_dst_shape.num_elements() == 0) {
        OP_REQUIRES_OK(context, context->allocate_output(
                                    static_cast<const int>(kOutputIdx),
                                    filter_shape, &diff_filter_tensor));
        ITEX_CHECK_NOTNULL(diff_filter_tensor);
        // If output tensor has more than 0 elements, we need to 0 them out.
        if (filter_shape.num_elements() > 0) {
          DeviceMemset<Device>(
              diff_filter_tensor->flat<T>().data(), 0,
              diff_filter_tensor->shape().num_elements() * sizeof(T),
              context->GetDeviceStream());
        }
        return;
      }

      memory::dims diff_bias_dims = {};
      int64 depth = 0;
      if (bias_enabled) {
        depth = (this->data_format_ == FORMAT_NCHW)
                    ? diff_dst_shape.dim_size(1)
                    : diff_dst_shape.dim_size(this->is_conv2d_ ? 3 : 4);
        diff_bias_dims = {static_cast<int>(depth)};
      }

      // Memory dimensions
      memory::dims fwd_src_dims, fwd_filter_dims, diff_dst_dims;
      memory::dims pad_left_dims, pad_right_dims, dilation_dims, stride_dims,
          bias_dims;
      memory::dims dst_dims_tf, dst_dims_onednn;

      OneDnnConvUtil conv_util(context, this->data_format_, this->strides_,
                               this->dilations_, this->padding_,
                               this->explicit_paddings_, this->is_conv2d_,
                               is_depthwise);

      if (pad_enabled) {
        const int kPadIndex = kOutbpropIdx + 1;
        conv_util.InitPadWithFusion(kPadIndex, true);
      }
      bool is_grouped_convolution;
      conv_util.InitFwdDimensions(
          src_shape, filter_shape, &fwd_src_dims, &fwd_filter_dims,
          &stride_dims, &dilation_dims, &dst_dims_tf, &dst_dims_onednn,
          &pad_left_dims, &pad_right_dims, &is_grouped_convolution);
      conv_util.GetInputDimension(diff_dst_shape, &diff_dst_dims);

      OneDnnTensorFormat data_format_onednn =
          TFDataFormatToOneDnnDataFormat(this->data_format_, this->is_conv2d_);
      memory::format_tag data_layout =
          OneDnnTensorFormatToTag(data_format_onednn);
      auto fwd_src_md =
          memory::desc(fwd_src_dims, OneDnnType<T>(), data_layout);
      auto diff_dst_md =
          memory::desc(diff_dst_dims, OneDnnType<T>(), data_layout);
      // OneDNN dilations start from 0.
      for (int i = 0; i < dilation_dims.size(); ++i) {
        --dilation_dims[i];
      }

      // allocate output tensors: diff_filter
      auto diff_filter_dims = fwd_filter_dims;

      const std::vector<DNNL_SIZE_DTYPE>& diff_filter_array =
          this->is_conv2d_
              ? (is_depthwise
                     ? std::vector<DNNL_SIZE_DTYPE>(
                           {diff_filter_dims
                                [FilterGroupDims::GROUP_FILTER_DIM_H],
                            diff_filter_dims
                                [FilterGroupDims::GROUP_FILTER_DIM_W],
                            diff_filter_dims
                                [FilterGroupDims::GROUP_FILTER_DIM_G],
                            diff_filter_dims
                                [FilterGroupDims::GROUP_FILTER_DIM_O]})
                     : (is_grouped_convolution
                            ? std::vector<DNNL_SIZE_DTYPE>(
                                  {diff_filter_dims
                                       [FilterGroupDims::GROUP_FILTER_DIM_H],
                                   diff_filter_dims
                                       [FilterGroupDims::GROUP_FILTER_DIM_W],
                                   diff_filter_dims
                                       [FilterGroupDims::GROUP_FILTER_DIM_I],
                                   diff_filter_dims[FilterGroupDims::
                                                        GROUP_FILTER_DIM_O] *
                                       diff_filter_dims
                                           [FilterGroupDims::
                                                GROUP_FILTER_DIM_G]})
                            : std::vector<DNNL_SIZE_DTYPE>(
                                  {diff_filter_dims[DimensionIndex::Dim_H],
                                   diff_filter_dims[DimensionIndex::Dim_W],
                                   diff_filter_dims[DimensionIndex::Dim_I],
                                   diff_filter_dims[DimensionIndex::Dim_O]})))
              : std::vector<DNNL_SIZE_DTYPE>(
                    {diff_filter_dims[DimensionIndex3D::Dim3d_D],
                     diff_filter_dims[DimensionIndex3D::Dim3d_H],
                     diff_filter_dims[DimensionIndex3D::Dim3d_W],
                     diff_filter_dims[DimensionIndex3D::Dim3d_I],
                     diff_filter_dims[DimensionIndex3D::Dim3d_O]});

      TensorShape diff_filter_shape(
          {diff_filter_array.data(), diff_filter_array.size()});

      const auto diff_filter_format =
          this->is_conv2d_ ? (is_depthwise || is_grouped_convolution
                                  ? memory::format_tag::hwigo
                                  : memory::format_tag::hwio)
                           : memory::format_tag::dhwio;
      auto diff_filter_md =
          memory::desc(diff_filter_dims, OneDnnType<T>(), diff_filter_format);
      auto diff_filter_md_prefer = memory::desc(
          diff_filter_dims, OneDnnType<T>(), memory::format_tag::any);

      memory::desc diff_bias_md = memory::desc(
          {diff_bias_dims}, OneDnnType<T>(), dnnl::memory::format_tag::x);

      OP_REQUIRES_OK(context, context->allocate_output(
                                  static_cast<const int>(kOutputIdx),
                                  diff_filter_shape, &diff_filter_tensor));

      // the convolution primitive is optimized for NHWC
      auto format_tag_opt = this->is_conv2d_ ? memory::format_tag::nhwc
                                             : memory::format_tag::ndhwc;
      auto fwd_src_md_opt =
          memory::desc(fwd_src_dims, OneDnnType<T>(), format_tag_opt);
      auto diff_dst_md_opt =
          memory::desc(diff_dst_dims, OneDnnType<T>(), format_tag_opt);

      // Create descriptor and primitive descriptor for convolution forward.

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<T, float>::value) {
        attr.set_fpmath_mode(this->fp32_math_mode_);
      }
      ConvFwdPd fwd_pd;
      ConvBwdFilterPd bwd_filter_pd;
      if (bias_enabled) {
        fwd_pd = ConvFwdPd(onednn_engine, prop_kind::forward,
                           dnnl::algorithm::convolution_direct, fwd_src_md_opt,
                           diff_filter_md_prefer, diff_bias_md, diff_dst_md_opt,
                           stride_dims, dilation_dims, pad_left_dims,
                           pad_right_dims, attr);
        bwd_filter_pd = ConvBwdFilterPd(
            onednn_engine, dnnl::algorithm::convolution_direct, fwd_src_md_opt,
            diff_filter_md_prefer, diff_bias_md, diff_dst_md_opt, stride_dims,
            dilation_dims, pad_left_dims, pad_right_dims, fwd_pd, attr);
      } else {
        fwd_pd = ConvFwdPd(onednn_engine, prop_kind::forward,
                           dnnl::algorithm::convolution_direct, fwd_src_md_opt,
                           diff_filter_md_prefer, diff_dst_md_opt, stride_dims,
                           dilation_dims, pad_left_dims, pad_right_dims, attr);
        bwd_filter_pd = ConvBwdFilterPd(
            onednn_engine, dnnl::algorithm::convolution_direct, fwd_src_md_opt,
            diff_filter_md_prefer, diff_dst_md_opt, stride_dims, dilation_dims,
            pad_left_dims, pad_right_dims, fwd_pd, attr);
      }

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          bwd_filter_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(bwd_filter_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      // Create memory.
      auto src_mem = CreateDnnlMemory(fwd_src_md, onednn_engine,
                                      GetTensorBuffer<T>(&src_tensor));
      auto diff_dst_mem = CreateDnnlMemory(
          diff_dst_md, onednn_engine, GetTensorBuffer<T>(&diff_dst_tensor));

      // reorder src/diff_dst to NHWC if needed
      memory src_mem_opt = src_mem, diff_dst_mem_opt = diff_dst_mem;
      // keep tensor out of if block to avoid of being deallocated
      Tensor src_tensor_opt, diff_dst_tensor_opt;
      if (data_layout != format_tag_opt) {
        int64 src_nums = bwd_filter_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                       TensorShape({src_nums}),
                                                       &src_tensor_opt));
        src_mem_opt = CreateDnnlMemory(fwd_src_md_opt, onednn_engine,
                                       GetTensorBuffer<T>(&src_tensor_opt));
        ReorderMemory(*context, &src_mem, &src_mem_opt, onednn_engine);

        int64 diff_dst_nums =
            bwd_filter_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({diff_dst_nums}),
                                              &diff_dst_tensor_opt));
        diff_dst_mem_opt =
            CreateDnnlMemory(diff_dst_md_opt, onednn_engine,
                             GetTensorBuffer<T>(&diff_dst_tensor_opt));
        ReorderMemory(*context, &diff_dst_mem, &diff_dst_mem_opt,
                      onednn_engine);
      }

      // Check diff filter reorder
      Tensor tmp_diff_weight;
      T* diff_filter_data =
          const_cast<T*>(diff_filter_tensor->flat<T>().data());
      auto diff_filter_mem = CreateDnnlMemory(
          diff_filter_md, onednn_engine, static_cast<void*>(diff_filter_data));
      memory diff_filter_mem_reordered = diff_filter_mem;
      bool is_diff_filter_reordered =
          (bwd_filter_pd.diff_weights_desc() != diff_filter_md);

      dnnl::memory diff_bias_mem;
      if (bias_enabled) {
        TensorShape diff_bias_tf_shape({depth});
        OP_REQUIRES_OK(context, context->allocate_output(kDiffBiasIndex,
                                                         diff_bias_tf_shape,
                                                         &diff_bias_tensor));
        void* diff_bias_data = GetTensorBuffer<T>(diff_bias_tensor);

        diff_bias_mem =
            CreateDnnlMemory(diff_bias_md, onednn_engine, diff_bias_data);
      }

      if (is_diff_filter_reordered) {
        // allocate temporay reorder tensor
        int64 reorder_diff_filter_data_size =
            bwd_filter_pd.diff_weights_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(
            context,
            context->allocate_temp(DataTypeToEnum<T>::v(),
                                   TensorShape({reorder_diff_filter_data_size}),
                                   &tmp_diff_weight));

        void* diff_filter_data_handle = GetTensorBuffer<T>(&tmp_diff_weight);
        diff_filter_mem_reordered =
            CreateDnnlMemory(bwd_filter_pd.diff_weights_desc(), onednn_engine,
                             diff_filter_data_handle);
      }

      std::unordered_map<int, memory> bwd_filter_primitive_args;
      bwd_filter_primitive_args.insert({DNNL_ARG_SRC, src_mem_opt});
      bwd_filter_primitive_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem_opt});
      bwd_filter_primitive_args.insert(
          {DNNL_ARG_DIFF_WEIGHTS, diff_filter_mem_reordered});
      bwd_filter_primitive_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      if (bias_enabled) {
        bwd_filter_primitive_args.insert({DNNL_ARG_DIFF_BIAS, diff_bias_mem});
      }

      // Create convolution backward filter primitive and add it to the net.
      primitive bwd_filter_primitive = ConvBwdFilterPrimitive(bwd_filter_pd);
      bwd_filter_primitive.execute(onednn_stream, bwd_filter_primitive_args);
      primitive fwd_primitive = dnnl::convolution_forward(fwd_pd);

      if (is_diff_filter_reordered) {
        ReorderMemory(*context, &diff_filter_mem_reordered, &diff_filter_mem,
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
};

template <typename Device, typename T, bool is_depthwise = false,
          bool pad_enabled = false>
class ConvBackpropInputOp
    : public ConvBackpropCommonOp<Device, T, is_depthwise> {
 public:
  explicit ConvBackpropInputOp(OpKernelConstruction* context)
      : ConvBackpropCommonOp<Device, T, is_depthwise>(context) {
    if (pad_enabled) {
      OP_REQUIRES(
          context, this->padding_ == Padding::VALID,
          errors::InvalidArgument("Pad can only be fused with `VALID` Conv."));
    }
  }

  void Compute(OpKernelContext* context) override {
    const int kInputSizesIdx = 0, kFilterIdx = 1, kOutBackpropIdx = 2;
    const int kOutputIdx = 0;
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      // Input tensors
      const Tensor& src_sizes_tensor = context->input(kInputSizesIdx);
      const Tensor& filter_tensor = context->input(kFilterIdx);
      const Tensor& diff_dst_tensor = context->input(kOutBackpropIdx);

      const TensorShape& filter_shape = filter_tensor.shape();
      const TensorShape& diff_dst_shape = diff_dst_tensor.shape();
      // V2 op has shape tensor, otherwise need to resolve shape from filter.
      TensorShape src_shape;
      if (src_sizes_tensor.dtype() == DT_INT32 ||
          src_sizes_tensor.dtype() == DT_INT64) {
        // TODO(itex): rename GetTensorShape to CreateTensorShape
        src_shape = GetTensorShape(src_sizes_tensor);
      } else {
        src_shape = src_sizes_tensor.shape();
      }
      OneDnnConvUtil conv_util(context, this->data_format_, this->strides_,
                               this->dilations_, this->padding_,
                               this->explicit_paddings_, this->is_conv2d_,
                               is_depthwise);

      if (pad_enabled) {
        const int kSliceIdx = 3;
        conv_util.InitPadWithFusion(kSliceIdx, false);

        // If Slice is fused, we need to delete the padding for the input.
        auto size_tensor_vec = context->input(kSliceIdx + 1).vec<int32>();
        for (int i = 0; i < src_shape.dims(); ++i) {
          src_shape.set_dim(i, size_tensor_vec(i));
        }
      }

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* diff_src_tensor = nullptr;
      TensorShape diff_src_shape = src_shape;
      if (src_shape.num_elements() == 0 || filter_shape.num_elements() == 0 ||
          diff_dst_shape.num_elements() == 0) {
        OP_REQUIRES_OK(context, context->allocate_output(
                                    static_cast<const int>(kOutputIdx),
                                    diff_src_shape, &diff_src_tensor));
        ITEX_CHECK_NOTNULL(diff_src_tensor);
        // If output tensor has more than 0 elements, we need to 0 them out.
        if (diff_src_shape.num_elements() > 0) {
          auto out = diff_src_tensor->flat<T>();
          auto d = context->eigen_device<Device>();
          out.device(d) = out.constant(T(0));
        }
        return;
      }

      // Memory dimensions
      memory::dims fwd_src_dims, fwd_filter_dims, diff_dst_dims;
      memory::dims pad_left_dims, pad_right_dims, dilation_dims, stride_dims,
          bias_dims;
      memory::dims dst_dims_tf, dst_dims_onednn;
      bool is_grouped_convolution;

      conv_util.InitFwdDimensions(
          src_shape, filter_shape, &fwd_src_dims, &fwd_filter_dims,
          &stride_dims, &dilation_dims, &dst_dims_tf, &dst_dims_onednn,
          &pad_left_dims, &pad_right_dims, &is_grouped_convolution);
      conv_util.GetInputDimension(diff_dst_shape, &diff_dst_dims);

      // OneDNN dilations start from 0.
      for (int i = 0; i < dilation_dims.size(); ++i) {
        --dilation_dims[i];
      }

      if (is_depthwise) {
        OP_REQUIRES(context, this->is_conv2d_,
                    errors::InvalidArgument(
                        "Only 2D convolution is supported for depthwise."));
      }

      OneDnnTensorFormat data_format_onednn =
          TFDataFormatToOneDnnDataFormat(this->data_format_, this->is_conv2d_);
      memory::format_tag data_layout =
          OneDnnTensorFormatToTag(data_format_onednn);
      const auto filter_format = this->is_conv2d_
                                     ? (is_depthwise || is_grouped_convolution
                                            ? memory::format_tag::hwigo
                                            : memory::format_tag::hwio)
                                     : memory::format_tag::dhwio;
      auto filter_md =
          memory::desc(fwd_filter_dims, OneDnnType<T>(), filter_format);
      auto filter_md_prefer = memory::desc(fwd_filter_dims, OneDnnType<T>(),
                                           memory::format_tag::any);
      auto diff_dst_md =
          memory::desc(diff_dst_dims, OneDnnType<T>(), data_layout);
      auto diff_src_dims = fwd_src_dims;
      auto diff_src_md =
          memory::desc(diff_src_dims, OneDnnType<T>(), data_layout);
      OP_REQUIRES_OK(
          context, context->allocate_output(static_cast<const int>(kOutputIdx),
                                            diff_src_shape, &diff_src_tensor));

      // the convolution primitive is optimized for NHWC
      auto format_tag_opt = this->is_conv2d_ ? memory::format_tag::nhwc
                                             : memory::format_tag::ndhwc;
      auto diff_dst_md_opt =
          memory::desc(diff_dst_dims, OneDnnType<T>(), format_tag_opt);
      auto diff_src_md_opt =
          memory::desc(diff_src_dims, OneDnnType<T>(), format_tag_opt);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<T, float>::value) {
        attr.set_fpmath_mode(this->fp32_math_mode_);
      }
      ConvFwdPd fwd_pd =
          ConvFwdPd(onednn_engine, prop_kind::forward,
                    dnnl::algorithm::convolution_direct, diff_src_md_opt,
                    filter_md_prefer, diff_dst_md_opt, stride_dims,
                    dilation_dims, pad_left_dims, pad_right_dims, attr);
      ConvBwdInputPd bwd_input_pd = ConvBwdInputPd(
          onednn_engine, dnnl::algorithm::convolution_direct, diff_src_md_opt,
          filter_md_prefer, diff_dst_md_opt, stride_dims, dilation_dims,
          pad_left_dims, pad_right_dims, fwd_pd, attr);

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          bwd_input_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(bwd_input_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      // Create memory.
      auto diff_dst_mem = CreateDnnlMemory(
          diff_dst_md, onednn_engine, GetTensorBuffer<T>(&diff_dst_tensor));
      auto diff_src_mem = CreateDnnlMemory(diff_src_md, onednn_engine,
                                           GetTensorBuffer<T>(diff_src_tensor));

      // reorder diff_dst to NHWC if needed
      memory diff_dst_mem_opt = diff_dst_mem, diff_src_mem_opt = diff_src_mem;
      // keep tensor out of if block to avoid of being deallocated
      Tensor diff_dst_tensor_opt, diff_src_tensor_opt;
      if (data_layout != format_tag_opt) {
        int64 diff_dst_nums =
            bwd_input_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({diff_dst_nums}),
                                              &diff_dst_tensor_opt));
        diff_dst_mem_opt =
            CreateDnnlMemory(diff_dst_md_opt, onednn_engine,
                             GetTensorBuffer<T>(&diff_dst_tensor_opt));
        ReorderMemory(*context, &diff_dst_mem, &diff_dst_mem_opt,
                      onednn_engine);
        // allocate diff_src memory for reorder back later
        int64 diff_src_nums =
            bwd_input_pd.diff_src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({diff_src_nums}),
                                              &diff_src_tensor_opt));
        diff_src_mem_opt =
            CreateDnnlMemory(diff_src_md_opt, onednn_engine,
                             GetTensorBuffer<T>(&diff_src_tensor_opt));
      }

      // Check filter reorder
      Tensor tmp_weight;
      T* filter_data = const_cast<T*>(filter_tensor.flat<T>().data());
      auto filter_mem = CreateDnnlMemory(filter_md, onednn_engine,
                                         static_cast<void*>(filter_data));
      bool is_filter_reordered = (bwd_input_pd.weights_desc() != filter_md);

      if (is_filter_reordered) {
        // TODO(itex): add supports for costant filter
        bool is_filter_cached = false;
        if (!is_filter_cached) {
          // allocate temporay reorder tensor
          int reorder_filter_data_size =
              bwd_input_pd.weights_desc().get_size() / sizeof(T);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DataTypeToEnum<T>::v(),
                                      TensorShape({reorder_filter_data_size}),
                                      &tmp_weight));
          void* filter_data_handle = GetTensorBuffer<T>(&tmp_weight);
          auto filter_mem_reordered = CreateDnnlMemory(
              bwd_input_pd.weights_desc(), onednn_engine, filter_data_handle);
          ReorderMemory(*context, &filter_mem, &filter_mem_reordered,
                        onednn_engine);
          filter_data = static_cast<T*>(filter_mem_reordered.get_data_handle());
          filter_mem = filter_mem_reordered;
        }
      }

      std::unordered_map<int, memory> bwd_input_primitive_args;
      bwd_input_primitive_args.insert({DNNL_ARG_WEIGHTS, filter_mem});
      bwd_input_primitive_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem_opt});
      bwd_input_primitive_args.insert({DNNL_ARG_DIFF_SRC, diff_src_mem_opt});
      bwd_input_primitive_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      // Create convolution backward input primitive and add it to the net.
      primitive bwd_input_primitive = ConvBwdInputPrimitive(bwd_input_pd);
      bwd_input_primitive.execute(onednn_stream, bwd_input_primitive_args);
      primitive fwd_primitive = dnnl::convolution_forward(fwd_pd);

      // reorder back if needed
      if (data_layout != format_tag_opt) {
        ReorderMemory(*context, &diff_src_mem_opt, &diff_src_mem,
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
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_CONV_GRAD_OPS_H_
