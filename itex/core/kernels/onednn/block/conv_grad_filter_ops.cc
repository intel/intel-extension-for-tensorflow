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

#include <limits>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/conv_grad_ops.h"
#include "itex/core/kernels/common/conv_ops.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"

namespace itex {

using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;

template <typename Device, typename T, bool bias_enabled = false,
          bool is_depthwise = false>
class OneDnnConvBackpropFilterOp
    : public ConvBackpropCommonOp<Device, T, is_depthwise> {
 public:
  explicit OneDnnConvBackpropFilterOp(OpKernelConstruction* context)
      : ConvBackpropCommonOp<Device, T, is_depthwise>(context) {
    if (bias_enabled == true) {
      std::vector<string> fused_ops_;
      OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops_));
      OP_REQUIRES(context, fused_ops_.size() == 1,
                  errors::InvalidArgument(
                      "OneDnnConvBackpropFilter must have 1 post-arguments at "
                      "most."));
      OP_REQUIRES(context, fused_ops_[0] == "BiasAddGrad",
                  errors::InvalidArgument(
                      "The 1st post-argument of OneDnnConvBackpropFilter must "
                      "be BiasAddGrad."));
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

      // Get shapes of input & filter tensors
      OneDnnShape src_onednn_shape, diff_dst_onednn_shape;
      GetOneDnnShape(context, kInputIdx, &src_onednn_shape);
      GetOneDnnShape(context, kOutbpropIdx, &diff_dst_onednn_shape);

      // TODO(itex): rename GetTensorShape to CreateTensorShape
      const TensorShape& src_shape = src_onednn_shape.IsOneDnnTensor()
                                         ? src_onednn_shape.GetTfShape()
                                         : src_tensor.shape();
      const TensorShape& filter_shape = GetTensorShape(filter_sizes_tensor);
      const TensorShape& diff_dst_shape =
          diff_dst_onednn_shape.IsOneDnnTensor()
              ? diff_dst_onednn_shape.GetTfShape()
              : diff_dst_tensor.shape();

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* diff_filter_tensor = nullptr;
      Tensor* diff_bias_tensor = nullptr;
      OneDnnShape diff_filter_onednn_shape;
      if (src_shape.num_elements() == 0 || filter_shape.num_elements() == 0 ||
          diff_dst_shape.num_elements() == 0) {
        TensorShape diff_filter_shape = filter_shape;
        diff_filter_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kOutputIdx, &diff_filter_tensor,
                                     diff_filter_shape,
                                     diff_filter_onednn_shape);

        ITEX_CHECK_NOTNULL(diff_filter_tensor);
        // If output tensor has more than 0 elements, we need to 0 them out.
        if (diff_filter_shape.num_elements() > 0) {
          DeviceFill<Device, T>(diff_filter_tensor->flat<T>().data(), T(0),
                                diff_filter_shape.num_elements(),
                                context->GetDeviceStream());
        }
        return;
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
      // OneDnn dims
      conv_util.InitFwdDimensions(
          src_shape, filter_shape, &fwd_src_dims, &fwd_filter_dims,
          &stride_dims, &dilation_dims, &dst_dims_tf, &dst_dims_onednn,
          &pad_left_dims, &pad_right_dims);
      conv_util.GetInputDimension(diff_dst_shape, &diff_dst_dims);

      if (is_depthwise) {
        OP_REQUIRES(context, this->is_conv2d_,
                    errors::InvalidArgument(
                        "Only 2D convolution is supported for depthwise."));
      }

      memory::dims diff_filter_dims = fwd_filter_dims;

      OneDnnTensorFormat data_fmt_onednn =
          TFDataFormatToOneDnnDataFormat(this->data_format_, this->is_conv2d_);
      memory::format_tag data_layout = OneDnnTensorFormatToTag(data_fmt_onednn);
      OP_REQUIRES(context, data_layout != memory::format_tag::undef,
                  errors::InvalidArgument("Invalid data format"));
      memory::format_tag filter_layout =
          this->is_conv2d_ ? (is_depthwise ? memory::format_tag::hwigo
                                           : memory::format_tag::hwio)
                           : memory::format_tag::dhwio;

      memory::dims diff_bias_dims = {};
      int64 depth = 0;
      if (bias_enabled) {
        depth = (this->data_format_ == FORMAT_NCHW)
                    ? diff_dst_shape.dim_size(1)
                    : diff_dst_shape.dim_size(this->is_conv2d_ ? 3 : 4);
        diff_bias_dims = {static_cast<int>(depth)};
      }

      // OneDnn memory desc
      memory::desc fwd_src_md =
          src_onednn_shape.IsOneDnnTensor()
              ? src_onednn_shape.GetOneDnnLayout()
              : memory::desc(fwd_src_dims, OneDnnType<T>(), data_layout);
      memory::desc diff_dst_md =
          diff_dst_onednn_shape.IsOneDnnTensor()
              ? diff_dst_onednn_shape.GetOneDnnLayout()
              : memory::desc(diff_dst_dims, OneDnnType<T>(), data_layout);
      memory::desc diff_filter_md =
          memory::desc(diff_filter_dims, OneDnnType<T>(), filter_layout);
      memory::desc diff_bias_md = memory::desc(
          {diff_bias_dims}, OneDnnType<T>(), dnnl::memory::format_tag::x);

      // OneDnn preferred memory desc
      memory::desc fwd_src_md_prefer =
          memory::desc(fwd_src_dims, OneDnnType<T>(), memory::format_tag::any);
      memory::desc diff_dst_md_prefer =
          memory::desc(diff_dst_dims, OneDnnType<T>(), memory::format_tag::any);
      memory::desc diff_filter_md_prefer = memory::desc(
          diff_filter_dims, OneDnnType<T>(), memory::format_tag::any);

      // OneDNN dilations start from 0.
      for (int i = 0; i < dilation_dims.size(); ++i) {
        --dilation_dims[i];
      }

      ConvFwdDesc fwd_desc = ConvFwdDesc(
          prop_kind::forward, dnnl::algorithm::convolution_direct,
          fwd_src_md_prefer, diff_filter_md_prefer, diff_dst_md_prefer,
          stride_dims, dilation_dims, pad_left_dims, pad_right_dims);

      if (bias_enabled) {
        fwd_desc =
            ConvFwdDesc(prop_kind::forward, dnnl::algorithm::convolution_direct,
                        fwd_src_md_prefer, diff_filter_md_prefer, diff_bias_md,
                        diff_dst_md_prefer, stride_dims, dilation_dims,
                        pad_left_dims, pad_right_dims);
      }

      // Create descriptor and primitive descriptor for convolution bwd filter.
      ConvBwdFilterDesc bwd_filter_desc = ConvBwdFilterDesc(
          dnnl::algorithm::convolution_direct, fwd_src_md_prefer,
          diff_filter_md_prefer, diff_dst_md_prefer, stride_dims, dilation_dims,
          pad_left_dims, pad_right_dims);

      if (bias_enabled) {
        bwd_filter_desc = ConvBwdFilterDesc(
            dnnl::algorithm::convolution_direct, fwd_src_md_prefer,
            diff_filter_md_prefer, diff_bias_md, diff_dst_md_prefer,
            stride_dims, dilation_dims, pad_left_dims, pad_right_dims);
      }

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<T, float>::value) {
        attr.set_fpmath_mode(this->fp32_math_mode_);
      }

      ConvFwdPd fwd_pd = ConvFwdPd(fwd_desc, attr, onednn_engine);
      ConvBwdFilterPd bwd_filter_pd =
          ConvBwdFilterPd(bwd_filter_desc, attr, onednn_engine, fwd_pd);

      // Allocate output tensors: diff_fitler.
      TensorShape diff_filter_shape;
      if (this->is_conv2d_) {
        if (!is_depthwise) {
          diff_filter_shape =
              TensorShape({diff_filter_dims[DimensionIndex::Dim_H],
                           diff_filter_dims[DimensionIndex::Dim_W],
                           diff_filter_dims[DimensionIndex::Dim_I],
                           diff_filter_dims[DimensionIndex::Dim_O]});

        } else {
          // Depthwise Conv2d: diff_filter_dims is GOIHW format.
          //                  | TensorFlow       | oneDNN
          // ----------------------------------------------------------------
          // filter_out_depth | depth_multiplier | depth_multiplier *
          //                  |                  | group_count
          // ----------------------------------------------------------------
          // filter_in_depth  | in_depth         | in_depth / group_count
          // For depthwise convolution, we have group_count == in_depth.
          // So here G = original I, and I = 1.
          // And the GOIHW is oneDNN format, here we try to extract the TF
          // format, TF format is HWIO, as G = original I, so here is HWGO.
          diff_filter_shape = TensorShape(
              {diff_filter_dims[FilterGroupDims::GROUP_FILTER_DIM_H],
               diff_filter_dims[FilterGroupDims::GROUP_FILTER_DIM_W],
               diff_filter_dims[FilterGroupDims::GROUP_FILTER_DIM_G],
               diff_filter_dims[FilterGroupDims::GROUP_FILTER_DIM_O]});
        }
      } else {
        diff_filter_shape =
            TensorShape({diff_filter_dims[DimensionIndex3D::Dim3d_D],
                         diff_filter_dims[DimensionIndex3D::Dim3d_H],
                         diff_filter_dims[DimensionIndex3D::Dim3d_W],
                         diff_filter_dims[DimensionIndex3D::Dim3d_I],
                         diff_filter_dims[DimensionIndex3D::Dim3d_O]});
      }

      diff_filter_onednn_shape.SetOneDnnTensor(false);
      AllocateOutputSetOneDnnShape(context, kOutputIdx, &diff_filter_tensor,
                                   diff_filter_shape, diff_filter_onednn_shape);

      if (bias_enabled == true) {
        TensorShape bias_tensor_shape(diff_bias_dims);
        OneDnnShape bias_onednn_shape;
        bias_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kDiffBiasIndex, &diff_bias_tensor,
                                     bias_tensor_shape, bias_onednn_shape);
      }

      // Check whether src and diff_dst need to be reordered.
      bool is_src_reordered = (fwd_src_md != bwd_filter_pd.src_desc());
      auto src_mem = CreateDnnlMemory(fwd_src_md, onednn_engine,
                                      GetTensorBuffer<T>(&src_tensor));

      Tensor tmp_src;
      memory src_mem_reordered;
      if (is_src_reordered) {
        int64 reorder_src_size =
            bwd_filter_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::v(),
                                    TensorShape({reorder_src_size}), &tmp_src));
        src_mem_reordered =
            CreateDnnlMemory(bwd_filter_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<T>(&tmp_src));
        ReorderMemory(*context, &src_mem, &src_mem_reordered, onednn_engine);
        src_mem = src_mem_reordered;
      }

      dnnl::memory diff_bias_mem;
      if (bias_enabled == true) {
        void* diff_bias_data = GetTensorBuffer<T>(diff_bias_tensor);
        diff_bias_mem =
            CreateDnnlMemory(diff_bias_md, onednn_engine, diff_bias_data);
      }
      // Check diff_dst need to be reordered.
      bool is_diff_dst_reordered =
          (diff_dst_md != bwd_filter_pd.diff_dst_desc());
      auto diff_dst_mem = CreateDnnlMemory(
          diff_dst_md, onednn_engine, GetTensorBuffer<T>(&diff_dst_tensor));

      Tensor tmp_diff_dst;
      memory diff_dst_mem_reordered;
      if (is_diff_dst_reordered) {
        int64 reorder_diff_dst_size =
            bwd_filter_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(
            context, context->allocate_temp(
                         DataTypeToEnum<T>::v(),
                         TensorShape({reorder_diff_dst_size}), &tmp_diff_dst));
        diff_dst_mem_reordered =
            CreateDnnlMemory(bwd_filter_pd.diff_dst_desc(), onednn_engine,
                             GetTensorBuffer<T>(&tmp_diff_dst));
        ReorderMemory(*context, &diff_dst_mem, &diff_dst_mem_reordered,
                      onednn_engine);
        diff_dst_mem = diff_dst_mem_reordered;
      }

      // Prepare tmp buffer for diff_weight.
      Tensor tmp_diff_filter;
      int64 reorder_diff_filter_size =
          bwd_filter_pd.diff_weights_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::v(),
                                  TensorShape({reorder_diff_filter_size}),
                                  &tmp_diff_filter));
      auto diff_filter_mem_prefer =
          CreateDnnlMemory(bwd_filter_pd.diff_weights_desc(), onednn_engine,
                           GetTensorBuffer<T>(&tmp_diff_filter));

      // Execute.
      std::unordered_map<int, memory> bwd_filter_primitive_args;
      bwd_filter_primitive_args.insert({DNNL_ARG_SRC, src_mem});
      bwd_filter_primitive_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem});
      bwd_filter_primitive_args.insert(
          {DNNL_ARG_DIFF_WEIGHTS, diff_filter_mem_prefer});

      if (bias_enabled == true) {
        bwd_filter_primitive_args.insert({DNNL_ARG_DIFF_BIAS, diff_bias_mem});
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
      bwd_filter_primitive_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      primitive bwd_filter_primitive = ConvBwdFilterPrimitive(bwd_filter_pd);
      bwd_filter_primitive.execute(onednn_stream, bwd_filter_primitive_args);

      // Reorder diff_weight.
      auto diff_filter_mem =
          CreateDnnlMemory(diff_filter_md, onednn_engine,
                           GetTensorBuffer<T>(diff_filter_tensor));
      ReorderMemory(*context, &diff_filter_mem_prefer, &diff_filter_mem,
                    onednn_engine);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
    return;
  }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(T)                                                  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv2DBackpropFilter")               \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<T>("T")                       \
                              .HostMemory("filter_sizes")                   \
                              .HostMemory("input_meta")                     \
                              .HostMemory("filter_sizes_meta")              \
                              .HostMemory("out_backprop_meta")              \
                              .HostMemory("output_meta"),                   \
                          OneDnnConvBackpropFilterOp<GPUDevice, T, false>); \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv3DBackpropFilterV2")             \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<T>("T")                       \
                              .HostMemory("filter_sizes")                   \
                              .HostMemory("input_meta")                     \
                              .HostMemory("filter_sizes_meta")              \
                              .HostMemory("out_backprop_meta")              \
                              .HostMemory("output_meta"),                   \
                          OneDnnConvBackpropFilterOp<GPUDevice, T, false>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_OneDnnDepthwiseConv2dNativeBackpropFilter")                    \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<T>("T")                                           \
          .HostMemory("filter_sizes")                                       \
          .HostMemory("input_meta")                                         \
          .HostMemory("filter_sizes_meta")                                  \
          .HostMemory("out_backprop_meta")                                  \
          .HostMemory("output_meta"),                                       \
      OneDnnConvBackpropFilterOp<GPUDevice, T, false, true>);

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);

#define REGISTER_KERNEL_FILTER_WITH_BIAS(T)                                \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv2DBackpropFilterWithBias")      \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .HostMemory("filter_sizes")                  \
                              .HostMemory("input_meta")                    \
                              .HostMemory("filter_sizes_meta")             \
                              .HostMemory("out_backprop_meta")             \
                              .HostMemory("output_meta")                   \
                              .HostMemory("bias_grad_meta"),               \
                          OneDnnConvBackpropFilterOp<GPUDevice, T, true>); \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv3DBackpropFilterWithBias")      \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .HostMemory("filter_sizes")                  \
                              .HostMemory("input_meta")                    \
                              .HostMemory("filter_sizes_meta")             \
                              .HostMemory("out_backprop_meta")             \
                              .HostMemory("output_meta")                   \
                              .HostMemory("bias_grad_meta"),               \
                          OneDnnConvBackpropFilterOp<GPUDevice, T, true>);

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL_FILTER_WITH_BIAS);

#else
#define REGISTER_KERNEL(T)                                                  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv2DBackpropFilter")               \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<T>("T"),                      \
                          OneDnnConvBackpropFilterOp<CPUDevice, T, false>); \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv3DBackpropFilterV2")             \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<T>("T"),                      \
                          OneDnnConvBackpropFilterOp<CPUDevice, T, false>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_OneDnnDepthwiseConv2dNativeBackpropFilter")                    \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<T>("T"),                                          \
      OneDnnConvBackpropFilterOp<CPUDevice, T, false, true>);               \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv2DBackpropFilterWithBias")       \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<T>("T"),                      \
                          OneDnnConvBackpropFilterOp<CPUDevice, T, true>);  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv3DBackpropFilterWithBias")       \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<T>("T"),                      \
                          OneDnnConvBackpropFilterOp<CPUDevice, T, true>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#endif  // INTEL_CPU_ONLY

}  // namespace itex
