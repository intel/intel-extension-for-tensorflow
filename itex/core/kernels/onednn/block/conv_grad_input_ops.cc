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

template <typename Device, typename T, bool is_depthwise = false,
          bool pad_enabled = false>
class OneDnnConvBackpropInputOp
    : public ConvBackpropCommonOp<Device, T, is_depthwise> {
 public:
  explicit OneDnnConvBackpropInputOp(OpKernelConstruction* context)
      : ConvBackpropCommonOp<Device, T, is_depthwise>(context) {
    if (pad_enabled) {
      OP_REQUIRES(
          context, this->padding_ == Padding::VALID,
          errors::InvalidArgument("Pad can only be fused with `VALID` Conv."));
    }
  }

  void Compute(OpKernelContext* context) override {
    const int kInputIndex_Src = 0, kInputIndex_Filter = 1,
              kInputIndex_DiffDst = 2;
    const int kOutputIndex_DiffSrc = 0;

    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      // Input tensors
      const Tensor& src_sizes_tensor = context->input(kInputIndex_Src);
      const Tensor& filter_tensor = context->input(kInputIndex_Filter);
      const Tensor& diff_dst_tensor = context->input(kInputIndex_DiffDst);

      // Get shapes of filter and diff_dst.
      OneDnnShape filter_onednn_shape, diff_dst_onednn_shape;
      GetOneDnnShape(context, kInputIndex_Filter, &filter_onednn_shape);
      GetOneDnnShape(context, kInputIndex_DiffDst, &diff_dst_onednn_shape);

      // TODO(itex): rename GetTensorShape to CreateTensorShape
      TensorShape src_shape = GetTensorShape(src_sizes_tensor);
      const TensorShape& filter_shape = filter_onednn_shape.IsOneDnnTensor()
                                            ? filter_onednn_shape.GetTfShape()
                                            : filter_tensor.shape();
      const TensorShape& diff_dst_shape =
          diff_dst_onednn_shape.IsOneDnnTensor()
              ? diff_dst_onednn_shape.GetTfShape()
              : diff_dst_tensor.shape();

      OneDnnConvUtil conv_util(context, this->data_format_, this->strides_,
                               this->dilations_, this->padding_,
                               this->explicit_paddings_, this->is_conv2d_,
                               is_depthwise);

      if (pad_enabled) {
        const int kSliceIdx = 3;
        conv_util.InitPadWithFusion(kSliceIdx, false);

        // If Slice is fused, we need to delete the padding for the input.
        auto size_tensor_vec = context->input(kSliceIdx + 1).vec<int32>();
        for (int i = 0; i < src_shape.dims(); i++) {
          src_shape.set_dim(i, size_tensor_vec(i));
        }
      }

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* diff_src_tensor = nullptr;
      OneDnnShape diff_src_onednn_shape;
      TensorShape diff_src_shape = src_shape;
      if (src_shape.num_elements() == 0 || filter_shape.num_elements() == 0 ||
          diff_dst_shape.num_elements() == 0) {
        diff_src_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kOutputIndex_DiffSrc,
                                     &diff_src_tensor, diff_src_shape,
                                     diff_src_onednn_shape);

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

      OneDnnTensorFormat data_fmt_onednn =
          TFDataFormatToOneDnnDataFormat(this->data_format_, this->is_conv2d_);
      memory::format_tag data_layout = OneDnnTensorFormatToTag(data_fmt_onednn);
      OP_REQUIRES(context, data_layout != memory::format_tag::undef,
                  errors::InvalidArgument("Invalid data format"));

      auto filter_layout = this->is_conv2d_
                               ? (is_depthwise || is_grouped_convolution
                                      ? memory::format_tag::hwigo
                                      : memory::format_tag::hwio)
                               : memory::format_tag::dhwio;
      memory::desc filter_md =
          memory::desc(fwd_filter_dims, OneDnnType<T>(), filter_layout);
      memory::desc filter_md_prefer = memory::desc(
          fwd_filter_dims, OneDnnType<T>(), memory::format_tag::any);

      memory::desc diff_dst_md =
          diff_dst_onednn_shape.IsOneDnnTensor()
              ? diff_dst_onednn_shape.GetOneDnnLayout()
              : memory::desc(diff_dst_dims, OneDnnType<T>(), data_layout);
      memory::desc diff_dst_md_prefer =
          memory::desc(diff_dst_dims, OneDnnType<T>(), memory::format_tag::any);

      memory::dims diff_src_dims = fwd_src_dims;
      memory::desc diff_src_md_prefer =
          memory::desc(diff_src_dims, OneDnnType<T>(), memory::format_tag::any);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<T, float>::value) {
        attr.set_fpmath_mode(this->fp32_math_mode_);
      }

#ifdef ITEX_ONEDNN_3_0
      ConvFwdPd fwd_pd =
          ConvFwdPd(onednn_engine, prop_kind::forward,
                    dnnl::algorithm::convolution_direct, diff_src_md_prefer,
                    filter_md_prefer, diff_dst_md_prefer, stride_dims,
                    dilation_dims, pad_left_dims, pad_right_dims, attr);
      ConvBwdInputPd bwd_input_pd = ConvBwdInputPd(
          onednn_engine, dnnl::algorithm::convolution_direct,
          diff_src_md_prefer, filter_md_prefer, diff_dst_md_prefer, stride_dims,
          dilation_dims, pad_left_dims, pad_right_dims, fwd_pd, attr);
#else
      // Create descriptor and primitive descriptor for convolution forward.
      ConvFwdDesc fwd_desc = ConvFwdDesc(
          prop_kind::forward, dnnl::algorithm::convolution_direct,
          diff_src_md_prefer, filter_md_prefer, diff_dst_md_prefer, stride_dims,
          dilation_dims, pad_left_dims, pad_right_dims);
      // Create descriptor and primitive descriptor for convolution bwd filter.
      ConvBwdInputDesc bwd_input_desc = ConvBwdInputDesc(
          dnnl::algorithm::convolution_direct, diff_src_md_prefer,
          filter_md_prefer, diff_dst_md_prefer, stride_dims, dilation_dims,
          pad_left_dims, pad_right_dims);

      ConvFwdPd fwd_pd = ConvFwdPd(fwd_desc, attr, onednn_engine);
      ConvBwdInputPd bwd_input_pd =
          ConvBwdInputPd(bwd_input_desc, attr, onednn_engine, fwd_pd);
#endif
      // Check whether filter and diff_dst need to be reordered.
      bool is_filter_reordered = (filter_md != bwd_input_pd.weights_desc());
      auto filter_mem = CreateDnnlMemory(filter_md, onednn_engine,
                                         GetTensorBuffer<T>(&filter_tensor));

      Tensor tmp_filter;
      memory filter_mem_reordered;
      if (is_filter_reordered) {
        int64 reorder_filter_data_size =
            bwd_input_pd.weights_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(
            context, context->allocate_temp(
                         DataTypeToEnum<T>::v(),
                         TensorShape({reorder_filter_data_size}), &tmp_filter));
        filter_mem_reordered =
            CreateDnnlMemory(bwd_input_pd.weights_desc(), onednn_engine,
                             GetTensorBuffer<T>(&tmp_filter));
        ReorderMemory(*context, &filter_mem, &filter_mem_reordered,
                      onednn_engine);
        filter_mem = filter_mem_reordered;
      }

      // Check whether src and filter need to be reordered.
      bool is_diff_dst_reordered =
          (diff_dst_md != bwd_input_pd.diff_dst_desc());
      auto diff_dst_mem = CreateDnnlMemory(
          diff_dst_md, onednn_engine, GetTensorBuffer<T>(&diff_dst_tensor));

      Tensor tmp_diff_dst;
      memory diff_dst_mem_reordered;
      if (is_diff_dst_reordered) {
        int64 reorder_diff_dst_size =
            bwd_input_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(
            context, context->allocate_temp(
                         DataTypeToEnum<T>::v(),
                         TensorShape({reorder_diff_dst_size}), &tmp_diff_dst));
        diff_dst_mem_reordered =
            CreateDnnlMemory(bwd_input_pd.diff_dst_desc(), onednn_engine,
                             GetTensorBuffer<T>(&tmp_diff_dst));
        ReorderMemory(*context, &diff_dst_mem, &diff_dst_mem_reordered,
                      onednn_engine);
        diff_dst_mem = diff_dst_mem_reordered;
      }

      // diff src mem.
      SetOutputTensorShape(bwd_input_pd.diff_src_desc(), data_fmt_onednn,
                           &diff_src_shape, &diff_src_onednn_shape, true);
      AllocateOutputSetOneDnnShape(context, kOutputIndex_DiffSrc,
                                   &diff_src_tensor, diff_src_shape,
                                   diff_src_onednn_shape);

      auto diff_src_mem =
          CreateDnnlMemory(bwd_input_pd.diff_src_desc(), onednn_engine,
                           GetTensorBuffer<T>(diff_src_tensor));

      std::unordered_map<int, memory> bwd_input_primitive_args;
      bwd_input_primitive_args.insert({DNNL_ARG_WEIGHTS, filter_mem});
      bwd_input_primitive_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem});
      bwd_input_primitive_args.insert({DNNL_ARG_DIFF_SRC, diff_src_mem});

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
      bwd_input_primitive_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      // Create convolution backward input primitive and add it to the net.
      primitive bwd_input_primitive = ConvBwdInputPrimitive(bwd_input_pd);
      bwd_input_primitive.execute(onednn_stream, bwd_input_primitive_args);
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
#define REGISTER_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv2DBackpropInput")        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .HostMemory("input_sizes")            \
                              .HostMemory("input_sizes_meta")       \
                              .HostMemory("filter_meta")            \
                              .HostMemory("out_backprop_meta")      \
                              .HostMemory("output_meta"),           \
                          OneDnnConvBackpropInputOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv3DBackpropInputV2")      \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .HostMemory("input_sizes")            \
                              .HostMemory("input_sizes_meta")       \
                              .HostMemory("filter_meta")            \
                              .HostMemory("out_backprop_meta")      \
                              .HostMemory("output_meta"),           \
                          OneDnnConvBackpropInputOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnConv2DBackpropInputWithSlice")                   \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<T>("T")                                   \
          .HostMemory("begin")                                      \
          .HostMemory("size")                                       \
          .HostMemory("input_sizes")                                \
          .HostMemory("input_sizes_meta")                           \
          .HostMemory("filter_meta")                                \
          .HostMemory("out_backprop_meta")                          \
          .HostMemory("output_meta")                                \
          .HostMemory("begin_meta")                                 \
          .HostMemory("size_meta"),                                 \
      OneDnnConvBackpropInputOp<GPUDevice, T, false, true>);        \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnConv3DBackpropInputV2WithSlice")                 \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<T>("T")                                   \
          .HostMemory("begin")                                      \
          .HostMemory("size")                                       \
          .HostMemory("input_sizes")                                \
          .HostMemory("input_sizes_meta")                           \
          .HostMemory("filter_meta")                                \
          .HostMemory("out_backprop_meta")                          \
          .HostMemory("output_meta")                                \
          .HostMemory("size_meta")                                  \
          .HostMemory("begin_meta"),                                \
      OneDnnConvBackpropInputOp<GPUDevice, T, false, true>);        \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnDepthwiseConv2dNativeBackpropInput")             \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<T>("T")                                   \
          .HostMemory("input_sizes")                                \
          .HostMemory("input_sizes_meta")                           \
          .HostMemory("filter_meta")                                \
          .HostMemory("out_backprop_meta")                          \
          .HostMemory("output_meta"),                               \
      OneDnnConvBackpropInputOp<GPUDevice, T, true, false>);
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);
#else
#define REGISTER_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv2DBackpropInput")        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T"),              \
                          OneDnnConvBackpropInputOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConv3DBackpropInputV2")      \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T"),              \
                          OneDnnConvBackpropInputOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnConv2DBackpropInputWithSlice")                   \
          .Device(DEVICE_CPU)                                       \
          .TypeConstraint<T>("T"),                                  \
      OneDnnConvBackpropInputOp<CPUDevice, T, false, true>);        \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnConv3DBackpropInputV2WithSlice")                 \
          .Device(DEVICE_CPU)                                       \
          .TypeConstraint<T>("T"),                                  \
      OneDnnConvBackpropInputOp<CPUDevice, T, false, true>);        \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnDepthwiseConv2dNativeBackpropInput")             \
          .Device(DEVICE_CPU)                                       \
          .TypeConstraint<T>("T"),                                  \
      OneDnnConvBackpropInputOp<CPUDevice, T, true, false>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

#endif  // INTEL_CPU_ONLY
}  // namespace itex
