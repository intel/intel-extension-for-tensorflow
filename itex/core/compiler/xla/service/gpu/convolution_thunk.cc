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

#include "itex/core/compiler/xla/service/gpu/convolution_thunk.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "itex/core/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "itex/core/compiler/xla/service/gpu/ir_emission_utils.h"
#include "itex/core/compiler/xla/service/hlo_casting_utils.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"
#include "itex/core/compiler/xla/types.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/logging.h"
namespace itex_xla {
namespace gpu {

using ConvFwdPd = dnnl::convolution_forward::primitive_desc;
using ConvBwdInputPd = dnnl::convolution_backward_data::primitive_desc;
using ConvBwdFilterPd = dnnl::convolution_backward_weights::primitive_desc;
using ConvBwdFilterPrimitive = dnnl::convolution_backward_weights;

int64_t GetVectCSize(DataLayout layout) {
  switch (layout) {
    case DataLayout::kBatchDepthYX4:
      return 4;
    case DataLayout::kBatchDepthYX32:
      return 32;
    default:
      return 1;
  }
}

int64_t GetVectCSize(FilterLayout layout) {
  switch (layout) {
    case FilterLayout::kOutputInputYX4:
      return 4;
    case FilterLayout::kOutputInputYX32:
      return 32;
    default:
      return 1;
  }
}

Status CreateOneDnnPrimitive(
    OneDnnConvPrimitive* onednn_primitive,  // NOLINT
    const GpuConvDescriptor& conv_descriptor,
    absl::Span<const se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer, const Thunk::ExecuteParams& params) {
  ITEX_GPUStream* dpcpp_stream = se::gpu::AsGpuStreamValue(params.stream);
  auto& buffer_allocations = *params.buffer_allocations;
  se::ScratchAllocator scratch_allocator(buffer_allocations.device_ordinal(),
                                         buffer_allocations.memory_allocator());
  onednn_primitive->engine = dnnl::sycl_interop::make_engine(
      dpcpp_stream->get_device(), dpcpp_stream->get_context());
  onednn_primitive->stream =
      dnnl::sycl_interop::make_stream(onednn_primitive->engine, *dpcpp_stream);
  DataLayout input_dl;
  FilterLayout filter_dl;
  DataLayout output_dl;

  Shape input_shape, filter_shape, output_shape;
  void* input_data;
  void* filter_data;
  void* output_data;
  void* bias_data = nullptr;
  void* side_input_data = nullptr;

  const CudnnConvBackendConfig& backend_config = conv_descriptor.backend_config;
  float conv_result_scale =
      static_cast<float>(backend_config.conv_result_scale());
  bool conv_result_scale_one = (fabs(conv_result_scale - 1.0f) < 1e-6);

  switch (conv_descriptor.kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
      input_shape = conv_descriptor.operand0_shape;
      filter_shape = conv_descriptor.operand1_shape;
      output_shape = conv_descriptor.result_shape;

      input_data = const_cast<void*>(operand_buffers[0].opaque());
      filter_data = const_cast<void*>(operand_buffers[1].opaque());
      output_data = const_cast<void*>(result_buffer.opaque());
      break;
    case CudnnConvKind::kBackwardInput:
      input_shape = conv_descriptor.result_shape;
      filter_shape = conv_descriptor.operand1_shape;
      output_shape = conv_descriptor.operand0_shape;

      input_data = const_cast<void*>(result_buffer.opaque());
      filter_data = const_cast<void*>(operand_buffers[1].opaque());
      output_data = const_cast<void*>(operand_buffers[0].opaque());

      break;
    case CudnnConvKind::kBackwardFilter:
      input_shape = conv_descriptor.operand0_shape;
      filter_shape = conv_descriptor.result_shape;
      output_shape = conv_descriptor.operand1_shape;

      input_data = const_cast<void*>(operand_buffers[0].opaque());
      filter_data = const_cast<void*>(result_buffer.opaque());
      output_data = const_cast<void*>(operand_buffers[1].opaque());

      break;
    default:
      return InternalError("Unkown convolution kind");
  }

  float side_input_scale;
  bool side_input_scale_zero;
  if (conv_descriptor.kind == CudnnConvKind::kForwardActivation) {
    bias_data = const_cast<void*>(operand_buffers[2].opaque());
    if (operand_buffers.size() >= 4) {
      side_input_data = const_cast<void*>(operand_buffers[3].opaque());
      side_input_scale = backend_config.side_input_scale();
      side_input_scale_zero = (fabs(side_input_scale - 0.0f) < 1e-6);
    }
  }

  const Window& window = conv_descriptor.window;
  const ConvolutionDimensionNumbers& dnums = conv_descriptor.dnums;

  TF_ASSIGN_OR_RETURN(std::tie(input_dl, filter_dl, output_dl),
                      XlaConvShapesToStreamExecutorLayouts(
                          dnums, input_shape, filter_shape, output_shape));

  const int num_dimensions = conv_descriptor.window.dimensions_size();
  ITEX_CHECK_LE(num_dimensions, 3);

  // OneDNN does not support 1D convolutions. We therefore express 1D
  // convolutions as 2D convolutions where the first spatial dimension is 1.
  // This matches the behavior of TF (see definition of conv1d in
  // tensorflow/python/ops/nn_ops.py).
  const int effective_num_dimensions = std::max(2, num_dimensions);

  int ic = GetVectCSize(input_dl) *
           input_shape.dimensions(dnums.input_feature_dimension());
  int n = input_shape.dimensions(dnums.input_batch_dimension());
  int id, ih, iw;
  if (num_dimensions == 3) {
    id = input_shape.dimensions(dnums.input_spatial_dimensions(0));
    ih = input_shape.dimensions(dnums.input_spatial_dimensions(1));
    iw = input_shape.dimensions(dnums.input_spatial_dimensions(2));
  } else if (num_dimensions == 2) {
    ih = input_shape.dimensions(dnums.input_spatial_dimensions(0));
    iw = input_shape.dimensions(dnums.input_spatial_dimensions(1));
  } else if (num_dimensions == 1) {
    ih = 1;
    iw = input_shape.dimensions(dnums.input_spatial_dimensions(0));
  } else {
    return InternalError("Invalid convolution dimension num");
  }

  int kd, kh, kw;
  if (num_dimensions == 3) {
    kd = filter_shape.dimensions(dnums.kernel_spatial_dimensions(0));
    kh = filter_shape.dimensions(dnums.kernel_spatial_dimensions(1));
    kw = filter_shape.dimensions(dnums.kernel_spatial_dimensions(2));
  } else if (num_dimensions == 2) {
    kh = filter_shape.dimensions(dnums.kernel_spatial_dimensions(0));
    kw = filter_shape.dimensions(dnums.kernel_spatial_dimensions(1));
  } else {
    kh = 1;
    kw = filter_shape.dimensions(dnums.kernel_spatial_dimensions(0));
  }

  // It is group-conv if filter_in != src_in
  // G = src_in/filter_in
  // O = filter_out/G
  // TODO(ITEX): depthwise-conv
  int filter_ic =
      filter_shape.dimensions(dnums.kernel_input_feature_dimension());
  int filter_oc =
      filter_shape.dimensions(dnums.kernel_output_feature_dimension());
  bool is_group_conv = ic != filter_ic;
  int kg = ic / filter_ic;  // kg for group-conv and depthwise-conv
  int ko = filter_oc / kg;
  int ki = filter_ic;

  int padding_d_l, padding_h_l, padding_w_l;
  int padding_d_h, padding_h_h, padding_w_h;
  int stride_d, stride_h, stride_w, dilate_d, dilate_h, dilate_w;

  if (num_dimensions == 3) {
    padding_d_l = window.dimensions(0).padding_low();
    padding_h_l = window.dimensions(1).padding_low();
    padding_w_l = window.dimensions(2).padding_low();
    padding_d_h = window.dimensions(0).padding_high();
    padding_h_h = window.dimensions(1).padding_high();
    padding_w_h = window.dimensions(2).padding_high();

    stride_d = window.dimensions(0).stride();
    stride_h = window.dimensions(1).stride();
    stride_w = window.dimensions(2).stride();

    dilate_d = window.dimensions(0).window_dilation();
    dilate_h = window.dimensions(1).window_dilation();
    dilate_w = window.dimensions(2).window_dilation();
  } else if (num_dimensions == 2) {
    padding_h_l = window.dimensions(0).padding_low();
    padding_w_l = window.dimensions(1).padding_low();
    padding_h_h = window.dimensions(0).padding_high();
    padding_w_h = window.dimensions(1).padding_high();

    stride_h = window.dimensions(0).stride();
    stride_w = window.dimensions(1).stride();

    dilate_h = window.dimensions(0).window_dilation();
    dilate_w = window.dimensions(1).window_dilation();
  } else if (num_dimensions == 1) {
    padding_h_l = 0;
    padding_w_l = window.dimensions(0).padding_low();
    padding_h_h = 0;
    padding_w_h = window.dimensions(0).padding_high();

    stride_h = 1;
    stride_w = window.dimensions(0).stride();

    dilate_h = 1;
    dilate_w = window.dimensions(0).window_dilation();
  }

  int od, oh, ow;
  int oc = output_shape.dimensions(dnums.output_feature_dimension());
  if (num_dimensions == 3) {
    od = output_shape.dimensions(dnums.output_spatial_dimensions(0));
    oh = output_shape.dimensions(dnums.output_spatial_dimensions(1));
    ow = output_shape.dimensions(dnums.output_spatial_dimensions(2));
  } else if (num_dimensions == 2) {
    oh = output_shape.dimensions(dnums.output_spatial_dimensions(0));
    ow = output_shape.dimensions(dnums.output_spatial_dimensions(1));
  } else if (num_dimensions == 1) {
    oh = 1;
    ow = output_shape.dimensions(dnums.output_spatial_dimensions(0));
  }
  bool is_conv3d = (num_dimensions == 3);
  try {
    dnnl::memory::dims src_dims, filter_dims, bias_dims, dst_dims, stride_dims,
        padding_dims_l, padding_dims_r, dilation_dims;
    dnnl::memory::format_tag src_fmt, weight_fmt, dst_fmt;
    if (!is_conv3d) {
      src_dims = {n, ic, ih, iw};
      if (is_group_conv)
        filter_dims = {kg, ko, ki, kh, kw};
      else
        filter_dims = {ko, ki, kh, kw};
      bias_dims = {oc};
      dst_dims = {n, oc, oh, ow};
      stride_dims = {stride_h, stride_w};
      padding_dims_l = {padding_h_l, padding_w_l};
      padding_dims_r = {padding_h_h, padding_w_h};
      dilation_dims = {dilate_h - 1, dilate_w - 1};

      switch (input_dl) {
        case DataLayout::kBatchDepthYX:
          src_fmt = dnnl::memory::format_tag::nchw;
          break;
        case DataLayout::kBatchYXDepth:
          src_fmt = dnnl::memory::format_tag::nhwc;
          break;
        default:
          return InternalError("Unsupported input format");
      }

      switch (filter_dl) {
        case FilterLayout::kOutputInputYX:
          weight_fmt = is_group_conv ? dnnl::memory::format_tag::goihw
                                     : dnnl::memory::format_tag::oihw;
          break;
        case FilterLayout::kOutputYXInput:
          weight_fmt = is_group_conv ? dnnl::memory::format_tag::gohwi
                                     : dnnl::memory::format_tag::ohwi;
          break;
        default:
          return InternalError("Unsupported weight format");
      }

      switch (output_dl) {
        case DataLayout::kBatchDepthYX:
          dst_fmt = dnnl::memory::format_tag::nchw;
          break;
        case DataLayout::kBatchYXDepth:
          dst_fmt = dnnl::memory::format_tag::nhwc;
          break;
        default:
          return InternalError("Unsupported input format");
      }
    } else {
      src_dims = {n, ic, id, ih, iw};
      filter_dims = {oc, ic, kd, kh, kw};
      bias_dims = {oc};
      dst_dims = {n, oc, od, oh, ow};
      stride_dims = {stride_d, stride_h, stride_w};
      padding_dims_l = {padding_d_l, padding_h_l, padding_w_l};
      padding_dims_r = {padding_d_h, padding_h_h, padding_w_h};
      dilation_dims = {dilate_d - 1, dilate_h - 1, dilate_w - 1};

      switch (input_dl) {
        case DataLayout::kBatchDepthYX:
          src_fmt = dnnl::memory::format_tag::ncdhw;
          break;
        case DataLayout::kBatchYXDepth:
          src_fmt = dnnl::memory::format_tag::ndhwc;
          break;
        default:
          return InternalError("Unsupported input format");
      }

      switch (filter_dl) {
        case FilterLayout::kOutputInputYX:
          weight_fmt = dnnl::memory::format_tag::oidhw;
          break;
        case FilterLayout::kOutputYXInput:
          weight_fmt = dnnl::memory::format_tag::odhwi;
          break;
        default:
          return InternalError("Unsupported weight format");
      }

      switch (output_dl) {
        case DataLayout::kBatchDepthYX:
          dst_fmt = dnnl::memory::format_tag::ncdhw;
          break;
        case DataLayout::kBatchYXDepth:
          dst_fmt = dnnl::memory::format_tag::ndhwc;
          break;
        default:
          return InternalError("Unsupported input format");
      }
    }

    auto kind = dnnl::sycl_interop::memory_kind::usm;

    dnnl::memory::data_type data_type;

    PrimitiveType input_type = input_shape.element_type();
    switch (input_type) {
      case BF16:
        data_type = dnnl::memory::data_type::bf16;
        break;
      case F32:
        data_type = dnnl::memory::data_type::f32;
        break;
      case F16:
        data_type = dnnl::memory::data_type::f16;
        break;
      case F64:
        data_type = dnnl::memory::data_type::f64;
        break;
      default:
        return InternalError("Unsupported input data type");
    }

    dnnl::memory::desc src_md =
        dnnl::memory::desc({src_dims}, data_type, src_fmt);
    dnnl::memory::desc filter_md =
        dnnl::memory::desc({filter_dims}, data_type, weight_fmt);
    dnnl::memory::desc dst_md =
        dnnl::memory::desc({dst_dims}, data_type, dst_fmt);

    bool flag = false;
    itex::ReadBoolFromEnvVar("ONEDNN_PLAIN_WEIGHT", false, &flag);
    dnnl::memory::desc filter_md_prefer = dnnl::memory::desc(
        {filter_dims}, data_type, dnnl::memory::format_tag::any);
    if (flag)
      filter_md_prefer =
          dnnl::memory::desc({filter_dims}, data_type, weight_fmt);

    onednn_primitive->src_memory = dnnl::sycl_interop::make_memory(
        src_md, onednn_primitive->engine, kind, input_data);
    onednn_primitive->filter_memory = dnnl::sycl_interop::make_memory(
        filter_md, onednn_primitive->engine, kind, filter_data);
    onednn_primitive->dst_memory = dnnl::sycl_interop::make_memory(
        dst_md, onednn_primitive->engine, kind, output_data);

    // if alpha is 1:
    //   out = activation(conv(x, w, bias) + beta * side)
    //   po.append_sum(beta)
    //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);
    // else:
    //   out = activation(alpha * conv(x, w) + beta * side + bias)
    //   po.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0);
    //   po.append_sum(beta)
    //   po.append_binary(1, bias);
    //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);
    dnnl::post_ops po;
    dnnl::primitive_attr post_ops_attr;
    if (!conv_result_scale_one)
      po.append_eltwise(dnnl::algorithm::eltwise_linear, conv_result_scale, 0);
    if (side_input_data && !side_input_scale_zero)
      po.append_sum(side_input_scale);
    if (!conv_result_scale_one && bias_data) {
      auto bias_post_md =
          dnnl::memory::desc(bias_dims, data_type, dnnl::memory::format_tag::x);
      po.append_binary(dnnl::algorithm::binary_add, bias_post_md);
      onednn_primitive->bias_memory = dnnl::sycl_interop::make_memory(
          bias_post_md, onednn_primitive->engine, kind, bias_data);
      onednn_primitive->fwd_primitives_args.insert(
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(po.len() - 1) | DNNL_ARG_SRC_1,
           onednn_primitive->bias_memory});
    }
    if (conv_descriptor.kind == CudnnConvKind::kForwardActivation) {
      switch (conv_descriptor.activation) {
        case mlir::lmhlo_gpu::Activation::Sigmoid:
          po.append_eltwise(dnnl::algorithm::eltwise_logistic, 1, 0);
          break;
        case mlir::lmhlo_gpu::Activation::Relu:
          po.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);
          break;
        case mlir::lmhlo_gpu::Activation::Relu6:
          po.append_eltwise(dnnl::algorithm::eltwise_clip_v2, 0, 6);
          break;
        case mlir::lmhlo_gpu::Activation::Tanh:
          po.append_eltwise(dnnl::algorithm::eltwise_tanh, 0, 0);
          break;
        case mlir::lmhlo_gpu::Activation::None:
          break;
        default:
          return InternalError("Unsupported Activation mode");
      }
    }
    post_ops_attr.set_post_ops(po);
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Set fp32 mode.
    dnnl::fpmath_mode fp32_math_mode = itex::GetFP32MathMode<itex::GPUDevice>();
    if (input_type == F32) {
      post_ops_attr.set_fpmath_mode(fp32_math_mode);
    }

    if (conv_descriptor.kind == CudnnConvKind::kForward ||
        conv_descriptor.kind == CudnnConvKind::kForwardActivation) {
      ConvFwdPd fwd_pd =
          ConvFwdPd(onednn_primitive->engine, dnnl::prop_kind::forward,
                    dnnl::algorithm::convolution_direct, src_md,
                    filter_md_prefer, dst_md, stride_dims, dilation_dims,
                    padding_dims_l, padding_dims_r, post_ops_attr);
      if (bias_data != nullptr && conv_result_scale_one) {
        auto bias_md = dnnl::memory::desc(bias_dims, data_type,
                                          dnnl::memory::format_tag::x);
        fwd_pd = ConvFwdPd(onednn_primitive->engine, dnnl::prop_kind::forward,
                           dnnl::algorithm::convolution_direct, src_md,
                           filter_md_prefer, bias_md, dst_md, stride_dims,
                           dilation_dims, padding_dims_l, padding_dims_r,
                           post_ops_attr);
        onednn_primitive->bias_memory = dnnl::sycl_interop::make_memory(
            bias_md, onednn_primitive->engine, kind, bias_data);
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_BIAS, onednn_primitive->bias_memory});
      }

      onednn_primitive->fwd_primitive = dnnl::convolution_forward(fwd_pd);
      size_t scratchpad_size = fwd_pd.scratchpad_desc().get_size();
      void* workspace;
      TF_RETURN_IF_ERROR(se::AllocateWorkspace(&workspace, &scratch_allocator,
                                               scratchpad_size));
      onednn_primitive->scratchpad_memory = dnnl::memory(
          fwd_pd.scratchpad_desc(), onednn_primitive->engine, workspace);

      bool is_filter_reordered = (filter_md != fwd_pd.weights_desc());
      if (is_filter_reordered) {
        onednn_primitive->has_reorder = true;
        size_t reorder_filter_data_size = fwd_pd.weights_desc().get_size();
        void* reorder_filter;
        TF_RETURN_IF_ERROR(se::AllocateWorkspace(
            &reorder_filter, &scratch_allocator, reorder_filter_data_size));

        onednn_primitive->internal_filter_memory = dnnl::memory(
            fwd_pd.weights_desc(), onednn_primitive->engine, reorder_filter);
        onednn_primitive->filter_reorder_primitive =
            dnnl::reorder(onednn_primitive->filter_memory,
                          onednn_primitive->internal_filter_memory);
        onednn_primitive->reorder_args = {
            {DNNL_ARG_SRC, onednn_primitive->filter_memory},
            {DNNL_ARG_DST, onednn_primitive->internal_filter_memory}};

        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_WEIGHTS, onednn_primitive->internal_filter_memory});
      } else {
        onednn_primitive->has_reorder = false;
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_WEIGHTS, onednn_primitive->filter_memory});
      }
      onednn_primitive->fwd_primitives_args.insert(
          {DNNL_ARG_SRC, onednn_primitive->src_memory});
      onednn_primitive->fwd_primitives_args.insert(
          {DNNL_ARG_DST, onednn_primitive->dst_memory});
      onednn_primitive->fwd_primitives_args.insert(
          {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory});

    } else if (conv_descriptor.kind == CudnnConvKind::kBackwardInput) {
      // TODO(ITEX): handle post_ops_attr.
      ConvFwdPd fwd_pd = ConvFwdPd(
          onednn_primitive->engine, dnnl::prop_kind::forward,
          dnnl::algorithm::convolution_direct, src_md, filter_md_prefer, dst_md,
          stride_dims, dilation_dims, padding_dims_l, padding_dims_r);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      ConvBwdInputPd bwd_input_pd = ConvBwdInputPd(
          onednn_primitive->engine, dnnl::algorithm::convolution_direct, src_md,
          filter_md_prefer, dst_md, stride_dims, dilation_dims, padding_dims_l,
          padding_dims_r, fwd_pd, attr);

      size_t scratchpad_size = bwd_input_pd.scratchpad_desc().get_size();
      void* workspace;
      TF_RETURN_IF_ERROR(se::AllocateWorkspace(&workspace, &scratch_allocator,
                                               scratchpad_size));
      onednn_primitive->scratchpad_memory = dnnl::memory(
          bwd_input_pd.scratchpad_desc(), onednn_primitive->engine, workspace);

      bool is_filter_reordered = (filter_md != bwd_input_pd.weights_desc());
      if (is_filter_reordered) {
        size_t reorder_filter_data_size =
            bwd_input_pd.weights_desc().get_size();
        void* reorder_filter;
        TF_RETURN_IF_ERROR(se::AllocateWorkspace(
            &reorder_filter, &scratch_allocator, reorder_filter_data_size));

        onednn_primitive->internal_filter_memory =
            dnnl::memory(bwd_input_pd.weights_desc(), onednn_primitive->engine,
                         reorder_filter);
        onednn_primitive->filter_reorder_primitive =
            dnnl::reorder(onednn_primitive->filter_memory,
                          onednn_primitive->internal_filter_memory);
        onednn_primitive->reorder_args = {
            {DNNL_ARG_SRC, onednn_primitive->filter_memory},
            {DNNL_ARG_DST, onednn_primitive->internal_filter_memory}};
        onednn_primitive->bwd_input_primitive_args.insert(
            {DNNL_ARG_WEIGHTS, onednn_primitive->internal_filter_memory});
        onednn_primitive->has_reorder = true;
      } else {
        onednn_primitive->bwd_input_primitive_args.insert(
            {DNNL_ARG_WEIGHTS, onednn_primitive->filter_memory});
        onednn_primitive->has_reorder = false;
      }

      onednn_primitive->bwd_input_primitive_args.insert(
          {DNNL_ARG_DIFF_DST, onednn_primitive->dst_memory});
      onednn_primitive->bwd_input_primitive_args.insert(
          {DNNL_ARG_DIFF_SRC, onednn_primitive->src_memory});
      onednn_primitive->bwd_input_primitive_args.insert(
          {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory});

      onednn_primitive->bwd_input_primitive =
          dnnl::convolution_backward_data(bwd_input_pd);

    } else if (conv_descriptor.kind == CudnnConvKind::kBackwardFilter) {
      // TODO(ITEX): handle post_ops_attr.
      ConvFwdPd fwd_pd = ConvFwdPd(
          onednn_primitive->engine, dnnl::prop_kind::forward,
          dnnl::algorithm::convolution_direct, src_md, filter_md_prefer, dst_md,
          stride_dims, dilation_dims, padding_dims_l, padding_dims_r);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      ConvBwdFilterPd bwd_filter_pd = ConvBwdFilterPd(
          onednn_primitive->engine, dnnl::algorithm::convolution_direct, src_md,
          filter_md_prefer, dst_md, stride_dims, dilation_dims, padding_dims_l,
          padding_dims_r, fwd_pd, attr);

      size_t scratchpad_size = bwd_filter_pd.scratchpad_desc().get_size();
      void* workspace;
      TF_RETURN_IF_ERROR(se::AllocateWorkspace(&workspace, &scratch_allocator,
                                               scratchpad_size));
      onednn_primitive->scratchpad_memory = dnnl::memory(
          bwd_filter_pd.scratchpad_desc(), onednn_primitive->engine, workspace);

      bool is_filter_reordered =
          (filter_md != bwd_filter_pd.diff_weights_desc());
      if (is_filter_reordered) {
        onednn_primitive->has_reorder = true;
        size_t reorder_filter_data_size =
            bwd_filter_pd.diff_weights_desc().get_size();
        void* prefer_filter;
        TF_RETURN_IF_ERROR(se::AllocateWorkspace(
            &prefer_filter, &scratch_allocator, reorder_filter_data_size));

        onednn_primitive->internal_filter_memory =
            dnnl::memory(bwd_filter_pd.diff_weights_desc(),
                         onednn_primitive->engine, prefer_filter);
        onednn_primitive->filter_reorder_primitive =
            dnnl::reorder(onednn_primitive->internal_filter_memory,
                          onednn_primitive->filter_memory);
        onednn_primitive->reorder_args = {
            {DNNL_ARG_SRC, onednn_primitive->internal_filter_memory},
            {DNNL_ARG_DST, onednn_primitive->filter_memory}};

        onednn_primitive->bwd_filter_primitive_args.insert(
            {DNNL_ARG_DIFF_WEIGHTS, onednn_primitive->internal_filter_memory});
      } else {
        onednn_primitive->has_reorder = false;
        onednn_primitive->bwd_filter_primitive_args.insert(
            {DNNL_ARG_DIFF_WEIGHTS, onednn_primitive->filter_memory});
      }

      onednn_primitive->bwd_filter_primitive_args.insert(
          {DNNL_ARG_SRC, onednn_primitive->src_memory});
      onednn_primitive->bwd_filter_primitive_args.insert(
          {DNNL_ARG_DIFF_DST, onednn_primitive->dst_memory});
      onednn_primitive->bwd_filter_primitive_args.insert(
          {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory});

      onednn_primitive->bwd_filter_primitive =
          ConvBwdFilterPrimitive(bwd_filter_pd);

    } else {
      return InternalError("Unkown convolutuion kind");
    }
  } catch (dnnl::error& e) {
    std::string error_msg = "Status: " + std::to_string(e.status) +
                            ", message: " + std::string(e.message) +
                            ", in file " + std::string(__FILE__) + ":" +
                            std::to_string(__LINE__);
    std::cout << error_msg << std::endl;
  }
}  // NOLINT

OneDnnConvPrimitive ConvolutionThunk::GetOrCreateOneDnnConvPrimitive(
    se::Stream* stream,
    const std::vector<se::DeviceMemoryBase>& operand_se_buffers,
    const se::DeviceMemoryBase& result_buffer, const ExecuteParams& params) {
  OneDnnConvPrimitive primitive;
  CreateOneDnnPrimitive(&primitive, descriptor_,
                        absl::MakeSpan(operand_se_buffers), result_buffer,
                        params);
  return primitive;
}

ConvolutionThunk::ConvolutionThunk(
    ThunkInfo thunk_info, GpuConvDescriptor descriptor,
    std::vector<BufferAllocation::Slice> operand_slices,
    BufferAllocation::Slice result_slice, BufferAllocation::Slice scratch_slice)
    : Thunk(Kind::kConvolution, thunk_info),
      operand_buffers_(std::move(operand_slices)),
      result_buffer_(result_slice),
      scratch_buffer_(scratch_slice),
      descriptor_(std::move(descriptor)) {}

Status ConvolutionThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  std::vector<se::DeviceMemoryBase> operand_se_buffers;
  for (const auto& buffer : operand_buffers_) {
    operand_se_buffers.push_back(buffer_allocations.GetDeviceAddress(buffer));
  }

  se::DeviceMemoryBase result_buffer =
      buffer_allocations.GetDeviceAddress(result_buffer_);

  se::DeviceMemoryBase scratch =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  auto stream = params.stream;
  auto conv_primitive = GetOrCreateOneDnnConvPrimitive(
      stream, operand_se_buffers, result_buffer, params);

  TF_RETURN_IF_ERROR(RunGpuConv(conv_primitive, descriptor_,
                                absl::MakeSpan(operand_se_buffers),
                                result_buffer, params));

  // Note:: Convolution has a tuple buffer as an output, but we don't need t
  // populate it as no one should be reading from the tuple directly.
  //  if (!params.stream->ok()) {
  //   return InternalError("ConvolutionThunk::ExecuteOnStream failed.");
  // }
  return Status::OK();
}

}  // namespace gpu

}  // namespace itex_xla
