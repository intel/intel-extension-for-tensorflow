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

#ifndef ITEX_CORE_KERNELS_COMMON_CONV_OPS_H_
#define ITEX_CORE_KERNELS_COMMON_CONV_OPS_H_

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/common_shape_fns.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/padding.h"
#include "itex/core/utils/tensor_format.h"

namespace itex {

using dnnl::convolution_forward;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using ConvFwdDesc = dnnl::convolution_forward::desc;
using ConvFwdPd = dnnl::convolution_forward::primitive_desc;

#define DNNL_SIZE_DTYPE int64_t

class OneDnnConvUtil {
 protected:
  OpKernelContext* context_;  // We don't own this.
  TensorFormat data_format_;
  std::vector<int32_t> strides_;
  std::vector<int32_t> dilations_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  bool is_conv2d_;
  bool is_depthwise_;

 public:
  OneDnnConvUtil(OpKernelContext* context, TensorFormat data_format,
                 std::vector<int32_t> strides, std::vector<int32_t> dilations,
                 Padding padding, std::vector<int64_t> explicit_paddings,
                 bool is_conv2d, bool is_depthwise)
      : context_(context),
        data_format_(data_format),
        strides_(strides),
        dilations_(dilations),
        padding_(padding),
        explicit_paddings_(explicit_paddings),
        is_conv2d_(is_conv2d),
        is_depthwise_(is_depthwise) {}

  virtual ~OneDnnConvUtil() { context_ = nullptr; }

  inline void GetGeneralDimensionHelper(const std::vector<int32_t>& attrs,
                                        TensorFormat tensor_format,
                                        dnnl::memory::dims* attr_dims) {
    if (is_conv2d_) {
      int rows = GetTensorDim(attrs, tensor_format, 'H');
      int cols = GetTensorDim(attrs, tensor_format, 'W');
      *attr_dims = {rows, cols};
    } else {
      int planes = GetTensorDim(attrs, tensor_format, '0');
      int rows = GetTensorDim(attrs, tensor_format, '1');
      int cols = GetTensorDim(attrs, tensor_format, '2');
      *attr_dims = {planes, rows, cols};
    }
  }

  // Calculate Convolution strides
  virtual inline void GetStrideDimension(dnnl::memory::dims* strides_dims) {
    // For now we take the stride from the second and third dimensions only
    // (we do not support striding on the batch or depth dimension).
    OP_REQUIRES(context_, strides_dims != nullptr,
                errors::InvalidArgument("strides shoud not be nullptr."));
    GetGeneralDimensionHelper(strides_, data_format_, strides_dims);
  }

  // Calculate Convolution dilations
  virtual inline void GetDilationDimension(dnnl::memory::dims* dilations_dims) {
    // For now we take the dilation from the second and third dimensions only
    // (we do not support dilation on the batch or depth dimension).
    OP_REQUIRES(context_, dilations_dims != nullptr,
                errors::InvalidArgument("dilations shoud not be nullptr."));
    GetGeneralDimensionHelper(dilations_, data_format_, dilations_dims);
  }

  // Calculate Convolution input size in OneDNN order. OneDNN
  // requires input in NCHW/NCDHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetInputDimension(const TensorShape& input_shape,
                                        dnnl::memory::dims* input_dims) {
#define CHECK_BOUNDS(val, err_msg)                                     \
  do {                                                                 \
    OP_REQUIRES(context_,                                              \
                FastBoundsCheck(val, std::numeric_limits<int>::max()), \
                errors::InvalidArgument(err_msg));                     \
  } while (0)

    OP_REQUIRES(context_, input_dims != nullptr,
                errors::InvalidArgument("input_dims shoud not be nullptr."));

    // Input channel
    int64 input_depth_raw = GetTensorDim(input_shape, data_format_, 'C');
    int input_depth = static_cast<int>(input_depth_raw);

    // Input batch
    int64 input_batch_raw = GetTensorDim(input_shape, data_format_, 'N');
    CHECK_BOUNDS(input_batch_raw, "Input batch too large");
    int input_batch = static_cast<int>(input_batch_raw);

    if (is_conv2d_) {  // NCHW format for Conv2D
      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, 'H');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, 'W');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // OneDNN always requires input in NCHW format Conv2D.
      std::vector<DNNL_SIZE_DTYPE> input_dims_tmp(4, -1);
      input_dims_tmp[DimensionIndex::Dim_N] = input_batch;
      input_dims_tmp[DimensionIndex::Dim_C] = input_depth;
      input_dims_tmp[DimensionIndex::Dim_H] = input_rows;
      input_dims_tmp[DimensionIndex::Dim_W] = input_cols;

      *input_dims = input_dims_tmp;
    } else {  // NCDHW format for Conv3D
      // Input planes/third-dimension
      int64 input_planes_raw = GetTensorDim(input_shape, data_format_, '0');
      CHECK_BOUNDS(input_planes_raw, "Input depth too large");
      int input_planes = static_cast<int>(input_planes_raw);

      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, '1');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, '2');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // OneDNN always requires input in NCDHW format for Conv3D.
      std::vector<DNNL_SIZE_DTYPE> input_dims_tmp(5, -1);
      input_dims_tmp[DimensionIndex3D::Dim3d_N] = input_batch;
      input_dims_tmp[DimensionIndex3D::Dim3d_C] = input_depth;
      input_dims_tmp[DimensionIndex3D::Dim3d_D] = input_planes;
      input_dims_tmp[DimensionIndex3D::Dim3d_H] = input_rows;
      input_dims_tmp[DimensionIndex3D::Dim3d_W] = input_cols;

      *input_dims = input_dims_tmp;
    }
#undef CHECK_BOUNDS
  }

  // Calculate Convolution filter size in OneDNN order.
  // OneDNN requires filter in OIHW (Conv2D) or OIDHW (Conv3D) format.
  // Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetFilterDimension(const TensorShape& input_shape,
                                         const TensorShape& filter_shape,
                                         dnnl::memory::dims* filter_dims) {
    OP_REQUIRES(context_, filter_dims != nullptr,
                errors::InvalidArgument("filter_dims shoud not be nullptr."));
    OP_REQUIRES(
        context_, filter_shape.dims() == strides_.size(),
        errors::InvalidArgument(is_conv2d_ ? "filter must be 4-dimensional: "
                                           : "filter must be 5-dimensional: ",
                                filter_shape.DebugString()));

    for (int i = 0; i < strides_.size(); i++) {
      OP_REQUIRES(context_,
                  FastBoundsCheck(filter_shape.dim_size(i),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    int input_depth = GetTensorDim(input_shape, data_format_, 'C');

    if (is_conv2d_) {  // Conv2D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(2),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(2)));

      // TF filter is always in (rows, cols, in_depth, out_depth) order.
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_I));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_O));
      // OneDNN always needs filter in OIHW format for regular convolutions
      // and GOIHW for grouped/depthwise convolutions,
      // OIHW = (out_depth, in_depth, rows, cols)
      // GOIHW = (group, out_depth, in_depth, rows, cols)
      // Specifically for depthwise G=filter_indepth, O=filter_outdepth, I=1
      if (is_depthwise_) {
        std::vector<DNNL_SIZE_DTYPE> filter_dims_tmp(5, -1);
        filter_dims_tmp[FilterGroupDims::GROUP_FILTER_DIM_G] = filter_in_depth;
        filter_dims_tmp[FilterGroupDims::GROUP_FILTER_DIM_O] = filter_out_depth;
        filter_dims_tmp[FilterGroupDims::GROUP_FILTER_DIM_I] = 1;
        filter_dims_tmp[FilterGroupDims::GROUP_FILTER_DIM_H] = filter_rows;
        filter_dims_tmp[FilterGroupDims::GROUP_FILTER_DIM_W] = filter_cols;
        *filter_dims = filter_dims_tmp;
      } else {
        std::vector<DNNL_SIZE_DTYPE> filter_dims_tmp(4, -1);
        filter_dims_tmp[DimensionIndex::Dim_O] = filter_out_depth;
        filter_dims_tmp[DimensionIndex::Dim_I] = filter_in_depth;
        filter_dims_tmp[DimensionIndex::Dim_H] = filter_rows;
        filter_dims_tmp[DimensionIndex::Dim_W] = filter_cols;
        *filter_dims = filter_dims_tmp;
      }
    } else {  // Conv3D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(3),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(3)));

      // TF filter is always in (planes, rows, cols, in_depth, out_depth) order.
      int filter_planes =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_P));
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_I));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_O));

      // OneDNN always needs filter in OIDHW format.
      // OIDHW = (out_depth, in_depth, planes, rows, cols)
      std::vector<DNNL_SIZE_DTYPE> filter_dims_tmp(5, -1);
      filter_dims_tmp[DimensionIndex3D::Dim3d_O] = filter_out_depth;
      filter_dims_tmp[DimensionIndex3D::Dim3d_I] = filter_in_depth;
      filter_dims_tmp[DimensionIndex3D::Dim3d_D] = filter_planes;
      filter_dims_tmp[DimensionIndex3D::Dim3d_H] = filter_rows;
      filter_dims_tmp[DimensionIndex3D::Dim3d_W] = filter_cols;
      *filter_dims = filter_dims_tmp;
    }
  }

  // Calculate Bias size for 2D or 3D Convolution. Function does not
  // return anything, but may set an error in context status.
  virtual inline void GetBiasDimension(const TensorShape& bias_shape,
                                       dnnl::memory::dims* bias_dims) {
    if (bias_shape.dims() > 1) {
      // Make sure all the dims except channel(last) is 1
      for (int i = 0; i < bias_shape.dims() - 1; i++) {
        OP_REQUIRES(
            context_, bias_shape.dim_size(i) == 1,
            errors::InvalidArgument("For bias_dims > 1, all except the last "
                                    "dimension (channel) must be 1: ",
                                    bias_shape.DebugString()));
      }
      *bias_dims = {
          static_cast<int>(bias_shape.dim_size(bias_shape.dims() - 1))};
    } else {
      *bias_dims = {static_cast<int>(bias_shape.dim_size(0))};
    }
  }

  // Function to calculate output and padding size for 2D/3D convolution.
  //
  // Calculate output shape of Convolution in OneDNN and TensorFlow order.
  // OneDNN uses NCHW(Conv2D) or NCDHW(Conv3D) for output order.
  // But TensorFlow output will be in NHWC||NCHW(Conv2D) or
  // NDHWC||NCDHW(Conv3D) format depending on data format.
  // Function also calculates left, right, top and bottom pads.
  // Function does not return any status which is set with context status.
  virtual inline void GetOutputAndPadDimension(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      const dnnl::memory::dims& strides, const dnnl::memory::dims& dilations,
      dnnl::memory::dims* output_dims_tf_order,
      dnnl::memory::dims* output_dims_onednn, dnnl::memory::dims* pad_left_dims,
      dnnl::memory::dims* pad_right_dims) {
    OP_REQUIRES(
        context_, output_dims_tf_order != nullptr,
        errors::InvalidArgument("output_dims_tf_order shoud not be nullptr."));
    OP_REQUIRES(
        context_, output_dims_onednn != nullptr,
        errors::InvalidArgument("output_dims_onednn shoud not be nullptr."));
    OP_REQUIRES(context_, pad_left_dims != nullptr,
                errors::InvalidArgument("pad_left_dims shoud not be nullptr."));
    OP_REQUIRES(
        context_, pad_right_dims != nullptr,
        errors::InvalidArgument("pad_right_dims shoud not be nullptr."));

    int input_planes = 0, input_rows = 0, input_cols = 0;
    if (is_conv2d_) {
      input_rows = GetTensorDim(input_shape, data_format_, 'H');
      input_cols = GetTensorDim(input_shape, data_format_, 'W');
    } else {
      input_planes = GetTensorDim(input_shape, data_format_, '0');
      input_rows = GetTensorDim(input_shape, data_format_, '1');
      input_cols = GetTensorDim(input_shape, data_format_, '2');
    }

    // Filter dimension
    // Conv2D:
    //    First dimension: rows/height.
    //    Second dimension: cols/width.
    // Conv3D:
    //    First dimension: planes/depth.
    //    Second dimension: rows/height.
    //    Third dimension: cols/width.

    int filter_planes = 0, filter_rows = 0, filter_cols = 0;
    if (is_conv2d_) {
      filter_rows = filter_shape.dim_size(TF_2DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_2DFILTER_DIM_W);
    } else {
      filter_planes = filter_shape.dim_size(TF_3DFILTER_DIM_P);
      filter_rows = filter_shape.dim_size(TF_3DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_3DFILTER_DIM_W);
    }

    int stride_planes = 0, stride_rows = 0, stride_cols = 0;
    int dilation_planes = 0, dilation_rows = 0, dilation_cols = 0;
    if (is_conv2d_) {
      // Conv2D stride is a vector of 2 elements: {s_r, s_c}
      stride_rows = strides[0];
      stride_cols = strides[1];
      dilation_rows = dilations[0];
      dilation_cols = dilations[1];
    } else {
      // Conv3D stride is a vector of 3 elements: {s_d, s_r, s_c}
      stride_planes = strides[0];
      stride_rows = strides[1];
      stride_cols = strides[2];
      dilation_planes = dilations[0];
      dilation_rows = dilations[1];
      dilation_cols = dilations[2];
    }

    // Output batch is same as input batch.
    int out_batch = GetTensorDim(input_shape, data_format_, 'N');
    int out_depth;

    // Output depth is same as last dimension for filters for regular
    // convolutions. For depthwise it is in_depth * channel_multiplier.
    // The channel_multiplier is the last dimension of TF filter for
    // depthwise convolutions.
    if (is_depthwise_) {
      out_depth = (filter_shape.dim_size(TF_2DFILTER_DIM_I) *
                   filter_shape.dim_size(TF_2DFILTER_DIM_O));
    } else {
      out_depth = filter_shape.dim_size(
          is_conv2d_ ? static_cast<int>(TF_2DFILTER_DIM_O)
                     : static_cast<int>(TF_3DFILTER_DIM_O));
    }

    int64 out_rows = 0, out_cols = 0, out_planes = 0;
    int64 pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    int64 pad_D1 = 0, pad_D2 = 0;

    // If the pad is fused, the pad type should be explicit to skip the
    // GetWindowedOutputSizeVerboseV2 to overwrite.
    Padding padding_type =
        explicit_paddings_.size() != 0 ? Padding::EXPLICIT : padding_;

    // Get padding value, there're 3 scenarios of padding:
    // 1. Pad + Conv fusion: The padding type must be `VALID` and its padding
    //    value will be stored in `explicit_paddings_` when call
    //    `InitPadWithFusion()`
    // 2. Conv with EXPLICIT: The padding value is stored in
    //    `explicit_paddings_` when initialized op
    // 3. INT8 Conv with VALID and explicit list: INT8 op may have VALID type
    //    and explicit padding value because of legacy issue, the padding value
    //    is stored in `explicit_paddings_` when call `Compute()`
    if (padding_type == Padding::EXPLICIT) {
      GetExplicitPaddingForDim(explicit_paddings_, data_format_, 'H', &pad_top,
                               &pad_bottom);
      GetExplicitPaddingForDim(explicit_paddings_, data_format_, 'W', &pad_left,
                               &pad_right);

      if (!is_conv2d_) {
        // '0' is for depth of Conv3D.
        GetExplicitPaddingForDim(explicit_paddings_, data_format_, '0', &pad_D1,
                                 &pad_D2);
      }
    }

    // Use padding value to initialize output info and dims.
    OP_REQUIRES_OK(context_,
                   GetWindowedOutputSizeVerboseV2(
                       input_rows, filter_rows, dilation_rows, stride_rows,
                       padding_type, &out_rows, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context_,
                   GetWindowedOutputSizeVerboseV2(
                       input_cols, filter_cols, dilation_cols, stride_cols,
                       padding_type, &out_cols, &pad_left, &pad_right));

    if (!is_conv2d_) {
      OP_REQUIRES_OK(context_, GetWindowedOutputSizeVerboseV2(
                                   input_planes, filter_planes, dilation_planes,
                                   stride_planes, padding_type, &out_planes,
                                   &pad_D1, &pad_D2));

      *pad_left_dims = {static_cast<int>(pad_D1), static_cast<int>(pad_top),
                        static_cast<int>(pad_left)};
      *pad_right_dims = {static_cast<int>(pad_D2), static_cast<int>(pad_bottom),
                         static_cast<int>(pad_right)};
    } else {
      *pad_left_dims = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
      *pad_right_dims = {static_cast<int>(pad_bottom),
                         static_cast<int>(pad_right)};
    }

    // Tensorflow output is in data_format order.
    //     Conv2D: NHWC or NCHW
    //     Conv3D: NDHWC or NCDHW
    // OneDNN uses asymmetric padding.
    TensorShape out_shape =
        is_conv2d_
            ? ShapeFromFormat(data_format_, out_batch, out_rows, out_cols,
                              out_depth)
            : ShapeFromFormat(data_format_, out_batch,
                              {{out_planes, out_rows, out_cols}}, out_depth);
    *output_dims_tf_order = TFShapeToOneDnnDims(out_shape);

    if (is_conv2d_) {
      // For Conv2D, OneDNN always needs output in NCHW format.
      std::vector<DNNL_SIZE_DTYPE> output_dims_onednn_tmp(4, -1);
      output_dims_onednn_tmp[DimensionIndex::Dim_N] = out_batch;
      output_dims_onednn_tmp[DimensionIndex::Dim_C] = out_depth;
      output_dims_onednn_tmp[DimensionIndex::Dim_H] =
          static_cast<int>(out_rows);
      output_dims_onednn_tmp[DimensionIndex::Dim_W] =
          static_cast<int>(out_cols);
      *output_dims_onednn = output_dims_onednn_tmp;
    } else {
      std::vector<DNNL_SIZE_DTYPE> output_dims_onednn_tmp(5, -1);
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_N] = out_batch;
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_C] = out_depth;
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_D] =
          static_cast<int>(out_planes);
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_H] =
          static_cast<int>(out_rows);
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_W] =
          static_cast<int>(out_cols);
      *output_dims_onednn = output_dims_onednn_tmp;
    }
  }

  // Wrapper function to calculate input, filter, and output sizes of
  // Conv2D/Conv3D in OneDNN order:
  //     Conv2D: NCHW for input and output; OIHW for filter.
  //     Conv3D: NCDHW for input and output; OIDHW for filter.
  // Function also calculates output shape in Tensorflow order.
  // Additionally, it also calculates strides and paddings.
  //
  // Function does not return anything, but sets error in context status.
  inline void InitFwdDimensions(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      dnnl::memory::dims* input_dims, dnnl::memory::dims* filter_dims,
      dnnl::memory::dims* strides, dnnl::memory::dims* dilations,
      dnnl::memory::dims* output_dims_tf_order,
      dnnl::memory::dims* output_dims_onednn, dnnl::memory::dims* pad_left_dims,
      dnnl::memory::dims* pad_right_dims) {
    GetInputDimension(input_shape, input_dims);
    GetFilterDimension(input_shape, filter_shape, filter_dims);
    GetStrideDimension(strides);
    GetDilationDimension(dilations);
    GetOutputAndPadDimension(input_shape, filter_shape, *strides, *dilations,
                             output_dims_tf_order, output_dims_onednn,
                             pad_left_dims, pad_right_dims);
  }

  // Helper function for Pad fusion. It will set fused Pad tensor to
  // internal `explicit_paddings` list.
  // @ input kPadIndex the index of Pad/Slice input
  // @ input is_forward indicate whether this fusion is forward op
  // @ return None
  void InitPadWithFusion(const int kPadIndex, bool is_forward) {
    const Tensor& pad_tensor = context_->input(kPadIndex);
    OP_REQUIRES(context_, explicit_paddings_.size() == 0,
                errors::InvalidArgument("explicit padding size must be 0 ",
                                        "when fuse with Pad"));

    // Flatten tensor to get individual paddings.
    int32_t* paddings =
        static_cast<int32_t*>(GetTensorBuffer<int32_t>(&pad_tensor));

    const int kSize = is_conv2d_ ? 8 : 10;
    explicit_paddings_.resize(kSize, 0);

    if (is_forward) {
      // If the data format is NHWC, indices 0, 1, 6 and 7 of paddings(_tf)
      // will be zero.
      // Example:
      // paddings_tf = [ [0, 0] [1, 2] [3, 4] [0, 0] ],
      // flat method = row-major, then:
      // paddings = {0, 0, 1, 2, 3, 4, 0, 0}.
      // Hence, the values are: top = 1, bottom = 2, left = 3, right = 4.
      //
      // Similarly, if the data format is NCHW, indices 0, 1, 2 and 3 of
      // paddings(_tf) will be zero.
      // i.e. for the above example, paddings = {0, 0, 0, 0, 1, 2, 3, 4}.
      //
      // For 5-D, the input's format is NDHWC or NCDHW. The padding format rule
      // is the same with 2-D when channel order is changed. E,g.
      //   1. NDHWC: {0, 0, D1, D2, HT, HB, WL, WR, 0, 0}
      //   2. NCDHW: {0, 0, 0, 0, D1, D2, HT, HB, WL, WR}
      OP_REQUIRES(context_, pad_tensor.dims() == 2,
                  errors::InvalidArgument("paddings must be 2-dimensional: ",
                                          pad_tensor.shape().DebugString()));
      OP_REQUIRES(
          context_, pad_tensor.NumElements() == kSize,
          errors::InvalidArgument("Pad size is not correct, expected ", kSize,
                                  "but got ", pad_tensor.NumElements()));

      for (int i = 0; i < kSize; ++i) {
        explicit_paddings_[i] = paddings[i];
      }
    } else {
      // The begin tensor is different with pad tensor of forwarding. It only
      // contains left side, we need to calculate the right side by sizes. For
      // example, if it's a 4-D tensor, [0, 2, 2, 0] with NHWC, means that
      //    + Padding Top is 2,
      //    + Padding Left is 2.
      // A valid padding will be [[0, 0], [2, x], [2, y], [0, 0]].
      // `x` and `y` is calculated by begin and sizes tensors.
      // The same with 5-D.
      OP_REQUIRES(
          context_, pad_tensor.NumElements() * 2 == kSize,
          errors::InvalidArgument("Slice size is not correct, expected ", kSize,
                                  "but got ", pad_tensor.NumElements() * 2));

      auto input_sizes = static_cast<int32*>(context_->input(0).data());
      auto slice_output_size =
          static_cast<int32*>(context_->input(kPadIndex + 1).data());
      // Start index in Slice `paddings` of 1st valid dim.
      const int kStart = (this->data_format_ == FORMAT_NHWC) ? 1 : 2;
      // Valid dim length in Slice `paddings`.
      const int kPaddingLength = is_conv2d_ ? 2 : 3;

      for (int i = kStart; i < kStart + kPaddingLength; ++i) {
        explicit_paddings_[i * 2] = paddings[i];
        explicit_paddings_[i * 2 + 1] =
            input_sizes[i] - slice_output_size[i] - paddings[i];
      }
    }
  }
};

template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Tsummand, bool pad_enabled = false,
          bool is_depthwise = false>
class ConvOpBase : public OpKernel {
 public:
  explicit ConvOpBase(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    is_conv2d_ = (strides_.size() == 4);

    string data_format_string;
    // TF raw op Conv INT8 old API, doesn't have data_format attribute
    if (context->HasAttr("data_format")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("data_format", &data_format_string));
    } else {
      data_format_string = "NHWC";
    }

    OP_REQUIRES(context, FormatFromString(data_format_string, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::Unimplemented("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));

    const int64 dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    const int64 dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));

    if (is_conv2d_) {
      OP_REQUIRES(context, dilations_.size() == 4,
                  errors::InvalidArgument("Sliding window dilations field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES(context, strides_.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 4 dimensions"));
      const int64 stride_h = GetTensorDim(strides_, data_format_, 'H');
      const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
      OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                  errors::InvalidArgument(
                      "Row and column strides should be larger than 0."));

      const int64 dilation_h = GetTensorDim(dilations_, data_format_, 'H');
      const int64 dilation_w = GetTensorDim(dilations_, data_format_, 'W');
      OP_REQUIRES(
          context, dilation_h > 0 && dilation_w > 0,
          errors::InvalidArgument("Dilated rates should be larger than 0."));
    } else {
      OP_REQUIRES(context, strides_.size() == 5,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 5 dimensions"));
      OP_REQUIRES(context, dilations_.size() == 5,
                  errors::InvalidArgument("Dilation rates field must "
                                          "specify 5 dimensions"));
      OP_REQUIRES(
          context,
          (GetTensorDim(strides_, data_format_, '0') > 0 &&
           GetTensorDim(strides_, data_format_, '1') > 0 &&
           GetTensorDim(strides_, data_format_, '2') > 0),
          errors::InvalidArgument("Spatial strides should be larger than 0."));
      OP_REQUIRES(
          context,
          (GetTensorDim(dilations_, data_format_, '0') > 0 &&
           GetTensorDim(dilations_, data_format_, '1') > 0 &&
           GetTensorDim(dilations_, data_format_, '2') > 0),
          errors::InvalidArgument("Dilated rates should be larger than 0."));
    }

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

    // Pad fusion check.
    if (pad_enabled) {
      OP_REQUIRES(
          context, padding_ == Padding::VALID,
          errors::InvalidArgument("Pad can only be fused with `VALID` Conv."));
    }

    // Add fusion check.
    if (context->HasAttr("inplace_sum")) {
      OP_REQUIRES_OK(context, context->GetAttr("inplace_sum", &inplace_sum_));
    }

    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));
    fp32_math_mode_ = GetFP32MathMode<Device>();
  }

  void InitOrSetMemory(OpKernelContext* context) {
    if (!(enable_cache_ && is_init_ && context->is_input_same(0, input_dims_) &&
          context->is_input_same(1, filter_dims_) && !is_format_reordered_)) {
      Init(context);
      return;
    }

    if (is_input_zero_) {
      OP_REQUIRES_OK(context, context->allocate_output(
                                  kDstIndex_, dst_tensor_shape_, &dst_tensor_));
      return;
    }

    src_mem_opt_.set_data_handle(context->tensor_data(kSrcIndex_));

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
      // TODO(itex): avoid to use context->input, which may call C API
      // more than once and cause overhead.
      const Tensor& bias_tensor = context->input(kBiasIndex_);
      // GetBiasHandle is needed for INT8 kernels, where bias scaling is
      // required.
      Tbias* bias_data = this->GetBiasHandle(context, bias_tensor);
      bias_mem_.set_data_handle(bias_data);
    }

    // Reallocate scratchpad memory.
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                          TensorShape({scratchpad_size_}),
                                          scratchpad_tensor_.get()));
    scratchpad_mem_.set_data_handle(
        GetTensorBuffer<Tinput>(scratchpad_tensor_.get()));

    Tensor dst_tensor_opt;
    AllocateOutputTensor(context, fwd_pd_, dst_dims_onednn_, dst_tensor_shape_,
                         &dst_tensor_, &dst_tensor_opt);

    // Set dst mem if output need reorder.
    // Here is trick to calculate INT8 conv + bias + add + relu, where
    // Tsummand is s8, and Toutput is u8
    dst_mem_opt_.set_data_handle(
        reinterpret_cast<Tsummand*>(GetTensorBuffer<Toutput>(dst_tensor_)));
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);
    onednn_engine_ = CreateDnnlEngine<Device>(*context);
    // onednn_stream has thread safety issue, need create a new one in
    // every compute.
    onednn_stream_ = CreateDnnlStream(*context, onednn_engine_);
    scratchpad_tensor_ = std::make_shared<Tensor>();
    InitOrSetMemory(context);

    // Skip primitive execution if the calculation is meaningless.
    if (is_filter_zero_ || is_input_zero_) {
      scratchpad_tensor_.reset();
      return;
    }

    if (!is_format_reordered_) {
      fwd_primitive_.execute(onednn_stream_, fwd_primitives_args_);
    }
    scratchpad_tensor_.reset();
  }

  void Init(OpKernelContext* context) {
    try {
      fwd_primitives_args_.clear();

      const Tensor& src_tensor = context->input(kSrcIndex_);
      const Tensor& filter_tensor = context->input(kFilterIndex_);

      // Corner cases: filter with 0 elements
      if (filter_tensor.NumElements() == 0) {
        context->CtxFailure(
            errors::InvalidArgument("filter must not have zero elements "
                                    "(i.e. all dimensions must be non-zero)"));
        is_filter_zero_ = true;
        return;
      }

      TensorShape src_tensor_shape = src_tensor.shape();
      TensorShape filter_tensor_shape = filter_tensor.shape();

      input_dims_.clear();
      for (int i = 0; i < src_tensor_shape.dims(); ++i) {
        input_dims_.push_back(src_tensor_shape.dim_size(i));
      }
      filter_dims_.clear();
      for (int i = 0; i < filter_tensor_shape.dims(); ++i) {
        filter_dims_.push_back(filter_tensor_shape.dim_size(i));
      }

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

      conv_util.InitFwdDimensions(
          src_tensor_shape, filter_tensor_shape, &src_dims, &filter_dims,
          &stride_dims, &dilation_dims, &dst_dims_tf, &dst_dims_onednn_,
          &pad_left_dims, &pad_right_dims);

      // OneDNN dilations start from 0.
      for (int i = 0; i < dilation_dims.size(); ++i) {
        --dilation_dims[i];
      }

      // output tensor shape.
      dst_tensor_shape_ = OneDnnDimsToTFShape(dst_dims_tf);

      // Corner cases: output with 0 elements and 0 batch size.
      if (dst_tensor_shape_.num_elements() == 0 || dst_dims_tf[0] == 0) {
        is_input_zero_ = true;
        OP_REQUIRES_OK(context,
                       context->allocate_output(kDstIndex_, dst_tensor_shape_,
                                                &dst_tensor_));
        is_init_ = true;
        return;
      }

      data_format_onednn_ =
          TFDataFormatToOneDnnDataFormat(data_format_, is_conv2d_);
      memory::format_tag data_layout =
          OneDnnTensorFormatToTag(data_format_onednn_);
      memory::desc src_md =
          memory::desc({src_dims}, OneDnnType<Tinput>(), data_layout);
      auto filter_format = is_conv2d_
                               ? (is_depthwise ? memory::format_tag::hwigo
                                               : memory::format_tag::hwio)
                               : memory::format_tag::dhwio;
      memory::desc filter_md =
          memory::desc({filter_dims}, OneDnnType<Tfilter>(), filter_format);
      memory::desc filter_md_prefer = memory::desc(
          {filter_dims}, OneDnnType<Tfilter>(), memory::format_tag::any);
      dst_md_ =
          memory::desc({dst_dims_onednn_}, OneDnnType<Tbias>(), data_layout);
      // the convolution primitive is optimized for NHWC
      memory::format_tag tag_opt =
          is_conv2d_ ? memory::format_tag::nhwc : memory::format_tag::ndhwc;
      memory::desc src_md_opt =
          memory::desc({src_dims}, OneDnnType<Tinput>(), tag_opt);
      memory::desc dst_md_opt;

      // TODO(itex): redesign the code for Tsummand
      // The reason for using Tsummand is to deal with the situation for int8
      // fusion conv + bias + add + relu fusion. Two inputs for add op may be
      // respectively quint8 and qint8.

      dst_md_ =
          memory::desc({dst_dims_onednn_}, OneDnnType<Tsummand>(), data_layout);
      dst_md_opt =
          memory::desc({dst_dims_onednn_}, OneDnnType<Tsummand>(), tag_opt);

      this->ExtendInt8PostOps(context);

      ConvFwdDesc fwd_desc =
          ConvFwdDesc(prop_kind::forward, dnnl::algorithm::convolution_direct,
                      src_md_opt, filter_md_prefer, dst_md_opt, stride_dims,
                      dilation_dims, pad_left_dims, pad_right_dims);

      if (post_op_util_.HasBias()) {
        const Tensor& bias_tensor = context->input(kBiasIndex_);
        TensorShape bias_tensor_shape = bias_tensor.shape();
        conv_util.GetBiasDimension(bias_tensor_shape, &bias_dims);
        auto bias_md =
            memory::desc(bias_dims, OneDnnType<Tbias>(), memory::format_tag::x);
        // GetBiasHandle is needed for INT8 kernels, where bias scaling is
        // required.
        Tbias* bias_data = this->GetBiasHandle(context, bias_tensor);
        bias_mem_ = CreateDnnlMemory(bias_md, onednn_engine_, bias_data);

        fwd_primitives_args_.insert({DNNL_ARG_BIAS, bias_mem_});
        fwd_desc = ConvFwdDesc(
            prop_kind::forward, dnnl::algorithm::convolution_direct, src_md_opt,
            filter_md_prefer, bias_md, dst_md_opt, stride_dims, dilation_dims,
            pad_left_dims, pad_right_dims);
      }

      // Set post op attribution.
      dnnl::primitive_attr post_ops_attr;
      post_op_util_.SetPostOpAttr(&post_ops_attr);
      post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<Tinput, float>::value) {
        post_ops_attr.set_fpmath_mode(fp32_math_mode_);
      }
      fwd_pd_ = ConvFwdPd(fwd_desc, post_ops_attr, onednn_engine_);

      // keep tensor out of if block to avoid of being deallocated
      is_format_reordered_ = data_layout != tag_opt;

      // This one for dnnl primitive output when output need reorder.
      Tensor dst_tensor_opt;
      // This one for dnnl primitive input when input need reorder.
      Tensor src_tensor_opt;

      dnnl::reorder src_reorder;
      dnnl::reorder dst_reorder;
      std::unordered_map<int, memory> src_reorder_args;
      std::unordered_map<int, memory> dst_reorder_args;

      if (is_format_reordered_) {
        // allocate dst memory for reorder back later
        int64 dst_nums = fwd_pd_.dst_desc().get_size() / sizeof(Tsummand);
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<Tsummand>::v(),
                                    TensorShape({dst_nums}), &dst_tensor_opt));
      }

      AllocateOutputTensor(context, fwd_pd_, dst_dims_onednn_,
                           dst_tensor_shape_, &dst_tensor_, &dst_tensor_opt);
      scratchpad_size_ = fwd_pd_.scratchpad_desc().get_size() / sizeof(Tinput);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                            TensorShape({scratchpad_size_}),
                                            scratchpad_tensor_.get()));
      scratchpad_mem_ =
          dnnl::memory(fwd_pd_.scratchpad_desc(), onednn_engine_,
                       GetTensorBuffer<Tinput>(scratchpad_tensor_.get()));

      fwd_primitive_ = convolution_forward(fwd_pd_);

      src_mem_ = CreateDnnlMemory(src_md, onednn_engine_,
                                  GetTensorBuffer<Tinput>(&src_tensor));
      dst_mem_ = CreateDnnlMemory(
          dst_md_, onednn_engine_,
          reinterpret_cast<Tsummand*>(GetTensorBuffer<Toutput>(dst_tensor_)));
      // reorder src/dst to NHWC if needed
      src_mem_opt_ = src_mem_;
      dst_mem_opt_ = dst_mem_;

      if (is_format_reordered_) {
        int64 src_nums = fwd_pd_.src_desc().get_size() / sizeof(Tinput);
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<Tinput>::v(),
                                    TensorShape({src_nums}), &src_tensor_opt));
        src_mem_opt_ =
            CreateDnnlMemory(src_md_opt, onednn_engine_,
                             GetTensorBuffer<Tinput>(&src_tensor_opt));

        src_reorder_args.insert({DNNL_ARG_SRC, src_mem_});
        src_reorder_args.insert({DNNL_ARG_DST, src_mem_opt_});
        src_reorder = dnnl::reorder(src_mem_, src_mem_opt_);
        src_reorder.execute(onednn_stream_, src_reorder_args);

        dst_mem_opt_ =
            CreateDnnlMemory(dst_md_opt, onednn_engine_,
                             GetTensorBuffer<Tsummand>(&dst_tensor_opt));

        dst_reorder_args.insert({DNNL_ARG_SRC, dst_mem_opt_});
        dst_reorder_args.insert({DNNL_ARG_DST, dst_mem_});
        dst_reorder = dnnl::reorder(dst_mem_opt_, dst_mem_);
      }

      // Check filter reorder and do cache if filter is const.
      filter_mem_input_ = CreateDnnlMemory(
          filter_md, onednn_engine_, GetTensorBuffer<Tfilter>(&filter_tensor));
      filter_md_prefer = fwd_pd_.weights_desc();
      is_filter_reordered_ = (filter_md_prefer != filter_md);
      if (is_filter_reordered_) {
        Tfilter* filter_cached_data = nullptr;
        if (is_filter_const_) {
          if (weight_cache_manager_.IsEmpty()) {
            weight_cache_manager_.SetCache(
                context, filter_md, filter_md_prefer,
                GetTensorBuffer<Tfilter>(&filter_tensor), onednn_engine_);
          }
          filter_cached_data =
              weight_cache_manager_.GetCache(context, filter_md_prefer);
          if (filter_cached_data != nullptr) {
            filter_mem_ = CreateDnnlMemory(filter_md_prefer, onednn_engine_,
                                           filter_cached_data);
          }
        }
        if (filter_cached_data == nullptr) {
          // allocate temporay tensor for reordering filter
          int64_t reorder_filter_data_size =
              fwd_pd_.weights_desc().get_size() / sizeof(Tfilter);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DataTypeToEnum<Tfilter>::v(),
                                      TensorShape({reorder_filter_data_size}),
                                      &tmp_weight_));
          void* filter_data_handle = GetTensorBuffer<Tfilter>(&tmp_weight_);
          filter_mem_ = CreateDnnlMemory(filter_md_prefer, onednn_engine_,
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

      // Execute convolution
      fwd_primitives_args_.insert({DNNL_ARG_SRC, src_mem_opt_});
      fwd_primitives_args_.insert({DNNL_ARG_WEIGHTS, filter_mem_});
      fwd_primitives_args_.insert({DNNL_ARG_DST, dst_mem_opt_});
      fwd_primitives_args_.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem_});

      // reorder back if needed
      if (is_format_reordered_) {
        fwd_primitive_.execute(onednn_stream_, fwd_primitives_args_);
        dst_reorder.execute(onednn_stream_, dst_reorder_args);
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
    return;
  }

 private:
  TensorFormat data_format_;
  std::vector<int32_t> strides_;
  std::vector<int32_t> dilations_;
  Padding padding_;

  bool inplace_sum_ = false;
  bool is_filter_const_ = false;

  // Weight cache manager
  WeightCacheManager<Tfilter> weight_cache_manager_;

  mutex mu_compute_;

 protected:
  std::vector<int64_t> explicit_paddings_;
  bool is_conv2d_;
  const int kSrcIndex_ = 0, kFilterIndex_ = 1, kBiasIndex_ = 2, kAddIndex_ = 3;
  const int kDstIndex_ = 0;
  PostOpUtil post_op_util_;

  // Cache oneDNN object and TF memory
  bool is_init_ = false;
  bool is_input_zero_ = false;
  bool is_filter_zero_ = false;
  bool is_format_reordered_ = false;
  bool is_filter_reordered_ = false;

  // This one for TF input when input need reorder.
  dnnl::memory src_mem_;
  // This one for dnnl primitive input
  dnnl::memory src_mem_opt_;
  // This one for TF output when output need reorder.
  dnnl::memory dst_mem_;
  // This one for dnnl primitive output
  dnnl::memory dst_mem_opt_;
  // This one for dnnl primitive weight
  dnnl::memory filter_mem_;
  // This one for TF weight when weight need reorder.
  dnnl::memory filter_mem_input_;
  dnnl::memory scratchpad_mem_;
  dnnl::memory bias_mem_;
  dnnl::memory::dims dst_dims_onednn_;

  memory::desc dst_md_;

  dnnl::stream onednn_stream_;
  dnnl::engine onednn_engine_;

  dnnl::reorder weight_reorder_;
  primitive fwd_primitive_;
  ConvFwdPd fwd_pd_;

  std::unordered_map<int, memory> fwd_primitives_args_;
  std::unordered_map<int, memory> weight_reorder_args_;

  TensorShape dst_tensor_shape_;
  OneDnnTensorFormat data_format_onednn_;
  std::vector<int64> input_dims_, filter_dims_;

  Tensor* dst_tensor_ = nullptr;
  // This one for dnnl primitive weight when weight need reorder.
  Tensor tmp_weight_;
  std::shared_ptr<Tensor> scratchpad_tensor_;
  int64_t scratchpad_size_ = 0;

  bool enable_cache_ = false;
  dnnl::fpmath_mode fp32_math_mode_ = dnnl::fpmath_mode::strict;

  // ExtendInt8PostOps is only used in Int8 ops.
  virtual void ExtendInt8PostOps(OpKernelContext* context) {}

  virtual void AllocateOutputTensor(
      OpKernelContext* context,
      const dnnl::convolution_forward::primitive_desc& conv_pd,
      const memory::dims& dst_dims_onednn, TensorShape dst_tensor_shape,
      Tensor** dst_tensor, Tensor* dst_tensor_opt) {
    ITEX_DCHECK(dst_tensor);
    auto dst_md_opt = conv_pd.dst_desc();

    // Handle INT8 fusion, where Tsummand s8 and Toutput u8
    if (!std::is_same<Tsummand, Toutput>::value) {
      dst_md_opt.data.data_type = memory::convert_to_c(OneDnnType<Toutput>());
    }

    if (post_op_util_.HasAdd() &&
        (std::is_same<Toutput, float>::value ||
         std::is_same<Toutput, Eigen::half>::value ||
         std::is_same<Toutput, Eigen::bfloat16>::value)) {
      // FP32/BF16/FP16 condition
      const Tensor& add_tensor = context->input(this->kAddIndex_);
      const int kUnsuccess = -1;
      int is_forward_success = kUnsuccess;

      // Try to do in-place.
      // TODO(itex): Remove this workaround when inplace works.
      if (!this->is_format_reordered_) {
        if (inplace_sum_) {
          context->set_output(this->kDstIndex_, add_tensor);
          dst_tensor_ = context->mutable_output(this->kDstIndex_);
          is_forward_success = this->kAddIndex_;
        } else {
          OP_REQUIRES_OK(context,
                         context->forward_input_or_allocate_output(
                             {this->kAddIndex_}, kDstIndex_, dst_tensor_shape,
                             dst_tensor, &is_forward_success));
        }
      } else {
        OP_REQUIRES_OK(context,
                       context->allocate_output(this->kDstIndex_,
                                                dst_tensor_shape, dst_tensor));
      }

      // Reorder is needed, forward is failed but dst has been allocated;
      if (is_forward_success == kUnsuccess) {
        // In-place do not success, need reorder.
        auto fuse_add_src =
            CreateDnnlMemory(this->dst_md_, this->onednn_engine_,
                             GetTensorBuffer<Toutput>(&add_tensor));
        auto fuse_add_dst =
            CreateDnnlMemory(dst_md_opt, this->onednn_engine_,
                             GetTensorBuffer<Toutput>(*dst_tensor));

        if (this->is_format_reordered_) {
          fuse_add_dst =
              CreateDnnlMemory(dst_md_opt, this->onednn_engine_,
                               GetTensorBuffer<Tsummand>(dst_tensor_opt));
        }
        ReorderMemory(*context, &fuse_add_src, &fuse_add_dst,
                      this->onednn_engine_);
      }
    } else {
      OP_REQUIRES(
          context,
          (!post_op_util_.HasAdd() ||
           (post_op_util_.HasAdd() && (std::is_same<Toutput, qint8>::value ||
                                       std::is_same<Toutput, quint8>::value ||
                                       std::is_same<Toutput, qint32>::value))),
          errors::InvalidArgument("ConvOp: Invalid data type in AddN fusion."));

      OP_REQUIRES_OK(
          context, context->allocate_output(this->kDstIndex_, dst_tensor_shape,
                                            dst_tensor));
    }

    return;
  }

  virtual Tbias* GetBiasHandle(OpKernelContext* context,
                               const Tensor& bias_tensor) {
    return static_cast<Tbias*>(
        const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ConvOpBase);
};

template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Tsummand, bool pad_enabled = false,
          bool is_depthwise = false>
class FusedConvOp : public ConvOpBase<Device, Tinput, Tfilter, Tbias, Toutput,
                                      Tsummand, pad_enabled, is_depthwise> {
 public:
  explicit FusedConvOp(OpKernelConstruction* context)
      : ConvOpBase<Device, Tinput, Tfilter, Tbias, Toutput, Tsummand,
                   pad_enabled, is_depthwise>(context) {
    int num_args;
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    OP_REQUIRES(
        context, !(fused_ops.empty()),
        errors::InvalidArgument("Fused Conv must have at least one fused op."));
    OP_REQUIRES(
        context, this->post_op_util_.AddOps(fused_ops),
        errors::InvalidArgument("Found unsupported fusion in Fused Conv2D."));

    // Set alpha if get `LeakyRelu` after adding ops.
    if (this->post_op_util_.HasLeakyRelu()) {
      float alpha;
      OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha", &alpha));
      this->post_op_util_.SetLeakyReluAlpha(alpha);
    }
  }
  TF_DISALLOW_COPY_AND_ASSIGN(FusedConvOp);
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_CONV_OPS_H_
