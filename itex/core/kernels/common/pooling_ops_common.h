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

#ifndef ITEX_CORE_KERNELS_COMMON_POOLING_OPS_COMMON_H_
#define ITEX_CORE_KERNELS_COMMON_POOLING_OPS_COMMON_H_

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/common_shape_fns.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/padding.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/types.h"

namespace itex {

using dnnl::memory;
using dnnl::prop_kind;

// TODO(itex): If OneDnn unifies the data types of workspace on CPU and GPU,
// the following parts can be deleted.
#ifndef INTEL_CPU_ONLY
typedef typename EnumToDataType<DT_UINT8>::Type Tws;
#else
typedef typename EnumToDataType<DT_INT32>::Type Tws;
#endif

struct OneDnnPoolParameters {
  int depth;

  int tensor_in_planes;  // Pool3D
  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_planes;  // Pool3D
  int window_rows;
  int window_cols;
  int depth_window;

  int planes_stride;  // Pool3D
  int row_stride;
  int col_stride;
  int depth_stride;

  int64 out_planes;  // Pool3D
  int64 out_height;
  int64 out_width;
  int out_depth;

  int64 pad_P1;  // Pool3D
  int64 pad_P2;  // Pool3D
  int64 pad_left;
  int64 pad_right;
  int64 pad_top;
  int64 pad_bottom;
  int pad_depth;

  TensorFormat data_format;
  OneDnnPoolParameters()
      : depth(0),
        tensor_in_planes(0),
        tensor_in_cols(0),
        tensor_in_rows(0),
        tensor_in_batch(0),
        window_planes(0),
        window_rows(0),
        window_cols(0),
        depth_window(0),
        planes_stride(0),
        row_stride(0),
        col_stride(0),
        depth_stride(0),
        out_planes(0),
        out_height(0),
        out_width(0),
        out_depth(0),
        pad_P1(0),
        pad_P2(0),
        pad_left(0),
        pad_right(0),
        pad_top(0),
        pad_bottom(0),
        pad_depth(0),
        data_format(TensorFormat::FORMAT_NCHW) {}

  // Updates context->status if there is an invalid input.
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            const std::vector<int32>& padding_list, TensorFormat data_format,
            const TensorShape& tensor_in_shape) {
    // For max pooling, tensor_in should have 4 or 5 dimensions.
    OP_REQUIRES(
        context, tensor_in_shape.dims() == 4 || tensor_in_shape.dims() == 5,
        errors::InvalidArgument("tensor_in must be 4 or 5-dimensional"));

    depth = GetTensorDim(tensor_in_shape, data_format, 'C');
    if (tensor_in_shape.dims() == 4) {
      // Pool2D
      tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
      tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
    } else {
      // Pool3D
      tensor_in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
      tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
      tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
    }
    tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');

    Init(context, ksize, stride, padding, padding_list, data_format);
  }

 private:
  // Common initialization for TensorFlow and oneDNN formats
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            const std::vector<int32>& padding_list, TensorFormat data_format) {
    // Get the data format.
    this->data_format = data_format;

    bool is_pool2d = (ksize.size() == 4);
    if (is_pool2d) {
      // Pool2D
      // Get the output sizes.
      window_rows = GetTensorDim(ksize, data_format, 'H');
      window_cols = GetTensorDim(ksize, data_format, 'W');
      depth_window = GetTensorDim(ksize, data_format, 'C');

      // Get the strides.
      row_stride = GetTensorDim(stride, data_format, 'H');
      col_stride = GetTensorDim(stride, data_format, 'W');
      depth_stride = GetTensorDim(stride, data_format, 'C');

      // We only support 2D pooling across width/height and depthwise
      // pooling, not a combination.
      OP_REQUIRES(context,
                  (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
                  errors::Unimplemented(
                      "MaxPooling supports exactly one of pooling across depth "
                      "or pooling across width/height."));
    } else {
      // Pool3D
      // Get the output sizes.
      window_planes = GetTensorDim(ksize, data_format, '0');
      window_rows = GetTensorDim(ksize, data_format, '1');
      window_cols = GetTensorDim(ksize, data_format, '2');
      depth_window = GetTensorDim(ksize, data_format, 'C');

      // Get the strides.
      planes_stride = GetTensorDim(stride, data_format, '0');
      row_stride = GetTensorDim(stride, data_format, '1');
      col_stride = GetTensorDim(stride, data_format, '2');
      depth_stride = GetTensorDim(stride, data_format, 'C');

      // We only support 3D pooling across depth/width/height and depthwise
      // pooling, not a combination.
      OP_REQUIRES(
          context,
          (depth_window == 1 ||
           (window_rows == 1 && window_cols == 1 && window_planes == 1)),
          errors::Unimplemented(
              "AvgPooling3D supports exactly one of pooling across depth "
              "or pooling across depth/width/height."));
    }

    if (depth_window == 1) {  // We are pooling in the D (Pool3D only), H and W.
      if (!is_pool2d) {
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSizeVerbose(
                           tensor_in_planes, window_planes, planes_stride,
                           padding, &out_planes, &pad_P1, &pad_P2));
      }
      if (padding == Padding::EXPLICIT) {
        if (data_format == FORMAT_NHWC) {
          pad_top = static_cast<int64>(padding_list[2]);
          pad_left = static_cast<int64>(padding_list[4]);
          pad_bottom = static_cast<int64>(padding_list[3]);
          pad_right = static_cast<int64>(padding_list[5]);
        } else if (data_format == FORMAT_NCHW) {
          pad_top = static_cast<int64>(padding_list[4]);
          pad_left = static_cast<int64>(padding_list[6]);
          pad_bottom = static_cast<int64>(padding_list[5]);
          pad_right = static_cast<int64>(padding_list[7]);
        }
      }
      OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                  tensor_in_rows, window_rows, row_stride,
                                  padding, &out_height, &pad_top, &pad_bottom));

      OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                  tensor_in_cols, window_cols, col_stride,
                                  padding, &out_width, &pad_left, &pad_right));

      // TF can work with int64, but dnnl only supports int32.
      // Fail if the depth, height or width are greater than MAX_INT.
      // We check depth only for 3D pooling case.
      if (!is_pool2d) {
        OP_REQUIRES(
            context,
            FastBoundsCheck(out_planes, std::numeric_limits<int>::max()),
            errors::InvalidArgument("output depth/planes is too large"));
      }

      OP_REQUIRES(context,
                  FastBoundsCheck(out_height, std::numeric_limits<int>::max()),
                  errors::InvalidArgument("output height is too large"));

      OP_REQUIRES(context,
                  FastBoundsCheck(out_width, std::numeric_limits<int>::max()),
                  errors::InvalidArgument("output width is too large"));

      out_depth = depth;  // Output will have the same depth as the input.
    } else {              // We are pooling in the depth dimension.
      // Our current version of depthwise max pooling does not support
      // any padding, and expects the depth_window to equal the depth
      // stride (no overlapping).
      OP_REQUIRES(context, depth % depth_window == 0,
                  errors::Unimplemented("Depthwise max pooling requires the"
                                        " depth window to evenly divide the"
                                        " input depth"));
      OP_REQUIRES(context, depth_stride == depth_window,
                  errors::Unimplemented("Depthwise max pooling requires the"
                                        " depth window to equal the depth"
                                        " stride"));
      out_depth = depth / depth_window;
    }
  }
};

template <typename T>
class PoolingOpBase : public OpKernel {
 public:
  explicit PoolingOpBase(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (std::is_same<T, qint8>::value || std::is_same<T, quint8>::value) {
      // Current quantized pooling doesn't have data_format attribute.
      data_format = "NHWC";
    } else {
      if (context->HasAttr("data_format")) {
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      } else {
        data_format = "NHWC";
      }
    }

    OP_REQUIRES(context, FormatFromString(data_format, &this->data_format_tf_),
                errors::InvalidArgument("Invalid data format"));

    if (context->HasAttr("ksize")) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &this->ksize_));
      OP_REQUIRES(context, this->ksize_.size() == 4 || this->ksize_.size() == 5,
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify 4 or 5 dimensions"));
    }
    if (context->HasAttr("strides")) {
      OP_REQUIRES_OK(context, context->GetAttr("strides", &this->stride_));
      OP_REQUIRES(context,
                  this->stride_.size() == 4 || this->stride_.size() == 5,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 4 or 5 dimensions"));
      OP_REQUIRES(context, this->ksize_[0] == 1 && this->stride_[0] == 1,
                  errors::Unimplemented("Pooling is not yet supported on the "
                                        "batch dimension."));
    }
    string padding;
    if (context->HasAttr("padding")) {
      OP_REQUIRES_OK(context, context->GetAttr("padding", &(this->padding_)));
      if (this->padding_ == Padding::EXPLICIT) {
        if (context->HasAttr("explicit_paddings")) {
          OP_REQUIRES_OK(context, context->GetAttr("explicit_paddings",
                                                   &this->padding_list_));
        }
        OP_REQUIRES(
            context, !this->padding_list_.empty(),
            errors::InvalidArgument(
                "explicit_paddings attribute must be empty if the padding "
                "attribute is "
                "not EXPLICIT"));
      }
    }
    if (context->HasAttr("include_batch_in_index")) {
      OP_REQUIRES_OK(context, context->GetAttr("include_batch_in_index",
                                               &include_batch_in_index_));
    }
  }

  void Compute(OpKernelContext* context) override = 0;

 protected:
  // Calculate output shape of pooling op in oneDNN and TensorFlow order.
  // OneDNN uses NCHW(Pool2D) or NCDHW(Pool3D) for output order.
  // But TensorFlow output will be in NHWC/NCHW(Pool2D) or
  // NDHWC/NCDHW(Pool3D) format depending on data format. Function expects
  // output height and width to have already been int32 bounds-checked.
  void GetOutputDims(const OneDnnPoolParameters& onednn_pool_params,
                     TensorFormat tf_format,
                     memory::dims* output_dims_onednn_order,
                     TensorShape* output_tf_shape) {
    if (this->ksize_.size() == 4) {
      // Pooling2D: OneDNN always needs output in NCHW format.
      *output_dims_onednn_order = {
          onednn_pool_params.tensor_in_batch, onednn_pool_params.out_depth,
          static_cast<int>(onednn_pool_params.out_height),
          static_cast<int>(onednn_pool_params.out_width)};

      if (tf_format == TensorFormat::FORMAT_NCHW) {
        output_tf_shape->AddDim(onednn_pool_params.tensor_in_batch);
        output_tf_shape->AddDim(onednn_pool_params.out_depth);
        output_tf_shape->AddDim(
            static_cast<int>(onednn_pool_params.out_height));
        output_tf_shape->AddDim(static_cast<int>(onednn_pool_params.out_width));
      } else {
        output_tf_shape->AddDim(onednn_pool_params.tensor_in_batch);
        output_tf_shape->AddDim(
            static_cast<int>(onednn_pool_params.out_height));
        output_tf_shape->AddDim(static_cast<int>(onednn_pool_params.out_width));
        output_tf_shape->AddDim(onednn_pool_params.out_depth);
      }
    } else {
      // Pooling3D: OneDNN always needs output in NCDHW format.
      *output_dims_onednn_order = {
          onednn_pool_params.tensor_in_batch, onednn_pool_params.out_depth,
          static_cast<int>(onednn_pool_params.out_planes),
          static_cast<int>(onednn_pool_params.out_height),
          static_cast<int>(onednn_pool_params.out_width)};
      if (tf_format == TensorFormat::FORMAT_NCHW) {
        output_tf_shape->AddDim(onednn_pool_params.tensor_in_batch);
        output_tf_shape->AddDim(onednn_pool_params.out_depth);
        output_tf_shape->AddDim(
            static_cast<int>(onednn_pool_params.out_planes));
        output_tf_shape->AddDim(
            static_cast<int>(onednn_pool_params.out_height));
        output_tf_shape->AddDim(static_cast<int>(onednn_pool_params.out_width));
      } else {
        output_tf_shape->AddDim(onednn_pool_params.tensor_in_batch);
        output_tf_shape->AddDim(
            static_cast<int>(onednn_pool_params.out_planes));
        output_tf_shape->AddDim(
            static_cast<int>(onednn_pool_params.out_height));
        output_tf_shape->AddDim(static_cast<int>(onednn_pool_params.out_width));
        output_tf_shape->AddDim(onednn_pool_params.out_depth);
      }
    }
  }

  void InitPoolParameters(OpKernelContext* context,
                          OneDnnPoolParameters* pool_params,
                          const TensorShape& input_tensor_shape,
                          const std::vector<int32>& padding_list) {
    pool_params->Init(context, this->ksize_, this->stride_, this->padding_,
                      padding_list, this->data_format_tf_, input_tensor_shape);
  }

  void PoolParamsToDims(const OneDnnPoolParameters* pool_params,
                        memory::dims* filter_dims, memory::dims* dilation_dims,
                        memory::dims* strides, memory::dims* padding_left,
                        memory::dims* padding_right, bool is_pool2d) {
    if (is_pool2d) {
      // Pool2D
      *filter_dims =
          memory::dims({pool_params->window_rows, pool_params->window_cols});
      *dilation_dims = memory::dims({0, 0});
      *strides =
          memory::dims({pool_params->row_stride, pool_params->col_stride});
      *padding_left = memory::dims({static_cast<int>(pool_params->pad_top),
                                    static_cast<int>(pool_params->pad_left)});
      *padding_right = memory::dims({static_cast<int>(pool_params->pad_bottom),
                                     static_cast<int>(pool_params->pad_right)});
    } else {
      // Pool3D
      *filter_dims =
          memory::dims({pool_params->window_planes, pool_params->window_rows,
                        pool_params->window_cols});
      *dilation_dims = memory::dims({0, 0, 0});
      *strides =
          memory::dims({pool_params->planes_stride, pool_params->row_stride,
                        pool_params->col_stride});

      *padding_left = memory::dims({static_cast<int>(pool_params->pad_P1),
                                    static_cast<int>(pool_params->pad_top),
                                    static_cast<int>(pool_params->pad_left)});
      *padding_right = memory::dims({static_cast<int>(pool_params->pad_P2),
                                     static_cast<int>(pool_params->pad_bottom),
                                     static_cast<int>(pool_params->pad_right)});
    }
  }

  void AllocateEmptyOutputTensor(OpKernelContext* context,
                                 const int kOutputIndex,
                                 OneDnnPoolParameters* pool_params,
                                 const memory::dims output_dims_onednn_order,
                                 Tensor** output_tensor) {
    TensorShape output_tf_shape;
    if (pool_params->data_format == TensorFormat::FORMAT_NCHW) {
      output_tf_shape = OneDnnDimsToTFShape(output_dims_onednn_order);
    } else {
      memory::dims output_dims_order;
      // determine Pooling2D (NHWC) or Pooling3D (NDHWC)
      if (this->ksize_.size() == 4) {
        output_dims_order = {pool_params->tensor_in_batch,
                             static_cast<int>(pool_params->out_height),
                             static_cast<int>(pool_params->out_width),
                             pool_params->out_depth};
      } else {
        output_dims_order = {pool_params->tensor_in_batch,
                             static_cast<int>(pool_params->out_planes),
                             static_cast<int>(pool_params->out_height),
                             static_cast<int>(pool_params->out_width),
                             pool_params->out_depth};
      }
      output_tf_shape = OneDnnDimsToTFShape(output_dims_order);
    }
    OP_REQUIRES_OK(context, context->allocate_output(
                                kOutputIndex, output_tf_shape, output_tensor));
    ITEX_DCHECK(output_tensor);
  }

  size_t GetNumTElements(const dnnl::memory::desc& pd) {
    size_t num_bytes = pd.get_size();
    size_t ret_val = num_bytes / sizeof(T);
    if (num_bytes % sizeof(T) != 0) {
      ret_val++;
    }
    return ret_val;
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  std::vector<int32> padding_list_;
  TensorFormat data_format_tf_;
  dnnl::memory::format_tag data_format_onednn_;
  bool include_batch_in_index_;
};

template <typename T>
class PoolingForwardOpBase : public PoolingOpBase<T> {
 public:
  explicit PoolingForwardOpBase<T>(OpKernelConstruction* context)
      : PoolingOpBase<T>(context) {}
  void Compute(OpKernelContext* context) override = 0;

 protected:
  void AllocateOutputTensor(OpKernelContext* context,
                            TensorShape* output_tf_shape,
                            Tensor** output_tensor) {
    ITEX_DCHECK(output_tensor);
    OP_REQUIRES_OK(
        context, context->allocate_output(0, *output_tf_shape, output_tensor));
    ITEX_DCHECK(*output_tensor);
  }

  void SanityCheckInput(OpKernelContext* context, const Tensor& input_tensor) {
    OP_REQUIRES(context, input_tensor.dims() == 4 || input_tensor.dims() == 5,
                errors::InvalidArgument("Input must be 4 or 5-dimensional"));
  }

  const int kInputTensorIndexInput = 0;
  const int kOutputTensorIndexOutput = 0;
};

template <typename T>
class PoolingBackwardOpBase : public PoolingOpBase<T> {
 public:
  explicit PoolingBackwardOpBase(OpKernelConstruction* context)
      : PoolingOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override = 0;

 protected:
  void AllocateOutputTensor(OpKernelContext* context,
                            TensorShape* output_tf_shape,
                            Tensor** output_tensor) {
    ITEX_DCHECK(output_tensor);
    OP_REQUIRES_OK(
        context, context->allocate_output(0, *output_tf_shape, output_tensor));
    ITEX_DCHECK(*output_tensor);
  }

  const int kOutputTensorIndexOutput = 0;
};

template <typename Device, typename T, dnnl::algorithm algo>
class PoolingOp : public PoolingForwardOpBase<T> {
 public:
  explicit PoolingOp(OpKernelConstruction* context)
      : PoolingForwardOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const Tensor& input_tensor = context->input(this->kInputTensorIndexInput);
      this->SanityCheckInput(context, input_tensor);

      std::vector<int32> ksize = this->ksize_;
      std::vector<int32> stride = this->stride_;

      // This code is actually for MaxPoolV2/_ITEXMaxPoolV2, where strides and
      // sizes are input tensors not attributes
      if (context->num_inputs() != 1 && !std::is_same<T, qint8>::value &&
          !std::is_same<T, quint8>::value) {
        const Tensor& tensor_ksize = context->input(1);
        auto value_ksize = tensor_ksize.flat<int32>();
        ksize.resize(tensor_ksize.shape().num_elements());
        std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

        const Tensor& tensor_stride = context->input(2);
        auto value_stride = tensor_stride.flat<int32>();
        stride.resize(tensor_stride.shape().num_elements());
        std::copy_n(&value_stride(0), stride.size(), stride.begin());
      }
      this->ksize_ = ksize;
      this->stride_ = stride;

      OP_REQUIRES(
          context, this->ksize_.size() == 4 || this->ksize_.size() == 5,
          errors::InvalidArgument("Sliding window ksize field must "
                                  "specify 4 dimensions or 5 dimensions"));
      OP_REQUIRES(
          context, this->stride_.size() == 4 || this->stride_.size() == 5,
          errors::InvalidArgument("Sliding window stride field must "
                                  "specify 4 dimensions or 5 dimensions"));
      const int32 ksize_n = GetTensorDim(ksize, this->data_format_tf_, 'N');
      const int32 stride_n = GetTensorDim(stride, this->data_format_tf_, 'N');
      OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));

      OneDnnPoolParameters pool_params;
      // Check whether pooling is 2D or 3D.
      bool is_pool2d = (this->ksize_.size() == 4);
      OneDnnTensorFormat tensor_format_onednn =
          TFDataFormatToOneDnnDataFormat(this->data_format_tf_, is_pool2d);
      this->data_format_onednn_ = OneDnnTensorFormatToTag(tensor_format_onednn);

      // Get the input tensor and initialize the pooling parameters.
      TensorShape input_tensor_shape = input_tensor.shape();
      this->InitPoolParameters(context, &pool_params, input_tensor_shape,
                               this->padding_list_);

      Tensor* output_tensor = nullptr;
      Tensor* ws_tensor = nullptr;
      dnnl::memory::dims dst_dims;
      TensorShape tf_output_shape;
      this->GetOutputDims(pool_params, this->data_format_tf_, &dst_dims,
                          &tf_output_shape);

      // int8 and fp16 only support inference
      bool only_forward_inference = std::is_same<T, qint8>::value ||
                                    std::is_same<T, quint8>::value ||
                                    std::is_same<T, Eigen::half>::value;
      // GPU MaxPool doesn't have workspace, and it's num_outputs == 1.
      bool workspace_enabled = (algo == dnnl::algorithm::pooling_max) &&
                               (context->num_outputs() != 1) &&
                               !only_forward_inference;

      // If input is an empty tensor, allocate an empty output tensor.
      if (input_tensor.NumElements() == 0) {
        this->AllocateEmptyOutputTensor(context, this->kOutputTensorIndexOutput,
                                        &pool_params, dst_dims, &output_tensor);
        if (workspace_enabled) {
          this->AllocateEmptyOutputTensor(context, 1, &pool_params, dst_dims,
                                          &ws_tensor);
        }
        return;
      }
      this->AllocateOutputTensor(context, &tf_output_shape, &output_tensor);
      ITEX_DCHECK(output_tensor);

      dnnl::memory::dims filter_dims, strides, padding_left, padding_right;
      dnnl::memory::dims dilation_dims;

      // Get src/filter/stride/padding information.
      this->PoolParamsToDims(&pool_params, &filter_dims, &dilation_dims,
                             &strides, &padding_left, &padding_right,
                             is_pool2d);

      // Get the input memory descriptor.
      dnnl::memory::dims src_dims = TFShapeToOneDnnDimsInNC(
          input_tensor.shape(), this->data_format_tf_, is_pool2d);

      dnnl::prop_kind pooling_prop_kind;
      if (workspace_enabled)
        pooling_prop_kind = prop_kind::forward_training;
      else
        pooling_prop_kind = prop_kind::forward_inference;

      dnnl::memory::desc src_md(src_dims, OneDnnType<T>(),
                                this->data_format_onednn_);
      dnnl::memory::desc dst_md(dst_dims, OneDnnType<T>(),
                                this->data_format_onednn_);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      dnnl::pooling_forward::primitive_desc fwd_pd(
          onednn_engine, pooling_prop_kind, algo, src_md, dst_md, strides,
          filter_dims, dilation_dims, padding_left, padding_right, attr);
#else
      dnnl::pooling_forward::desc fwd_desc(pooling_prop_kind, algo, src_md,
                                           dst_md, strides, filter_dims,
                                           padding_left, padding_right);

      dnnl::pooling_forward::primitive_desc fwd_pd(fwd_desc, attr,
                                                   onednn_engine);
#endif
      Tensor scratchpad_tensor;
      int64 scratchpad_size = fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      dnnl::pooling_forward fwd(fwd_pd);

      const T* src_data = input_tensor.flat<T>().data();
      T* dst_data = output_tensor->flat<T>().data();

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      // Create src and dst memory.
      auto src_mem =
          CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                           static_cast<void*>(const_cast<T*>(src_data)));
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      static_cast<void*>(dst_data));

      std::unordered_map<int, dnnl::memory> net_args(
          {{DNNL_ARG_SRC, src_mem},
           {DNNL_ARG_DST, dst_mem},
           {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

      if (workspace_enabled) {
        TensorShape ws_tensor_shape;
        ws_tensor_shape.AddDim(fwd_pd.workspace_desc().get_size());
        OP_REQUIRES_OK(
            context, context->allocate_output(1, ws_tensor_shape, &ws_tensor));
        dnnl::memory ws_mem;
        ws_mem = CreateDnnlMemory(fwd_pd.workspace_desc(), onednn_engine,
                                  GetTensorBuffer<Tws>(ws_tensor));
        net_args.insert({DNNL_ARG_WORKSPACE, ws_mem});
      }
      fwd.execute(onednn_stream, net_args);

      bool int8_forward_inference =
          std::is_same<T, qint8>::value || std::is_same<T, quint8>::value;

      if (int8_forward_inference) {
        // Pass min, max from input to output.
        const Tensor& min_input_t = context->input(1);
        const Tensor& max_input_t = context->input(2);
        const float min_input = min_input_t.flat<float>()(0);
        const float max_input = max_input_t.flat<float>()(0);

        Tensor* output_min = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(1, TensorShape({}), &output_min));
        output_min->flat<float>()(0) = min_input;

        Tensor* output_max = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(2, TensorShape({}), &output_max));
        output_max->flat<float>()(0) = max_input;
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

template <typename T>
class OneDnnPoolOpBase : public OpKernel {
 public:
  explicit OneDnnPoolOpBase(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    string padding;
    if (std::is_same<T, qint8>::value || std::is_same<T, quint8>::value) {
      // Current quantized pool doesn't have data_format attribute.
      data_format = "NHWC";
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    }
    OP_REQUIRES(context, FormatFromString(data_format, &this->data_format_tf_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &this->ksize_));
    OP_REQUIRES(context, this->ksize_.size() == 4 || this->ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 or 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &this->stride_));
    OP_REQUIRES(context, this->stride_.size() == 4 || this->stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 or 5 dimensions"));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &(this->padding_)));
    if (this->padding_ == Padding::EXPLICIT) {
      if (context->HasAttr("explicit_paddings")) {
        OP_REQUIRES_OK(context, context->GetAttr("explicit_paddings",
                                                 &this->padding_list_));
      }
      OP_REQUIRES(
          context, !this->padding_list_.empty(),
          errors::InvalidArgument(
              "explicit_paddings attribute must be empty if the padding "
              "attribute is "
              "not EXPLICIT"));
    }

    OP_REQUIRES(context, this->ksize_[0] == 1 && this->stride_[0] == 1,
                errors::Unimplemented("Pool is not yet supported on the "
                                      "batch dimension."));

    this->is_2d_ = (this->ksize_.size() == 4);
    this->tensor_format_onednn_ =
        TFDataFormatToOneDnnDataFormat(this->data_format_tf_, this->is_2d_);
    this->data_format_onednn_ =
        OneDnnTensorFormatToTag(this->tensor_format_onednn_);
  }
  void Compute(OpKernelContext* context) override = 0;

 protected:
  // Calculate output shape in OneDNN and TensorFlow order.
  // OneDNN uses NCHW(Pool2D) or NCDHW(Pool3D) for output order.
  // But TF output will be in NHWC/NCHW(Pool2D) or NDHWC/NCDHW(Pool3D) format
  // depending on data format. Function expects output height and width to
  // have already been int32 bounds-checked.
  void GetOutputDims(const OneDnnPoolParameters& pool_params,
                     memory::dims* dst_onednn_dims, TensorShape* dst_tf_shape) {
    if (this->is_2d_) {
      // Pooling2D: OneDNN always needs output in NCHW format.
      *dst_onednn_dims = {pool_params.tensor_in_batch, pool_params.out_depth,
                          static_cast<int>(pool_params.out_height),
                          static_cast<int>(pool_params.out_width)};
    } else {
      // Pooling3D: OneDNN always needs output in NCDHW format.
      *dst_onednn_dims = {pool_params.tensor_in_batch, pool_params.out_depth,
                          static_cast<int>(pool_params.out_planes),
                          static_cast<int>(pool_params.out_height),
                          static_cast<int>(pool_params.out_width)};
    }

    if (pool_params.data_format == TensorFormat::FORMAT_NCHW) {
      *dst_tf_shape = OneDnnDimsToTFShape(*dst_onednn_dims);
    } else {
      memory::dims dst_tf_dims;
      // Determine Pooling2D (NHWC) or Pooling3D (NDHWC).
      // Switch the 2nd dim and last dim to transform OneDnn order
      // (NCHW/NCDHW) to TF order (NHWC/NDHWC).
      if (this->is_2d_) {
        dst_tf_dims = {pool_params.tensor_in_batch,
                       static_cast<int>(pool_params.out_height),
                       static_cast<int>(pool_params.out_width),
                       pool_params.out_depth};
      } else {
        dst_tf_dims = {pool_params.tensor_in_batch,
                       static_cast<int>(pool_params.out_planes),
                       static_cast<int>(pool_params.out_height),
                       static_cast<int>(pool_params.out_width),
                       pool_params.out_depth};
      }
      *dst_tf_shape = OneDnnDimsToTFShape(dst_tf_dims);
    }
  }

  void PoolParamsToDims(const OneDnnPoolParameters* pool_params,
                        memory::dims* filter_dims, memory::dims* dilation_dims,
                        memory::dims* strides, memory::dims* padding_left,
                        memory::dims* padding_right) {
    if (this->is_2d_) {
      // Pool2D
      *filter_dims =
          memory::dims({pool_params->window_rows, pool_params->window_cols});
      *dilation_dims = memory::dims({0, 0});
      *strides =
          memory::dims({pool_params->row_stride, pool_params->col_stride});
      *padding_left = memory::dims({static_cast<int>(pool_params->pad_top),
                                    static_cast<int>(pool_params->pad_left)});
      *padding_right = memory::dims({static_cast<int>(pool_params->pad_bottom),
                                     static_cast<int>(pool_params->pad_right)});
    } else {
      // Pool3D
      *filter_dims =
          memory::dims({pool_params->window_planes, pool_params->window_rows,
                        pool_params->window_cols});
      *dilation_dims = memory::dims({0, 0, 0});
      *strides =
          memory::dims({pool_params->planes_stride, pool_params->row_stride,
                        pool_params->col_stride});

      *padding_left = memory::dims({static_cast<int>(pool_params->pad_P1),
                                    static_cast<int>(pool_params->pad_top),
                                    static_cast<int>(pool_params->pad_left)});
      *padding_right = memory::dims({static_cast<int>(pool_params->pad_P2),
                                     static_cast<int>(pool_params->pad_bottom),
                                     static_cast<int>(pool_params->pad_right)});
    }
  }

  bool is_2d_;
  std::vector<int32> ksize_;
  std::vector<int32> padding_list_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_tf_;
  OneDnnTensorFormat tensor_format_onednn_;
  memory::format_tag data_format_onednn_;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_POOLING_OPS_COMMON_H_
