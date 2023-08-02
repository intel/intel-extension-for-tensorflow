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

#ifndef ITEX_CORE_KERNELS_ONEDNN_BLOCK_RESIZE_OP_H_
#define ITEX_CORE_KERNELS_ONEDNN_BLOCK_RESIZE_OP_H_

#include <string>
#include <unordered_map>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::algorithm;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;

namespace itex {
template <typename Device, typename InputT, typename OutputT,
          dnnl::algorithm alg>
class OneDnnResizeOp : public OpKernel {
 public:
  explicit OneDnnResizeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
    ITEX_CHECK_EQ(align_corners_, false);
    ITEX_CHECK_EQ(half_pixel_centers_, true);
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const size_t src_index = 0;  // index of src input tensor
      const size_t dst_index = 0;  // index of dst output tensor
      const Tensor& src_tensor = context->input(src_index);
      const Tensor& size_tensor = context->input(1);

      OneDnnShape src_onednn_shape;
      GetOneDnnShape(context, src_index, &src_onednn_shape);
      TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                     ? src_onednn_shape.GetTfShape()
                                     : src_tensor.shape();

      OneDnnShape dst_onednn_shape;
      TensorShape dst_tf_shape;

      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_tf_shape.num_elements() == 0) {
        dst_onednn_shape.SetOneDnnTensor(false);
        dst_tf_shape = src_tf_shape;
        ForwardOrAllocateOutputSetOneDnnShape(context, src_index, dst_index,
                                              &dst_tensor, dst_tf_shape,
                                              dst_onednn_shape);
        return;
      }

      bool is_5d = size_tensor.NumElements() == 3;

      memory::dims src_dims;
      memory::desc src_md;
      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
        src_md = src_onednn_shape.GetOneDnnLayout();
      } else {
        src_dims = TFShapeToOneDnnDimsInNC(src_tf_shape, FORMAT_NHWC, !is_5d);
        // Create `plain` onednn memory descriptor
        // src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
        auto format = dnnl::memory::format_tag::nhwc;
        if (is_5d) {
          format = dnnl::memory::format_tag::ndhwc;
        }
        src_md = memory::desc(src_dims, OneDnnType<InputT>(), format);
      }

      auto batch_size = src_tf_shape.dim_size(0);
      auto channel = src_tf_shape.dim_size(is_5d ? 4 : 3);

      int output_depth, output_height, output_width;
      if (is_5d) {
        output_depth = size_tensor.vec<int32>()(0);
        output_height = size_tensor.vec<int32>()(1);
        output_width = size_tensor.vec<int32>()(2);
      } else {
        output_depth = 1;
        output_height = size_tensor.vec<int32>()(0);
        output_width = size_tensor.vec<int32>()(1);
      }

      memory::dims dst_dims;
      if (!is_5d) {
        dst_dims = {batch_size, channel, output_height, output_width};
        dst_tf_shape =
            TensorShape({batch_size, output_height, output_width, channel});

      } else {
        dst_dims = {batch_size, channel, output_depth, output_height,
                    output_width};
        dst_tf_shape = TensorShape(
            {batch_size, output_depth, output_height, output_width, channel});
      }
      memory::desc dst_md = memory::desc(dst_dims, OneDnnType<OutputT>(),
                                         dnnl::memory::format_tag::any);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      auto fwd_pd = dnnl::resampling_forward::primitive_desc(
          onednn_engine,
          prop_kind::forward_training
          /* training and inference is same*/,
          alg, src_md, dst_md, attr);
      Tensor scratchpad_tensor;
      dnnl::memory scratchpad_mem;
      int64 scratchpad_size =
          fwd_pd.scratchpad_desc().get_size() / sizeof(InputT);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<InputT>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<InputT>(&scratchpad_tensor));

      auto fwd_primitive = dnnl::resampling_forward(fwd_pd);

      dnnl::memory src_mem = dnnl::memory(src_md, onednn_engine,
                                          GetTensorBuffer<InputT>(&src_tensor));

      dnnl::memory reorder_mem;
      Tensor src_reorder_tensor;
      bool is_src_reordered = (src_md != fwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size = fwd_pd.src_desc().get_size() / sizeof(InputT);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<InputT>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));

        reorder_mem =
            CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<InputT>(&src_reorder_tensor));
        ReorderMemory(*context, &src_mem, &reorder_mem, onednn_engine);
      }

      // Allocate output data tensor and meta tensor
      SetOutputTensorShape(
          fwd_pd.dst_desc(), src_onednn_shape.GetTfDataFormat(), &dst_tf_shape,
          &dst_onednn_shape, src_onednn_shape.IsOneDnnTensor());

      AllocateOutputSetOneDnnShape(context, 0, &dst_tensor, dst_tf_shape,
                                   dst_onednn_shape);

      // Create dst memory
      OutputT* dst_data = dst_tensor->flat<OutputT>().data();
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      static_cast<void*>(dst_data));

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> fwd_primitive_args = {
          {DNNL_ARG_SRC, is_src_reordered ? reorder_mem : src_mem},
          {DNNL_ARG_DST, dst_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};

      fwd_primitive.execute(onednn_stream, fwd_primitive_args);
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
  bool align_corners_;
  bool half_pixel_centers_;
};

template <typename Device, typename T, dnnl::algorithm alg>
class OneDnnResizeGradOp : public OpKernel {
 public:
  ~OneDnnResizeGradOp() {}

  explicit OneDnnResizeGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));

    ITEX_CHECK_EQ(align_corners_, false);
    ITEX_CHECK_EQ(half_pixel_centers_, true);
  }

  void Compute(OpKernelContext* context) override {
    auto onednn_engine = CreateDnnlEngine<Device>(*context);

    const size_t diff_dst_index = 0;
    const size_t src_index = 1;

    const Tensor& diff_dst_tensor = context->input(0);
    OneDnnShape diff_dst_onednn_shape;
    GetOneDnnShape(context, diff_dst_index, &diff_dst_onednn_shape);
    TensorShape diff_dst_tf_shape = diff_dst_onednn_shape.IsOneDnnTensor()
                                        ? diff_dst_onednn_shape.GetTfShape()
                                        : diff_dst_tensor.shape();

    TensorShape diff_src_tf_shape;
    OneDnnShape diff_src_onednn_shape;
    Tensor* diff_src_tensor = nullptr;
    if (diff_dst_tensor.shape().num_elements() == 0) {
      diff_src_onednn_shape.SetOneDnnTensor(false);
      diff_src_tf_shape = diff_dst_tf_shape;
      AllocateOutputSetOneDnnShape(context, 0, &diff_src_tensor,
                                   diff_src_tf_shape, diff_src_onednn_shape);
      return;
    }

    auto create_dims_and_md = [](TensorShape& tf_shape, OneDnnShape& shape,
                                 memory::dims& dims, memory::desc& desc,
                                 dnnl::memory::data_type = OneDnnType<T>()) {
      bool is_5d = tf_shape.dims() == 5;
      if (shape.IsOneDnnTensor()) {
        dims = shape.GetSizesAsOneDnnDims();
        desc = shape.GetOneDnnLayout();
      } else {
        dims = TFShapeToOneDnnDimsInNC(tf_shape, FORMAT_NHWC, !is_5d);
        auto format = dnnl::memory::format_tag::nhwc;
        if (is_5d) {
          format = dnnl::memory::format_tag::ndhwc;
        }
        desc = memory::desc(dims, OneDnnType<T>(), format);
      }
    };

    try {
      memory::dims src_dims;
      memory::desc src_md;

      memory::dims diff_dst_dims;
      memory::desc diff_dst_md;

      if constexpr (alg == dnnl::algorithm::resampling_linear) {
        const Tensor& original_image = context->input(1);
        OneDnnShape src_onednn_shape;
        GetOneDnnShape(context, src_index, &src_onednn_shape);
        TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                       ? src_onednn_shape.GetTfShape()
                                       : original_image.shape();

        diff_src_tf_shape = src_tf_shape;
        diff_src_onednn_shape = src_onednn_shape;

        create_dims_and_md(src_tf_shape, src_onednn_shape, src_dims, src_md);
        create_dims_and_md(diff_dst_tf_shape, diff_dst_onednn_shape,
                           diff_dst_dims, diff_dst_md,
                           dnnl::memory::data_type::f32);

      } else {
        const Tensor& size_tensor = context->input(1);
        bool is_5d = size_tensor.NumElements() == 3;

        auto batch_size = diff_dst_tf_shape.dim_size(0);
        auto channel = diff_dst_tf_shape.dim_size(is_5d ? 4 : 3);

        int depth, height, width;
        if (is_5d) {
          depth = size_tensor.vec<int32>()(0);
          height = size_tensor.vec<int32>()(1);
          width = size_tensor.vec<int32>()(2);
          src_dims = {batch_size, channel, depth, height, width};
          src_md = memory::desc(src_dims, OneDnnType<T>(),
                                dnnl::memory::format_tag::ndhwc);
          diff_src_tf_shape =
              TensorShape({batch_size, depth, height, width, channel});
        } else {
          depth = 1;
          height = size_tensor.vec<int32>()(0);
          width = size_tensor.vec<int32>()(1);
          src_dims = {batch_size, channel, height, width};
          // TODO(itex) Should be any.
          src_md = memory::desc(src_dims, OneDnnType<T>(),
                                dnnl::memory::format_tag::nhwc);
          diff_src_tf_shape = TensorShape({batch_size, height, width, channel});
        }

        create_dims_and_md(diff_dst_tf_shape, diff_dst_onednn_shape,
                           diff_dst_dims, diff_dst_md);
      }

      memory::dims diff_src_dims = src_dims;
      memory::desc diff_src_md = src_md;

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      auto fwd_pd = dnnl::resampling_forward::primitive_desc(
          onednn_engine, dnnl::prop_kind::forward_training, alg, src_md,
          diff_dst_md);
      auto bwd_pd = dnnl::resampling_backward::primitive_desc(
          onednn_engine, alg, diff_src_md, diff_dst_md, fwd_pd, attr);
      Tensor scratchpad_tensor;
      dnnl::memory scratchpad_mem;
      int64 scratchpad_size = bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      scratchpad_mem = dnnl::memory(bwd_pd.scratchpad_desc(), onednn_engine,
                                    GetTensorBuffer<T>(&scratchpad_tensor));

      auto reorder_memory = [&](dnnl::memory& lhs, memory::desc& rhs_desc,
                                Tensor& rhs_tensor, dnnl::memory& rhs) -> bool {
        bool ret = lhs.get_desc() != rhs_desc;
        if (ret) {
          int64 size = rhs_desc.get_size() / sizeof(T);
          ITEX_CHECK_OK(context->allocate_temp(
              DataTypeToEnum<T>::v(), TensorShape({size}), &rhs_tensor));
          rhs = CreateDnnlMemory(rhs_desc, onednn_engine,
                                 GetTensorBuffer<T>(&rhs_tensor));
          ReorderMemory(*context, &lhs, &rhs, onednn_engine);
        }
        return ret;
      };

      dnnl::memory diff_dst_mem;
      if constexpr (alg == dnnl::algorithm::resampling_linear) {
        diff_dst_mem =
            CreateDnnlMemory(diff_dst_md, onednn_engine,
                             GetTensorBuffer<float>(&diff_dst_tensor));
      } else {
        diff_dst_mem = CreateDnnlMemory(diff_dst_md, onednn_engine,
                                        GetTensorBuffer<T>(&diff_dst_tensor));
      }
      Tensor diff_dst_reorder_tensor;
      dnnl::memory diff_dst_reorder_mem;
      dnnl::memory::desc diff_dst_desc = bwd_pd.diff_dst_desc();
      bool is_diff_dst_reordered =
          reorder_memory(diff_dst_mem, diff_dst_desc, diff_dst_reorder_tensor,
                         diff_dst_reorder_mem);

      dnnl::memory src_mem;
      if constexpr (alg == dnnl::algorithm::resampling_linear) {
        const Tensor& original_image = context->input(1);
        src_mem = dnnl::memory(src_md, onednn_engine,
                               GetTensorBuffer<T>(&original_image));
      }

      // Allocate output data tensor and meta tensor
      SetOutputTensorShape(bwd_pd.diff_src_desc(),
                           diff_dst_onednn_shape.GetTfDataFormat(),
                           &diff_src_tf_shape, &diff_src_onednn_shape,
                           diff_dst_onednn_shape.IsOneDnnTensor());

      AllocateOutputSetOneDnnShape(context, 0, &diff_src_tensor,
                                   diff_src_tf_shape, diff_src_onednn_shape);

      dnnl::memory diff_src_mem =
          dnnl::memory(bwd_pd.diff_src_desc(), onednn_engine,
                       GetTensorBuffer<T>(diff_src_tensor));

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> bwd_primitive_args = {
          {DNNL_ARG_DIFF_DST,
           is_diff_dst_reordered ? diff_dst_reorder_mem : diff_dst_mem},
          {DNNL_ARG_DIFF_SRC, diff_src_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};

      if constexpr (alg == dnnl::algorithm::resampling_linear) {
        bwd_primitive_args.insert({DNNL_ARG_SRC, src_mem});
      }

      auto bwd_primitive = dnnl::resampling_backward(bwd_pd);
      bwd_primitive.execute(onednn_stream, bwd_primitive_args);
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
  bool align_corners_;
  bool half_pixel_centers_;
};
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_ONEDNN_BLOCK_RESIZE_OP_H_
