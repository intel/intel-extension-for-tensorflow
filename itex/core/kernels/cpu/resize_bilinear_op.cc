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

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::algorithm;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;

namespace itex {
template <typename Device, typename T>
class ResizeBilinearOp : public OpKernel {
 public:
  explicit ResizeBilinearOp(OpKernelConstruction* context) : OpKernel(context) {
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

      TensorShape src_tf_shape = src_tensor.shape();

      TensorShape dst_tf_shape;

      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_tf_shape.num_elements() == 0) {
        OP_REQUIRES_OK(context,
                       context->forward_input_or_allocate_output(
                           {src_index}, dst_index, src_tf_shape, &dst_tensor));
        return;
      }

      memory::dims src_dims;
      memory::desc src_md;

      src_dims = TFShapeToOneDnnDimsInNC(src_tf_shape, FORMAT_NHWC);
      // Create `plain` onednn memory descriptor
      // src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
      src_md = memory::desc(src_dims, OneDnnType<T>(),
                            dnnl::memory::format_tag::nhwc);

      auto batch_size = src_tf_shape.dim_size(0);
      auto channel = src_tf_shape.dim_size(3);
      auto output_height = size_tensor.vec<int32>()(0);
      auto output_width = size_tensor.vec<int32>()(1);
      memory::dims dst_dims = {batch_size, channel, output_height,
                               output_width};
      memory::desc dst_md = memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                         dnnl::memory::format_tag::nhwc);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      auto fwd_pd = dnnl::resampling_forward::primitive_desc(
          onednn_engine,
          prop_kind::forward_training
          /* training and inference is same*/,
          algorithm::resampling_linear, src_md, dst_md, attr);
#else
      auto fwd_desc = dnnl::resampling_forward::desc(
          prop_kind::forward_training
          /* training and inference is same*/,
          algorithm::resampling_linear, src_md, dst_md);
      auto fwd_pd = dnnl::resampling_forward::primitive_desc(fwd_desc, attr,
                                                             onednn_engine);
#endif
      Tensor scratchpad_tensor;
      dnnl::memory scratchpad_mem;
      int64 scratchpad_size = fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      scratchpad_mem = dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                                    GetTensorBuffer<T>(&scratchpad_tensor));

      auto fwd_primitive = dnnl::resampling_forward(fwd_pd);

      dnnl::memory src_mem =
          dnnl::memory(src_md, onednn_engine, GetTensorBuffer<T>(&src_tensor));

      dnnl::memory reorder_mem;
      Tensor src_reorder_tensor;
      bool is_src_reordered = (src_md != fwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size = fwd_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));

        reorder_mem = CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                                       GetTensorBuffer<T>(&src_reorder_tensor));
        ReorderMemory(*context, &src_mem, &reorder_mem, onednn_engine);
      }

      // By default, the output format will be the same with input (NHWC).
      dst_tf_shape =
          TensorShape({batch_size, output_height, output_width, channel});

      OP_REQUIRES_OK(context, context->allocate_output(dst_index, dst_tf_shape,
                                                       &dst_tensor));

      // Create dst memory
      float* dst_data = dst_tensor->flat<float>().data();
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

template <typename Device, typename T>
class ResizeBilinearGradOp : public OpKernel {
 public:
  ~ResizeBilinearGradOp() {}

  explicit ResizeBilinearGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));

    ITEX_CHECK_EQ(align_corners_, false);
    ITEX_CHECK_EQ(half_pixel_centers_, true);
  }

  void Compute(OpKernelContext* context) override {
    auto onednn_engine = CreateDnnlEngine<Device>(*context);

    const Tensor& diff_dst_tensor = context->input(0);
    const Tensor& original_image = context->input(1);

    TensorShape diff_dst_tf_shape = diff_dst_tensor.shape();

    TensorShape src_tf_shape = original_image.shape();

    TensorShape diff_src_tf_shape = src_tf_shape;

    Tensor* diff_src_tensor = nullptr;
    if (diff_dst_tensor.shape().num_elements() == 0) {
      OP_REQUIRES_OK(
          context, context->allocate_output(0, src_tf_shape, &diff_src_tensor));
      return;
    }

    auto create_dims_and_md = [](const TensorShape& tf_shape,
                                 memory::dims* dims, memory::desc* desc,
                                 dnnl::memory::data_type = OneDnnType<T>()) {
      *dims = TFShapeToOneDnnDimsInNC(tf_shape, FORMAT_NHWC);
      *desc =
          memory::desc(*dims, OneDnnType<T>(), dnnl::memory::format_tag::nhwc);
    };

    try {
      memory::dims src_dims;
      memory::desc src_md;
      create_dims_and_md(src_tf_shape, &src_dims, &src_md);

      memory::dims diff_dst_dims;
      memory::desc diff_dst_md;
      create_dims_and_md(diff_dst_tf_shape, &diff_dst_dims, &diff_dst_md,
                         dnnl::memory::data_type::f32);

      memory::dims diff_src_dims;
      memory::desc diff_src_md;
      create_dims_and_md(diff_src_tf_shape, &diff_src_dims, &diff_src_md);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      auto fwd_pd = dnnl::resampling_forward::primitive_desc(
          onednn_engine, dnnl::prop_kind::forward_training,
          dnnl::algorithm::resampling_linear, src_md, diff_dst_md);
      auto bwd_pd = dnnl::resampling_backward::primitive_desc(
          onednn_engine, dnnl::algorithm::resampling_linear, diff_src_md,
          diff_dst_md, fwd_pd, attr);
#else
      auto bwd_desc = dnnl::resampling_backward::desc(
          dnnl::algorithm::resampling_linear, diff_src_md, diff_dst_md);

      // resampling needs a forward hint, we create it ourselve due to we can't
      // get the true one.
      auto fwd_desc = dnnl::resampling_forward::desc(
          dnnl::prop_kind::forward_training, dnnl::algorithm::resampling_linear,
          src_md, diff_dst_md);
      auto fwd_pd =
          dnnl::resampling_forward::primitive_desc(fwd_desc, onednn_engine);
      auto bwd_pd = dnnl::resampling_backward::primitive_desc(
          bwd_desc, attr, onednn_engine, fwd_pd);
#endif
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

      dnnl::memory diff_dst_mem = CreateDnnlMemory(
          diff_dst_md, onednn_engine, GetTensorBuffer<float>(&diff_dst_tensor));
      Tensor diff_dst_reorder_tensor;
      dnnl::memory diff_dst_reorder_mem;
      dnnl::memory::desc diff_dst_desc = bwd_pd.diff_dst_desc();
      bool is_diff_dst_reordered =
          reorder_memory(diff_dst_mem, diff_dst_desc, diff_dst_reorder_tensor,
                         diff_dst_reorder_mem);

      dnnl::memory src_mem = dnnl::memory(src_md, onednn_engine,
                                          GetTensorBuffer<T>(&original_image));

      OP_REQUIRES_OK(context, context->allocate_output(0, diff_src_tf_shape,
                                                       &diff_src_tensor));

      dnnl::memory diff_src_mem =
          dnnl::memory(bwd_pd.diff_src_desc(), onednn_engine,
                       GetTensorBuffer<T>(diff_src_tensor));

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> bwd_primitive_args = {
          {DNNL_ARG_DIFF_DST,
           is_diff_dst_reordered ? diff_dst_reorder_mem : diff_dst_mem},
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_DIFF_SRC, diff_src_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};

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

#define REGISTER_KERNEL(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXResizeBilinear").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ResizeBilinearOp<CPUDevice, T>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

#define REGISTER_GRAD_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("_ITEXResizeBilinearGrad") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T"),    \
                          ResizeBilinearGradOp<CPUDevice, T>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_GRAD_KERNEL);
#undef REGISTER_KERNEL
#undef REGISTER_GRAD_KENREL
}  // namespace itex
