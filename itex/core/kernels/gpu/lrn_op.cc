/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename T>
class LRNOp : public OpKernel {
 public:
  ~LRNOp() {}

  explicit LRNOp(OpKernelConstruction* context) : OpKernel(context) {
    int64 depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(
        context,
        FastBoundsCheck(depth_radius64, std::numeric_limits<int>::max()),
        errors::InvalidArgument("depth_radius = ", depth_radius64,
                                " larger than int max"));
    depth_radius_ = static_cast<size_t>(depth_radius64);

    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<GPUDevice>(*context);
      const Tensor& src_tensor = context->input(0);
      OP_REQUIRES(context, src_tensor.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional"));
      OP_REQUIRES(context,
                  FastBoundsCheck(src_tensor.NumElements(),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("argument to LRN too large"));

      // OneDNN has a notion of kernel_size and not depth_radius.
      int kernel_size = 2 * depth_radius_ + 1;
      float new_alpha = alpha_ * kernel_size;
      TensorShape src_tf_shape = src_tensor.shape();

      // Create memory for user input.
      // Since Tensorflow always performs normalization over last dimension,
      // and OneDNN performs normalization over Channel, we tell OneDNN
      // that input is in NHWC layout with Channel being the last dimension.
      auto src_dims = TFShapeToOneDnnDimsInNC(src_tf_shape, FORMAT_NHWC);
      auto src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(),
                                       dnnl::memory::format_tag::nhwc);

      // Create LRN primitive descriptor.
      // Tensorflow's normalization semantics is across channels.
      // OneDNN also supports normalization within channel.
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      auto fwd_pd = dnnl::lrn_forward::primitive_desc(
          onednn_engine, dnnl::prop_kind::forward,
          dnnl::algorithm::lrn_across_channels, src_md, src_md, kernel_size,
          new_alpha, beta_, bias_, attr);
#else
      auto lrn_desc = dnnl::lrn_forward::desc(
          dnnl::prop_kind::forward, dnnl::algorithm::lrn_across_channels,
          src_md, kernel_size, new_alpha, beta_, bias_);
      auto fwd_pd =
          dnnl::lrn_forward::primitive_desc(lrn_desc, attr, onednn_engine);
#endif
      dnnl::primitive fwd_primitive = dnnl::lrn_forward(fwd_pd);

      Tensor scratchpad_tensor;
      int64 scratchpad_size = fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      T* src_data =
          static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      auto src_mem = CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                                      static_cast<void*>(src_data));

      Tensor* dst_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(),
                                                       &dst_tensor));

      T* dst_data = dst_tensor->flat<T>().data();
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      static_cast<void*>(dst_data));

      Tensor ws_tensor;
      int64 ws_size = fwd_pd.workspace_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                          TensorShape({ws_size}), &ws_tensor));
      T* ws_data = ws_tensor.flat<T>().data();
      auto ws_mem = CreateDnnlMemory(fwd_pd.workspace_desc(), onednn_engine,
                                     static_cast<void*>(ws_data));

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, dnnl::memory> fwd_primitive_args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_DST, dst_mem},
          {DNNL_ARG_WORKSPACE, ws_mem},
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

 private:
  int depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

template <typename T>
class LRNGradOp : public OpKernel {
 public:
  explicit LRNGradOp(OpKernelConstruction* context) : OpKernel(context) {
    int64 depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(
        context,
        FastBoundsCheck(depth_radius64, std::numeric_limits<int>::max()),
        errors::InvalidArgument("depth_radius = ", depth_radius64,
                                " larger than int max"));
    depth_radius_ = static_cast<int>(depth_radius64);
    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<GPUDevice>(*context);
      const int kIdxGradient = 0, kIdxOrigInput = 1;

      const Tensor& diff_dst_tensor = context->input(kIdxGradient);
      const Tensor& src_tensor = context->input(kIdxOrigInput);
      const TensorShape& src_tf_shape = src_tensor.shape();

      OP_REQUIRES(
          context, diff_dst_tensor.dims() == 4,
          errors::InvalidArgument("input gradient must be 4-dimensional"));

      OP_REQUIRES(context, src_tensor.dims() == 4,
                  errors::InvalidArgument("input images must be "
                                          "4-dimensional"));

      // // OneDNN has a notion of kernel_size and not depth_radius.
      int kernel_size = 2 * depth_radius_ + 1;
      float new_alpha = alpha_ * kernel_size;

      // Create memory for user input.
      // Since Tensorflow always performs normalization over last dimension,
      // and OneDNN performs normalization over Channel, we tell OneDNN
      // that input is in NHWC layout with Channel being the last dimension.
      auto src_dims = TFShapeToOneDnnDimsInNC(src_tf_shape, FORMAT_NHWC);
      auto src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(),
                                       dnnl::memory::format_tag::nhwc);

      // Create LRN primitive descriptor.
      // Tensorflow's normalization semantics is across channels.
      // OneDNN also supports normalization within channel.
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      auto fwd_pd = dnnl::lrn_forward::primitive_desc(
          onednn_engine, dnnl::prop_kind::forward,
          dnnl::algorithm::lrn_across_channels, src_md, src_md, kernel_size,
          new_alpha, beta_, bias_, attr);

      auto bwd_pd = dnnl::lrn_backward::primitive_desc(
          onednn_engine, dnnl::algorithm::lrn_across_channels, src_md, src_md,
          src_md, kernel_size, new_alpha, beta_, bias_, fwd_pd, attr);
#else
      auto fwd_desc = dnnl::lrn_forward::desc(
          dnnl::prop_kind::forward, dnnl::algorithm::lrn_across_channels,
          src_md, kernel_size, new_alpha, beta_, bias_);
      auto fwd_pd =
          dnnl::lrn_forward::primitive_desc(fwd_desc, attr, onednn_engine);

      auto bwd_desc = dnnl::lrn_backward::desc(
          dnnl::algorithm::lrn_across_channels, src_md, src_md, kernel_size,
          new_alpha, beta_, bias_);
      auto bwd_pd = dnnl::lrn_backward::primitive_desc(bwd_desc, attr,
                                                       onednn_engine, fwd_pd);
#endif
      dnnl::primitive fwd_primitive = dnnl::lrn_forward(fwd_pd);
      dnnl::primitive bwd_primitive = dnnl::lrn_backward(bwd_pd);

      Tensor fwd_scratchpad_tensor;
      int64 fwd_scratchpad_size =
          fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({fwd_scratchpad_size}),
                                            &fwd_scratchpad_tensor));
      auto fwd_scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&fwd_scratchpad_tensor));

      T* src_data =
          static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      auto src_mem = CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                                      static_cast<void*>(src_data));

      Tensor dst_tensor_temp;
      int64 dst_tmp_size = fwd_pd.dst_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({dst_tmp_size}),
                                            &dst_tensor_temp));
      T* dst_tmp_data = dst_tensor_temp.flat<T>().data();
      auto dst_tmp_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                          static_cast<void*>(dst_tmp_data));

      Tensor ws_tensor;
      int64 ws_size = fwd_pd.workspace_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                          TensorShape({ws_size}), &ws_tensor));
      T* ws_data = ws_tensor.flat<T>().data();
      auto ws_mem = CreateDnnlMemory(fwd_pd.workspace_desc(), onednn_engine,
                                     static_cast<void*>(ws_data));

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, dnnl::memory> fwd_primitive_args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_DST, dst_tmp_mem},
          {DNNL_ARG_WORKSPACE, ws_mem},
          {DNNL_ARG_SCRATCHPAD, fwd_scratchpad_mem}};
      fwd_primitive.execute(onednn_stream, fwd_primitive_args);

      // Backward memory
      T* diff_dst_data =
          static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
      auto diff_dst_mem = CreateDnnlMemory(bwd_pd.src_desc(), onednn_engine,
                                           static_cast<void*>(diff_dst_data));

      Tensor bwd_scratchpad_tensor;
      int64 bwd_scratchpad_size =
          bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({bwd_scratchpad_size}),
                                            &bwd_scratchpad_tensor));
      auto bwd_scratchpad_mem =
          dnnl::memory(bwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&bwd_scratchpad_tensor));

      Tensor* diff_src_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(),
                                                       &diff_src_tensor));

      T* diff_src_data = diff_src_tensor->flat<T>().data();
      auto diff_src_mem = CreateDnnlMemory(bwd_pd.dst_desc(), onednn_engine,
                                           static_cast<void*>(diff_src_data));

      std::unordered_map<int, dnnl::memory> bwd_primitive_args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_DIFF_DST, diff_dst_mem},
          {DNNL_ARG_WORKSPACE, ws_mem},
          {DNNL_ARG_SCRATCHPAD, bwd_scratchpad_mem},
          {DNNL_ARG_DIFF_SRC, diff_src_mem},
      };
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

  int depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

#define REGISTER_LRN(T)    \
  REGISTER_KERNEL_BUILDER( \
      Name("LRN").Device(DEVICE_GPU).TypeConstraint<T>("T"), LRNOp<T>);
#define REGISTER_LRN_GRAD(T)                                     \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("LRNGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LRNGradOp<T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_LRN);
TF_CALL_float(REGISTER_LRN_GRAD);
TF_CALL_bfloat16(REGISTER_LRN_GRAD);
#undef REGISTER_LRN
#undef REGISTER_LRN_GRAD

}  // namespace itex
