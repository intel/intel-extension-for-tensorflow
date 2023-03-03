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

#ifndef ITEX_CORE_KERNELS_COMMON_ELTWISE_BASE_H_
#define ITEX_CORE_KERNELS_COMMON_ELTWISE_BASE_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "itex/core/utils/onednn/onednn_util.h"

using dnnl::algorithm;
using dnnl::eltwise_backward;
using dnnl::eltwise_forward;
using dnnl::engine;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;

namespace itex {

template <typename Device, typename T>
class EltwiseBaseOp : public OpKernel {
 public:
  explicit EltwiseBaseOp(OpKernelConstruction* ctx, dnnl::algorithm algo,
                         float alpha, float beta)
      : OpKernel(ctx), alg_kind_(algo), alpha_(alpha), beta_(beta) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      // index of src input tensor: "features"
      const size_t kSrcIndex = 0;
      // index of dst output tensor: "activations"
      const size_t kDstIndex = 0;
      const Tensor& src_tensor = context->input(kSrcIndex);

      if (std::is_same<T, qint8>::value) {
        OP_REQUIRES(
            context, src_tensor.NumElements() % 4 == 0,
            errors::InvalidArgument(
                "Tensor size must be a multiple of 4 for Relu<qint8>. Got ",
                src_tensor.NumElements()));
      }

      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_tensor.shape().num_elements() == 0) {
        OP_REQUIRES_OK(context,
                       context->allocate_output(kDstIndex, src_tensor.shape(),
                                                &dst_tensor));
        return;
      }

      // memory desc
      memory::desc src_md({}, memory::data_type::undef,
                          memory::format_tag::undef);
      memory::dims src_dims = TFShapeToOneDnnDims(src_tensor.shape());
      src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      // Create an eltwise forward descriptor and primitive descriptor
#ifdef ITEX_ONEDNN_3_0
      eltwise_forward::primitive_desc fwd_pd(onednn_engine, prop_kind::forward,
                                             alg_kind_, src_md, src_md, alpha_,
                                             beta_, attr);
#else
      eltwise_forward::desc fwd_desc(prop_kind::forward, alg_kind_, src_md,
                                     alpha_, beta_);
      eltwise_forward::primitive_desc fwd_pd(fwd_desc, attr, onednn_engine);
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

      primitive fwd_primitive(fwd_pd);

      // Create memory primitive
      T* src_data =
          static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      auto src_mem = CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                                      static_cast<void*>(src_data));

      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {static_cast<const int>(kSrcIndex)},
                                  static_cast<const int>(kDstIndex),
                                  src_tensor.shape(), &dst_tensor));

      T* dst_data = dst_tensor->flat<T>().data();
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      static_cast<void*>(dst_data));

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> fwd_primitive_args = {
          {DNNL_ARG_SRC, src_mem},
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
  algorithm alg_kind_ = algorithm::eltwise_relu;
  float alpha_ = 0.0f;
  float beta_ = 0.0f;
};

template <typename Device, typename T>
class EltwiseGradBaseOp : public OpKernel {
 public:
  explicit EltwiseGradBaseOp(OpKernelConstruction* context,
                             dnnl::algorithm algo, float alpha, float beta)
      : OpKernel(context), alg_kind_(algo), alpha_(alpha), beta_(beta) {}

  // All activation functions have dy at index 0 and x at index 1. Tanh is an
  // exception, it has y at index 0 and dy at index 1.
  //
  // If forward op is defined as: y = f(x), {Relu,Elu,Relu6,LeakyRelu}
  // Grad is: z = f_grad(dy, x), TanhGrad is: z = tanh_grad(y, dy)
  //
  // Src below refers to a tensor that gradient op receives from forward
  // operator. From Relu-family ops, it is 'x'; while for TanhGrad, it is 'y'.
  virtual int GetDiffDstIndex() const = 0;
  virtual int GetSrcIndex() const = 0;
  virtual int GetDiffSrcIndex() const = 0;

  // The type of input tensor that grad op receives from forward op. For
  // example, it is DNNL_ARG_SRC for ReLU.
  virtual int GetTypeOfInputTensorFromFwdOp() const = 0;

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine_ = CreateDnnlEngine<Device>(*context);

      const size_t diff_dst_index = GetDiffDstIndex();
      const size_t src_index = GetSrcIndex();
      const size_t diff_src_index = GetDiffSrcIndex();

      const Tensor& src_tensor = context->input(src_index);
      const Tensor& diff_dst_tensor = context->input(diff_dst_index);
      Tensor* diff_src_tensor = nullptr;

      // Nothing to compute, return.
      if (src_tensor.shape().num_elements() == 0) {
        OP_REQUIRES_OK(
            context, context->allocate_output(
                         diff_src_index, context->input(diff_src_index).shape(),
                         &diff_src_tensor));
        return;
      }

      memory::dims src_dims = {};
      memory::desc src_md, diff_dst_md;

      src_dims = TFShapeToOneDnnDims(src_tensor.shape());
      src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
      diff_dst_md = src_md;

      // Create forward eltwise primitive based on src/diff_dst md
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      eltwise_forward::primitive_desc fwd_pd(
          onednn_engine_, prop_kind::forward_training, alg_kind_, src_md,
          src_md, alpha_, beta_);
      eltwise_backward::primitive_desc bwd_pd(onednn_engine_, alg_kind_, src_md,
                                              diff_dst_md, src_md, alpha_,
                                              beta_, fwd_pd, attr);
#else
      eltwise_forward::desc fwd_desc(prop_kind::forward_training, alg_kind_,
                                     src_md, alpha_, beta_);
      eltwise_forward::primitive_desc fwd_pd(fwd_desc, onednn_engine_);
      eltwise_backward::desc bwd_desc(alg_kind_, src_md, diff_dst_md, alpha_,
                                      beta_);

      eltwise_backward::primitive_desc bwd_pd(bwd_desc, attr, onednn_engine_,
                                              fwd_pd);
#endif
      Tensor scratchpad_tensor;
      int64 scratchpad_size = bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(bwd_pd.scratchpad_desc(), onednn_engine_,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      primitive bwd_primitive(bwd_pd);

      T* src_data =
          static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      memory src_mem = CreateDnnlMemory(src_md, onednn_engine_,
                                        static_cast<void*>(src_data));

      T* diff_dst_data =
          static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
      memory diff_dst_mem =
          CreateDnnlMemory(bwd_pd.diff_dst_desc(), onednn_engine_,
                           static_cast<void*>(diff_dst_data));

      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {static_cast<const int>(diff_dst_index)},
                                  static_cast<const int>(diff_src_index),
                                  src_tensor.shape(), &diff_src_tensor));

      T* diff_src_data = diff_src_tensor->flat<T>().data();
      memory diff_src_mem =
          CreateDnnlMemory(bwd_pd.diff_src_desc(), onednn_engine_,
                           static_cast<void*>(diff_src_data));

      stream onednn_stream = CreateDnnlStream(*context, onednn_engine_);
      std::unordered_map<int, memory> bwd_primitive_args = {
          {GetTypeOfInputTensorFromFwdOp(), src_mem},
          {DNNL_ARG_DIFF_DST, diff_dst_mem},
          {DNNL_ARG_DIFF_SRC, diff_src_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
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
  algorithm alg_kind_ = algorithm::eltwise_relu;
  float alpha_ = 0.0f;
  float beta_ = 0.0f;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_ELTWISE_BASE_H_
