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

#ifndef ITEX_CORE_KERNELS_ONEDNN_BLOCK_ELTWISE_OP_H_
#define ITEX_CORE_KERNELS_ONEDNN_BLOCK_ELTWISE_OP_H_

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
using dnnl::eltwise_backward;
using dnnl::eltwise_forward;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;

namespace itex {
template <typename Device, typename T>
class OneDnnEltwiseBaseOp : public OpKernel {
 public:
  explicit OneDnnEltwiseBaseOp(OpKernelConstruction* context,
                               dnnl::algorithm algo, float alpha, float beta)
      : OpKernel(context), algo_(algo), alpha_(alpha), beta_(beta) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const size_t src_index = 0;  // index of src input tensor
      const size_t dst_index = 0;  // index of dst output tensor
      const Tensor& src_tensor = context->input(src_index);
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

      memory::dims src_dims;
      memory::desc src_md;

      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
        src_md = src_onednn_shape.GetOneDnnLayout();
      } else {
        src_dims = TFShapeToOneDnnDims(src_tf_shape);
        // Create `plain` onednn memory descriptor
        src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
      }

      // Create eltwise forward primitive
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      auto fwd_pd = eltwise_forward::primitive_desc(
          onednn_engine, prop_kind::forward, algo_, src_md, src_md, alpha_,
          beta_, attr);
#else
      auto fwd_desc = eltwise_forward::desc(prop_kind::forward, algo_, src_md,
                                            alpha_, beta_);
      auto fwd_pd =
          eltwise_forward::primitive_desc(fwd_desc, attr, onednn_engine);
#endif
      auto fwd_primitive = eltwise_forward(fwd_pd);

      // Create src memory, check if src needs to be reordered
      const T* src_data = src_tensor.flat<T>().data();

      // Actually we don't need check reorder code in eltwise op. Since eltwise
      // primitive is constructed from`src_md` instead of `any`, The primitive
      // memory desc is always the same as the src memory desc. Here's just demo
      // when there is inconsistency (such as conv), how to use `ReorderMemory`
      dnnl::memory src_mem = CreateDnnlMemory(
          src_md, onednn_engine, static_cast<void*>(const_cast<T*>(src_data)));

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

      // Allocate output data tensor and meta tensor
      dst_tf_shape = src_tf_shape;
      SetOutputTensorShape(
          fwd_pd.dst_desc(), src_onednn_shape.GetTfDataFormat(), &dst_tf_shape,
          &dst_onednn_shape, src_onednn_shape.IsOneDnnTensor());

      // If the input and output tensor shape is always the same, try to use
      // `ForwardOrAllocateOutputSetOneDnnShape(`. Otherwise, please use
      // `AllocateOutputSetOneDnnShape`
      ForwardOrAllocateOutputSetOneDnnShape(context, src_index, dst_index,
                                            &dst_tensor, dst_tf_shape,
                                            dst_onednn_shape);

      // Create dst memory
      T* dst_data = dst_tensor->flat<T>().data();
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      static_cast<void*>(dst_data));

      Tensor scratchpad_tensor;
      int64 scratchpad_size = fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      // execute eltwise
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
  dnnl::algorithm algo_;
  float alpha_;
  float beta_;
};

template <typename Device, typename T>
class OneDnnEltwiseGradBaseOp : public OpKernel {
 public:
  ~OneDnnEltwiseGradBaseOp() {}

  explicit OneDnnEltwiseGradBaseOp(OpKernelConstruction* context,
                                   dnnl::algorithm algo, float alpha,
                                   float beta)
      : OpKernel(context), algo_(algo), alpha_(alpha), beta_(beta) {}

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
    auto onednn_engine = CreateDnnlEngine<Device>(*context);

    const size_t diff_dst_index = GetDiffDstIndex();
    const size_t src_index = GetSrcIndex();
    const size_t diff_src_index = GetDiffSrcIndex();

    const Tensor& src_tensor = context->input(src_index);
    const Tensor& diff_dst_tensor = context->input(diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    OneDnnShape src_onednn_shape, diff_dst_onednn_shape;
    GetOneDnnShape(context, src_index, &src_onednn_shape);
    GetOneDnnShape(context, diff_dst_index, &diff_dst_onednn_shape);
    TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                   ? src_onednn_shape.GetTfShape()
                                   : src_tensor.shape();

    TensorShape diff_src_tf_shape;
    OneDnnShape diff_src_dnn_shape;
    // Nothing to compute, return.
    if (src_tensor.shape().num_elements() == 0) {
      diff_src_dnn_shape.SetOneDnnTensor(false);
      diff_src_tf_shape = src_tf_shape;
      ForwardOrAllocateOutputSetOneDnnShape(context, src_index, diff_src_index,
                                            &diff_src_tensor, diff_src_tf_shape,
                                            diff_src_dnn_shape);
      return;
    }

    try {
      // get a eltwise bwd from primitive pool
      memory::dims src_dims = {};
      memory::desc src_md({}, memory::data_type::undef,
                          memory::format_tag::undef);
      memory::dims diff_dst_dims = {};
      memory::desc diff_dst_md({}, memory::data_type::undef,
                               memory::format_tag::undef);
      if (!src_onednn_shape.IsOneDnnTensor() &&
          !diff_dst_onednn_shape.IsOneDnnTensor()) {
        src_dims = TFShapeToOneDnnDims(src_tensor.shape());
        src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
        diff_dst_md = src_md;
      } else if (src_onednn_shape.IsOneDnnTensor() &&
                 !diff_dst_onednn_shape.IsOneDnnTensor()) {
        src_md = src_onednn_shape.GetOneDnnLayout();
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();

        OneDnnTensorFormat src_onednn_data_format =
            src_onednn_shape.GetTfDataFormat();
        if (diff_dst_tensor.dims() == 4 || diff_dst_tensor.dims() == 5) {
          auto src_tf_data_format =
              OneDnnDataFormatToTFDataFormat(src_onednn_data_format);
          diff_dst_dims = TFShapeToOneDnnDimsInNC(diff_dst_tensor.shape(),
                                                  src_tf_data_format,
                                                  diff_dst_tensor.dims() == 4);
          diff_dst_md =
              memory::desc(diff_dst_dims, OneDnnType<T>(),
                           OneDnnTensorFormatToTag(src_onednn_data_format));
        } else {
          diff_dst_dims = TFShapeToOneDnnDims(diff_dst_tensor.shape());
          diff_dst_md = CreatePlainMemDescWithFormatTag<T>(diff_dst_dims);
        }
      } else if (!src_onednn_shape.IsOneDnnTensor() &&
                 diff_dst_onednn_shape.IsOneDnnTensor()) {
        diff_dst_md = diff_dst_onednn_shape.GetOneDnnLayout();

        OneDnnTensorFormat diff_dst_onednn_data_format =
            diff_dst_onednn_shape.GetTfDataFormat();
        if (src_tensor.dims() == 4 || src_tensor.dims() == 5) {
          auto diff_dst_tf_data_format =
              OneDnnDataFormatToTFDataFormat(diff_dst_onednn_data_format);

          src_dims = TFShapeToOneDnnDimsInNC(src_tensor.shape(),
                                             diff_dst_tf_data_format,
                                             src_tensor.dims() == 4);
          src_md = memory::desc(
              src_dims, OneDnnType<T>(),
              OneDnnTensorFormatToTag(diff_dst_onednn_data_format));
        } else {
          src_dims = TFShapeToOneDnnDims(src_tensor.shape());
          src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
        }
      } else {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
        src_md = src_onednn_shape.GetOneDnnLayout();
        diff_dst_md = diff_dst_onednn_shape.GetOneDnnLayout();
      }

      // Temporarily fix ReluGrad for CPU
      // If one of the input is in BLOCK format, reorder the other one as BLOCK
      // Otherwise the primitive will run into the reference path.
      // Also for GPU, GPU would crash.
      // TODO(itex): Remove this fix once the issue is solved by OneDNN
      memory::desc _src_md({}, memory::data_type::undef,
                           memory::format_tag::undef);
      memory::desc _diff_dst_md({}, memory::data_type::undef,
                                memory::format_tag::undef);
      _src_md = src_md;
      _diff_dst_md = diff_dst_md;
      if (src_onednn_shape.IsOneDnnTensor() &&
          !diff_dst_onednn_shape.IsOneDnnTensor()) {
        _diff_dst_md = _src_md;
      } else if (!src_onednn_shape.IsOneDnnTensor() &&
                 diff_dst_onednn_shape.IsOneDnnTensor()) {
        _src_md = _diff_dst_md;
      }

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      auto fwd_pd = eltwise_forward::primitive_desc(
          onednn_engine, prop_kind::forward_training, algo_, _src_md, _src_md,
          alpha_, beta_, attr);
      auto eltwise_bwd_pd = eltwise_backward::primitive_desc(
          onednn_engine, algo_, _src_md, _src_md, _src_md, alpha_, beta_,
          fwd_pd, attr);
#else
#ifdef INTEL_CPU_ONLY
      auto fwd_desc = eltwise_forward::desc(prop_kind::forward_training, algo_,
                                            _src_md, alpha_, beta_);
      auto fwd_pd =
          eltwise_forward::primitive_desc(fwd_desc, attr, onednn_engine);
      auto bwd_desc =
          eltwise_backward::desc(algo_, _diff_dst_md, _src_md, alpha_, beta_);
#else
      auto fwd_desc = eltwise_forward::desc(prop_kind::forward_training, algo_,
                                            src_md, alpha_, beta_);
      auto fwd_pd =
          eltwise_forward::primitive_desc(fwd_desc, attr, onednn_engine);
      auto bwd_desc =
          eltwise_backward::desc(algo_, diff_dst_md, src_md, alpha_, beta_);
#endif  // INTEL_CPU_ONLY

      auto eltwise_bwd_pd = eltwise_backward::primitive_desc(
          bwd_desc, attr, onednn_engine, fwd_pd);
#endif
      auto eltwise_bwd_primitive = eltwise_backward(eltwise_bwd_pd);

      dnnl::memory src_mem = CreateDnnlMemory(src_md, onednn_engine,
                                              GetTensorBuffer<T>(&src_tensor));
      dnnl::memory diff_dst_mem = CreateDnnlMemory(
          diff_dst_md, onednn_engine, GetTensorBuffer<T>(&diff_dst_tensor));

      // check whether need reorder for src / diff_dst
      dnnl::memory src_reorder_mem;
      Tensor src_reorder_tensor;
      bool is_src_reordered = (src_md != eltwise_bwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size =
            eltwise_bwd_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));
        src_reorder_mem =
            CreateDnnlMemory(eltwise_bwd_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<T>(&src_reorder_tensor));
        ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
      }

      dnnl::memory diff_dst_reorder_mem;
      Tensor diff_dst_reorder_tensor;
      bool is_diff_dst_reordered =
          (diff_dst_md != eltwise_bwd_pd.diff_dst_desc());
      if (is_diff_dst_reordered) {
        int64 diff_dst_reorder_size =
            eltwise_bwd_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::v(),
                                    TensorShape({diff_dst_reorder_size}),
                                    &diff_dst_reorder_tensor));
        diff_dst_reorder_mem =
            CreateDnnlMemory(eltwise_bwd_pd.diff_dst_desc(), onednn_engine,
                             GetTensorBuffer<T>(&diff_dst_reorder_tensor));
        ReorderMemory(*context, &diff_dst_mem, &diff_dst_reorder_mem,
                      onednn_engine);
      }

      // allocate diff_src tensor
      if (src_onednn_shape.IsOneDnnTensor() ||
          diff_dst_onednn_shape.IsOneDnnTensor()) {
        auto diff_src_pd = eltwise_bwd_pd.diff_src_desc();
        diff_src_dnn_shape.SetOneDnnTensor(true);
        diff_src_dnn_shape.SetOneDnnLayout(diff_src_pd);
        if (src_onednn_shape.IsOneDnnTensor()) {
          diff_src_dnn_shape.SetTfDataFormat(
              src_onednn_shape.GetTfDataFormat());
        } else {
          diff_src_dnn_shape.SetTfDataFormat(
              diff_dst_onednn_shape.GetTfDataFormat());
        }
        diff_src_tf_shape.AddDim(diff_src_pd.get_size() / sizeof(T));
      } else {
        diff_src_dnn_shape.SetOneDnnTensor(false);
        diff_src_tf_shape = src_tensor.shape();
      }
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {static_cast<int>(src_index)}, diff_src_index,
                                  diff_src_tf_shape, &diff_src_tensor));
      AllocateMetaData(context, diff_src_index, diff_src_dnn_shape);

      dnnl::memory diff_src_mem =
          CreateDnnlMemory(eltwise_bwd_pd.diff_src_desc(), onednn_engine,
                           GetTensorBuffer<T>(diff_src_tensor));

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          eltwise_bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(eltwise_bwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      // execute eltwise bwd
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> bwd_primitives_args = {
          {GetTypeOfInputTensorFromFwdOp(),
           is_src_reordered ? src_reorder_mem : src_mem},
          {DNNL_ARG_DIFF_DST,
           is_diff_dst_reordered ? diff_dst_reorder_mem : diff_dst_mem},
          {DNNL_ARG_DIFF_SRC, diff_src_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
      eltwise_bwd_primitive.execute(onednn_stream, bwd_primitives_args);
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
  dnnl::algorithm algo_;
  float alpha_;
  float beta_;
};
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_ONEDNN_BLOCK_ELTWISE_OP_H_
