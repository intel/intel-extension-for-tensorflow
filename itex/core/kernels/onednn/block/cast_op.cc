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

namespace itex {

template <typename Device, typename SrcT, typename DstT>
class OneDnnCastOp : public OpKernel {
 public:
  ~OneDnnCastOp() {}
  explicit OneDnnCastOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("SrcT", &src_dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("DstT", &dst_dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("Truncate", &use_truncation_));

    OP_REQUIRES(
        context,
        ((src_dtype_ == DT_FLOAT || src_dtype_ == DT_BFLOAT16 ||
          src_dtype_ == DT_HALF) &&
         (dst_dtype_ == DT_FLOAT || dst_dtype_ == DT_BFLOAT16 ||
          dst_dtype_ == DT_HALF)),
        errors::InvalidArgument("_OneDnnCastOp supports casting between"
                                "fp32, bf16, half and vice-versa only."));
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      const Tensor& src_tensor = context->input(0);
      OneDnnShape src_onednn_shape;
      GetOneDnnShape(context, 0, &src_onednn_shape);
      TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                     ? src_onednn_shape.GetTfShape()
                                     : src_tensor.shape();

      OneDnnShape dst_onednn_shape;
      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_tf_shape.num_elements() == 0) {
        dst_onednn_shape.SetOneDnnTensor(false);
        ForwardOrAllocateOutputSetOneDnnShape(context, 0, 0, &dst_tensor,
                                              src_tf_shape, dst_onednn_shape);
        return;
      }

      dnnl::memory::dims src_dims;
      dnnl::memory::desc src_md, dst_md;
      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
        src_md = src_onednn_shape.GetOneDnnLayout();
        // OneDNN 3.0 doesn't support format::any as dst format in Reorder,
        // so simply set src TF format to it.
        // FIXME(itex): Change it to format::any to propagate block format
        //              to next op once oneDNN has suppported it.
        dst_md = dnnl::memory::desc(src_dims, OneDnnType<DstT>(),
                                    src_onednn_shape.GetFormatTag());
      } else {
        src_dims = TFShapeToOneDnnDims(src_tf_shape);
        src_md = CreatePlainMemDescWithFormatTag<SrcT>(src_dims);
        dst_md = CreatePlainMemDescWithFormatTag<DstT>(src_dims);
      }
      auto reorder_pd = dnnl::reorder::primitive_desc(onednn_engine, src_md,
                                                      onednn_engine, dst_md);
      auto reorder_primitive = dnnl::reorder(reorder_pd);

      OneDnnShape output_onednn_shape;
      TensorShape output_tf_shape = src_tf_shape;
      SetOutputTensorShape(reorder_pd.dst_desc(),
                           src_onednn_shape.GetTfDataFormat(), &output_tf_shape,
                           &output_onednn_shape,
                           src_onednn_shape.IsOneDnnTensor());
      AllocateOutputSetOneDnnShape(context, 0, &dst_tensor, output_tf_shape,
                                   output_onednn_shape);

      auto src_mem = CreateDnnlMemory(src_md, onednn_engine,
                                      GetTensorBuffer<SrcT>(&src_tensor));
      auto dst_mem = CreateDnnlMemory(dst_md, onednn_engine,
                                      GetTensorBuffer<DstT>(dst_tensor));
      dnnl::stream onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, dnnl::memory> reorder_args = {
          {DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}};
      reorder_primitive.execute(onednn_stream, reorder_args);
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
  DataType src_dtype_;
  DataType dst_dtype_;
  bool use_truncation_;
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_CAST(SrcT, DstT)                            \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnCast")                \
                              .Device(DEVICE_GPU)            \
                              .HostMemory("x_meta")          \
                              .HostMemory("y_meta")          \
                              .TypeConstraint<SrcT>("SrcT")  \
                              .TypeConstraint<DstT>("DstT"), \
                          OneDnnCastOp<GPUDevice, SrcT, DstT>);
REGISTER_CAST(float, Eigen::bfloat16);
REGISTER_CAST(float, Eigen::half);
REGISTER_CAST(Eigen::bfloat16, float);
REGISTER_CAST(Eigen::bfloat16, Eigen::half);
REGISTER_CAST(Eigen::half, float);
REGISTER_CAST(Eigen::half, Eigen::bfloat16);
#undef REGISTER_CAST
#else
#define REGISTER_CAST(SrcT, DstT)                            \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnCast")                \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<SrcT>("SrcT")  \
                              .TypeConstraint<DstT>("DstT"), \
                          OneDnnCastOp<CPUDevice, SrcT, DstT>);
REGISTER_CAST(float, Eigen::bfloat16);
REGISTER_CAST(float, Eigen::half);
REGISTER_CAST(Eigen::bfloat16, float);
REGISTER_CAST(Eigen::bfloat16, Eigen::half);
REGISTER_CAST(Eigen::half, float);
REGISTER_CAST(Eigen::half, Eigen::bfloat16);
#undef REGISTER_CAST
#endif
}  // namespace itex
