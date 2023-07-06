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

#include "itex/core/kernels/common/cast_op.h"

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
class CastOp : public OpKernel {
 public:
  ~CastOp() {}
  explicit CastOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("SrcT", &src_dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("DstT", &dst_dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("Truncate", &use_truncation_));

    OP_REQUIRES(
        context,
        ((src_dtype_ == DT_FLOAT || src_dtype_ == DT_BFLOAT16 ||
          src_dtype_ == DT_HALF) &&
         (dst_dtype_ == DT_FLOAT || dst_dtype_ == DT_BFLOAT16 ||
          dst_dtype_ == DT_HALF)),
        errors::InvalidArgument("_ITEXCastOp supports casting between"
                                "fp32, bf16, half and vice-versa only."));
  }

  void Compute(OpKernelContext* context) override {
    try {
      const Tensor& src_tensor = context->input(0);
      TensorShape src_tf_shape = src_tensor.shape();

      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_tf_shape.num_elements() == 0) {
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {0}, 0, src_tf_shape, &dst_tensor));
        return;
      }
      TensorShape output_tf_shape = src_tf_shape;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_tf_shape, &dst_tensor));

      const Device& d = context->eigen_device<Device>();
#ifdef INTEL_CPU_ONLY
      CastDataType<Device, SrcT, DstT>{}(
          d, const_cast<const Tensor&>(src_tensor).flat<SrcT>(),
          dst_tensor->flat<DstT>());
#else
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      dnnl::memory::dims src_dims;
      dnnl::memory::desc src_md, dst_md;
      src_dims = TFShapeToOneDnnDims(src_tf_shape);
      src_md = CreatePlainMemDescWithFormatTag<SrcT>(src_dims);
      dst_md = CreatePlainMemDescWithFormatTag<DstT>(src_dims);
      auto reorder_pd = dnnl::reorder::primitive_desc(onednn_engine, src_md,
                                                      onednn_engine, dst_md);
      auto reorder_primitive = dnnl::reorder(reorder_pd);

      OneDnnShape output_onednn_shape;
      TensorShape output_tf_shape = src_tf_shape;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_tf_shape, &dst_tensor));

      auto src_mem = CreateDnnlMemory(src_md, onednn_engine,
                                      GetTensorBuffer<SrcT>(&src_tensor));
      auto dst_mem = CreateDnnlMemory(dst_md, onednn_engine,
                                      GetTensorBuffer<DstT>(dst_tensor));
      dnnl::stream onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, dnnl::memory> reorder_args = {
          {DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}};
      reorder_primitive.execute(onednn_stream, reorder_args);
#endif
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

#define REGISTER_CAST(SrcT, DstT)                            \
  REGISTER_KERNEL_BUILDER(Name("_ITEXCast")                  \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<SrcT>("SrcT")  \
                              .TypeConstraint<DstT>("DstT"), \
                          CastOp<CPUDevice, SrcT, DstT>);
REGISTER_CAST(float, Eigen::bfloat16);
REGISTER_CAST(float, Eigen::half);
REGISTER_CAST(Eigen::bfloat16, float);
REGISTER_CAST(Eigen::bfloat16, Eigen::half);
REGISTER_CAST(Eigen::half, float);
REGISTER_CAST(Eigen::half, Eigen::bfloat16);
#undef REGISTER_CAST

}  // namespace itex
