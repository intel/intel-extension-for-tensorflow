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

#ifndef ITEX_CORE_KERNELS_COMMON_SOFTMAX_OP_H_
#define ITEX_CORE_KERNELS_COMMON_SOFTMAX_OP_H_

#include <string>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/str_util.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
// Softmax op implementation is based on OneDnn kernel
template <typename Device, typename T>
class SoftmaxOp : public OpKernel {
 public:
  ~SoftmaxOp() {}

  explicit SoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {
    is_inplace_ = false;
    if (context->HasAttr("is_inplace")) {
      OP_REQUIRES_OK(context, context->GetAttr("is_inplace", &is_inplace_));
    }
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      const int kSrcIndex = 0;
      const Tensor& src_tensor = context->input(kSrcIndex);
      auto src_tf_shape = src_tensor.shape();
      const int input_dims = src_tf_shape.dims();
      dnnl::memory::dims src_dims = TFShapeToOneDnnDims(src_tf_shape);
      int axis = input_dims - 1;
      auto src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      auto fwd_desc = dnnl::softmax_forward::desc(
          dnnl::prop_kind::forward_training, src_md, axis);
      auto fwd_pd =
          dnnl::softmax_forward::primitive_desc(fwd_desc, attr, onednn_engine);
      auto src_mem =
          dnnl::memory(src_md, onednn_engine, GetTensorBuffer<T>(&src_tensor));

      // Prepare for creating output tensor.
      Tensor* output_tensor = nullptr;
      if (is_inplace_) {
        context->set_output(0, src_tensor);
        output_tensor = context->mutable_output(0);
      } else {
        OP_REQUIRES_OK(
            context, context->allocate_output(0, src_tf_shape, &output_tensor));
      }
      auto dst_mem = dnnl::memory(fwd_pd.dst_desc(), onednn_engine,
                                  GetTensorBuffer<T>(output_tensor));

      // Prepare for creating scratchpad tensor.
      Tensor scratchpad_tensor;
      int64 scratchpad_size = fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      auto softmax_fwd = dnnl::softmax_forward(fwd_pd);
      softmax_fwd.execute(onednn_stream,
                          {
                              {DNNL_ARG_SRC, src_mem},
                              {DNNL_ARG_DST, dst_mem},
                              {DNNL_ARG_SCRATCHPAD, scratchpad_mem},
                          });
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
  bool is_inplace_;
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_SOFTMAX_OP_H_
