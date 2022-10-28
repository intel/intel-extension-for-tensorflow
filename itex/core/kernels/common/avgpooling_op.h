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

#ifndef ITEX_CORE_KERNELS_COMMON_AVGPOOLING_OP_H_
#define ITEX_CORE_KERNELS_COMMON_AVGPOOLING_OP_H_
#include <string>
#include <unordered_map>

#include "itex/core/kernels/common/pooling_ops_common.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename Device, typename T, dnnl::prop_kind prop>
class AvgPoolGradOp : public PoolingBackwardOpBase<T> {
 public:
  explicit AvgPoolGradOp(OpKernelConstruction* context)
      : PoolingBackwardOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const Tensor& orig_input_tensor =
          context->input(this->kInputTensorIndexInputShape);
      const Tensor& grad_tensor =
          context->input(this->kInputTensorIndexInputGradient);

      auto shape_vec = orig_input_tensor.vec<int32>();
      TensorShape orig_input_shape;
      for (int i = 0; i < orig_input_tensor.NumElements(); i++) {
        orig_input_shape.AddDim(shape_vec(i));
      }

      OneDnnPoolParameters pool_params;

      TensorShape grad_shape = grad_tensor.shape();

      bool is_pool2d = (this->ksize_.size() == 4);
      OneDnnTensorFormat tensor_format_onednn =
          TFDataFormatToOneDnnDataFormat(this->data_format_tf_, is_pool2d);
      this->data_format_onednn_ = OneDnnTensorFormatToTag(tensor_format_onednn);

      this->InitPoolParameters(context, &pool_params, orig_input_shape,
                               this->padding_list_);

      dnnl::memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right, is_pool2d);

      dnnl::memory::dims orig_input_dims_order = TFShapeToOneDnnDimsInNC(
          orig_input_shape, this->data_format_tf_, is_pool2d);
      dnnl::memory::dims diff_dst_dims = TFShapeToOneDnnDimsInNC(
          grad_tensor.shape(), this->data_format_tf_, is_pool2d);

      dnnl::memory::desc src_md(orig_input_dims_order, OneDnnType<T>(),
                                this->data_format_onednn_);
      dnnl::memory::desc diff_dst_md(diff_dst_dims, OneDnnType<T>(),
                                     this->data_format_onednn_);
      dnnl::pooling_backward::desc pooling_bwd_desc(
          dnnl::algorithm::pooling_avg_exclude_padding, src_md, diff_dst_md,
          strides, filter_dims, padding_left, padding_right);
      dnnl::pooling_forward::desc pooling_fwd_desc(
          prop, dnnl::algorithm::pooling_avg_exclude_padding, src_md,
          diff_dst_md, strides, filter_dims, padding_left, padding_right);
      dnnl::pooling_forward::primitive_desc pooling_fwd_pd(pooling_fwd_desc,
                                                           onednn_engine);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      dnnl::pooling_backward::primitive_desc pooling_bwd_pd(
          pooling_bwd_desc, attr, onednn_engine, pooling_fwd_pd);

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          pooling_bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(pooling_bwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      dnnl::pooling_backward pooling_bwd_primitive(pooling_bwd_pd);

      Tensor* output_tensor = nullptr;
      this->AllocateOutputTensor(context, &orig_input_shape, &output_tensor);
      ITEX_DCHECK(output_tensor);

      T* diff_src_data = output_tensor->flat<T>().data();
      T* diff_dst_data =
          static_cast<T*>(const_cast<T*>(grad_tensor.flat<T>().data()));

      auto diff_src_mem =
          CreateDnnlMemory(pooling_bwd_pd.diff_src_desc(), onednn_engine,
                           static_cast<void*>(diff_src_data));

      auto diff_dst_mem =
          CreateDnnlMemory(pooling_bwd_pd.diff_dst_desc(), onednn_engine,
                           static_cast<void*>(diff_dst_data));

      std::unordered_map<int, dnnl::memory> bwd_net_args(
          {{DNNL_ARG_DIFF_SRC, diff_src_mem},
           {DNNL_ARG_DIFF_DST, diff_dst_mem},
           {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      pooling_bwd_primitive.execute(onednn_stream, bwd_net_args);
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
  const int kInputTensorIndexInputShape = 0;
  const int kInputTensorIndexInputGradient = 1;
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_AVGPOOLING_OP_H_
