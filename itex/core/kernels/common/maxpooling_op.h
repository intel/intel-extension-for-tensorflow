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

#ifndef ITEX_CORE_KERNELS_COMMON_MAXPOOLING_OP_H_
#define ITEX_CORE_KERNELS_COMMON_MAXPOOLING_OP_H_
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/kernels/common/pooling_ops_common.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
template <typename Device, typename T, dnnl::prop_kind prop>
class MaxPoolGradOp : public PoolingBackwardOpBase<T> {
 public:
  explicit MaxPoolGradOp(OpKernelConstruction* context)
      : PoolingBackwardOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const Tensor& orig_input_tensor =
          context->input(this->kInputTensorIndexOrigInput);
      const Tensor& grad_tensor =
          context->input(this->kInputTensorIndexGradient);
      OneDnnPoolParameters pool_params;

      TensorShape orig_input_shape = orig_input_tensor.shape();
      TensorShape grad_shape = grad_tensor.shape();
      std::vector<int32> ksize = this->ksize_;
      std::vector<int32> stride = this->stride_;
      if (context->num_inputs() == 5) {
        const Tensor& tensor_ksize = context->input(3);
        auto value_ksize = tensor_ksize.flat<int32>();
        ksize.resize(tensor_ksize.shape().num_elements());
        std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

        const Tensor& tensor_stride = context->input(4);
        auto value_stride = tensor_stride.flat<int32>();
        stride.resize(tensor_stride.shape().num_elements());
        std::copy_n(&value_stride(0), stride.size(), stride.begin());
      }
      this->ksize_ = ksize;
      this->stride_ = stride;
      OP_REQUIRES(
          context, ksize.size() == 4 || ksize.size() == 5,
          errors::InvalidArgument("Sliding window ksize field must "
                                  "specify 4 dimensions or 5 dimensions"));
      OP_REQUIRES(
          context, stride.size() == 4 || stride.size() == 5,
          errors::InvalidArgument("Sliding window stride field must "
                                  "specify 4 dimensions or 5 dimensions"));
      bool is_pool2d = ksize.size() == 4;
      const int32 ksize_n = GetTensorDim(ksize, this->data_format_tf_, 'N');
      const int32 stride_n = GetTensorDim(stride, this->data_format_tf_, 'N');
      OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));
      pool_params.Init(context, ksize, stride, this->padding_,
                       this->padding_list_, this->data_format_tf_,
                       orig_input_shape);

      OneDnnTensorFormat tensor_format_onednn =
          TFDataFormatToOneDnnDataFormat(this->data_format_tf_, is_pool2d);
      this->data_format_onednn_ = OneDnnTensorFormatToTag(tensor_format_onednn);

      dnnl::memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right, is_pool2d);
      dnnl::memory::dims orig_input_dims_order = TFShapeToOneDnnDimsInNC(
          orig_input_tensor.shape(), this->data_format_tf_, is_pool2d);
      dnnl::memory::dims diff_dst_dims = TFShapeToOneDnnDimsInNC(
          grad_tensor.shape(), this->data_format_tf_, is_pool2d);
      dnnl::memory::desc src_md(orig_input_dims_order, OneDnnType<T>(),
                                this->data_format_onednn_);
      dnnl::memory::desc diff_dst_md(diff_dst_dims, OneDnnType<T>(),
                                     this->data_format_onednn_);
      dnnl::pooling_backward::desc pooling_bwd_desc(
          dnnl::algorithm::pooling_max, src_md, diff_dst_md, strides,
          filter_dims, padding_left, padding_right);
      dnnl::pooling_forward::desc pooling_fwd_desc(
          prop, dnnl::algorithm::pooling_max, src_md, diff_dst_md, strides,
          filter_dims, padding_left, padding_right);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      dnnl::pooling_forward::primitive_desc pooling_fwd_pd(pooling_fwd_desc,
                                                           attr, onednn_engine);
      Tensor scratchpad_tensor_fwd;
      int64 scratchpad_size_fwd =
          pooling_fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size_fwd}),
                                            &scratchpad_tensor_fwd));
      auto scratchpad_mem_fwd =
          dnnl::memory(pooling_fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor_fwd));
      dnnl::pooling_backward::primitive_desc pooling_bwd_pd(
          pooling_bwd_desc, attr, onednn_engine, pooling_fwd_pd);
      Tensor scratchpad_tensor_bwd;
      int64 scratchpad_size_bwd =
          pooling_bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size_bwd}),
                                            &scratchpad_tensor_bwd));
      auto scratchpad_mem_bwd =
          dnnl::memory(pooling_bwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor_bwd));

      dnnl::pooling_backward pooling_bwd_primitive(pooling_bwd_pd);
      Tensor* output_tensor = nullptr;
      this->AllocateOutputTensor(context, &orig_input_shape, &output_tensor);
      ITEX_DCHECK(output_tensor);

      T* diff_src_data = output_tensor->flat<T>().data();
      T* diff_dst_data =
          static_cast<T*>(const_cast<T*>(grad_tensor.flat<T>().data()));
      void* ws_data = nullptr;

      auto diff_src_mem =
          CreateDnnlMemory(pooling_bwd_pd.diff_src_desc(), onednn_engine,
                           static_cast<void*>(diff_src_data));

      auto diff_dst_mem =
          CreateDnnlMemory(pooling_bwd_pd.diff_dst_desc(), onednn_engine,
                           static_cast<void*>(diff_dst_data));
      std::unordered_map<int, dnnl::memory> bwd_net_args(
          {{DNNL_ARG_DIFF_SRC, diff_src_mem},
           {DNNL_ARG_DIFF_DST, diff_dst_mem}});

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      /*Execute fwd primitive first to get workspace_data*/
      const Tensor& orig_output_tensor =
          context->input(this->kInputTensorIndexOrigOutput);
      T* src_data =
          static_cast<T*>(const_cast<T*>(orig_input_tensor.flat<T>().data()));
      T* dst_data =
          static_cast<T*>(const_cast<T*>(orig_output_tensor.flat<T>().data()));
      auto fwd_src_mem =
          CreateDnnlMemory(pooling_fwd_pd.src_desc(), onednn_engine,
                           static_cast<void*>(src_data));

      auto fwd_dst_mem =
          CreateDnnlMemory(pooling_fwd_pd.dst_desc(), onednn_engine,
                           static_cast<void*>(dst_data));
      std::unordered_map<int, dnnl::memory> fwd_net_args(
          {{DNNL_ARG_SRC, fwd_src_mem}, {DNNL_ARG_DST, fwd_dst_mem}});

      dnnl::pooling_forward pooling_fwd_primitive(pooling_fwd_pd);
      Tensor ws_tensor;
      TensorShape ws_tensor_shape;
      dnnl::memory::desc ws_desc = pooling_fwd_pd.workspace_desc();
      size_t ws_size = ws_desc.get_size();
      ws_tensor_shape.AddDim(ws_size);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT8, ws_tensor_shape,
                                                     &ws_tensor));

      ws_data = static_cast<void*>(ws_tensor.flat<uint8>().data());
      dnnl::memory ws_mem = CreateDnnlMemory(ws_desc, onednn_engine, ws_data);

      fwd_net_args.insert({DNNL_ARG_WORKSPACE, ws_mem});
      fwd_net_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem_fwd});
      pooling_fwd_primitive.execute(onednn_stream, fwd_net_args);
      bwd_net_args.insert({DNNL_ARG_WORKSPACE, ws_mem});
      bwd_net_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem_bwd});
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
  const int kInputTensorIndexOrigInput = 0;
  const int kInputTensorIndexOrigOutput = 1;
  const int kInputTensorIndexGradient = 2;
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_MAXPOOLING_OP_H_
