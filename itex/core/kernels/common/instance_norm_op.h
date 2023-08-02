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

#ifndef ITEX_CORE_KERNELS_COMMON_INSTANCE_NORM_OP_H_
#define ITEX_CORE_KERNELS_COMMON_INSTANCE_NORM_OP_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename Device, typename T, typename U, bool fuse_activation = false>
class InstanceNormOp : public OpKernel {
 public:
  explicit InstanceNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));

    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));

    if (fuse_activation) {
      std::vector<string> fused_ops;
      string activation_mode_;
      OP_REQUIRES_OK(context,
                     context->GetAttr("activation_mode", &activation_mode_));

      if (activation_mode_ == "Relu") {
        leakyrelu_alpha_ = 0.0f;
      } else if (activation_mode_ == "LeakyRelu") {
        if (!Eigen::internal::is_same<Device, CPUDevice>::value) {
          OP_REQUIRES(
              context, false,
              errors::InvalidArgument("_OneDnnFusedInstanceNorm kernel do "
                                      "not support leakyrelu fusion on GPU"));
        }

        OP_REQUIRES_OK(context,
                       context->GetAttr("leakyrelu_alpha", &leakyrelu_alpha_));
      } else {
        OP_REQUIRES(
            context, false,
            errors::Unimplemented("_OneDnnFusedInstanceNorm activation_mode "
                                  "only support Relu and LeakyRelu"));
      }
    }
    is_inplace_ = false;
    if (context->HasAttr("is_inplace")) {
      OP_REQUIRES_OK(context, context->GetAttr("is_inplace", &is_inplace_));
    }
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      const size_t kSrcIndex = 0;    // index of src input tensor
      const size_t kScaleIndex = 1;  // index of scale tensor
      const size_t kShiftIndex = 2;  // index of shift tensor
      const Tensor& src_tensor = context->input(kSrcIndex);
      const Tensor& scale_tensor = context->input(kScaleIndex);
      const Tensor& shift_tensor = context->input(kShiftIndex);

      TensorShape src_tf_shape = src_tensor.shape();
      const int ndims = src_tf_shape.dims();

      OP_REQUIRES(context, ndims == 4 || ndims == 5,
                  errors::InvalidArgument(
                      "input must be 4-dimensional or 5-dimensional",
                      src_tensor.shape().DebugString()));

      const int batch_size = src_tensor.shape().dim_size(0);
      const int64_t elems_per_batch =
          src_tensor.shape().num_elements() / batch_size;

      // Handle the special case: input with 0 element and 0 layer size.
      Tensor* dst_tensor = nullptr;
      TensorShape workspace_tf_shape;
      if (src_tf_shape.num_elements() == 0) {
        workspace_tf_shape.AddDim(0);
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {0}, 0, src_tf_shape, &dst_tensor));
        ITEX_DCHECK(dst_tensor);
        return;
      } else {
        if (is_inplace_) {
          context->set_output(0, src_tensor);
          dst_tensor = context->mutable_output(0);
        } else {
          OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                      {0}, 0, src_tensor.shape(), &dst_tensor));
        }
      }

      int num_elements_scale = scale_tensor.dim_size(0);
      int num_elements_shift = shift_tensor.dim_size(0);
      if (scale_tensor.dims() > 1 && shift_tensor.dims() > 1) {
        if (data_format == "NCHW" || data_format == "NCDHW") {
          num_elements_scale = scale_tensor.dim_size(1);
          num_elements_shift = shift_tensor.dim_size(1);
        } else {
          int dims = scale_tensor.dims();
          num_elements_scale = scale_tensor.dim_size(dims - 1);
          num_elements_shift = shift_tensor.dim_size(dims - 1);
        }
      }

      OP_REQUIRES(
          context, num_elements_scale == num_elements_shift,
          errors::InvalidArgument("Number of elements in scale and shift",
                                  "tensors are not same."));

      bool use_3d_format = src_tensor.dims() == 5;

      OneDnnTensorFormat dnnl_tensor_fmt =
          TFDataFormatToOneDnnDataFormat(tensor_format_, !use_3d_format);
      dnnl::memory::format_tag dnn_fmt =
          OneDnnTensorFormatToTag(dnnl_tensor_fmt);

      // Set src memory descriptor.
      dnnl::memory::dims src_dims = TFShapeToOneDnnDimsInNC(
          src_tensor.shape(), tensor_format_, !use_3d_format);

      src_dims[0] = 1;
      auto src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(), dnn_fmt);
      auto scale_md =
          dnnl::memory::desc({static_cast<int64>(num_elements_scale)},
                             OneDnnType<U>(), dnnl::memory::format_tag::a);
      auto shift_md =
          dnnl::memory::desc({static_cast<int64>(num_elements_shift)},
                             OneDnnType<U>(), dnnl::memory::format_tag::a);
      // Create fwd primitive.
      auto propagation = dnnl::prop_kind::forward_inference;
      auto flags = dnnl::normalization_flags::use_scale |
                   dnnl::normalization_flags::use_shift;

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      dnnl::batch_normalization_forward::primitive_desc bn_fwd_pd;

      if (fuse_activation) {
        dnnl::post_ops post_ops;
        post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, leakyrelu_alpha_,
                                0.0);
        attr.set_post_ops(post_ops);
      }
      bn_fwd_pd = dnnl::batch_normalization_forward::primitive_desc(
          onednn_engine, propagation, src_md, src_md, epsilon_, flags, attr);

      dnnl::batch_normalization_forward bn_fwd_primitive(bn_fwd_pd);

      void* scale_data = GetTensorBuffer<U>(&scale_tensor);
      void* shift_data = GetTensorBuffer<U>(&shift_tensor);

      // Create onednn memory.
      auto scale_mem = CreateDnnlMemory(scale_md, onednn_engine, scale_data);
      auto shift_mem = CreateDnnlMemory(shift_md, onednn_engine, shift_data);

      // Create onednn memory for input and output.
      dnnl::memory dst_mem(bn_fwd_pd.dst_desc(), onednn_engine,
                           static_cast<char*>(nullptr));
      dnnl::memory src_mem(src_md, onednn_engine, static_cast<char*>(nullptr));

      const T* src_buf_batch = const_cast<T*>(src_tensor.flat<T>().data());
      const T* dst_buf_batch = const_cast<T*>(dst_tensor->flat<T>().data());

      // Execute.
      std::unordered_map<int, dnnl::memory> args;
      args.insert({DNNL_ARG_SRC, src_mem});
      args.insert({DNNL_ARG_DST, dst_mem});
      args.insert({DNNL_ARG_SCALE, scale_mem});
      args.insert({DNNL_ARG_SHIFT, shift_mem});

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          bn_fwd_pd.scratchpad_desc().get_size() / sizeof(U);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(bn_fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<U>(&scratchpad_tensor));
      args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      // Perform batchnorm computation for each batch in input
      for (int i = 0; i < batch_size; i++) {
        src_mem.set_data_handle(static_cast<void*>(
            const_cast<T*>(src_buf_batch + i * elems_per_batch)));
        dst_mem.set_data_handle(static_cast<void*>(
            const_cast<T*>(dst_buf_batch + i * elems_per_batch)));
        bn_fwd_primitive.execute(onednn_stream, args);
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status:" + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  bool is_inplace_;
  float epsilon_;
  float leakyrelu_alpha_;
  TensorFormat tensor_format_;
  string data_format;
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_INSTANCE_NORM_OP_H_
