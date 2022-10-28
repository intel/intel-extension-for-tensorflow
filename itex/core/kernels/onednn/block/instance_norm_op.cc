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

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

// oneDNN has no direct support for instancenorm, use batch_normalization
// primitive to simulate instance norm.
template <typename Device, typename T, typename U, bool fuse_activation = false>
class OneDnnInstanceNormOp : public OpKernel {
 public:
  explicit OneDnnInstanceNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
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
              errors::InvalidArgument("_OneDnnFusedInstanceNorm gpu kernel do "
                                      "not support leakyrelu fusion"));
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
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      const size_t kSrcIndex = 0;    // index of src input tensor
      const size_t kScaleIndex = 1;  // index of scale tensor
      const size_t kShiftIndex = 2;  // index of shift tensor
      const size_t kDstIndex = 0;

      const Tensor& src_tensor = context->input(kSrcIndex);
      const Tensor& scale_tensor = context->input(kScaleIndex);
      const Tensor& shift_tensor = context->input(kShiftIndex);

      OneDnnShape src_onednn_shape;
      GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
      TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                     ? src_onednn_shape.GetTfShape()
                                     : src_tensor.shape();

      OP_REQUIRES(context, src_tf_shape.dims() == 4 || src_tf_shape.dims() == 5,
                  errors::InvalidArgument(
                      "input must be 4-dimensional or 5-dimensional",
                      src_tensor.shape().DebugString()));

      // Get scale shape and shift shape. because for instancenorm fusion,
      // scale and shift could be multiple dimensional rather than 1D.
      OneDnnShape scale_onednn_shape, shift_onednn_shape;
      GetOneDnnShape(context, kScaleIndex, &scale_onednn_shape);
      GetOneDnnShape(context, kShiftIndex, &shift_onednn_shape);

      TensorShape scale_tf_shape = scale_onednn_shape.IsOneDnnTensor()
                                       ? scale_onednn_shape.GetTfShape()
                                       : scale_tensor.shape();
      TensorShape shift_tf_shape = shift_onednn_shape.IsOneDnnTensor()
                                       ? shift_onednn_shape.GetTfShape()
                                       : shift_tensor.shape();

      int num_elements_scale = scale_tf_shape.dim_size(0);
      int num_elements_shift = shift_tf_shape.dim_size(0);
      if (scale_tf_shape.dims() > 1 && shift_tf_shape.dims() > 1) {
        if (data_format == "NCHW" || data_format == "NCDHW") {
          num_elements_scale = scale_tf_shape.dim_size(1);
          num_elements_shift = shift_tf_shape.dim_size(1);
        } else {
          int dims = scale_tensor.dims();
          num_elements_scale = scale_tf_shape.dim_size(dims - 1);
          num_elements_shift = shift_tf_shape.dim_size(dims - 1);
        }
      }

      OP_REQUIRES(
          context, num_elements_scale == num_elements_shift,
          errors::InvalidArgument("Number of elements in scale and shift",
                                  "tensors are not same."));

      // Handle the special case: input with 0 element and 0 batch size.
      Tensor* dst_tensor = nullptr;
      OneDnnShape dst_onednn_shape;

      TensorShape workspace_tf_shape;
      if (src_tf_shape.num_elements() == 0) {
        size_t workspace_bytes = 0;
        workspace_tf_shape.AddDim(workspace_bytes);
        dst_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kDstIndex, &dst_tensor,
                                     src_tf_shape, dst_onednn_shape);
        ITEX_DCHECK(dst_tensor);
        // TODO(itex): replace with DeviceFill?
        DeviceMemset<Device>(
            const_cast<char*>(dst_tensor->tensor_data().data()), 0,
            dst_tensor->tensor_data().size(), context->GetDeviceStream());
        return;
      }

      // Record the inputs batch size and elements of each batch size.
      int batch_size = src_tf_shape.dim_size(0);
      const int64_t elems_per_batch = src_tf_shape.num_elements() / batch_size;

      dnnl::memory::dims src_dims;
      bool use_3d_format = src_tf_shape.dims() == 5;

      dnnl::memory::format_tag onednn_tag = dnnl::memory::format_tag::undef;
      OneDnnTensorFormat onednn_tensor_fmt;

      if (data_format == "NCHW" || data_format == "NCDHW") {
        onednn_tag = (src_tf_shape.dims() == 5)
                         ? dnnl::memory::format_tag::ncdhw
                         : dnnl::memory::format_tag::nchw;
        onednn_tensor_fmt = (src_tf_shape.dims() == 5)
                                ? OneDnnTensorFormat::FORMAT_NCDHW
                                : OneDnnTensorFormat::FORMAT_NCHW;
      } else {
        onednn_tag = (src_tf_shape.dims() == 5)
                         ? dnnl::memory::format_tag::ndhwc
                         : dnnl::memory::format_tag::nhwc;
        onednn_tensor_fmt = (src_tf_shape.dims() == 5)
                                ? OneDnnTensorFormat::FORMAT_NDHWC
                                : OneDnnTensorFormat::FORMAT_NHWC;
      }

      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
      } else {
        src_dims = TFShapeToOneDnnDimsInNC(src_tensor.shape(), tensor_format_,
                                           !use_3d_format);
      }

      auto src_input_md =
          src_onednn_shape.IsOneDnnTensor()
              ? src_onednn_shape.GetOneDnnLayout()
              : dnnl::memory::desc(src_dims, OneDnnType<T>(), onednn_tag);
      auto scale_md =
          dnnl::memory::desc({static_cast<int64>(num_elements_scale)},
                             OneDnnType<U>(), dnnl::memory::format_tag::a);
      auto shift_md =
          dnnl::memory::desc({static_cast<int64>(num_elements_shift)},
                             OneDnnType<U>(), dnnl::memory::format_tag::a);

      auto src_md_order =
          dnnl::memory::desc(src_dims, OneDnnType<T>(), onednn_tag);

      // oneDNN has no direct support for instancenorm, use a workaround
      // with performing multiple batchnorm computations for each batch
      // in the input. set batch size 1.
      src_dims[0] = 1;
      auto src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(), onednn_tag);

      void* src_data = GetTensorBuffer<T>(&src_tensor);
      auto src_input_mem =
          CreateDnnlMemory(src_input_md, onednn_engine, src_data);
      // Reorder for src tensor. use plain format to compute InstanceNorm.
      // because we use new src_dims, and src_md is the actual memory desc.
      dnnl::memory src_reorder_mem;
      Tensor src_reorder_tensor;
      if (src_onednn_shape.IsOneDnnTensor()) {
        int64 src_reorder_size = src_input_md.get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));
        src_reorder_mem =
            CreateDnnlMemory(src_md_order, onednn_engine,
                             GetTensorBuffer<T>(&src_reorder_tensor));
        ReorderMemory(*context, &src_input_mem, &src_reorder_mem,
                      onednn_engine);
      }

      // Create fwd primitive.
      auto propagation = dnnl::prop_kind::forward_inference;
      auto flags = dnnl::normalization_flags::use_scale |
                   dnnl::normalization_flags::use_shift;

      dnnl::batch_normalization_forward::desc bn_fwd_desc(propagation, src_md,
                                                          epsilon_, flags);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      dnnl::batch_normalization_forward::primitive_desc bn_fwd_pd;

      if (fuse_activation) {
        dnnl::post_ops post_ops;
        post_ops.append_eltwise(1.0, dnnl::algorithm::eltwise_relu,
                                leakyrelu_alpha_, 0.0);
        attr.set_post_ops(post_ops);
        bn_fwd_pd = dnnl::batch_normalization_forward::primitive_desc(
            bn_fwd_desc, attr, onednn_engine);
      } else {
        bn_fwd_pd = dnnl::batch_normalization_forward::primitive_desc(
            bn_fwd_desc, onednn_engine);
      }

      dnnl::batch_normalization_forward bn_fwd_primitive(bn_fwd_pd);

      // Allocate output dst tensor.
      TensorShape dst_tf_shape = src_tf_shape;
      SetOutputTensorShape(src_md_order, onednn_tensor_fmt, &dst_tf_shape,
                           &dst_onednn_shape, true);
      AllocateOutputSetOneDnnShape(context, kDstIndex, &dst_tensor,
                                   dst_tf_shape, dst_onednn_shape);

      void* scale_data = GetTensorBuffer<U>(&scale_tensor);
      void* shift_data = GetTensorBuffer<U>(&shift_tensor);

      // Create onednn memory.
      auto scale_mem = CreateDnnlMemory(scale_md, onednn_engine, scale_data);
      auto shift_mem = CreateDnnlMemory(shift_md, onednn_engine, shift_data);

      // Create onednn memory for input and output.
      auto dst_mem = dnnl::memory(bn_fwd_pd.dst_desc(), onednn_engine);
      auto src_mem = dnnl::memory(src_md, onednn_engine);

      const T* src_buf_batch =
          src_onednn_shape.IsOneDnnTensor()
              ? const_cast<T*>(src_reorder_tensor.flat<T>().data())
              : const_cast<T*>(src_tensor.flat<T>().data());
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

      // TODO(itex): Add parallel_for for both CPU and GPU
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
  float epsilon_;
  float leakyrelu_alpha_;
  TensorFormat tensor_format_;
  string data_format;
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_INSTANCE_NORM_GPU(T, U)                                 \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnInstanceNorm")                    \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("x_meta")                      \
                              .HostMemory("scale_meta")                  \
                              .HostMemory("offset_meta")                 \
                              .HostMemory("y_meta")                      \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<U>("U"),                   \
                          OneDnnInstanceNormOp<GPUDevice, T, U, false>); \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedInstanceNorm")               \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("x_meta")                      \
                              .HostMemory("scale_meta")                  \
                              .HostMemory("offset_meta")                 \
                              .HostMemory("y_meta")                      \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<U>("U"),                   \
                          OneDnnInstanceNormOp<GPUDevice, T, U, true>);
REGISTER_INSTANCE_NORM_GPU(float, float);
REGISTER_INSTANCE_NORM_GPU(Eigen::bfloat16, float);
REGISTER_INSTANCE_NORM_GPU(Eigen::half, float);
#undef REGISTER_INSTANCE_NORM_GPU
#else
#define REGISTER_INSTANCE_NORM_CPU(T, U)                                 \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnInstanceNorm")                    \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<U>("U"),                   \
                          OneDnnInstanceNormOp<CPUDevice, T, U, false>); \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedInstanceNorm")               \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<U>("U"),                   \
                          OneDnnInstanceNormOp<CPUDevice, T, U, true>);
REGISTER_INSTANCE_NORM_CPU(float, float);
REGISTER_INSTANCE_NORM_CPU(Eigen::bfloat16, float);
#undef REGISTER_INSTANCE_NORM_CPU

#endif
}  // namespace itex
