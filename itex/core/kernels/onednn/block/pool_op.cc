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

#include "itex/core/kernels/common/no_ops.h"
#include "itex/core/kernels/common/pooling_ops_common.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/register_types.h"

namespace itex {

using algorithm = dnnl::algorithm;
using pooling_forward = dnnl::pooling_forward;
using pooling_backward = dnnl::pooling_backward;
using prop_kind = dnnl::prop_kind;

template <typename Device, typename T, dnnl::algorithm alg>
class OneDnnPoolOp : public OneDnnPoolOpBase<T> {
 public:
  explicit OneDnnPoolOp(OpKernelConstruction* context)
      : OneDnnPoolOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    const int kSrcIndex = 0;
    const int kDstIndex = 0;
    const int kDstWorkspaceIndex = 1; /* only used in maxpool */
    const Tensor& src_tensor = context->input(kSrcIndex);
    OneDnnShape src_onednn_shape;
    GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
    const TensorShape& src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                          ? src_onednn_shape.GetTfShape()
                                          : src_tensor.shape();

    OP_REQUIRES(context, src_tf_shape.dims() == 4 || src_tf_shape.dims() == 5,
                errors::InvalidArgument("Input must be 4 or 5-dimensional"));

    // Initialize shape variables.
    OneDnnPoolParameters pool_params;
    pool_params.Init(context, this->ksize_, this->stride_, this->padding_,
                     this->padding_list_, this->data_format_tf_, src_tf_shape);
    OP_REQUIRES_OK(context, context->status());

    // Declare output tensor, and set OneDNN dims with NCHW/NCDHW order.
    Tensor* dst_tensor = nullptr;
    Tensor* dst_ws_tensor = nullptr; /* only used in maxpool */
    OneDnnShape dst_onednn_shape, dst_ws_onednn_shape;
    TensorShape dst_tf_shape, dst_ws_tf_shape;
    memory::dims dst_onednn_dims;
    this->GetOutputDims(pool_params, &dst_onednn_dims, &dst_tf_shape);

    // Return with TF format if nothing to compute. Need to change the
    // shape from OneDNN NCHW/NCDHW to original TF format.
    if (src_tf_shape.num_elements() == 0) {
      dst_onednn_shape.SetOneDnnTensor(false);
      AllocateOutputSetOneDnnShape(context, kDstIndex, &dst_tensor,
                                   dst_tf_shape, dst_onednn_shape);

      // int8 and fp16 only support inference
      bool only_forward_inference = std::is_same<T, qint8>::value ||
                                    std::is_same<T, quint8>::value ||
                                    std::is_same<T, Eigen::half>::value;
      if (!only_forward_inference && alg == dnnl::algorithm::pooling_max) {
        // dst_ws_tensor is not really used, so using dst_onednn_shape
        AllocateOutputSetOneDnnShape(context, kDstWorkspaceIndex,
                                     &dst_ws_tensor, dst_tf_shape,
                                     dst_onednn_shape);
      }
      return;
    }

    // Create primitive and execute op.
    // Since TF MaxPool will be NCHW/NCDHW or NHWC/NDHWC, here use dims and
    // format to describe memory to avoid explicit Reorder.
    try {
      // Create src and dst memory desc.
      auto dst_md = memory::desc(dst_onednn_dims, OneDnnType<T>(),
                                 memory::format_tag::any);
      memory::desc src_md;

      if (src_onednn_shape.IsOneDnnTensor()) {
        src_md = src_onednn_shape.GetOneDnnLayout();
      } else {
        auto src_dims = TFShapeToOneDnnDimsInNC(
            src_tensor.shape(), this->data_format_tf_, this->is_2d_);
        src_md =
            memory::desc(src_dims, OneDnnType<T>(), this->data_format_onednn_);
      }

      memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right);

      // Create forward primitive.
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      prop_kind pooling_prop_kind;
      bool int8_forward_inference =
          std::is_same<T, qint8>::value || std::is_same<T, quint8>::value;
      if (int8_forward_inference || std::is_same<T, Eigen::half>::value)
        pooling_prop_kind = prop_kind::forward_inference;
      else
        pooling_prop_kind = prop_kind::forward_training;
      auto fwd_desc =
          pooling_forward::desc(pooling_prop_kind, alg, src_md, dst_md, strides,
                                filter_dims, padding_left, padding_right);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      auto fwd_pd =
          pooling_forward::primitive_desc(fwd_desc, attr, onednn_engine);

      Tensor scratchpad_tensor;
      int64 scratchpad_size = fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      auto fwd_primitive = pooling_forward(fwd_pd);

      // Allocate output.
      // MaxPool may prefer plain format as its primitive format.
      // Need to record this info in meta data to reorder the data correctly.
      SetOutputTensorShape(fwd_pd.dst_desc(), this->tensor_format_onednn_,
                           &dst_tf_shape, &dst_onednn_shape,
                           true /*is_onednn*/);
      AllocateOutputSetOneDnnShape(context, kDstIndex, &dst_tensor,
                                   dst_tf_shape, dst_onednn_shape);

      if (alg == dnnl::algorithm::pooling_max) {
        dst_ws_onednn_shape.SetOneDnnTensor(false);
        dst_ws_tf_shape.AddDim(fwd_pd.workspace_desc().get_size());
        AllocateOutputSetOneDnnShape(context, kDstWorkspaceIndex,
                                     &dst_ws_tensor, dst_ws_tf_shape,
                                     dst_ws_onednn_shape);
      }

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      // Create src and dst memory.
      auto src_mem = CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                                      GetTensorBuffer<T>(&src_tensor));
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      GetTensorBuffer<T>(dst_tensor));

      // Execute primitive.
      if (alg == dnnl::algorithm::pooling_max) {
        auto ws_mem = CreateDnnlMemory(fwd_pd.workspace_desc(), onednn_engine,
                                       GetTensorBuffer<uint8>(dst_ws_tensor));

        std::unordered_map<int, memory> fwd_primitive_args = {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem},
            {DNNL_ARG_WORKSPACE, ws_mem},
            {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
        fwd_primitive.execute(onednn_stream, fwd_primitive_args);
      } else if (alg == dnnl::algorithm::pooling_avg) {
        std::unordered_map<int, memory> fwd_primitive_args = {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem},
            {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
        fwd_primitive.execute(onednn_stream, fwd_primitive_args);
      } else {
        ITEX_LOG(FATAL) << "Unsupported pooling algorithm";
      }

      if (int8_forward_inference) {
        // Pass min, max from input to output.
        const Tensor& min_input_t = context->input(1);
        const Tensor& max_input_t = context->input(2);
        const float min_input = min_input_t.flat<float>()(0);
        const float max_input = max_input_t.flat<float>()(0);

        Tensor* output_min = nullptr;
        Tensor* output_max = nullptr;
        OneDnnShape output_min_onednn_shape, output_max_onednn_shape;
        output_min_onednn_shape.SetOneDnnTensor(false);
        output_max_onednn_shape.SetOneDnnTensor(false);

        // Allocate output min as host memory
        AllocateOutputSetOneDnnShape(context, 1, &output_min, {},
                                     output_min_onednn_shape);
        AllocateOutputSetOneDnnShape(context, 2, &output_max, {},
                                     output_max_onednn_shape);
        output_min->flat<float>()(0) = min_input;
        output_max->flat<float>()(0) = max_input;
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }
};

template <typename Device, typename T, dnnl::algorithm alg>
class OneDnnPoolGradOp : public OneDnnPoolOpBase<T> {
 public:
  explicit OneDnnPoolGradOp(OpKernelConstruction* context)
      : OneDnnPoolOpBase<T>(context) {}
  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      const int kSrcIndex = 0;
      const int kDiffDstIndex = (alg == dnnl::algorithm::pooling_max) ? 2 : 1;
      const int kWorkspaceIndex = 3; /* only used in maxpoolgrad */
      const int kDstIndex = 0;

      const Tensor& src_tensor = context->input(kSrcIndex);
      const Tensor& diff_dst_tensor = context->input(kDiffDstIndex);

      OneDnnShape src_onednn_shape, diff_dst_onednn_shape;
      GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
      GetOneDnnShape(context, kDiffDstIndex, &diff_dst_onednn_shape);

      OneDnnPoolParameters pool_params;
      TensorShape src_tf_shape;
      if (alg == dnnl::algorithm::pooling_max) {
        src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                           ? src_onednn_shape.GetTfShape()
                           : src_tensor.shape();
      } else if (alg == dnnl::algorithm::pooling_avg) {
        // For AvgPoolGrad, the 1st input is src shape
        auto shape_vec = src_tensor.vec<int32>();
        for (int i = 0; i < src_tensor.NumElements(); i++) {
          src_tf_shape.AddDim(shape_vec(i));
        }
      } else {
        ITEX_LOG(FATAL) << "Unsupported pooling algorithm";
      }
      pool_params.Init(context, this->ksize_, this->stride_, this->padding_,
                       this->padding_list_, this->data_format_tf_,
                       src_tf_shape);
      OP_REQUIRES_OK(context, context->status());
      memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right);

      bool is_pool2d = (this->ksize_.size() == 4);
      memory::dims src_dims =
          src_onednn_shape.IsOneDnnTensor()
              ? src_onednn_shape.GetSizesAsOneDnnDims()
              : TFShapeToOneDnnDimsInNC(src_tf_shape, this->data_format_tf_,
                                        is_pool2d);
      memory::dims diff_dst_dims =
          diff_dst_onednn_shape.IsOneDnnTensor()
              ? diff_dst_onednn_shape.GetSizesAsOneDnnDims()
              : TFShapeToOneDnnDimsInNC(diff_dst_tensor.shape(),
                                        this->data_format_tf_, is_pool2d);

      memory::desc src_md = src_onednn_shape.IsOneDnnTensor()
                                ? src_onednn_shape.GetOneDnnLayout()
                                : memory::desc(src_dims, OneDnnType<T>(),
                                               this->data_format_onednn_);
      memory::desc diff_dst_md =
          diff_dst_onednn_shape.IsOneDnnTensor()
              ? diff_dst_onednn_shape.GetOneDnnLayout()
              : memory::desc(diff_dst_dims, OneDnnType<T>(),
                             this->data_format_onednn_);
      memory::desc diff_dst_md_any =
          memory::desc(diff_dst_dims, OneDnnType<T>(), memory::format_tag::any);

      // Create primitive.
      auto bwd_desc =
          pooling_backward::desc(alg, src_md, diff_dst_md_any, strides,
                                 filter_dims, padding_left, padding_right);
      auto fwd_desc = pooling_forward::desc(
          prop_kind::forward_training, alg, src_md, diff_dst_md_any, strides,
          filter_dims, padding_left, padding_right);
      auto fwd_pd = pooling_forward::primitive_desc(fwd_desc, onednn_engine);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      auto pooling_bwd_pd = pooling_backward::primitive_desc(
          bwd_desc, attr, onednn_engine, fwd_pd);

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

      auto bwd_primitive = pooling_backward(pooling_bwd_pd);

      // Allocate output tensor.
      Tensor* diff_src_tensor = nullptr;
      TensorShape diff_src_tf_shape;
      OneDnnShape diff_src_onednn_shape;
      SetOutputTensorShape(pooling_bwd_pd.diff_src_desc(),
                           this->tensor_format_onednn_, &diff_src_tf_shape,
                           &diff_src_onednn_shape, true /*is_onednn*/);
      AllocateOutputSetOneDnnShape(context, kDstIndex, &diff_src_tensor,
                                   diff_src_tf_shape, diff_src_onednn_shape);

      // Create memory primitive.
      dnnl::memory diff_src_mem = CreateDnnlMemory(
          src_md, onednn_engine, GetTensorBuffer<T>(diff_src_tensor));
      dnnl::memory diff_dst_mem = CreateDnnlMemory(
          diff_dst_md, onednn_engine, GetTensorBuffer<T>(&diff_dst_tensor));

      // Reorder.
      dnnl::memory diff_dst_reorder_mem;
      Tensor diff_dst_reorder_tensor;
      bool is_diff_dst_reordered =
          (diff_dst_md != pooling_bwd_pd.diff_dst_desc());
      if (is_diff_dst_reordered) {
        int diff_dst_reorder_size =
            pooling_bwd_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::v(),
                                    TensorShape({diff_dst_reorder_size}),
                                    &diff_dst_reorder_tensor));

        diff_dst_reorder_mem =
            CreateDnnlMemory(pooling_bwd_pd.diff_dst_desc(), onednn_engine,
                             GetTensorBuffer<T>(&diff_dst_reorder_tensor));
        ReorderMemory(*context, &diff_dst_mem, &diff_dst_reorder_mem,
                      onednn_engine);
      }

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      // Execute.
      if (alg == dnnl::algorithm::pooling_max) {
        const Tensor& workspace_tensor = context->input(kWorkspaceIndex);
        dnnl::memory ws_mem =
            CreateDnnlMemory(pooling_bwd_pd.workspace_desc(), onednn_engine,
                             GetTensorBuffer<uint8>(&workspace_tensor));

        std::unordered_map<int, memory> bwd_primitive_args = {
            {{DNNL_ARG_DIFF_DST,
              is_diff_dst_reordered ? diff_dst_reorder_mem : diff_dst_mem},
             {DNNL_ARG_WORKSPACE, ws_mem},
             {DNNL_ARG_DIFF_SRC, diff_src_mem},
             {DNNL_ARG_SCRATCHPAD, scratchpad_mem}}};
        bwd_primitive.execute(onednn_stream, bwd_primitive_args);
      } else if (alg == dnnl::algorithm::pooling_avg) {
        std::unordered_map<int, memory> bwd_primitive_args = {
            {{DNNL_ARG_DIFF_DST,
              is_diff_dst_reordered ? diff_dst_reorder_mem : diff_dst_mem},
             {DNNL_ARG_DIFF_SRC, diff_src_mem},
             {DNNL_ARG_SCRATCHPAD, scratchpad_mem}}};
        bwd_primitive.execute(onednn_stream, bwd_primitive_args);
      } else {
        ITEX_LOG(FATAL) << "Unsupported pooling algorithm";
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status:" + std::to_string(e.status) +
                         ", message: " + string(e.message) + ". in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }
};

// MaxPooling ops registration
#ifdef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnMaxPool").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),     \
      OneDnnPoolOp<CPUDevice, TYPE, dnnl::algorithm::pooling_max>)             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnMaxPool3D").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      OneDnnPoolOp<CPUDevice, TYPE, dnnl::algorithm::pooling_max>)             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnMaxPoolGrad").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnPoolGradOp<CPUDevice, TYPE, dnnl::algorithm::pooling_max>);        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnMaxPool3DGrad")                                             \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<TYPE>("T"),                                          \
      OneDnnPoolGradOp<CPUDevice, TYPE, dnnl::algorithm::pooling_max>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#else
#define REGISTER_KERNEL(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnMaxPool")                                        \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<TYPE>("T")                                \
          .HostMemory("input_meta")                                 \
          .HostMemory("output_meta")                                \
          .HostMemory("workspace_meta"),                            \
      OneDnnPoolOp<GPUDevice, TYPE, dnnl::algorithm::pooling_max>); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnMaxPool3D")                                      \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<TYPE>("T")                                \
          .HostMemory("input_meta")                                 \
          .HostMemory("output_meta")                                \
          .HostMemory("workspace_meta"),                            \
      OneDnnPoolOp<GPUDevice, TYPE, dnnl::algorithm::pooling_max>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OneDnnMaxPoolGrad")                                        \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<TYPE>("T")                                    \
          .HostMemory("orig_input_meta")                                \
          .HostMemory("orig_output_meta")                               \
          .HostMemory("grad_meta")                                      \
          .HostMemory("workspace_meta")                                 \
          .HostMemory("output_meta"),                                   \
      OneDnnPoolGradOp<GPUDevice, TYPE, dnnl::algorithm::pooling_max>); \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OneDnnMaxPool3DGrad")                                      \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<TYPE>("T")                                    \
          .HostMemory("orig_input_meta")                                \
          .HostMemory("orig_output_meta")                               \
          .HostMemory("grad_meta")                                      \
          .HostMemory("workspace_meta")                                 \
          .HostMemory("output_meta"),                                   \
      OneDnnPoolGradOp<GPUDevice, TYPE, dnnl::algorithm::pooling_max>);
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

// AvgPooling ops registration
#ifdef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnAvgPool").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),     \
      OneDnnPoolOp<CPUDevice, TYPE, dnnl::algorithm::pooling_avg>)             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnAvgPool3D").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      OneDnnPoolOp<CPUDevice, TYPE, dnnl::algorithm::pooling_avg>)             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnAvgPoolGrad").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnPoolGradOp<CPUDevice, TYPE, dnnl::algorithm::pooling_avg>);        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnAvgPool3DGrad")                                             \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<TYPE>("T"),                                          \
      OneDnnPoolGradOp<CPUDevice, TYPE, dnnl::algorithm::pooling_avg>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(TYPE)         \
  REGISTER_KERNEL_BUILDER(            \
      Name("_OneDnnQuantizedAvgPool") \
          .Device(DEVICE_CPU)         \
          .TypeConstraint<TYPE>("T"), \
      OneDnnPoolOp<CPUDevice, TYPE, dnnl::algorithm::pooling_avg>)
TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#else  // Below is GPU part.
#define REGISTER_KERNEL(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnAvgPool")                                        \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<TYPE>("T")                                \
          .HostMemory("input_meta")                                 \
          .HostMemory("output_meta"),                               \
      OneDnnPoolOp<GPUDevice, TYPE, dnnl::algorithm::pooling_avg>); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("_OneDnnAvgPool3D")                                      \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<TYPE>("T")                                \
          .HostMemory("input_meta")                                 \
          .HostMemory("output_meta"),                               \
      OneDnnPoolOp<GPUDevice, TYPE, dnnl::algorithm::pooling_avg>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(TYPE)             \
  REGISTER_KERNEL_BUILDER(                \
      Name("_OneDnnQuantizedAvgPool")     \
          .Device(DEVICE_GPU)             \
          .TypeConstraint<TYPE>("T")      \
          .HostMemory("min_input")        \
          .HostMemory("max_input")        \
          .HostMemory("min_output")       \
          .HostMemory("max_output")       \
          .HostMemory("input_meta")       \
          .HostMemory("output_meta")      \
          .HostMemory("min_input_meta")   \
          .HostMemory("max_input_meta")   \
          .HostMemory("min_output_meta")  \
          .HostMemory("max_output_meta"), \
      OneDnnPoolOp<GPUDevice, TYPE, dnnl::algorithm::pooling_avg>);
TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OneDnnAvgPoolGrad")                                        \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<TYPE>("T")                                    \
          .HostMemory("orig_input_shape")                               \
          .HostMemory("orig_input_meta")                                \
          .HostMemory("grad_meta")                                      \
          .HostMemory("output_meta"),                                   \
      OneDnnPoolGradOp<GPUDevice, TYPE, dnnl::algorithm::pooling_avg>); \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OneDnnAvgPool3DGrad")                                      \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<TYPE>("T")                                    \
          .HostMemory("orig_input_shape")                               \
          .HostMemory("orig_input_meta")                                \
          .HostMemory("grad_meta")                                      \
          .HostMemory("output_meta"),                                   \
      OneDnnPoolGradOp<GPUDevice, TYPE, dnnl::algorithm::pooling_avg>);
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

}  // namespace itex
