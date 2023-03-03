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

using dnnl::layer_normalization_backward;
using dnnl::layer_normalization_forward;
using dnnl::prop_kind;
using dnnl::stream;

using LayerNormFwdPd = dnnl::layer_normalization_forward::primitive_desc;
using LayerNormBwdPd = dnnl::layer_normalization_backward::primitive_desc;

namespace itex {

template <typename Device, typename T, typename U, bool is_inteltf_ln = false>
class OneDnnLayerNormOp : public OpKernel {
 public:
  explicit OneDnnLayerNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    if (context->HasAttr("is_training")) {
      OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
    } else {
      // MklLayernorm op used in Intel-TF
      is_training_ = false;
    }
    if (context->HasAttr("data_format")) {
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    } else {
      // MklLayernorm op used in Intel-TF
      tensor_format = "NHWC";
    }
    OP_REQUIRES(
        context, tensor_format == "NHWC",
        errors::InvalidArgument("Invalid data format, only support NHWC"));
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

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
      const int ndims = src_tf_shape.dims();

      OP_REQUIRES(context, ndims >= 2 && ndims <= 4,
                  errors::InvalidArgument("input must be 2, 3 or 4-dimensional",
                                          src_tensor.shape().DebugString()));
      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1-dimensional",
                                          scale_tensor.shape().DebugString()));
      OP_REQUIRES(context, shift_tensor.dims() == 1,
                  errors::InvalidArgument("offset must be 1-dimensional",
                                          shift_tensor.shape().DebugString()));

      // Handle the special case: input with 0 element and 0 layer size.
      Tensor* dst_tensor = nullptr;
      OneDnnShape dst_onednn_shape;
      // Allocate 2 output TF tensors.
      Tensor* layer_mean_tensor = nullptr;
      Tensor* layer_variance_tensor = nullptr;

      dnnl::memory::dims mean_var_dims = {};
      for (int i = 0; i < ndims - 1; ++i)
        mean_var_dims.push_back(src_tf_shape.dim_size(i));

      TensorShape mean_var_shape;
      mean_var_shape = OneDnnDimsToTFShape(mean_var_dims);

      if (src_tf_shape.num_elements() == 0) {
        dst_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kDstIndex, &dst_tensor,
                                     src_tf_shape, dst_onednn_shape);
        ITEX_DCHECK(dst_tensor);
        DeviceFill<Device, char>(
            const_cast<char*>(dst_tensor->tensor_data().data()), 0,
            dst_tensor->tensor_data().size(), context->GetDeviceStream());
        // _MklLayernorm doesn't have mean/variance output tensor
        if (!is_inteltf_ln) {
          AllocateTFOutputs(context, mean_var_shape, &layer_mean_tensor,
                            &layer_variance_tensor, true);
        }
        return;
      }

      int depth_ = scale_tensor.shape().dim_size(0);

      // Set src memory descriptor.
      dnnl::memory::format_tag onednn_tag = dnnl::memory::format_tag::undef;
      OneDnnTensorFormat onednn_tensor_fmt = OneDnnTensorFormat::FORMAT_INVALID;
      if (src_onednn_shape.IsOneDnnTensor()) {
        onednn_tensor_fmt = src_onednn_shape.GetTfDataFormat();
      } else {
        if (ndims == 2) {
          onednn_tensor_fmt = OneDnnTensorFormat::FORMAT_NC;
        } else if (ndims == 3) {
          onednn_tensor_fmt = OneDnnTensorFormat::FORMAT_TNC;
        } else if (ndims == 4) {
          // Now, data foramt only support NHWC(LDNC), for onednn, ldnc alias to
          // nchw.
          if (tensor_format == "NHWC") {
            onednn_tensor_fmt = OneDnnTensorFormat::FORMAT_NCHW;
          }
        }
        onednn_tag = OneDnnTensorFormatToTag(onednn_tensor_fmt);
      }

      dnnl::memory::dims src_dims = {};
      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
      } else {
        for (int i = 0; i < ndims; ++i) {
          src_dims.push_back(src_tf_shape.dim_size(i));
        }
      }
      auto src_md =
          src_onednn_shape.IsOneDnnTensor()
              ? src_onednn_shape.GetOneDnnLayout()
              : dnnl::memory::desc(src_dims, OneDnnType<T>(), onednn_tag);
      auto scale_md =
          dnnl::memory::desc({static_cast<int64>(depth_)}, OneDnnType<U>(),
                             dnnl::memory::format_tag::a);
      auto shift_md =
          dnnl::memory::desc({static_cast<int64>(depth_)}, OneDnnType<U>(),
                             dnnl::memory::format_tag::a);

      bool set_onednn_tensor = true;
      if (ndims == 4 && !src_onednn_shape.IsOneDnnTensor())
        set_onednn_tensor = false;
      // Create fwd primitive.
      auto propagation = is_training_ ? dnnl::prop_kind::forward_training
                                      : dnnl::prop_kind::forward_inference;
      auto flags = dnnl::normalization_flags::use_scale |
                   dnnl::normalization_flags::use_shift;

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      dnnl::layer_normalization_forward::primitive_desc ln_fwd_pd(
          onednn_engine, propagation, src_md, src_md, epsilon_, flags, attr);
#else
      dnnl::layer_normalization_forward::desc ln_fwd_desc(propagation, src_md,
                                                          epsilon_, flags);
      dnnl::layer_normalization_forward::primitive_desc ln_fwd_pd(
          ln_fwd_desc, attr, onednn_engine);
#endif
      dnnl::layer_normalization_forward ln_fwd_primitive(ln_fwd_pd);

      // Allocate output dst tensor.
      TensorShape dst_tf_shape = src_tensor.shape();
      SetOutputTensorShape(ln_fwd_pd.dst_desc(), onednn_tensor_fmt,
                           &dst_tf_shape, &dst_onednn_shape, set_onednn_tensor);
      AllocateOutputSetOneDnnShape(context, kDstIndex, &dst_tensor,
                                   dst_tf_shape, dst_onednn_shape);
      // _MklLayernorm doesn't have mean/variance output tensor
      if (!is_inteltf_ln) {
        AllocateTFOutputs(context, mean_var_shape, &layer_mean_tensor,
                          &layer_variance_tensor);
      }

      // Create onednn memory.
      void* src_data = GetTensorBuffer<T>(&src_tensor);
      void* mean_op_data =
          is_training_ ? GetTensorBuffer<U>(layer_mean_tensor) : nullptr;
      void* variance_op_data =
          is_training_ ? GetTensorBuffer<U>(layer_variance_tensor) : nullptr;
      void* dst_data = GetTensorBuffer<T>(dst_tensor);
      void* scale_data = GetTensorBuffer<U>(&scale_tensor);
      void* shift_data = GetTensorBuffer<U>(&shift_tensor);

      auto src_mem = CreateDnnlMemory(src_md, onednn_engine, src_data);
      auto dst_mem =
          CreateDnnlMemory(ln_fwd_pd.dst_desc(), onednn_engine, dst_data);
      auto scale_mem = CreateDnnlMemory(scale_md, onednn_engine, scale_data);
      auto shift_mem = CreateDnnlMemory(shift_md, onednn_engine, shift_data);

      dnnl::memory scale_cached_mem, shift_cached_mem;

      if (IsScaleShiftBF16()) {
        const dnnl::memory::desc scale_fp32_md = dnnl::memory::desc(
            {static_cast<int64>(depth_)}, OneDnnType<float>(),
            dnnl::memory::format_tag::a);
        if (scale_cache_manager_.IsEmpty()) {
          // Cache fp32 scale
          scale_cache_manager_.SetCache(context, scale_md, scale_fp32_md,
                                        scale_data, onednn_engine);
        }

        float* scale_cached_data =
            scale_cache_manager_.GetCache(context, scale_fp32_md);

        if (scale_cached_data != nullptr) {
          scale_cached_mem =
              CreateDnnlMemory(scale_fp32_md, onednn_engine, scale_cached_data);

        } else {
          ITEX_LOG(FATAL) << "Wrong cache for _OneDnnMklLayerNorm scale tensor";
        }

        const dnnl::memory::desc shift_fp32_md = dnnl::memory::desc(
            {static_cast<int64>(depth_)}, OneDnnType<float>(),
            dnnl::memory::format_tag::a);
        if (shift_cache_manager_.IsEmpty()) {
          // Cache fp32 shift
          shift_cache_manager_.SetCache(context, shift_md, shift_fp32_md,
                                        shift_data, onednn_engine);
        }
        float* shift_cached_data =
            shift_cache_manager_.GetCache(context, shift_fp32_md);
        if (shift_cached_data != nullptr) {
          shift_cached_mem =
              CreateDnnlMemory(shift_fp32_md, onednn_engine, shift_cached_data);
        } else {
          ITEX_LOG(FATAL) << "Wrong cache for _OneDnnMklLayerNorm shift tensor";
        }
      }

      auto mean_memory =
          CreateDnnlMemory(ln_fwd_pd.mean_desc(), onednn_engine, mean_op_data);
      auto var_memory = CreateDnnlMemory(ln_fwd_pd.variance_desc(),
                                         onednn_engine, variance_op_data);
      // Reorder for src tensor.
      dnnl::memory src_reorder_mem;
      Tensor src_reorder_tensor;
      bool is_src_reordered = (src_md != ln_fwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size = ln_fwd_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));

        src_reorder_mem =
            CreateDnnlMemory(ln_fwd_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<T>(&src_reorder_tensor));
        ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
      }

      // Execute.
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, dnnl::memory> args = {
          {DNNL_ARG_SRC, is_src_reordered ? src_reorder_mem : src_mem},
          {DNNL_ARG_DST, dst_mem}};
      if (static_cast<bool>(flags & dnnl::normalization_flags::use_scale))
        args.insert({DNNL_ARG_SCALE,
                     IsScaleShiftBF16() ? scale_cached_mem : scale_mem});
      if (static_cast<bool>(flags & dnnl::normalization_flags::use_shift))
        args.insert({DNNL_ARG_SHIFT,
                     IsScaleShiftBF16() ? shift_cached_mem : shift_mem});

      if (is_training_) {
        args.insert({DNNL_ARG_MEAN, mean_memory});
        args.insert({DNNL_ARG_VARIANCE, var_memory});
      }
      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          ln_fwd_pd.scratchpad_desc().get_size() / sizeof(U);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(ln_fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<U>(&scratchpad_tensor));
      args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      ln_fwd_primitive.execute(onednn_stream, args);
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
  bool is_training_;
  string tensor_format;

  WeightCacheManager<float> scale_cache_manager_;
  WeightCacheManager<float> shift_cache_manager_;

  bool IsScaleShiftBF16() {
    return (is_inteltf_ln && std::is_same<U, Eigen::bfloat16>::value);
  }

  void AllocateTFOutputs(OpKernelContext* context, TensorShape mean_var_shape,
                         Tensor** layer_mean_tensor,
                         Tensor** layer_variance_tensor,
                         bool init_val = false) {
    ITEX_DCHECK(layer_mean_tensor);
    ITEX_DCHECK(layer_variance_tensor);

    const size_t kLayerMeanIndex = 1;
    const size_t kLayerVarianceIndex = 2;

    // Allocate layer mean output tensor.
    OneDnnShape onednn_shape_layer_mean;
    onednn_shape_layer_mean.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kLayerMeanIndex, layer_mean_tensor,
                                 mean_var_shape, onednn_shape_layer_mean);
    ITEX_DCHECK(*layer_mean_tensor);

    // Allocate layer variance output tensor.
    OneDnnShape onednn_shape_layer_variance;
    onednn_shape_layer_variance.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kLayerVarianceIndex,
                                 layer_variance_tensor, mean_var_shape,
                                 onednn_shape_layer_variance);
    ITEX_DCHECK(*layer_variance_tensor);

    if (init_val) {
      char nan = Eigen::NumTraits<char>::quiet_NaN();
      const int kSize = mean_var_shape.num_elements();
      auto* stream = context->GetDeviceStream();

      DeviceFill<Device, char>(
          const_cast<char*>((*layer_mean_tensor)->tensor_data().data()), nan,
          kSize, stream);
      DeviceFill<Device, char>(
          const_cast<char*>((*layer_variance_tensor)->tensor_data().data()),
          nan, kSize, stream);
    }
  }
};

template <typename Device, typename T, typename U>
class OneDnnLayerNormGradOp : public OpKernel {
 public:
  explicit OneDnnLayerNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(
        context, tensor_format == "NHWC",
        errors::InvalidArgument("Invalid data format, only support NHWC"));
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const size_t kDiffDstIndex = 0;   // index of diff_dst tensor
      const size_t kSrcIndex = 1;       // index of src input tensor
      const size_t kScaleIndex = 2;     // index of scale tensor
      const size_t kMeanIndex = 3;      // index of saved_mean tensor
      const size_t kVarianceIndex = 4;  // index of saved_variance tensor
      const size_t kDiffSrcIndex = 0;   // index of diff_src tensor

      const Tensor& diff_dst_tensor = context->input(kDiffDstIndex);
      const Tensor& src_tensor = context->input(kSrcIndex);
      const Tensor& scale_tensor = context->input(kScaleIndex);
      const Tensor& saved_mean_tensor = context->input(kMeanIndex);
      const Tensor& saved_variance_tensor = context->input(kVarianceIndex);
      Tensor* diff_src_tensor = nullptr;

      OneDnnShape src_onednn_shape, diff_dst_onednn_shape;
      GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
      GetOneDnnShape(context, kDiffDstIndex, &diff_dst_onednn_shape);
      TensorShape diff_dst_tf_shape = diff_dst_onednn_shape.IsOneDnnTensor()
                                          ? diff_dst_onednn_shape.GetTfShape()
                                          : diff_dst_tensor.shape();
      TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                     ? src_onednn_shape.GetTfShape()
                                     : src_tensor.shape();

      const int diff_dst_shape_dims = diff_dst_tf_shape.dims();
      const int src_shape_dims = src_tf_shape.dims();

      OP_REQUIRES(
          context, diff_dst_shape_dims >= 2 && diff_dst_shape_dims <= 4,
          errors::InvalidArgument("input must be 2, 3 or 4-dimensional",
                                  diff_dst_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, src_shape_dims == diff_dst_shape_dims,
          errors::InvalidArgument("src and gard input shape must be same",
                                  src_tensor.shape().DebugString()));
      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1-dimensional",
                                          scale_tensor.shape().DebugString()));
      // Allocate output TF tensors diff_scale and diff_shift.
      OneDnnShape diff_src_onednn_shape;
      Tensor* diff_scale_tensor = nullptr;
      Tensor* diff_shift_tensor = nullptr;

      // Handle the special case: input with 0 element and 0 layer size.
      if (src_tf_shape.num_elements() == 0 ||
          diff_dst_tf_shape.num_elements() == 0) {
        diff_src_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kDiffSrcIndex, &diff_src_tensor,
                                     src_tf_shape, diff_src_onednn_shape);
        DeviceFill<Device, T>(diff_src_tensor->flat<T>().data(), T(0),
                              diff_src_tensor->shape().num_elements(),
                              context->GetDeviceStream());
        AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                          &diff_shift_tensor, true);

        return;
      }

      const int depth_ = scale_tensor.shape().dim_size(0);

      OneDnnTensorFormat onednn_tensor_fmt = OneDnnTensorFormat::FORMAT_INVALID;
      if (src_onednn_shape.IsOneDnnTensor()) {
        onednn_tensor_fmt = src_onednn_shape.GetTfDataFormat();
      } else {
        if (diff_dst_shape_dims == 2) {
          onednn_tensor_fmt = OneDnnTensorFormat::FORMAT_NC;
        } else if (diff_dst_shape_dims == 3) {
          onednn_tensor_fmt = OneDnnTensorFormat::FORMAT_TNC;
        } else if (diff_dst_shape_dims == 4) {
          if (tensor_format == "NHWC") {
            onednn_tensor_fmt = OneDnnTensorFormat::FORMAT_NCHW;
          }
        }
      }

      dnnl::memory::format_tag onednn_tag =
          OneDnnTensorFormatToTag(onednn_tensor_fmt);

      dnnl::memory::dims src_dims = {};
      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
      } else {
        for (int i = 0; i < diff_dst_shape_dims; ++i) {
          src_dims.push_back(src_tf_shape.dim_size(i));
        }
      }

      dnnl::memory::dims diff_dst_dims = src_dims;

      auto src_md =
          src_onednn_shape.IsOneDnnTensor()
              ? src_onednn_shape.GetOneDnnLayout()
              : dnnl::memory::desc(src_dims, OneDnnType<T>(), onednn_tag);
      auto diff_dst_md =
          diff_dst_onednn_shape.IsOneDnnTensor()
              ? diff_dst_onednn_shape.GetOneDnnLayout()
              : dnnl::memory::desc(diff_dst_dims, OneDnnType<T>(), onednn_tag);
      auto diff_dst_md_any = dnnl::memory::desc(diff_dst_dims, OneDnnType<T>(),
                                                dnnl::memory::format_tag::any);
      auto scale_md = dnnl::memory::desc({depth_}, OneDnnType<U>(),
                                         dnnl::memory::format_tag::a);
      auto shift_md = dnnl::memory::desc({depth_}, OneDnnType<U>(),
                                         dnnl::memory::format_tag::a);

      // Prepare primitives.
      auto propagation_fwd = dnnl::prop_kind::forward_training;
      auto propagation_bwd = dnnl::prop_kind::backward;
      auto flags = dnnl::normalization_flags::use_scale |
                   dnnl::normalization_flags::use_shift;
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      dnnl::layer_normalization_forward::primitive_desc ln_fwd_pd(
          onednn_engine, propagation_fwd, src_md, src_md, epsilon_, flags);
      dnnl::layer_normalization_backward::primitive_desc ln_bwd_pd(
          onednn_engine, propagation_bwd, diff_dst_md_any, diff_dst_md_any,
          src_md, epsilon_, flags, ln_fwd_pd, attr);
#else
      dnnl::layer_normalization_forward::desc ln_fwd_desc(
          propagation_fwd, src_md, epsilon_, flags);
      dnnl::layer_normalization_forward::primitive_desc ln_fwd_pd(
          ln_fwd_desc, onednn_engine);
      dnnl::layer_normalization_backward::desc ln_bwd_desc(
          propagation_bwd, diff_dst_md_any, src_md, epsilon_, flags);
      dnnl::layer_normalization_backward::primitive_desc ln_bwd_pd(
          ln_bwd_desc, attr, onednn_engine, ln_fwd_pd);
#endif
      dnnl::layer_normalization_backward ln_bwd_primitive(ln_bwd_pd);

      bool set_onednn_tensor = true;
      if (diff_dst_shape_dims == 4 && !src_onednn_shape.IsOneDnnTensor())
        set_onednn_tensor = false;
      // Allocate diff_src tensor.
      TensorShape diff_src_tf_shape = diff_dst_tensor.shape();
      SetOutputTensorShape(ln_bwd_pd.diff_src_desc(), onednn_tensor_fmt,
                           &diff_src_tf_shape, &diff_src_onednn_shape,
                           set_onednn_tensor);
      AllocateOutputSetOneDnnShape(context, kDiffSrcIndex, &diff_src_tensor,
                                   diff_src_tf_shape, diff_src_onednn_shape);

      AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                        &diff_shift_tensor);

      // OneDnn requests an empty shift tensor.
      Tensor shift_tensor;
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<U>::v(),
                                          scale_tensor.shape(), &shift_tensor));

      // Create onednn memory.
      void* src_data = GetTensorBuffer<T>(&src_tensor);
      void* diff_dst_data = GetTensorBuffer<T>(&diff_dst_tensor);
      void* mean_data = GetTensorBuffer<U>(&saved_mean_tensor);
      void* variance_data = GetTensorBuffer<U>(&saved_variance_tensor);
      void* scale_data = GetTensorBuffer<U>(&scale_tensor);
      void* shift_data = GetTensorBuffer<U>(&shift_tensor);
      void* diff_src_data = GetTensorBuffer<T>(diff_src_tensor);
      void* diff_scale_data = GetTensorBuffer<U>(diff_scale_tensor);
      void* diff_shift_data = GetTensorBuffer<U>(diff_shift_tensor);

      auto src_mem = CreateDnnlMemory(src_md, onednn_engine, src_data);
      auto scale_mem = CreateDnnlMemory(scale_md, onednn_engine, scale_data);
      auto shift_mem = CreateDnnlMemory(shift_md, onednn_engine, shift_data);
      auto mean_mem =
          CreateDnnlMemory(ln_bwd_pd.mean_desc(), onednn_engine, mean_data);
      auto variance_mem = CreateDnnlMemory(ln_bwd_pd.variance_desc(),
                                           onednn_engine, variance_data);
      auto diff_src_mem = CreateDnnlMemory(ln_bwd_pd.diff_src_desc(),
                                           onednn_engine, diff_src_data);
      auto diff_dst_mem =
          CreateDnnlMemory(diff_dst_md, onednn_engine, diff_dst_data);
      auto diff_scale_mem =
          CreateDnnlMemory(scale_md, onednn_engine, diff_scale_data);
      auto diff_shift_mem =
          CreateDnnlMemory(shift_md, onednn_engine, diff_shift_data);

      // Reorder for src/diff_dst tensor.
      dnnl::memory src_reorder_mem;
      Tensor src_reorder_tensor;
      bool is_src_reordered = (src_md != ln_bwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size = ln_bwd_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));

        src_reorder_mem =
            CreateDnnlMemory(ln_bwd_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<T>(&src_reorder_tensor));
        ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
      }

      dnnl::memory diff_dst_reorder_mem;
      Tensor diff_dst_reorder_tensor;
      bool is_diff_dst_reordered = (diff_dst_md != ln_bwd_pd.diff_dst_desc());
      if (is_diff_dst_reordered) {
        int64 diff_dst_reorder_size =
            ln_bwd_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::v(),
                                    TensorShape({diff_dst_reorder_size}),
                                    &diff_dst_reorder_tensor));

        diff_dst_reorder_mem =
            CreateDnnlMemory(ln_bwd_pd.diff_dst_desc(), onednn_engine,
                             GetTensorBuffer<T>(&diff_dst_reorder_tensor));
        ReorderMemory(*context, &diff_dst_mem, &diff_dst_reorder_mem,
                      onednn_engine);
      }

      // Execute.
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, dnnl::memory> args = {
          {DNNL_ARG_SRC, is_src_reordered ? src_reorder_mem : src_mem},
          {DNNL_ARG_MEAN, mean_mem},
          {DNNL_ARG_VARIANCE, variance_mem},
          {DNNL_ARG_DIFF_DST,
           is_diff_dst_reordered ? diff_dst_reorder_mem : diff_dst_mem},
          {DNNL_ARG_SCALE, scale_mem},
          {DNNL_ARG_SHIFT, shift_mem},
          {DNNL_ARG_DIFF_SRC, diff_src_mem},
          {DNNL_ARG_DIFF_SCALE, diff_scale_mem},
          {DNNL_ARG_DIFF_SHIFT, diff_shift_mem}};

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          ln_bwd_pd.scratchpad_desc().get_size() / sizeof(U);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(ln_bwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<U>(&scratchpad_tensor));
      args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      ln_bwd_primitive.execute(onednn_stream, args);
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
  bool is_training_;
  string tensor_format;

  void AllocateTFOutputs(OpKernelContext* context,
                         TensorShape tf_shape_scale_shift,
                         Tensor** diff_scale_tensor, Tensor** diff_shift_tensor,
                         bool init_val = false) {
    ITEX_DCHECK(diff_scale_tensor);
    ITEX_DCHECK(diff_shift_tensor);

    const size_t kDiffScaleIndex = 1;
    const size_t kDiffShiftIndex = 2;

    // Separate out scale and shift grad and copy to individual tensors
    OneDnnShape onednn_shape_diff_scale;
    onednn_shape_diff_scale.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kDiffScaleIndex, diff_scale_tensor,
                                 tf_shape_scale_shift, onednn_shape_diff_scale);
    ITEX_DCHECK(*diff_scale_tensor);

    OneDnnShape onednn_shape_diff_shift;
    onednn_shape_diff_shift.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kDiffShiftIndex, diff_shift_tensor,
                                 tf_shape_scale_shift, onednn_shape_diff_shift);
    ITEX_DCHECK(*diff_shift_tensor);

    if (init_val) {
      const int kSize = (*diff_scale_tensor)->shape().num_elements();
      auto* stream = context->GetDeviceStream();

      DeviceFill<Device, U>((*diff_scale_tensor)->flat<U>().data(), 0, kSize,
                            stream);
      DeviceFill<Device, U>((*diff_shift_tensor)->flat<U>().data(), 0, kSize,
                            stream);
    }
  }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_LAYERNORM_GPU(T, U)                             \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnLayerNorm")               \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("x_meta")              \
                              .HostMemory("scale_meta")          \
                              .HostMemory("offset_meta")         \
                              .HostMemory("y_meta")              \
                              .HostMemory("layer_mean_meta")     \
                              .HostMemory("layer_variance_meta") \
                              .TypeConstraint<T>("T")            \
                              .TypeConstraint<U>("U"),           \
                          OneDnnLayerNormOp<GPUDevice, T, U>);
REGISTER_LAYERNORM_GPU(float, float);
REGISTER_LAYERNORM_GPU(Eigen::bfloat16, float);
REGISTER_LAYERNORM_GPU(Eigen::half, float);
#undef REGISTER_LAYERNORM_GPU

#define REGISTER_LAYERNORM_GRAD_GPU(T, U)                         \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnLayerNormGrad")            \
                              .Device(DEVICE_GPU)                 \
                              .HostMemory("y_backprop_meta")      \
                              .HostMemory("x_meta")               \
                              .HostMemory("scale_meta")           \
                              .HostMemory("reserve_space_1_meta") \
                              .HostMemory("reserve_space_2_meta") \
                              .HostMemory("x_backprop_meta")      \
                              .HostMemory("scale_backprop_meta")  \
                              .HostMemory("offset_backprop_meta") \
                              .HostMemory("reserve_space_3_meta") \
                              .HostMemory("reserve_space_4_meta") \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<U>("U"),            \
                          OneDnnLayerNormGradOp<GPUDevice, T, U>);
REGISTER_LAYERNORM_GRAD_GPU(float, float);
REGISTER_LAYERNORM_GRAD_GPU(Eigen::bfloat16, float);
#undef REGISTER_LAYERNORM_GRAD_GPU

#define REGISTER_MKLLAYERNORM_GPU(T)                     \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnMklLayerNorm")    \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("x_meta")      \
                              .HostMemory("scale_meta")  \
                              .HostMemory("offset_meta") \
                              .HostMemory("y_meta")      \
                              .TypeConstraint<T>("T"),   \
                          OneDnnLayerNormOp<GPUDevice, T, T, true>);
REGISTER_MKLLAYERNORM_GPU(float);
REGISTER_MKLLAYERNORM_GPU(Eigen::bfloat16);
REGISTER_MKLLAYERNORM_GPU(Eigen::half);
#undef REGISTER_MKLLAYERNORM_GPU

#else
#define REGISTER_LAYERNORM_CPU(T, U)                   \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnLayerNorm")     \
                              .Device(DEVICE_CPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          OneDnnLayerNormOp<CPUDevice, T, U>);
REGISTER_LAYERNORM_CPU(float, float);
REGISTER_LAYERNORM_CPU(Eigen::bfloat16, float);
#undef REGISTER_LAYERNORM_CPU

#define REGISTER_LAYERNORM_GRAD_CPU(T, U)              \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnLayerNormGrad") \
                              .Device(DEVICE_CPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          OneDnnLayerNormGradOp<CPUDevice, T, U>);
REGISTER_LAYERNORM_GRAD_CPU(float, float);
REGISTER_LAYERNORM_GRAD_CPU(Eigen::bfloat16, float);
#undef REGISTER_LAYERNORM_GRAD_CPU

#define REGISTER_MKLLAYERNORM_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_OneDnnMklLayerNorm").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OneDnnLayerNormOp<CPUDevice, T, T, true>);
REGISTER_MKLLAYERNORM_CPU(float);
REGISTER_MKLLAYERNORM_CPU(Eigen::bfloat16);
#undef REGISTER_MKLLAYERNORM_CPU

#endif

}  // namespace itex
