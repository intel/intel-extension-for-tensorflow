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
#include "itex/core/kernels/common/fused_batch_norm_functor.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::batch_normalization_backward;
using dnnl::batch_normalization_forward;
using dnnl::prop_kind;
using dnnl::stream;

namespace itex {

// Adding a third parameter to the template to support oneDNN
// FusedBatchNormV3. This is different from default where the classes are
// derived. Moves enabling to compile-time rather than runtime.
template <typename Device, typename T, typename U, bool reserved_space,
          bool is_batch_norm_ex = false>
class OneDnnFusedBatchNormOp : public OpKernel {
 public:
  explicit OneDnnFusedBatchNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    float exponential_avg_factor;
    OP_REQUIRES_OK(context, context->GetAttr("exponential_avg_factor",
                                             &exponential_avg_factor));
    exponential_avg_factor_ = static_cast<U>(exponential_avg_factor);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));

    if (is_batch_norm_ex) {
      int num_side_inputs;
      FbnActivationMode activation_mode;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      if (num_side_inputs > 0) has_side_input_ = true;

      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode));
      OP_REQUIRES(context, activation_mode == FbnActivationMode::kRelu,
                  errors::InvalidArgument(
                      "_OneDnnFusedBatchNorm only support Relu activation"));
    }
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const size_t kSrcIndex = 0;       // index of src input tensor
      const size_t kScaleIndex = 1;     // index of scale tensor
      const size_t kShiftIndex = 2;     // index of shift tensor
      const size_t kMeanIndex = 3;      // index of est_mean tensor
      const size_t kVarianceIndex = 4;  // index of est_variance tensor
      const size_t kSrc1Index = 5;      // index of side input tensor
      const size_t kDstIndex = 0;

      const Tensor& src_tensor = context->input(kSrcIndex);
      const Tensor& scale_tensor = context->input(kScaleIndex);
      const Tensor& shift_tensor = context->input(kShiftIndex);
      const Tensor& est_mean_tensor = context->input(kMeanIndex);
      const Tensor& est_variance_tensor = context->input(kVarianceIndex);

      OneDnnShape src_onednn_shape;
      GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
      TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                     ? src_onednn_shape.GetTfShape()
                                     : src_tensor.shape();

      OP_REQUIRES(context, src_tf_shape.dims() == 4 || src_tf_shape.dims() == 5,
                  errors::InvalidArgument(
                      "input must be 4-dimensional or 5-dimensional",
                      src_tensor.shape().DebugString()));
      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1-dimensional",
                                          scale_tensor.shape().DebugString()));
      OP_REQUIRES(context, shift_tensor.dims() == 1,
                  errors::InvalidArgument("offset must be 1-dimensional",
                                          shift_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, est_mean_tensor.dims() == 1,
          errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                  est_mean_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, est_variance_tensor.dims() == 1,
          errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                  est_variance_tensor.shape().DebugString()));

      // Handle the special case: input with 0 element and 0 batch size.
      Tensor* dst_tensor = nullptr;
      OneDnnShape dst_onednn_shape;
      // Allocate 5 output TF tensors.
      Tensor* batch_mean_tensor = nullptr;
      Tensor* batch_variance_tensor = nullptr;
      Tensor* saved_mean_tensor = nullptr;
      Tensor* saved_variance_tensor = nullptr;
      Tensor* reserved_space_tensor = nullptr;

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
        AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                          &batch_mean_tensor, &batch_variance_tensor,
                          &saved_mean_tensor, &saved_variance_tensor,
                          &reserved_space_tensor, true);
        return;
      }

      int depth_ = 0;
      bool use_3d_format = src_tf_shape.dims() == 5;

      if (src_onednn_shape.IsOneDnnTensor())
        depth_ = src_onednn_shape.GetSizesAsOneDnnDims()[DimensionIndex::Dim_C];
      else
        depth_ =
            static_cast<int>(GetTensorDim(src_tensor, tensor_format_, 'C'));

      // Set src memory descriptor.
      dnnl::memory::format_tag onednn_tag = dnnl::memory::format_tag::undef;
      OneDnnTensorFormat onednn_tensor_fmt;
      dnnl::memory::dims src_dims;

      if (src_onednn_shape.IsOneDnnTensor()) {
        onednn_tensor_fmt = src_onednn_shape.GetTfDataFormat();
      } else {
        onednn_tensor_fmt =
            TFDataFormatToOneDnnDataFormat(tensor_format_, !use_3d_format);
        onednn_tag = OneDnnTensorFormatToTag(onednn_tensor_fmt);
      }

      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
      } else {
        src_dims = TFShapeToOneDnnDimsInNC(src_tensor.shape(), tensor_format_,
                                           !use_3d_format);
      }

      auto src_md =
          src_onednn_shape.IsOneDnnTensor()
              ? src_onednn_shape.GetOneDnnLayout()
              : dnnl::memory::desc(src_dims, OneDnnType<T>(), onednn_tag);

      auto scale_md = dnnl::memory::desc({depth_}, OneDnnType<U>(),
                                         dnnl::memory::format_tag::a);
      auto shift_md = dnnl::memory::desc({depth_}, OneDnnType<U>(),
                                         dnnl::memory::format_tag::a);

      // Create fwd primitive.
      auto propagation = (is_training_ || is_batch_norm_ex)
                             ? dnnl::prop_kind::forward_training
                             : dnnl::prop_kind::forward_scoring;

      auto flags = dnnl::normalization_flags::use_scale |
                   dnnl::normalization_flags::use_shift;
      if (!is_training_) {
        flags |= dnnl::normalization_flags::use_global_stats;
      }
      if (is_batch_norm_ex) {
        if (has_side_input_) {
          flags |= dnnl::normalization_flags::fuse_norm_add_relu;
        } else {
          flags |= dnnl::normalization_flags::fuse_norm_relu;
        }
      }

      dnnl::batch_normalization_forward::desc bn_fwd_desc(propagation, src_md,
                                                          epsilon_, flags);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      dnnl::batch_normalization_forward::primitive_desc bn_fwd_pd(
          bn_fwd_desc, attr, onednn_engine);
      dnnl::batch_normalization_forward bn_fwd_primitive(bn_fwd_pd);

      // Allocate output dst tensor.
      TensorShape dst_tf_shape;
      SetOutputTensorShape(bn_fwd_pd.dst_desc(), onednn_tensor_fmt,
                           &dst_tf_shape, &dst_onednn_shape, true);
      AllocateOutputSetOneDnnShape(context, kDstIndex, &dst_tensor,
                                   dst_tf_shape, dst_onednn_shape);

      if (is_batch_norm_ex) {
        dnnl::memory::desc workspace_md = bn_fwd_pd.workspace_desc();
        size_t workspace_bytes = workspace_md.get_size();
        workspace_tf_shape.AddDim(workspace_bytes / sizeof(U));

        AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                          &batch_mean_tensor, &batch_variance_tensor,
                          &saved_mean_tensor, &saved_variance_tensor,
                          &reserved_space_tensor);
      } else {
        // There is actually no workspace tensor out, so we make a dummy one.
        size_t workspace_bytes = 0;
        workspace_tf_shape.AddDim(workspace_bytes);
        AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                          &batch_mean_tensor, &batch_variance_tensor,
                          &saved_mean_tensor, &saved_variance_tensor,
                          &reserved_space_tensor);
      }

      // Create onednn memory.
      void* src_data = GetTensorBuffer<T>(&src_tensor);
      void *mean_op_data, *variance_op_data;
      if (is_training_) {
        mean_op_data = GetTensorBuffer<U>(saved_mean_tensor);
        variance_op_data = GetTensorBuffer<U>(saved_variance_tensor);
      } else {
        mean_op_data = GetTensorBuffer<U>(&est_mean_tensor);
        variance_op_data = GetTensorBuffer<U>(&est_variance_tensor);
      }
      void* dst_data = GetTensorBuffer<T>(dst_tensor);
      void* ws_op_data =
          reserved_space ? GetTensorBuffer<U>(reserved_space_tensor) : nullptr;
      void* scale_data = GetTensorBuffer<U>(&scale_tensor);
      void* shift_data = GetTensorBuffer<U>(&shift_tensor);

      auto src_mem = CreateDnnlMemory(src_md, onednn_engine, src_data);
      auto dst_mem =
          CreateDnnlMemory(bn_fwd_pd.dst_desc(), onednn_engine, dst_data);
      auto scale_mem = CreateDnnlMemory(scale_md, onednn_engine, scale_data);
      auto shift_mem = CreateDnnlMemory(shift_md, onednn_engine, shift_data);
      auto mean_memory =
          CreateDnnlMemory(bn_fwd_pd.mean_desc(), onednn_engine, mean_op_data);
      auto var_memory = CreateDnnlMemory(bn_fwd_pd.variance_desc(),
                                         onednn_engine, variance_op_data);
      dnnl::memory ws_memory;
      if (is_batch_norm_ex)
        ws_memory = CreateDnnlMemory(bn_fwd_pd.workspace_desc(), onednn_engine,
                                     ws_op_data);

      // Reorder for src tensor.
      dnnl::memory src_reorder_mem;
      Tensor src_reorder_tensor;
      bool is_src_reordered = (src_md != bn_fwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size = bn_fwd_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));

        src_reorder_mem =
            CreateDnnlMemory(bn_fwd_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<T>(&src_reorder_tensor));
        ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
      }

      // Reorder for src1 tensor.
      dnnl::memory src1_mem;
      dnnl::memory src1_reorder_mem;
      Tensor src1_reorder_tensor;
      bool is_src1_reordered = false;
      if (has_side_input_) {
        const Tensor& src1_tensor = context->input(kSrc1Index);
        OneDnnShape src1_onednn_shape;
        GetOneDnnShape(context, kSrc1Index, &src1_onednn_shape);
        dnnl::memory::dims src1_dims;
        if (src1_onednn_shape.IsOneDnnTensor()) {
          src1_dims = src1_onednn_shape.GetSizesAsOneDnnDims();
        } else {
          src1_dims = TFShapeToOneDnnDimsInNC(src1_tensor.shape(),
                                              tensor_format_, !use_3d_format);
        }
        auto src1_md =
            src1_onednn_shape.IsOneDnnTensor()
                ? src1_onednn_shape.GetOneDnnLayout()
                : dnnl::memory::desc(src1_dims, OneDnnType<T>(), onednn_tag);
        void* src1_data = GetTensorBuffer<T>(&src1_tensor);
        src1_mem = CreateDnnlMemory(src1_md, onednn_engine, src1_data);
        // This fusion, oneDNN need input and side input has same format.
        is_src1_reordered = (src1_md != bn_fwd_pd.src_desc());
        if (is_src1_reordered) {
          int64 src1_reorder_size = bn_fwd_pd.src_desc().get_size() / sizeof(T);
          OP_REQUIRES_OK(
              context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src1_reorder_size}),
                                              &src1_reorder_tensor));

          src1_reorder_mem =
              CreateDnnlMemory(bn_fwd_pd.src_desc(), onednn_engine,
                               GetTensorBuffer<T>(&src1_reorder_tensor));
          ReorderMemory(*context, &src1_mem, &src1_reorder_mem, onednn_engine);
        }
      }

      // Execute.
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, dnnl::memory> args = {
          {DNNL_ARG_SRC, is_src_reordered ? src_reorder_mem : src_mem},
          {DNNL_ARG_DST, dst_mem}};
      if (static_cast<bool>(flags & dnnl::normalization_flags::use_scale))
        args.insert({DNNL_ARG_SCALE, scale_mem});
      if (static_cast<bool>(flags & dnnl::normalization_flags::use_shift))
        args.insert({DNNL_ARG_SHIFT, shift_mem});
      if (is_training_ ||
          static_cast<bool>(flags &
                            dnnl::normalization_flags::use_global_stats)) {
        args.insert({DNNL_ARG_MEAN, mean_memory});
        args.insert({DNNL_ARG_VARIANCE, var_memory});
      }
      if (is_batch_norm_ex) args.insert({DNNL_ARG_WORKSPACE, ws_memory});
      if (has_side_input_) {
        args.insert(
            {DNNL_ARG_SRC_1, is_src1_reordered ? src1_reorder_mem : src1_mem});
      }

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

      bn_fwd_primitive.execute(onednn_stream, args);
      float adjust_factor = 1.0;
      if (is_training_) {
        size_t orig_size = src_dims[0] * src_dims[2] * src_dims[3];
        if (use_3d_format) orig_size *= src_dims[4];
        size_t adjust_size = (orig_size > 1) ? (orig_size - 1) : 1;
        adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
      }

      // Calculate running mean/var.
      U *mean_data = nullptr, *variance_data = nullptr;
      if (is_training_) {
        mean_data = saved_mean_tensor->flat<U>().data();
        variance_data = saved_variance_tensor->flat<U>().data();
      }
      auto batch_mean_data = batch_mean_tensor->flat<U>().data();
      auto batch_variance_data = batch_variance_tensor->flat<U>().data();
      auto est_mean_data = est_mean_tensor.flat<U>().data();
      auto est_variance_data = est_variance_tensor.flat<U>().data();

#ifndef INTEL_CPU_ONLY
      auto* stream = context->GetDeviceStream();
      if (Eigen::internal::is_same<Device, GPUDevice>::value) {
        auto total_threads =
            stream->get_device()
                .template get_info<sycl::info::device::max_work_group_size>();
        if (is_training_) {
          if (exponential_avg_factor_ == U(1.0)) {
            stream->submit([&](sycl::handler& cgh) {
              auto batch_mean_data_ptr = static_cast<U*>(batch_mean_data);
              auto mean_data_ptr = static_cast<U*>(mean_data);
              auto batch_variance_data_ptr =
                  static_cast<U*>(batch_variance_data);
              auto variance_data_ptr = static_cast<U*>(variance_data);
              int depth = depth_;

              cgh.parallel_for<
                  VarAdjust<T, U, reserved_space, is_batch_norm_ex>>(
                  sycl::range<1>(total_threads), [=](sycl::item<1> item) {
                    auto id = item.get_id(0);
                    for (auto k = id; k < depth; k += total_threads) {
                      batch_mean_data_ptr[k] = mean_data_ptr[k];
                      batch_variance_data_ptr[k] =
                          static_cast<U>(adjust_factor) * variance_data_ptr[k];
                    }
                  });
            });
          } else {
            U one_minus_factor = U(1.0) - exponential_avg_factor_;
            U exponential_avg_factor_tmp = exponential_avg_factor_;
            stream->submit([&](sycl::handler& cgh) {
              auto batch_mean_data_ptr = batch_mean_data;
              auto est_mean_data_ptr = est_mean_data;
              auto mean_data_ptr = mean_data;
              auto batch_variance_data_ptr = batch_variance_data;
              auto est_variance_data_ptr = est_variance_data;
              auto variance_data_ptr = variance_data;
              int depth = depth_;

              cgh.parallel_for<
                  VarAdjustMinus<T, U, reserved_space, is_batch_norm_ex>>(
                  sycl::range<1>(total_threads), [=](sycl::item<1> item) {
                    auto id = item.get_id(0);
                    for (auto k = id; k < depth; k += total_threads) {
                      batch_mean_data_ptr[k] =
                          one_minus_factor * est_mean_data_ptr[k] +
                          exponential_avg_factor_tmp * mean_data_ptr[k];
                      batch_variance_data_ptr[k] =
                          one_minus_factor * est_variance_data_ptr[k] +
                          exponential_avg_factor_tmp *
                              static_cast<U>(adjust_factor) *
                              variance_data_ptr[k];
                    }
                  });
            });
          }
        }
      }
#else
      // TODO(itex): Parallel_for for CPU code.
      if (Eigen::internal::is_same<Device, CPUDevice>::value) {
        if (is_training_) {
          if (exponential_avg_factor_ == U(1.0)) {
            for (int k = 0; k < depth_; k++) {
              batch_mean_data[k] = mean_data[k];
              batch_variance_data[k] =
                  static_cast<U>(adjust_factor) * variance_data[k];
            }
          } else {
            U one_minus_factor = U(1.0) - exponential_avg_factor_;
            for (int k = 0; k < depth_; k++) {
              batch_mean_data[k] = one_minus_factor * est_mean_data[k] +
                                   exponential_avg_factor_ * mean_data[k];
              batch_variance_data[k] = one_minus_factor * est_variance_data[k] +
                                       exponential_avg_factor_ *
                                           static_cast<U>(adjust_factor) *
                                           variance_data[k];
            }
          }
        }
      }
#endif
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
  U exponential_avg_factor_;
  TensorFormat tensor_format_;
  bool is_training_;
  bool has_side_input_ = false;

  void AllocateTFOutputs(OpKernelContext* context, TensorShape tf_shape_scale,
                         TensorShape workspace_tf_shape,
                         Tensor** batch_mean_tensor,
                         Tensor** batch_variance_tensor,
                         Tensor** saved_mean_tensor,
                         Tensor** saved_variance_tensor,
                         Tensor** reserved_space_tensor,
                         bool init_val = false) {
    ITEX_DCHECK(batch_mean_tensor);
    ITEX_DCHECK(batch_variance_tensor);
    ITEX_DCHECK(saved_mean_tensor);
    ITEX_DCHECK(saved_variance_tensor);

    const size_t kBatchMeanIndex = 1;
    const size_t kBatchVarianceIndex = 2;
    const size_t kSavedMeanIndex = 3;
    const size_t kSavedVarianceIndex = 4;
    const size_t kReservedSpaceIndex = 5;

    // Allocate batch mean output tensor.
    OneDnnShape onednn_shape_batch_mean;
    onednn_shape_batch_mean.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kBatchMeanIndex, batch_mean_tensor,
                                 tf_shape_scale, onednn_shape_batch_mean);
    ITEX_DCHECK(*batch_mean_tensor);

    // Allocate batch variance output tensor.
    OneDnnShape onednn_shape_batch_variance;
    onednn_shape_batch_variance.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kBatchVarianceIndex,
                                 batch_variance_tensor, tf_shape_scale,
                                 onednn_shape_batch_variance);
    ITEX_DCHECK(*batch_variance_tensor);

    // Mean and variance (without Bessel's correction) saved for backward
    // computation to serve as pre-computed mean and variance.
    OneDnnShape onednn_shape_saved_mean;
    onednn_shape_saved_mean.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kSavedMeanIndex, saved_mean_tensor,
                                 tf_shape_scale, onednn_shape_saved_mean);
    ITEX_DCHECK(*saved_mean_tensor);

    OneDnnShape onednn_shape_saved_variance;
    onednn_shape_saved_variance.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kSavedVarianceIndex,
                                 saved_variance_tensor, tf_shape_scale,
                                 onednn_shape_saved_variance);
    ITEX_DCHECK(*saved_variance_tensor);

    if (init_val) {
      U nan = Eigen::NumTraits<U>::quiet_NaN();
      auto* stream = context->GetDeviceStream();
      const int kSize = tf_shape_scale.num_elements();
      DeviceFill<Device, U>((*batch_mean_tensor)->flat<U>().data(), nan, kSize,
                            stream);
      DeviceFill<Device, U>((*batch_variance_tensor)->flat<U>().data(), nan,
                            kSize, stream);
      DeviceFill<Device, U>((*saved_mean_tensor)->flat<U>().data(), U(0), kSize,
                            stream);
      DeviceFill<Device, U>((*saved_variance_tensor)->flat<U>().data(), U(0),
                            kSize, stream);
    }

    // Changes to support reserved_space_3 parameter in FusedBatchNormV3.
    if (reserved_space) {
      ITEX_DCHECK(reserved_space_tensor != nullptr);

      OneDnnShape onednn_shape_reserved_space;
      onednn_shape_reserved_space.SetOneDnnTensor(false);
      AllocateOutputSetOneDnnShape(context, kReservedSpaceIndex,
                                   reserved_space_tensor, workspace_tf_shape,
                                   onednn_shape_reserved_space);
      ITEX_DCHECK((*reserved_space_tensor) != nullptr);
    }
  }
};

template <typename Device, typename T, typename U, bool reserved_space,
          bool is_batch_norm_ex = false>
class OneDnnFusedBatchNormGradOp : public OpKernel {
 public:
  explicit OneDnnFusedBatchNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));

    if (is_batch_norm_ex) {
      FbnActivationMode activation_mode;
      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      if (num_side_inputs > 0) has_side_input_ = true;
      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode));
      OP_REQUIRES(context, activation_mode == FbnActivationMode::kReluGrad,
                  errors::InvalidArgument(
                      "OneDNN FusedBatchNorm only support Relu activation"));
    }
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      const size_t kDiffDstIndex = 0;        // index of diff_dst tensor
      const size_t kSrcIndex = 1;            // index of src input tensor
      const size_t kScaleIndex = 2;          // index of scale tensor
      const size_t kMeanIndex = 3;           // index of saved_mean tensor
      const size_t kVarianceIndex = 4;       // index of saved_variance tensor
      const size_t kReservedSpaceIndex = 5;  // index of reserved space 3 tensor
      const size_t kDiffSrcIndex = 0;        // index of diff_src tensor
      const size_t kDiffSrc1Index = 5;       // index of diff side tensor

      const Tensor& diff_dst_tensor = context->input(kDiffDstIndex);
      const Tensor& src_tensor = context->input(kSrcIndex);
      const Tensor& scale_tensor = context->input(kScaleIndex);
      const Tensor& saved_mean_tensor = context->input(kMeanIndex);
      const Tensor& saved_variance_tensor = context->input(kVarianceIndex);
      const Tensor* reserved_space_tensor =
          reserved_space ? &context->input(kReservedSpaceIndex) : nullptr;
      Tensor* diff_src_tensor = nullptr;
      Tensor* diff_side_input_tensor = nullptr;

      OneDnnShape src_onednn_shape, diff_dst_onednn_shape;
      GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
      GetOneDnnShape(context, kDiffDstIndex, &diff_dst_onednn_shape);
      TensorShape diff_dst_tf_shape = diff_dst_onednn_shape.IsOneDnnTensor()
                                          ? diff_dst_onednn_shape.GetTfShape()
                                          : diff_dst_tensor.shape();
      TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                     ? src_onednn_shape.GetTfShape()
                                     : src_tensor.shape();

      OP_REQUIRES(
          context,
          diff_dst_tf_shape.dims() == 4 || diff_dst_tf_shape.dims() == 5,
          errors::InvalidArgument(
              "input must be 4-dimensional or 5-dimensional",
              diff_dst_tensor.shape().DebugString()));
      OP_REQUIRES(context, src_tf_shape.dims() == 4 || src_tf_shape.dims() == 5,
                  errors::InvalidArgument(
                      "input must be 4-dimensional or 5-dimensional",
                      src_tensor.shape().DebugString()));
      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1-dimensional",
                                          scale_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, saved_mean_tensor.dims() == 1,
          errors::InvalidArgument("saved mean must be 1-dimensional",
                                  saved_mean_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, saved_variance_tensor.dims() == 1,
          errors::InvalidArgument("saved variance must be 1-dimensional",
                                  saved_variance_tensor.shape().DebugString()));

      // Allocate output TF tensors diff_scale and diff_shift.
      OneDnnShape diff_src_onednn_shape;
      Tensor* diff_scale_tensor = nullptr;
      Tensor* diff_shift_tensor = nullptr;

      // Handle the special case: input with 0 element and 0 batch size.
      if (src_tf_shape.num_elements() == 0 ||
          diff_dst_tf_shape.num_elements() == 0) {
        diff_src_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kDiffSrcIndex, &diff_src_tensor,
                                     src_tf_shape, diff_src_onednn_shape);
        // TODO(itex): replace with DeviceFill?
        DeviceMemset<Device>(
            diff_src_tensor->flat<T>().data(), 0,
            diff_src_tensor->shape().num_elements() * sizeof(T),
            context->GetDeviceStream());
        AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                          &diff_shift_tensor, true);
        return;
      }

      int depth_ = 0;
      bool use_3d_format = src_tf_shape.dims() == 5;

      if (src_onednn_shape.IsOneDnnTensor()) {
        depth_ = src_onednn_shape.GetSizesAsOneDnnDims()[DimensionIndex::Dim_C];
      } else if (diff_dst_onednn_shape.IsOneDnnTensor()) {
        depth_ =
            diff_dst_onednn_shape.GetSizesAsOneDnnDims()[DimensionIndex::Dim_C];
      } else {
        depth_ =
            static_cast<int>(GetTensorDim(src_tensor, tensor_format_, 'C'));
      }

      // Set src and diff_dst primitive descriptors.
      OneDnnTensorFormat onednn_tensor_fmt;
      if (src_onednn_shape.IsOneDnnTensor()) {
        onednn_tensor_fmt = src_onednn_shape.GetTfDataFormat();
      } else {
        onednn_tensor_fmt =
            TFDataFormatToOneDnnDataFormat(tensor_format_, !use_3d_format);
      }

      dnnl::memory::format_tag onednn_tag =
          OneDnnTensorFormatToTag(onednn_tensor_fmt);

      dnnl::memory::dims src_dims;
      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
      } else {
        src_dims = TFShapeToOneDnnDimsInNC(src_tensor.shape(), tensor_format_,
                                           !use_3d_format);
      }

      dnnl::memory::dims diff_dst_dims;
      if (diff_dst_onednn_shape.IsOneDnnTensor()) {
        diff_dst_dims = diff_dst_onednn_shape.GetSizesAsOneDnnDims();
      } else {
        diff_dst_dims = TFShapeToOneDnnDimsInNC(diff_dst_tensor.shape(),
                                                tensor_format_, !use_3d_format);
      }

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
      if (!is_training_) {
        flags |= dnnl::normalization_flags::use_global_stats;
      }
      if (is_batch_norm_ex) {
        if (has_side_input_) {
          flags |= dnnl::normalization_flags::fuse_norm_add_relu;
        } else {
          flags |= dnnl::normalization_flags::fuse_norm_relu;
        }
      }

      dnnl::batch_normalization_forward::desc bn_fwd_desc(
          propagation_fwd, src_md, epsilon_, flags);
      dnnl::batch_normalization_backward::desc bn_bwd_desc(
          propagation_bwd, diff_dst_md_any, src_md, epsilon_, flags);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      dnnl::batch_normalization_forward::primitive_desc bn_fwd_pd(
          bn_fwd_desc, attr, onednn_engine);
      dnnl::batch_normalization_backward::primitive_desc bn_bwd_pd(
          bn_bwd_desc, attr, onednn_engine, bn_fwd_pd);
      dnnl::batch_normalization_backward bn_bwd_primitive(bn_bwd_pd);

      // Allocate diff_src tensor.
      TensorShape diff_src_tf_shape;
      SetOutputTensorShape(bn_bwd_pd.diff_src_desc(), onednn_tensor_fmt,
                           &diff_src_tf_shape, &diff_src_onednn_shape, true);
      AllocateOutputSetOneDnnShape(context, kDiffSrcIndex, &diff_src_tensor,
                                   diff_src_tf_shape, diff_src_onednn_shape);

      AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                        &diff_shift_tensor);

      if (has_side_input_) {
        AllocateOutputSetOneDnnShape(context, kDiffSrc1Index,
                                     &diff_side_input_tensor, diff_src_tf_shape,
                                     diff_src_onednn_shape);
      }

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
      void* diff_src1_data = nullptr;
      if (has_side_input_) {
        diff_src1_data = GetTensorBuffer<T>(diff_side_input_tensor);
      }
      void* diff_scale_data = GetTensorBuffer<U>(diff_scale_tensor);
      void* diff_shift_data = GetTensorBuffer<U>(diff_shift_tensor);

      void* res_space_data = (is_batch_norm_ex)
                                 ? GetTensorBuffer<U>(reserved_space_tensor)
                                 : nullptr;

      auto src_mem = CreateDnnlMemory(src_md, onednn_engine, src_data);
      auto scale_mem = CreateDnnlMemory(scale_md, onednn_engine, scale_data);
      auto shift_mem = CreateDnnlMemory(shift_md, onednn_engine, shift_data);
      auto mean_mem =
          CreateDnnlMemory(bn_bwd_pd.mean_desc(), onednn_engine, mean_data);
      auto variance_mem = CreateDnnlMemory(bn_bwd_pd.variance_desc(),
                                           onednn_engine, variance_data);
      auto diff_src_mem = CreateDnnlMemory(bn_bwd_pd.diff_src_desc(),
                                           onednn_engine, diff_src_data);
      auto diff_dst_mem =
          CreateDnnlMemory(diff_dst_md, onednn_engine, diff_dst_data);
      auto diff_scale_mem =
          CreateDnnlMemory(scale_md, onednn_engine, diff_scale_data);
      auto diff_shift_mem =
          CreateDnnlMemory(shift_md, onednn_engine, diff_shift_data);
      dnnl::memory ws_mem;
      if (is_batch_norm_ex)
        ws_mem = CreateDnnlMemory(bn_fwd_pd.workspace_desc(), onednn_engine,
                                  res_space_data);
      dnnl::memory diff_src1_mem;
      if (has_side_input_) {
        diff_src1_mem =
            CreateDnnlMemory(diff_dst_md, onednn_engine, diff_src1_data);
      }

      // Reorder for src/diff_dst tensor.
      dnnl::memory src_reorder_mem;
      Tensor src_reorder_tensor;
      bool is_src_reordered = (src_md != bn_bwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size = bn_bwd_pd.src_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));

        src_reorder_mem =
            CreateDnnlMemory(bn_bwd_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<T>(&src_reorder_tensor));
        ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
      }

      dnnl::memory diff_dst_reorder_mem;
      Tensor diff_dst_reorder_tensor;
      bool is_diff_dst_reordered = (diff_dst_md != bn_bwd_pd.diff_dst_desc());
      if (is_diff_dst_reordered) {
        int64 diff_dst_reorder_size =
            bn_bwd_pd.diff_dst_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::v(),
                                    TensorShape({diff_dst_reorder_size}),
                                    &diff_dst_reorder_tensor));

        diff_dst_reorder_mem =
            CreateDnnlMemory(bn_bwd_pd.diff_dst_desc(), onednn_engine,
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
      if (is_batch_norm_ex) args.insert({DNNL_ARG_WORKSPACE, ws_mem});
      if (has_side_input_) args.insert({DNNL_ARG_DIFF_SRC_1, diff_src1_mem});
      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          bn_bwd_pd.scratchpad_desc().get_size() / sizeof(U);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(bn_bwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<U>(&scratchpad_tensor));
      args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});
      bn_bwd_primitive.execute(onednn_stream, args);
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
  TensorFormat tensor_format_;
  bool is_training_;
  bool has_side_input_ = false;

  void AllocateTFOutputs(OpKernelContext* context,
                         TensorShape tf_shape_scale_shift,
                         Tensor** diff_scale_tensor, Tensor** diff_shift_tensor,
                         bool init_val = false) {
    ITEX_DCHECK(diff_scale_tensor);
    ITEX_DCHECK(diff_shift_tensor);

    const size_t kDiffScaleIndex = 1;
    const size_t kDiffShiftIndex = 2;
    const size_t kP1Index = 3;
    const size_t kP2Index = 4;

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

    // Placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    Tensor *p1_tensor = nullptr, *p2_tensor = nullptr;
    OneDnnShape onednn_shape_p;
    onednn_shape_p.SetOneDnnTensor(false);
    AllocateOutputSetOneDnnShape(context, kP1Index, &p1_tensor, TensorShape({}),
                                 onednn_shape_p);
    AllocateOutputSetOneDnnShape(context, kP2Index, &p2_tensor, TensorShape({}),
                                 onednn_shape_p);

    if (init_val) {
      const int kSize = (*diff_scale_tensor)->shape().num_elements();
      auto* stream = context->GetDeviceStream();

      DeviceFill<Device, U>((*diff_scale_tensor)->flat<U>().data(), U(0), kSize,
                            stream);
      DeviceFill<Device, U>((*diff_shift_tensor)->flat<U>().data(), U(0), kSize,
                            stream);
      DeviceFill<Device, U>(p1_tensor->flat<U>().data(), U(0),
                            p1_tensor->shape().num_elements(), stream);
      DeviceFill<Device, U>(p2_tensor->flat<U>().data(), U(0),
                            p2_tensor->shape().num_elements(), stream);
    }
  }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_FUSED_BATCHNORM_GPU(T, U)                    \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("_OneDnnFusedBatchNorm")                           \
          .Device(DEVICE_GPU)                                 \
          .TypeConstraint<T>("T")                             \
          .HostMemory("x_meta")                               \
          .HostMemory("scale_meta")                           \
          .HostMemory("offset_meta")                          \
          .HostMemory("mean_meta")                            \
          .HostMemory("variance_meta")                        \
          .HostMemory("y_meta")                               \
          .HostMemory("batch_mean_meta")                      \
          .HostMemory("batch_variance_meta")                  \
          .HostMemory("reserve_space_1_meta")                 \
          .HostMemory("reserve_space_2_meta"),                \
      OneDnnFusedBatchNormOp<GPUDevice, T, U, false, false>); \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("_OneDnnFusedBatchNormV2")                         \
          .Device(DEVICE_GPU)                                 \
          .HostMemory("x_meta")                               \
          .HostMemory("scale_meta")                           \
          .HostMemory("offset_meta")                          \
          .HostMemory("mean_meta")                            \
          .HostMemory("variance_meta")                        \
          .HostMemory("y_meta")                               \
          .HostMemory("batch_mean_meta")                      \
          .HostMemory("batch_variance_meta")                  \
          .HostMemory("reserve_space_1_meta")                 \
          .HostMemory("reserve_space_2_meta")                 \
          .TypeConstraint<T>("T")                             \
          .TypeConstraint<U>("U"),                            \
      OneDnnFusedBatchNormOp<GPUDevice, T, U, false, false>); \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("_OneDnnFusedBatchNormV3")                         \
          .Device(DEVICE_GPU)                                 \
          .HostMemory("x_meta")                               \
          .HostMemory("scale_meta")                           \
          .HostMemory("offset_meta")                          \
          .HostMemory("mean_meta")                            \
          .HostMemory("variance_meta")                        \
          .HostMemory("y_meta")                               \
          .HostMemory("batch_mean_meta")                      \
          .HostMemory("batch_variance_meta")                  \
          .HostMemory("reserve_space_1_meta")                 \
          .HostMemory("reserve_space_2_meta")                 \
          .HostMemory("reserve_space_3_meta")                 \
          .TypeConstraint<T>("T")                             \
          .TypeConstraint<U>("U"),                            \
      OneDnnFusedBatchNormOp<GPUDevice, T, U, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("_OneDnnFusedBatchNormEx")                         \
          .Device(DEVICE_GPU)                                 \
          .HostMemory("x_meta")                               \
          .HostMemory("scale_meta")                           \
          .HostMemory("offset_meta")                          \
          .HostMemory("mean_meta")                            \
          .HostMemory("variance_meta")                        \
          .HostMemory("side_input_meta")                      \
          .HostMemory("y_meta")                               \
          .HostMemory("batch_mean_meta")                      \
          .HostMemory("batch_variance_meta")                  \
          .HostMemory("reserve_space_1_meta")                 \
          .HostMemory("reserve_space_2_meta")                 \
          .HostMemory("reserve_space_3_meta")                 \
          .TypeConstraint<T>("T")                             \
          .TypeConstraint<U>("U"),                            \
      OneDnnFusedBatchNormOp<GPUDevice, T, U, true, true>);
REGISTER_FUSED_BATCHNORM_GPU(float, float);
REGISTER_FUSED_BATCHNORM_GPU(Eigen::bfloat16, float);
REGISTER_FUSED_BATCHNORM_GPU(Eigen::half, float);
#undef REGISTER_FUSED_BATCHNORM_GPU

#define REGISTER_FUSED_BATCHNORM_GRAD_GPU(T, U)                   \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_OneDnnFusedBatchNormGrad")                           \
          .Device(DEVICE_GPU)                                     \
          .HostMemory("y_backprop_meta")                          \
          .HostMemory("x_meta")                                   \
          .HostMemory("scale_meta")                               \
          .HostMemory("reserve_space_1_meta")                     \
          .HostMemory("reserve_space_2_meta")                     \
          .HostMemory("x_backprop_meta")                          \
          .HostMemory("scale_backprop_meta")                      \
          .HostMemory("offset_backprop_meta")                     \
          .HostMemory("reserve_space_3_meta")                     \
          .HostMemory("reserve_space_4_meta")                     \
          .TypeConstraint<T>("T"),                                \
      OneDnnFusedBatchNormGradOp<GPUDevice, T, U, false, false>); \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_OneDnnFusedBatchNormGradV2")                         \
          .Device(DEVICE_GPU)                                     \
          .HostMemory("y_backprop_meta")                          \
          .HostMemory("x_meta")                                   \
          .HostMemory("scale_meta")                               \
          .HostMemory("reserve_space_1_meta")                     \
          .HostMemory("reserve_space_2_meta")                     \
          .HostMemory("x_backprop_meta")                          \
          .HostMemory("scale_backprop_meta")                      \
          .HostMemory("offset_backprop_meta")                     \
          .HostMemory("reserve_space_3_meta")                     \
          .HostMemory("reserve_space_4_meta")                     \
          .TypeConstraint<T>("T")                                 \
          .TypeConstraint<U>("U"),                                \
      OneDnnFusedBatchNormGradOp<GPUDevice, T, U, false, false>); \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_OneDnnFusedBatchNormGradV3")                         \
          .Device(DEVICE_GPU)                                     \
          .HostMemory("y_backprop_meta")                          \
          .HostMemory("x_meta")                                   \
          .HostMemory("scale_meta")                               \
          .HostMemory("reserve_space_1_meta")                     \
          .HostMemory("reserve_space_2_meta")                     \
          .HostMemory("reserve_space_3_meta")                     \
          .HostMemory("x_backprop_meta")                          \
          .HostMemory("scale_backprop_meta")                      \
          .HostMemory("offset_backprop_meta")                     \
          .HostMemory("reserve_space_4_meta")                     \
          .HostMemory("reserve_space_5_meta")                     \
          .TypeConstraint<T>("T")                                 \
          .TypeConstraint<U>("U"),                                \
      OneDnnFusedBatchNormGradOp<GPUDevice, T, U, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_OneDnnFusedBatchNormExGrad")                         \
          .Device(DEVICE_GPU)                                     \
          .HostMemory("y_backprop_meta")                          \
          .HostMemory("x_meta")                                   \
          .HostMemory("scale_meta")                               \
          .HostMemory("reserve_space_1_meta")                     \
          .HostMemory("reserve_space_2_meta")                     \
          .HostMemory("reserve_space_3_meta")                     \
          .HostMemory("offset_meta")                              \
          .HostMemory("y_meta")                                   \
          .HostMemory("x_backprop_meta")                          \
          .HostMemory("scale_backprop_meta")                      \
          .HostMemory("offset_backprop_meta")                     \
          .HostMemory("reserve_space_4_meta")                     \
          .HostMemory("reserve_space_5_meta")                     \
          .HostMemory("side_input_backprop_meta")                 \
          .TypeConstraint<T>("T")                                 \
          .TypeConstraint<U>("U"),                                \
      OneDnnFusedBatchNormGradOp<GPUDevice, T, U, true, true>);
REGISTER_FUSED_BATCHNORM_GRAD_GPU(float, float);
REGISTER_FUSED_BATCHNORM_GRAD_GPU(Eigen::bfloat16, float);
#undef REGISTER_FUSED_BATCHNORM_GRAD_GPU
#else
#define REGISTER_FUSED_BATCHNORM_CPU(T, U)                                     \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnFusedBatchNorm").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OneDnnFusedBatchNormOp<CPUDevice, T, T, false, false>);                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnFusedBatchNormV2")                                          \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<U>("U"),                                             \
      OneDnnFusedBatchNormOp<CPUDevice, T, U, false, false>);                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnFusedBatchNormV3")                                          \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<U>("U"),                                             \
      OneDnnFusedBatchNormOp<CPUDevice, T, U, true, false>);                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnFusedBatchNormEx")                                          \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<U>("U"),                                             \
      OneDnnFusedBatchNormOp<CPUDevice, T, U, true, true>);
REGISTER_FUSED_BATCHNORM_CPU(float, float);
REGISTER_FUSED_BATCHNORM_CPU(Eigen::bfloat16, float);
#undef REGISTER_FUSED_BATCHNORM_CPU

#define REGISTER_FUSED_BATCHNORM_GRAD_CPU(T, U)                   \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_OneDnnFusedBatchNormGrad")                           \
          .Device(DEVICE_CPU)                                     \
          .TypeConstraint<T>("T"),                                \
      OneDnnFusedBatchNormGradOp<CPUDevice, T, T, false, false>); \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_OneDnnFusedBatchNormGradV2")                         \
          .Device(DEVICE_CPU)                                     \
          .TypeConstraint<T>("T")                                 \
          .TypeConstraint<U>("U"),                                \
      OneDnnFusedBatchNormGradOp<CPUDevice, T, U, false, false>); \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_OneDnnFusedBatchNormGradV3")                         \
          .Device(DEVICE_CPU)                                     \
          .TypeConstraint<T>("T")                                 \
          .TypeConstraint<U>("U"),                                \
      OneDnnFusedBatchNormGradOp<CPUDevice, T, U, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_OneDnnFusedBatchNormExGrad")                         \
          .Device(DEVICE_CPU)                                     \
          .TypeConstraint<T>("T")                                 \
          .TypeConstraint<U>("U"),                                \
      OneDnnFusedBatchNormGradOp<CPUDevice, T, U, true, true>);
REGISTER_FUSED_BATCHNORM_GRAD_CPU(float, float);
REGISTER_FUSED_BATCHNORM_GRAD_CPU(Eigen::bfloat16, float);
#undef REGISTER_FUSED_BATCHNORM_GRAD_CPU
#endif
}  // namespace itex
