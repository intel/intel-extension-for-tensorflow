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

#ifndef ITEX_CORE_KERNELS_COMMON_FUSED_BATCH_NORM_OP_H_
#define ITEX_CORE_KERNELS_COMMON_FUSED_BATCH_NORM_OP_H_

#include <algorithm>
#include <string>
#include <unordered_map>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/kernels/common/fused_batch_norm_functor.h"
#include "itex/core/kernels/common/host_data_cache.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace itex {

// "is_batch_norm_ex" template argument is not used in kernel actually, since
// single _QuantizedBatchNorm INT8 op represent both with & without fusion
// condition. We cannot distinguish whether INT8 BN op contain fusion or not,
// simply based on its Op name. We now use class member is_batch_norm_ex_ to
// distinguish, based on Attr "activation_mode"
template <typename Device, typename T, typename U, bool reserved_space,
          bool is_batch_norm_ex>
class FusedBatchNormOp : public OpKernel {
  static constexpr bool use_reserved_space = reserved_space;

 public:
  explicit FusedBatchNormOp(OpKernelConstruction* context) : OpKernel(context) {
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

    if (context->HasAttr("num_side_inputs")) {
      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      if (num_side_inputs > 0) has_side_input_ = true;
    }

    if (context->HasAttr("activation_mode")) {
      FbnActivationMode activation_mode;
      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode));
      OP_REQUIRES(context,
                  activation_mode == FbnActivationMode::kRelu ||
                      activation_mode == FbnActivationMode::kIdentity,
                  errors::InvalidArgument(
                      "FusedBatchNorm only support Relu activation"));
      if (activation_mode == FbnActivationMode::kRelu) {
        is_batch_norm_ex_ = true;
      }
    }
    is_inplace_ = false;
    if (context->HasAttr("is_inplace")) {
      OP_REQUIRES_OK(context, context->GetAttr("is_inplace", &is_inplace_));
    }
  }
  // If use_reserved_space is true, we need to handle the 5th output (a reserved
  // space).
  // If use_reserved_space is false, we don't have 5th output.
  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      const size_t kSrcIndex = 0;       // index of src input tensor
      const size_t kScaleIndex = 1;     // index of scale tensor
      const size_t kShiftIndex = 2;     // index of shift tensor
      const size_t kMeanIndex = 3;      // index of est_mean tensor
      const size_t kVarianceIndex = 4;  // index of est_variance tensor
      const size_t kSrc1Index = 5;      // index of src side input tensor

      const Tensor& src_tensor = context->input(kSrcIndex);
      const Tensor& scale_tensor = context->input(kScaleIndex);
      const Tensor& shift_tensor = context->input(kShiftIndex);
      const Tensor& est_mean_tensor = context->input(kMeanIndex);
      const Tensor& est_variance_tensor = context->input(kVarianceIndex);

      OP_REQUIRES(context, src_tensor.dims() == 4 || src_tensor.dims() == 5,
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

      // Allocate 5 output TF tensors.
      Tensor* batch_mean_tensor = nullptr;
      Tensor* batch_variance_tensor = nullptr;
      Tensor* saved_mean_tensor = nullptr;
      Tensor* saved_variance_tensor = nullptr;
      Tensor* reserved_space_tensor = nullptr;

      // Handle the special case: input with 0 elements and 0 batch size.
      Tensor* dst_tensor = nullptr;
      TensorShape tf_shape_src = src_tensor.shape();
      TensorShape workspace_tf_shape;
      if (tf_shape_src.num_elements() == 0) {
        workspace_tf_shape.AddDim(0);
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {0}, 0, tf_shape_src, &dst_tensor));
        ITEX_DCHECK(dst_tensor);
        AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                          &batch_mean_tensor, &batch_variance_tensor,
                          &saved_mean_tensor, &saved_variance_tensor,
                          &reserved_space_tensor, true);

        return;
      } else {
        if (is_inplace_ && !std::is_same<T, qint8>::value) {
          context->set_output(0, src_tensor);
          dst_tensor = context->mutable_output(0);
        } else {
          OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                      {0}, 0, src_tensor.shape(), &dst_tensor));
        }
      }

      size_t depth =
          static_cast<size_t>(GetTensorDim(src_tensor, tensor_format_, 'C'));

      bool use_3d_format = src_tensor.dims() == 5;
      OneDnnTensorFormat dnnl_tensor_fmt =
          TFDataFormatToOneDnnDataFormat(tensor_format_, !use_3d_format);
      dnnl::memory::format_tag dnn_fmt =
          OneDnnTensorFormatToTag(dnnl_tensor_fmt);

      // Set src memory descriptor.
      dnnl::memory::dims src_dims = TFShapeToOneDnnDimsInNC(
          src_tensor.shape(), tensor_format_, !use_3d_format);

      auto src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(), dnn_fmt);
      auto scale_md =
          dnnl::memory::desc({static_cast<int64_t>(depth)}, OneDnnType<U>(),
                             dnnl::memory::format_tag::a);
      auto shift_md =
          dnnl::memory::desc({static_cast<int64_t>(depth)}, OneDnnType<U>(),
                             dnnl::memory::format_tag::a);
      auto propagation = (is_training_ || is_batch_norm_ex_)
                             ? dnnl::prop_kind::forward_training
#ifdef ITEX_ONEDNN_3_0
                             : dnnl::prop_kind::forward_inference;
#else
                             : dnnl::prop_kind::forward_scoring;
#endif
      auto flag = dnnl::normalization_flags::use_scale |
                  dnnl::normalization_flags::use_shift;
      if (!is_training_) {
        flag |= dnnl::normalization_flags::use_global_stats;
      }
      if (is_batch_norm_ex_) {
        if (has_side_input_) {
          flag |= dnnl::normalization_flags::fuse_norm_add_relu;
        } else {
          flag |= dnnl::normalization_flags::fuse_norm_relu;
        }
      }

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
      dnnl::batch_normalization_forward::primitive_desc bn_fwd_pd(
          onednn_engine, propagation, src_md, src_md, epsilon_, flag, attr);
#else
      dnnl::batch_normalization_forward::desc bn_fwd_desc(propagation, src_md,
                                                          epsilon_, flag);
      dnnl::batch_normalization_forward::primitive_desc bn_fwd_pd(
          bn_fwd_desc, attr, onednn_engine);
#endif

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          bn_fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(bn_fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      dnnl::batch_normalization_forward bn_fwd_primitive(bn_fwd_pd);

      if (is_batch_norm_ex_) {
        dnnl::memory::desc workspace_md = bn_fwd_pd.workspace_desc();
        size_t workspace_bytes = workspace_md.get_size();
        // Notice we need use ceiling here, since the required bytes may not
        // divisible by 4
        int num_elem = std::ceil(static_cast<float>(workspace_bytes) /
                                 static_cast<float>(sizeof(U)));
        workspace_tf_shape.AddDim(num_elem);

        AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                          &batch_mean_tensor, &batch_variance_tensor,
                          &saved_mean_tensor, &saved_variance_tensor,
                          &reserved_space_tensor);
      } else {
        // There is actually no workspace tensor out, so we make a dummy one.
        workspace_tf_shape.AddDim(0);
        AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                          &batch_mean_tensor, &batch_variance_tensor,
                          &saved_mean_tensor, &saved_variance_tensor,
                          &reserved_space_tensor);
      }

      void* src_data = GetTensorBuffer<T>(&src_tensor);
      void* dst_data = GetTensorBuffer<T>(dst_tensor);
      void* ws_op_data =
          reserved_space ? GetTensorBuffer<U>(reserved_space_tensor) : nullptr;

      void *scale_data, *shift_data;
      Tensor quantized_scaled_shift_tensor;
      scale_data = GetTensorBuffer<U>(&scale_tensor);
      if (IsQuantizedInput()) {
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<U>::value,
                                              shift_tensor.shape(),
                                              &quantized_scaled_shift_tensor));
        // For INT8 BN, we need to scale the shift tensor
        ScaleShiftOrMean(context, shift_tensor, &quantized_scaled_shift_tensor,
                         onednn_engine, onednn_stream);
        shift_data = GetTensorBuffer<U>(&quantized_scaled_shift_tensor);
      } else {
        shift_data = GetTensorBuffer<U>(&shift_tensor);
      }

      void *mean_op_data, *variance_op_data;
      Tensor quantized_scaled_mean_tensor;
      if (is_training_) {
        mean_op_data = GetTensorBuffer<U>(saved_mean_tensor);
        variance_op_data = GetTensorBuffer<U>(saved_variance_tensor);
      } else {
        if (IsQuantizedInput()) {
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<U>::value,
                                                est_mean_tensor.shape(),
                                                &quantized_scaled_mean_tensor));
          // For INT8 BN, we need to scale the mean tensor
          ScaleShiftOrMean(context, est_mean_tensor,
                           &quantized_scaled_mean_tensor, onednn_engine,
                           onednn_stream);
          mean_op_data = GetTensorBuffer<U>(&quantized_scaled_mean_tensor);
        } else {
          mean_op_data = GetTensorBuffer<U>(&est_mean_tensor);
        }

        variance_op_data = GetTensorBuffer<U>(&est_variance_tensor);
      }

      auto src_mem =
          CreateDnnlMemory(bn_fwd_pd.src_desc(), onednn_engine, src_data);
      auto dst_mem =
          CreateDnnlMemory(bn_fwd_pd.dst_desc(), onednn_engine, dst_data);
      auto scale_mem = CreateDnnlMemory(scale_md, onednn_engine, scale_data);
      auto shift_mem = CreateDnnlMemory(shift_md, onednn_engine, shift_data);
      auto mean_memory =
          CreateDnnlMemory(bn_fwd_pd.mean_desc(), onednn_engine, mean_op_data);
      auto var_memory = CreateDnnlMemory(bn_fwd_pd.variance_desc(),
                                         onednn_engine, variance_op_data);

      dnnl::memory ws_memory;
      if (is_batch_norm_ex_)
        ws_memory = CreateDnnlMemory(bn_fwd_pd.workspace_desc(), onednn_engine,
                                     ws_op_data);
      dnnl::memory src1_mem;
      if (has_side_input_) {
        const Tensor& src1_tensor = context->input(kSrc1Index);
        void* src1_data = GetTensorBuffer<T>(&src1_tensor);
        src1_mem =
            CreateDnnlMemory(bn_fwd_pd.src_desc(), onednn_engine, src1_data);
      }

      // Execute
      std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_SRC, src_mem},
                                                    {DNNL_ARG_DST, dst_mem}};
      if (static_cast<bool>(flag & dnnl::normalization_flags::use_scale))
        args.insert({DNNL_ARG_SCALE, scale_mem});
      if (static_cast<bool>(flag & dnnl::normalization_flags::use_shift))
        args.insert({DNNL_ARG_SHIFT, shift_mem});
      if (is_batch_norm_ex_) args.insert({DNNL_ARG_WORKSPACE, ws_memory});

      args.insert({DNNL_ARG_MEAN, mean_memory});
      args.insert({DNNL_ARG_VARIANCE, var_memory});
      args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      if (has_side_input_) {
        args.insert({DNNL_ARG_SRC_1, src1_mem});
      }

      bn_fwd_primitive.execute(onednn_stream, args);

      // For inference, we don't need to calculate running mean/var.
      if (!is_training_) return;

      // Calculate running mean/var.
      float adjust_factor = 1.0;
      size_t orig_size = src_dims[0] * src_dims[2] * src_dims[3];
      if (use_3d_format) orig_size *= src_dims[4];
      size_t adjust_size = (orig_size > 1) ? (orig_size - 1) : 1;
      adjust_factor = (static_cast<float>(orig_size)) / adjust_size;

      U *mean_data = nullptr, *variance_data = nullptr;
      mean_data = saved_mean_tensor->flat<U>().data();
      variance_data = saved_variance_tensor->flat<U>().data();
      auto batch_mean_data = batch_mean_tensor->flat<U>().data();
      auto batch_variance_data = batch_variance_tensor->flat<U>().data();
      auto est_mean_data = est_mean_tensor.flat<U>().data();
      auto est_variance_data = est_variance_tensor.flat<U>().data();

#ifndef INTEL_CPU_ONLY
      auto* gpu_stream = context->GetDeviceStream();
      auto total_threads =
          gpu_stream->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      if (exponential_avg_factor_ == U(1.0)) {
        gpu_stream->submit([&](sycl::handler& cgh) {
          auto batch_mean_data_ptr = static_cast<U*>(batch_mean_data);
          auto mean_data_ptr = static_cast<U*>(mean_data);
          auto batch_variance_data_ptr = static_cast<U*>(batch_variance_data);
          auto variance_data_ptr = static_cast<U*>(variance_data);
          int local_depth = depth;

          cgh.parallel_for<
              VarAdjust<T, U, use_reserved_space, is_batch_norm_ex>>(
              sycl::range<1>(total_threads), [=](sycl::item<1> item) {
                auto id = item.get_id(0);
                for (auto k = id; k < local_depth; k += total_threads) {
                  batch_mean_data_ptr[k] = mean_data_ptr[k];
                  batch_variance_data_ptr[k] =
                      static_cast<U>(adjust_factor) * variance_data_ptr[k];
                }
              });
        });
      } else {
        U one_minus_factor = U(1.0) - exponential_avg_factor_;
        U exponential_avg_factor = exponential_avg_factor_;
        gpu_stream->submit([&](sycl::handler& cgh) {
          auto batch_mean_data_ptr = batch_mean_data;
          auto est_mean_data_ptr = est_mean_data;
          auto mean_data_ptr = mean_data;
          auto batch_variance_data_ptr = batch_variance_data;
          auto est_variance_data_ptr = est_variance_data;
          auto variance_data_ptr = variance_data;
          int local_depth = depth;

          cgh.parallel_for<
              VarAdjustMinus<T, U, use_reserved_space, is_batch_norm_ex>>(
              sycl::range<1>(total_threads), [=](sycl::item<1> item) {
                auto id = item.get_id(0);
                for (auto k = id; k < local_depth; k += total_threads) {
                  batch_mean_data_ptr[k] =
                      one_minus_factor * est_mean_data_ptr[k] +
                      exponential_avg_factor * mean_data_ptr[k];
                  batch_variance_data_ptr[k] =
                      one_minus_factor * est_variance_data_ptr[k] +
                      exponential_avg_factor * static_cast<U>(adjust_factor) *
                          variance_data_ptr[k];
                }
              });
        });
      }
#else
      // TODO(itex): Parallel_for for CPU code.
      if (exponential_avg_factor_ == U(1.0)) {
        for (int k = 0; k < depth; k++) {
          batch_mean_data[k] = mean_data[k];
          batch_variance_data[k] =
              static_cast<U>(adjust_factor) * variance_data[k];
        }
      } else {
        U one_minus_factor = U(1.0) - exponential_avg_factor_;
        for (int k = 0; k < depth; k++) {
          batch_mean_data[k] = one_minus_factor * est_mean_data[k] +
                               exponential_avg_factor_ * mean_data[k];
          batch_variance_data[k] = one_minus_factor * est_variance_data[k] +
                                   exponential_avg_factor_ *
                                       static_cast<U>(adjust_factor) *
                                       variance_data[k];
        }
      }
#endif  // INTEL_CPU_ONLY
    } catch (dnnl::error& e) {
      string error_msg = "Status:" + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  // This function is only used in INT8 kernels
  virtual void ScaleShiftOrMean(OpKernelContext* context,
                                const Tensor& tensor_in, Tensor* tensor_out,
                                const dnnl::engine& engine,
                                const dnnl::stream& stream) {}

 protected:
  const bool IsQuantizedInput() { return is_quantized_input_; }
  void SetQuantizedInput(const bool is_quantized_input) {
    is_quantized_input_ = is_quantized_input;
  }

 private:
  bool is_inplace_;
  float epsilon_;
  U exponential_avg_factor_;
  TensorFormat tensor_format_;
  bool is_training_;
  bool has_side_input_ = false;
  bool is_quantized_input_ = false;
  bool is_batch_norm_ex_ = false;

  virtual void AllocateTFOutputs(
      OpKernelContext* context, TensorShape tf_shape_scale,
      TensorShape workspace_tf_shape, Tensor** batch_mean_tensor,
      Tensor** batch_variance_tensor, Tensor** saved_mean_tensor,
      Tensor** saved_variance_tensor, Tensor** reserved_space_tensor,
      bool init_val = false) {
    ITEX_DCHECK(batch_mean_tensor);
    ITEX_DCHECK(batch_variance_tensor);
    ITEX_DCHECK(saved_mean_tensor);
    ITEX_DCHECK(saved_variance_tensor);

    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {3}, 1, tf_shape_scale, batch_mean_tensor));
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {4}, 2, tf_shape_scale, batch_variance_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(3, tf_shape_scale,
                                                     saved_mean_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(4, tf_shape_scale,
                                                     saved_variance_tensor));

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

    if (use_reserved_space)
      OP_REQUIRES_OK(context, context->allocate_output(5, workspace_tf_shape,
                                                       reserved_space_tensor));
  }
};

template <typename Device, typename T, typename U, bool reserved_space,
          bool is_batch_norm_ex>
class QuantizedFusedBatchNormOp
    : public FusedBatchNormOp<Device, T, U, reserved_space, is_batch_norm_ex> {
 public:
  explicit QuantizedFusedBatchNormOp(OpKernelConstruction* context)
      : FusedBatchNormOp<Device, T, U, reserved_space, is_batch_norm_ex>(
            context) {
    DataType input_dt;
    OP_REQUIRES_OK(context, context->GetAttr("T", &input_dt));
    OP_REQUIRES(
        context, input_dt == DT_QINT8,
        errors::InvalidArgument(
            "_QuantizedFusedBatchNorm only supports qint8 input data type."));
    this->SetQuantizedInput(true);
    OP_REQUIRES_OK(context, context->GetAttr("Tout", &out_dt_));
    OP_REQUIRES(
        context, out_dt_ == DT_QINT8,
        errors::InvalidArgument("_QuantizedFusedBatchNorm output data type "
                                "should be either qint8."));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(
        context, data_format == "NHWC",
        errors::InvalidArgument("_QuantizedFusedBatchNorm only supports NHWC "
                                "data format."));
    // Inputs to this op are expected as follows, {} means optional inputs:
    // 0. x
    // 1. scale
    // 2. offset
    // 3. mean
    // 4. variance
    // 5. x_min
    // 6. x_max
    // 7. {output_min}
    // 8. {output_max}
  }

  void Compute(OpKernelContext* context) override {
    FusedBatchNormOp<Device, T, U, reserved_space, is_batch_norm_ex>::Compute(
        context);
    if (out_dt_ == DT_QINT8) {
      // TODO(itex): here code may has some bugs. but we just follow Intel-TF
      // implementation. It assumes the min/max of input & output of Batchnorm
      // are the same. In reality, the assumption is not always true, but
      // currently, we don't receive model accuracy issue report.
      context->set_output(1, context->input(5));
      context->set_output(2, context->input(6));
    }
  }

  void ScaleShiftOrMean(OpKernelContext* context, const Tensor& tensor_in,
                        Tensor* tensor_out, const dnnl::engine& engine,
                        const dnnl::stream& stream) override {
    if (out_dt_ == DT_FLOAT) {
      return;
    }
    // Scale offset or mean tensor via reorder

#ifdef INTEL_CPU_ONLY
    float min = context->input(5).flat<float>()(0);
    float max = context->input(6).flat<float>()(0);
#else
    // Intel TF BN INT8 op doesn't set min/max as host tensor
    float min, max;
    void* min_host_data = static_cast<void*>(&min);
    void* max_host_data = static_cast<void*>(&max);
    const void* min_device_data = context->input(5).data();
    const void* max_device_data = context->input(6).data();

    auto* gpu_stream = context->GetDeviceStream();
    DeviceMemcpy<Device>(min_host_data, min_device_data, 1 * sizeof(float),
                         gpu_stream);
    DeviceMemcpy<Device>(max_host_data, max_device_data, 1 * sizeof(float),
                         gpu_stream);
#endif  // INTEL_CPU_ONLY

    const float max_abs = std::max(std::abs(min), std::abs(max));
    float scale = 127.0f / max_abs;
    dnnl::primitive_attr scale_attr;
#ifdef ITEX_ONEDNN_3_0
    scale_attr.set_scales_mask(DNNL_ARG_SRC, 0);
#else
    scale_attr.set_output_scales(0, {scale});
#endif
    auto input_md =
        dnnl::memory::desc({tensor_in.NumElements()}, OneDnnType<U>(),
                           dnnl::memory::format_tag::x);
    dnnl::memory input_mem =
        dnnl::memory(input_md, engine, GetTensorBuffer<U>(&tensor_in));
    dnnl::memory scaled_input_mem =
        dnnl::memory(input_md, engine, GetTensorBuffer<U>(tensor_out));
#ifdef ITEX_ONEDNN_3_0
    float* output_scale_ptr =
        output_scale_cache_.GetCachedPtr(context, &scale, 1);
    dnnl::memory scale_mem(
        {{1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x},
        engine, output_scale_ptr);
#endif
    dnnl::reorder reorder_pd =
        dnnl::reorder(input_mem, scaled_input_mem, scale_attr);
    std::unordered_map<int, dnnl::memory> reorder_args = {
        {DNNL_ARG_SRC, input_mem},
        {DNNL_ARG_DST, scaled_input_mem},
#ifdef ITEX_ONEDNN_3_0
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_mem},
#endif
    };
    reorder_pd.execute(stream, reorder_args);
  }

  void AllocateTFOutputs(OpKernelContext* context, TensorShape tf_shape_scale,
                         TensorShape workspace_tf_shape,
                         Tensor** batch_mean_tensor,
                         Tensor** batch_variance_tensor,
                         Tensor** saved_mean_tensor,
                         Tensor** saved_variance_tensor,
                         Tensor** reserved_space_tensor,
                         bool init_val = false) override {
    // No additional outputs are needed for quantized kernel.
    return;
  }

 protected:
  DataType out_dt_;
#ifdef ITEX_ONEDNN_3_0
  HostDataCache<Device, float> output_scale_cache_;
#endif
};

template <typename Device, typename T, typename U, bool reserved_space,
          bool is_batch_norm_ex = false>
class FusedBatchNormGradOp : public OpKernel {
 public:
  explicit FusedBatchNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));

    if (context->HasAttr("num_side_inputs")) {
      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      if (num_side_inputs > 0) has_side_input_ = true;
    }

    if (context->HasAttr("activation_mode")) {
      FbnActivationMode activation_mode;
      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode));
      OP_REQUIRES(context,
                  activation_mode == FbnActivationMode::kReluGrad ||
                      activation_mode == FbnActivationMode::kIdentity,
                  errors::InvalidArgument(
                      "FusedBatchNorm only support ReluGrad activation"));
      if (activation_mode == FbnActivationMode::kReluGrad) {
        is_batch_norm_ex_ = true;
      }
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
      // const Tensor& reserved_space_tensor = (reserved_space) ?
      // context->input(kReservedSpaceIndex) : Tensor();
      const Tensor* reserved_space_tensor;
      Tensor* diff_src_tensor = nullptr;
      Tensor* diff_side_input_tensor = nullptr;

      if (reserved_space)
        reserved_space_tensor = &context->input(kReservedSpaceIndex);
      else
        reserved_space_tensor = new Tensor;

      TensorShape tf_shape_src = src_tensor.shape();
      TensorShape tf_shape_diff_dst = diff_dst_tensor.shape();

      OP_REQUIRES(context,
                  diff_dst_tensor.dims() == 4 || diff_dst_tensor.dims() == 5,
                  errors::InvalidArgument(
                      "input must be 4-dimensional or 5-dimensional",
                      diff_dst_tensor.shape().DebugString()));
      OP_REQUIRES(context, src_tensor.dims() == 4 || src_tensor.dims() == 5,
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
      Tensor* diff_scale_tensor = nullptr;
      Tensor* diff_shift_tensor = nullptr;

      // Handle the special case: input with 0 element and 0 batch size.
      if (tf_shape_src.num_elements() == 0 ||
          tf_shape_diff_dst.num_elements() == 0) {
        OP_REQUIRES_OK(context,
                       context->allocate_output(kDiffSrcIndex, tf_shape_src,
                                                &diff_src_tensor));
        ITEX_DCHECK(diff_src_tensor);

        auto diff_src_data = diff_src_tensor->flat<T>().data();
        std::fill_n(diff_src_data, diff_src_tensor->shape().num_elements(),
                    static_cast<T>(0));
        AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                          &diff_shift_tensor, true);

        return;
      } else {
        OP_REQUIRES_OK(
            context, context->allocate_output(kDiffSrcIndex, src_tensor.shape(),
                                              &diff_src_tensor));
        // Allocate output TF tensors diff_scale and diff_shift.
        AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                          &diff_shift_tensor);
        if (has_side_input_) {
          OP_REQUIRES_OK(context, context->allocate_output(
                                      kDiffSrc1Index, src_tensor.shape(),
                                      &diff_side_input_tensor));
        }
      }

      size_t depth =
          static_cast<size_t>(GetTensorDim(src_tensor, tensor_format_, 'C'));

      bool use_3d_format = src_tensor.dims() == 5;
      OneDnnTensorFormat dnnl_tensor_fmt =
          TFDataFormatToOneDnnDataFormat(tensor_format_, !use_3d_format);
      dnnl::memory::format_tag dnn_fmt =
          OneDnnTensorFormatToTag(dnnl_tensor_fmt);
      dnnl::memory::dims src_dims = TFShapeToOneDnnDimsInNC(
          src_tensor.shape(), tensor_format_, !use_3d_format);
      dnnl::memory::dims diff_dst_dims = TFShapeToOneDnnDimsInNC(
          diff_dst_tensor.shape(), tensor_format_, !use_3d_format);

      // Set src and diff_dst primitive descriptors.
      auto src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(), dnn_fmt);
      auto diff_dst_md =
          dnnl::memory::desc(diff_dst_dims, OneDnnType<T>(), dnn_fmt);
      auto scale_md =
          dnnl::memory::desc({static_cast<int64_t>(depth)}, OneDnnType<U>(),
                             dnnl::memory::format_tag::a);
      auto shift_md =
          dnnl::memory::desc({static_cast<int64_t>(depth)}, OneDnnType<U>(),
                             dnnl::memory::format_tag::a);
      auto propagation_fwd = dnnl::prop_kind::forward_training;
      auto propagation_bwd = dnnl::prop_kind::backward;

      auto flag = dnnl::normalization_flags::use_scale |
                  dnnl::normalization_flags::use_shift;
      if (!is_training_) {
        flag |= dnnl::normalization_flags::use_global_stats;
      }
      if (is_batch_norm_ex_) {
        if (has_side_input_) {
          flag |= dnnl::normalization_flags::fuse_norm_add_relu;
        } else {
          flag |= dnnl::normalization_flags::fuse_norm_relu;
        }
      }

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef ITEX_ONEDNN_3_0
      dnnl::batch_normalization_forward::primitive_desc bn_fwd_pd(
          onednn_engine, propagation_fwd, src_md, src_md, epsilon_, flag, attr);
      dnnl::batch_normalization_backward::primitive_desc bn_bwd_pd(
          onednn_engine, propagation_bwd, diff_dst_md, diff_dst_md, src_md,
          epsilon_, flag, bn_fwd_pd, attr);
#else
      dnnl::batch_normalization_forward::desc bn_fwd_desc(
          propagation_fwd, src_md, epsilon_, flag);
      dnnl::batch_normalization_backward::desc bn_bwd_desc(
          propagation_bwd, diff_dst_md, src_md, epsilon_, flag);

      dnnl::batch_normalization_forward::primitive_desc bn_fwd_pd(
          bn_fwd_desc, attr, onednn_engine);
      dnnl::batch_normalization_backward::primitive_desc bn_bwd_pd(
          bn_bwd_desc, attr, onednn_engine, bn_fwd_pd);
#endif

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          bn_bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(bn_bwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      dnnl::batch_normalization_backward bn_bwd_primitive(bn_bwd_pd);
#ifndef ITEX_ONEDNN_3_0
      // OneDnn requests an empty shift tensor.
      Tensor shift_tensor;
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<U>::v(),
                                          scale_tensor.shape(), &shift_tensor));
#endif
      void* src_data = GetTensorBuffer<T>(&src_tensor);
      void* diff_dst_data = GetTensorBuffer<T>(&diff_dst_tensor);
      void* mean_data = GetTensorBuffer<U>(&saved_mean_tensor);
      void* variance_data = GetTensorBuffer<U>(&saved_variance_tensor);
      void* scale_data = GetTensorBuffer<U>(&scale_tensor);
#ifndef ITEX_ONEDNN_3_0
      void* shift_data = GetTensorBuffer<U>(&shift_tensor);
#endif
      void* diff_src_data = GetTensorBuffer<T>(diff_src_tensor);
      void* diff_src1_data = nullptr;
      if (has_side_input_) {
        diff_src1_data = GetTensorBuffer<T>(diff_side_input_tensor);
      }
      void* diff_scale_data = GetTensorBuffer<U>(diff_scale_tensor);
      void* diff_shift_data = GetTensorBuffer<U>(diff_shift_tensor);

      void* res_space_data = is_batch_norm_ex_
                                 ? GetTensorBuffer<U>(reserved_space_tensor)
                                 : nullptr;
      auto src_mem =
          CreateDnnlMemory(bn_bwd_pd.src_desc(), onednn_engine, src_data);
      auto scale_mem = CreateDnnlMemory(scale_md, onednn_engine, scale_data);
#ifndef ITEX_ONEDNN_3_0
      auto shift_mem = CreateDnnlMemory(shift_md, onednn_engine, shift_data);
#endif
      auto mean_mem =
          CreateDnnlMemory(bn_bwd_pd.mean_desc(), onednn_engine, mean_data);
      auto variance_mem = CreateDnnlMemory(bn_bwd_pd.variance_desc(),
                                           onednn_engine, variance_data);
      auto diff_src_mem = CreateDnnlMemory(bn_bwd_pd.diff_src_desc(),
                                           onednn_engine, diff_src_data);
      auto diff_dst_mem = CreateDnnlMemory(bn_bwd_pd.diff_dst_desc(),
                                           onednn_engine, diff_dst_data);
      auto diff_scale_mem =
          CreateDnnlMemory(scale_md, onednn_engine, diff_scale_data);
      auto diff_shift_mem =
          CreateDnnlMemory(shift_md, onednn_engine, diff_shift_data);
      auto ws_mem = CreateDnnlMemory(bn_bwd_pd.workspace_desc(), onednn_engine,
                                     res_space_data);

      dnnl::memory diff_src1_mem;
      if (has_side_input_) {
        diff_src1_mem =
            CreateDnnlMemory(diff_dst_md, onednn_engine, diff_src1_data);
      }
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, dnnl::memory> args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_MEAN, mean_mem},
          {DNNL_ARG_VARIANCE, variance_mem},
          {DNNL_ARG_DIFF_DST, diff_dst_mem},
          {DNNL_ARG_DIFF_SRC, diff_src_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem},
          {DNNL_ARG_SCALE, scale_mem},
#ifndef ITEX_ONEDNN_3_0
          // https://github.com/intel-innersource/libraries.performance.math.onednn/commit/dccfdc25f7504a620a1fc2dc9602eefa24258147#diff-12751859c2f7964388e0bb75f7db610b4cc9f3320c9aac6c7167312fb5c5fbccR302
          // when calculate n_inputs, they remove use_shift()
          {DNNL_ARG_SHIFT, shift_mem},
#endif
          {DNNL_ARG_DIFF_SCALE, diff_scale_mem},
          {DNNL_ARG_DIFF_SHIFT, diff_shift_mem}};

      if (is_batch_norm_ex_) args.insert({DNNL_ARG_WORKSPACE, ws_mem});
      if (has_side_input_) args.insert({DNNL_ARG_DIFF_SRC_1, diff_src1_mem});

      bn_bwd_primitive.execute(onednn_stream, args);

      if (!reserved_space) delete reserved_space_tensor;
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
  bool is_batch_norm_ex_ = false;

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

    functor::SetZeroFunctor<Device, U> f_zero;

    // Separate out scale and shift grad and copy to individual tensors
    OP_REQUIRES_OK(
        context, context->allocate_output(kDiffScaleIndex, tf_shape_scale_shift,
                                          diff_scale_tensor));
    ITEX_DCHECK(*diff_scale_tensor);

    OP_REQUIRES_OK(
        context, context->allocate_output(kDiffShiftIndex, tf_shape_scale_shift,
                                          diff_shift_tensor));
    ITEX_DCHECK(*diff_shift_tensor);

    // Placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    Tensor *p1_tensor = nullptr, *p2_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(kP1Index, TensorShape({}),
                                                     &p1_tensor));
    ITEX_DCHECK(p1_tensor);
    OP_REQUIRES_OK(context, context->allocate_output(kP2Index, TensorShape({}),
                                                     &p2_tensor));
    ITEX_DCHECK(p2_tensor);

    if (init_val) {
      f_zero(context->eigen_device<Device>(), (*diff_scale_tensor)->flat<U>());
      f_zero(context->eigen_device<Device>(), (*diff_shift_tensor)->flat<U>());
      f_zero(context->eigen_device<Device>(), (p1_tensor)->flat<U>());
      f_zero(context->eigen_device<Device>(), (p2_tensor)->flat<U>());
    }
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_FUSED_BATCH_NORM_OP_H_
