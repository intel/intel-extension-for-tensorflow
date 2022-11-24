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

#ifndef ITEX_CORE_KERNELS_COMMON_QUANTIZED_MATMUL_COMMON_H_
#define ITEX_CORE_KERNELS_COMMON_QUANTIZED_MATMUL_COMMON_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/kernels/common/no_ops.h"
#include "itex/core/kernels/onednn/block/quantized_ops.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/quantization_util.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

using memory = dnnl::memory;

template <typename Device, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class LegacyQuantizedMatMulOpBase : public OpKernel {
 public:
  explicit LegacyQuantizedMatMulOpBase(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_a", &this->transpose_a_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_b", &this->transpose_b_));
    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));
  }
  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);
    dst_tensor_ = nullptr;
    onednn_engine_ = CreateDnnlEngine<Device>(*context);
    // onednn_stream has thread safety issue, need create a new one in
    // every compute.
    onednn_stream_ = CreateDnnlStream(*context, onednn_engine_);
    scratchpad_tensor_ = std::make_shared<Tensor>();
    InitOrSetMemory(context);

    if (is_input_zero_) {
      functor::SetZeroFunctor<Device, Toutput> f;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  kOutputIndex_Dst, dst_shape_, &dst_tensor_));
      f(context->eigen_device<Device>(), dst_tensor_->flat<Toutput>());
      AllocateNativeOutputMinMax<Tinput, Tweight, Toutput>(
          context, kSrcMinRangeIndex, kSrcMaxRangeIndex, kFilterMinRangeIndex,
          kFilterMaxRangeIndex, kMinFreezedIndex, kMaxFreezedIndex,
          kDstMinRangeIndex, kDstMaxRangeIndex);

      scratchpad_tensor_.reset();
      return;
    }

    fwd_primitive_.execute(onednn_stream_, fwd_primitive_args_);
    scratchpad_tensor_.reset();

    AllocateNativeOutputMinMax<Tinput, Tweight, Toutput>(
        context, kSrcMinRangeIndex, kSrcMaxRangeIndex, kFilterMinRangeIndex,
        kFilterMaxRangeIndex, kMinFreezedIndex, kMaxFreezedIndex,
        kDstMinRangeIndex, kDstMaxRangeIndex);
  }

  void InitOrSetMemory(OpKernelContext* context) {
    if (enable_cache_ && is_init_ && context->is_input_same(0, input_dims_)) {
      ITEX_VLOG(3) << "Hit ITEX native MatMul INT8 object cache";
      src_mem_.set_data_handle(context->tensor_data(kInputIndex_Src));

      if (is_weight_reorder_) {
        if (!is_weight_const_) {
          weight_mem_.set_data_handle(context->tensor_data(kInputIndex_Filter));
          weight_mem_opt_.set_data_handle(
              GetTensorBuffer<Tweight>(&tmp_weight_));
          ReorderMemory(*context, &weight_mem_, &weight_mem_opt_,
                        onednn_engine_);
          weight_mem_ = weight_mem_opt_;
        }
      } else {
        weight_mem_.set_data_handle(context->tensor_data(kInputIndex_Filter));
      }

      if (post_op_util_.HasBias()) {
        // TODO(itex): avoid to use context->input, which may can C API and
        // trigger new twice, causing overhead
        const Tensor& bias_tensor = context->input(kInputIndex_Bias);

        // Note: The asymmetric compensation is calculate in bias handle
        // TODO(itex): improve the code immplementation here
        Tensor scaled_bias_tensor;
        Tbias* scaled_bias_data;
        if (std::is_same<Tweight, qint8>::value) {
          scaled_bias_data = this->GetScaledBias(context, fwd_pd_, bias_tensor,
                                                 &scaled_bias_tensor);
        }

        Tbias* bias_data =
            std::is_same<Tweight, qint8>::value
                ? scaled_bias_data
                : const_cast<Tbias*>(bias_tensor.flat<Tbias>().data());

        bias_mem_.set_data_handle(bias_data);
      }

      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                            TensorShape({scratchpad_size_}),
                                            scratchpad_tensor_.get()));
      scratchpad_mem_.set_data_handle(
          GetTensorBuffer<Tinput>(scratchpad_tensor_.get()));

      AllocateOutputTensor(context, fwd_pd_, dst_dims_onednn_, dst_shape_,
                           &dst_tensor_);

      // Set dst mem if output need reorder.
      // Here is trick to calculate INT8 conv + bias + add + relu, where
      // Tsummand is s8, and Toutput is u8
      dst_mem_.set_data_handle(GetTensorBuffer<Toutput>(dst_tensor_));

    } else {
      Init(context);
    }
  }

  void Init(OpKernelContext* context) {
    try {
      // Input tensors
      const Tensor& src_tensor = context->input(this->kInputIndex_Src);
      const Tensor& weight_tensor = context->input(this->kInputIndex_Filter);
      const Tensor& bias_tensor = context->input(this->kInputIndex_Bias);

      fwd_primitive_args_.clear();

      // Get shapes of input & filter tensors
      TensorShape src_tf_shape = src_tensor.shape();
      TensorShape weight_tf_shape = weight_tensor.shape();

      input_dims_.clear();
      for (int i = 0; i < src_tf_shape.dims(); ++i) {
        input_dims_.push_back(src_tf_shape.dim_size(i));
      }

      memory::dims src_dims, weight_dims;

      const int batch = this->transpose_a_ ? src_tf_shape.dim_size(1)
                                           : src_tf_shape.dim_size(0);
      const int k = this->transpose_a_ ? src_tf_shape.dim_size(0)
                                       : src_tf_shape.dim_size(1);
      const int channel = this->transpose_b_ ? weight_tf_shape.dim_size(0)
                                             : weight_tf_shape.dim_size(1);

      src_dims = {batch, k};
      weight_dims = {channel, k};
      dst_dims_onednn_ = {batch, channel};

      // Create memory for user data.
      // Describe how the inputs and outputs of inner-product look like. Also
      // specify buffers containing actual input and output data.
      auto src_md =
          memory::desc(src_dims, OneDnnType<Tinput>(), memory::format_tag::nc);

      auto weight_md = memory::desc(
          weight_dims, OneDnnType<Tweight>(),
          this->transpose_b_ ? memory::format_tag::oi : memory::format_tag::io);

      auto weight_exec_md = memory::desc(weight_dims, OneDnnType<Tweight>(),
                                         memory::format_tag::any);

      dnnl::memory::dims bias_dims = {
          static_cast<int>(bias_tensor.dim_size(0))};

      auto bias_md =
          memory::desc(bias_dims, OneDnnType<Tbias>(), memory::format_tag::x);

      auto dst_md = memory::desc(dst_dims_onednn_, OneDnnType<Toutput>(),
                                 memory::format_tag::nc);

      // Note: Extend the basic parameters for data types and fusions.
      this->ExtendInt8PostOps(context);

      auto fwd_desc = dnnl::inner_product_forward::desc(
          dnnl::prop_kind::forward_inference, src_md, weight_exec_md, bias_md,
          dst_md);

      // Set post op attribution.
      dnnl::primitive_attr post_ops_attr;
      this->post_op_util_.SetPostOpAttr(&post_ops_attr);
      post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      fwd_pd_ = dnnl::inner_product_forward::primitive_desc(
          fwd_desc, post_ops_attr, onednn_engine_);
      fwd_primitive_ = dnnl::inner_product_forward(fwd_pd_);

      // Allocate output Tensor.
      dst_shape_ = TensorShape({batch, channel});

      this->AllocateOutputTensor(context, fwd_pd_, dst_dims_onednn_, dst_shape_,
                                 &dst_tensor_);

      // Create src memory, check if src needs to be reordered
      src_mem_ = CreateDnnlMemory(src_md, onednn_engine_,
                                  GetTensorBuffer<Tinput>(&src_tensor));

      const Tweight* weight_data = weight_tensor.flat<Tweight>().data();
      memory::desc expected_md = fwd_pd_.weights_desc();

      is_weight_reorder_ = (weight_md != expected_md);
      if (is_weight_reorder_) {
        if (this->weight_cache_manager.IsEmpty()) {
          // Cache weight in first time executing this node
          this->weight_cache_manager.SetCache(
              context, weight_md, expected_md,
              static_cast<void*>(const_cast<Tweight*>(weight_data)),
              onednn_engine_);
        }
        Tweight* weight_cached_data =
            this->weight_cache_manager.GetCache(context, expected_md);

        if (weight_cached_data != nullptr) {
          weight_mem_ =
              CreateDnnlMemory(expected_md, onednn_engine_, weight_cached_data);
        } else {
          // Reorder if cache is failed since pd has already used any format.
          int64_t reorder_size = expected_md.get_size() / sizeof(Tweight);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<Tweight>::v(),
                                                TensorShape({reorder_size}),
                                                &tmp_weight_));
          void* data_handle = GetTensorBuffer<Tweight>(&tmp_weight_);
          weight_mem_opt_ =
              CreateDnnlMemory(expected_md, onednn_engine_, data_handle);
          ReorderMemory(*context, &weight_mem_, &weight_mem_opt_,
                        onednn_engine_);
          weight_mem_ = weight_mem_opt_;
        }
      } else {
        // No reorder needed
        weight_mem_ = CreateDnnlMemory(
            weight_md, onednn_engine_,
            static_cast<void*>(const_cast<Tweight*>(weight_data)));
      }

      // Create dst memory
      Toutput* dst_data = dst_tensor_->flat<Toutput>().data();
      dst_mem_ = CreateDnnlMemory(fwd_pd_.dst_desc(), onednn_engine_,
                                  static_cast<void*>(dst_data));

      scratchpad_size_ = fwd_pd_.scratchpad_desc().get_size() / sizeof(Tinput);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                            TensorShape({scratchpad_size_}),
                                            scratchpad_tensor_.get()));
      scratchpad_mem_ =
          dnnl::memory(fwd_pd_.scratchpad_desc(), onednn_engine_,
                       GetTensorBuffer<Tinput>(scratchpad_tensor_.get()));

      // Execute MatMul INT8
      fwd_primitive_args_ = {{DNNL_ARG_SRC, src_mem_},
                             {DNNL_ARG_WEIGHTS, weight_mem_},
                             {DNNL_ARG_DST, dst_mem_},
                             {DNNL_ARG_SCRATCHPAD, scratchpad_mem_}};

      // Note: The asymmetric compensation is calculate in bias handle
      Tensor scaled_bias_tensor;
      Tbias* scaled_bias_data;
      if (std::is_same<Tweight, qint8>::value) {
        scaled_bias_data = this->GetScaledBias(context, fwd_pd_, bias_tensor,
                                               &scaled_bias_tensor);
      }

      Tbias* bias_data =
          std::is_same<Tweight, qint8>::value
              ? scaled_bias_data
              : const_cast<Tbias*>(bias_tensor.flat<Tbias>().data());
      // Create bias memory, since it is 1-dimension, no reordered needed
      bias_mem_ =
          CreateDnnlMemory(fwd_pd_.bias_desc(), onednn_engine_, bias_data);

      fwd_primitive_args_.emplace(DNNL_ARG_BIAS, bias_mem_);

      is_init_ = true;
    } catch (dnnl::error& e) {
      string error_msg = itex::strings::StrCat(
          "Status: ", e.status, ", message: ", string(e.message), ", in file ",
          __FILE__, ":", __LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  // MatMul + Bias + Add handling
  void SumPostopHandling(
      OpKernelContext* context,
      const dnnl::inner_product_forward::primitive_desc& matmul_pd,
      const dnnl::memory::dims& dst_dims_onednn, TensorShape tensor_shape,
      Tensor** dst_tensor) {
    if (!(std::is_same<Toutput, float>::value ||
          std::is_same<Toutput, Eigen::bfloat16>::value ||
          std::is_same<Toutput, Eigen::half>::value)) {
      ITEX_LOG(FATAL) << "Currently, we only support MatMul + Bias + Add INT8 "
                         "fusion with float/half/bfloat16 output";
    }

    auto dst_md = matmul_pd.dst_desc();
    const int kInputIndex_Add = 3;
    const Tensor& add_tensor = context->input(kInputIndex_Add);

    TensorShape add_tf_shape = add_tensor.shape();
    // Check if reorder is needed.
    if (add_tf_shape == tensor_shape) {
      // TODO(itex): Add inplace check
      if (true) {
        context->set_output(kOutputIndex_Dst, add_tensor);
        *dst_tensor = context->mutable_output(kOutputIndex_Dst);
        return;
      }
      const int kUnsuccess = -1;
      int is_forward_success = kUnsuccess;
      OP_REQUIRES_OK(context,
                     context->forward_input_or_allocate_output(
                         {kInputIndex_Add}, kOutputIndex_Dst, tensor_shape,
                         dst_tensor, &is_forward_success));

      // Everything is done if forward succeed.
      if (is_forward_success != kUnsuccess) return;
    }

    // Reorder is needed. Check `*dst_tensor` first:
    //   1) nullptr, add shape is different with dst shape;
    //   2) not nullptr, forward is failed but dst has been allocated;
    if (*dst_tensor == nullptr) {
      OP_REQUIRES_OK(context, context->allocate_output(
                                  kOutputIndex_Dst, tensor_shape, dst_tensor));
    }

    auto onednn_engine_ = CreateDnnlEngine<Device>(*context);
    auto add_md = dst_md;
    memory fuse_add_src =
        memory(add_md, onednn_engine_, GetTensorBuffer<Toutput>(&add_tensor));
    memory fuse_add_dst =
        memory(dst_md, onednn_engine_, GetTensorBuffer<Toutput>(*dst_tensor));
    ReorderMemory(*context, &fuse_add_src, &fuse_add_dst, onednn_engine_);
  }

  // Allocate output tensor.
  virtual void AllocateOutputTensor(
      OpKernelContext* context,
      const dnnl::inner_product_forward::primitive_desc& matmul_pd,
      const dnnl::memory::dims& dst_dims_onednn, TensorShape tensor_shape,
      Tensor** dst_tensor) {
    ITEX_DCHECK(dst_tensor);

    if (this->post_op_util_.HasAdd()) {
      SumPostopHandling(context, matmul_pd, dst_dims_onednn, tensor_shape,
                        dst_tensor);
    } else {
      OP_REQUIRES_OK(context, context->allocate_output(
                                  kOutputIndex_Dst, tensor_shape, dst_tensor));
    }
  }

  void ComputeOutputRangeForInt32(OpKernelContext* context,
                                  float* min_output_value,
                                  float* max_output_value) {
    const float min_input = context->input(kSrcMinRangeIndex).flat<float>()(0);
    const float max_input = context->input(kSrcMaxRangeIndex).flat<float>()(0);
    const float min_weight =
        context->input(kFilterMinRangeIndex).flat<float>()(0);
    const float max_weight =
        context->input(kFilterMaxRangeIndex).flat<float>()(0);
    OneDnnQuantizationRangeForMultiplication<quint8, qint8, qint32>(
        min_input, max_input, min_weight, max_weight, min_output_value,
        max_output_value);
  }

  virtual void ExtendInt8PostOps(OpKernelContext* context) = 0;

  bool IsBiasCacheEmpty() TF_LOCKS_EXCLUDED(bias_cache_mutex_) {
    tf_shared_lock lock(&bias_cache_mutex_);
    // TODO(itex): investigate why bias_cached_data_.NumElements() == 1
    // instead of 0,  when bias_cached_data_.IsInitialized() == True
    return (!bias_cached_data_.IsInitialized());
  }

  void CacheBias(OpKernelContext* context,
                 const Tensor& temp_scaled_bias_tensor) {
    mutex_lock lock(&bias_cache_mutex_);
    if (bias_cached_data_.IsInitialized()) {
      return;
    }
    Tensor* bias_cached_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_persistent(
                                temp_scaled_bias_tensor.dtype(),
                                temp_scaled_bias_tensor.shape(),
                                &bias_cached_data_, &bias_cached_tensor));

    auto* stream = context->GetDeviceStream();
    const void* input_data = temp_scaled_bias_tensor.flat<Tbias>().data();
    void* output_data = bias_cached_tensor->flat<Tbias>().data();
    DeviceMemcpy<Device>(output_data, input_data,
                         temp_scaled_bias_tensor.NumElements() * sizeof(Tbias),
                         stream);
  }

  Tbias* GetCachedBias(OpKernelContext* context) {
    tf_shared_lock lock(&bias_cache_mutex_);
    const Tensor* cached_bias_data = bias_cached_data_.AccessTensor(context);
    return const_cast<Tbias*>(cached_bias_data->flat<Tbias>().data());
  }

  bool IsCachedBiasValid(float current_min_input, float current_max_input) {
    if (this->is_bias_const_ && this->is_weight_const_ &&
        std::abs(current_min_input - saved_min_input_) < 1e-5f &&
        std::abs(current_max_input - saved_max_input_) < 1e-5f)
      return true;
    return false;
  }

  virtual Tbias* GetScaledBias(
      OpKernelContext* context,
      const dnnl::inner_product_forward::primitive_desc& matmul_pd,
      const Tensor& bias_tensor, Tensor* scaled_bias_tensor) {
    if (std::is_same<Tbias, qint32>::value) {
      return static_cast<Tbias*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
    }

    const float min_input = context->input(kSrcMinRangeIndex).flat<float>()(0);
    const float max_input = context->input(kSrcMaxRangeIndex).flat<float>()(0);
    const Tensor& min_weight_tensor = context->input(kFilterMinRangeIndex);
    const Tensor& max_weight_tensor = context->input(kFilterMaxRangeIndex);
    const float* min_weight = min_weight_tensor.flat<float>().data();
    const float* max_weight = max_weight_tensor.flat<float>().data();
    // We can use cached bias which has been scaled only when (i) bias is
    // constant (ii) weight is constant (iii) min_input is same as saved one
    // (iv) max_input is same as saved one (v) BiasCache is not empty.
    if (this->IsBiasCacheEmpty() ||
        !this->IsCachedBiasValid(min_input, max_input)) {
      void* input_bias_buf = static_cast<void*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
      auto scaled_bias_md = matmul_pd.bias_desc();
      TensorShape scaled_bias_shape;
      scaled_bias_shape.AddDim((scaled_bias_md.get_size() / sizeof(Tbias)));
      auto weight_md = matmul_pd.weights_desc();
      TensorShape weight_shape;
      weight_shape.AddDim((weight_md.get_size() / sizeof(Tweight)));

      AllocatorAttributes alloc_attr;
      alloc_attr.set_on_host(true);
      TF_ABORT_IF_ERROR(context->allocate_temp(DataTypeToEnum<Tbias>::v(),
                                               scaled_bias_shape,
                                               scaled_bias_tensor, alloc_attr));
      void* scaled_bias_buf =
          static_cast<void*>(scaled_bias_tensor->flat<Tbias>().data());

      const float max_int8_input =
          (std::is_same<Tinput, quint8>::value) ? 255.0f : 127.0f;
      const float max_int8_weight =
          (std::is_same<Tweight, quint8>::value) ? 255.0f : 127.0f;
      const float range_input =
          (mode_ == QuantizeMode::MIN_FIRST)
              ? max_input - min_input
              : std::max(std::abs(min_input), std::abs(max_input));
      const size_t num_weight_scales = min_weight_tensor.NumElements();
      std::vector<float> bias_scales(num_weight_scales, 1.0);
      for (size_t i = 0; i < num_weight_scales; ++i) {
        float range_weight =
            std::max(std::abs(min_weight[i]), std::abs(max_weight[i]));
        float scale_factor =
            (max_int8_input * max_int8_weight) / (range_input * range_weight);
        bias_scales[i] = scale_factor;
      }
      if (mode_ == QuantizeMode::MIN_FIRST) {
        const Tensor& weight_tensor = context->input(1);

        int k = weight_tensor.dim_size(0);
        int n = weight_tensor.dim_size(1);
#ifdef INTEL_CPU_ONLY
        Tbias* input_bias = static_cast<Tbias*>(input_bias_buf);
        Tweight* wt_buf =
            const_cast<Tweight*>(weight_tensor.flat<Tweight>().data());

#else
        // For GPU, copy bias tensor to host, for easy implementation
        void* input_weight_buf = GetTensorBuffer<Tweight>(&weight_tensor);

        Tensor bias_host_tensor, weight_host_tensor;
        TF_ABORT_IF_ERROR(context->allocate_temp(
            DataTypeToEnum<Tbias>::v(), scaled_bias_shape, &bias_host_tensor,
            alloc_attr));
        TF_ABORT_IF_ERROR(
            context->allocate_temp(DataTypeToEnum<Tweight>::v(), weight_shape,
                                   &weight_host_tensor, alloc_attr));
        void* input_bias_host_buf = GetTensorBuffer<Tbias>(&bias_host_tensor);
        void* input_weight_host_buf =
            GetTensorBuffer<Tweight>(&weight_host_tensor);

        auto* dpcpp_stream = context->GetDeviceStream();
        dpcpp_stream
            ->memcpy(input_bias_host_buf, input_bias_buf, n * sizeof(Tbias))
            .wait();
        dpcpp_stream
            ->memcpy(input_weight_host_buf, input_weight_buf,
                     k * n * sizeof(Tweight))
            .wait();
        auto* input_bias = static_cast<Tbias*>(input_bias_host_buf);
        auto* wt_buf = static_cast<Tweight*>(input_weight_host_buf);
#endif  // INTEL_CPU_ONLY
        auto* adjusted_bias = static_cast<Tbias*>(scaled_bias_buf);
        float q_min_input = max_int8_input * min_input / range_input;

        // Scales needs to expanded to number of output channels by the values
        // of bias_scales.
        std::vector<float> scales(n);
        if (num_weight_scales != n)  // weights quanitzed per_tensor
          std::fill(scales.begin(), scales.end(), bias_scales[0]);
        else
          scales = bias_scales;  // Expensive copy
#ifdef INTEL_CPU_ONLY
#pragma omp parallel for schedule(static)
#endif  // INTEL_CPU_ONLY
        for (int j = 0; j < n; ++j) {
          int sum = 0;
          for (int i = 0; i < k; ++i) {
            sum += wt_buf[i * n + j];
          }
          adjusted_bias[j] =
              static_cast<Tbias>(static_cast<float>(input_bias[j]) * scales[j] +
                                 (sum * q_min_input));
        }
      } else {
        dnnl::primitive_attr bias_attr;
        (num_weight_scales == 1) ? bias_attr.set_output_scales(0, bias_scales)
                                 : bias_attr.set_output_scales(1, bias_scales);
        memory::dims input_bias_dims =
            memory::dims({1, bias_tensor.shape().dim_size(0)});
        auto input_bias_md = dnnl::memory::desc(
            input_bias_dims, OneDnnType<Tbias>(), memory::format_tag::ab);

        auto onednn_engine_ = CreateDnnlEngine<Device>(*context);

        auto input_bias_mem =
            dnnl::memory(input_bias_md, onednn_engine_, input_bias_buf);

        auto scaled_bias_mem =
            dnnl::memory(scaled_bias_md, onednn_engine_, scaled_bias_buf);

        auto reorder_prim =
            dnnl::reorder(input_bias_mem, scaled_bias_mem, bias_attr);
        std::unordered_map<int, memory> reorder_net_args = {
            {DNNL_ARG_SRC, input_bias_mem}, {DNNL_ARG_DST, scaled_bias_mem}};
        auto onednn_stream = CreateDnnlStream(*context, onednn_engine_);
        reorder_prim.execute(onednn_stream, reorder_net_args);
      }

      // Cache the scaled bias
      if (this->is_bias_const_ && this->is_weight_const_) {
#ifdef INTEL_CPU_ONLY
        this->CacheBias(context, *scaled_bias_tensor);
#else
        Tensor scaled_bias_device_tensor;
        TF_ABORT_IF_ERROR(context->allocate_temp(DataTypeToEnum<Tbias>::v(),
                                                 scaled_bias_shape,
                                                 &scaled_bias_device_tensor));
        void* scaled_bias_device_buf =
            GetTensorBuffer<Tbias>(&scaled_bias_device_tensor);

        auto* dpcpp_stream = context->GetDeviceStream();
        dpcpp_stream
            ->memcpy(scaled_bias_device_buf, scaled_bias_buf,
                     scaled_bias_md.get_size() * sizeof(Tbias))
            .wait();
        this->CacheBias(context, scaled_bias_device_tensor);
#endif  // INTEL_CPU_ONLY
        this->saved_min_input_ = min_input;
        this->saved_max_input_ = max_input;
      }
    }
    return this->GetCachedBias(context);
  }

 protected:
  bool is_weight_const_;
  bool is_bias_const_;

  bool transpose_a_;
  bool transpose_b_;

  // Note: Legacy MatMul INT8 kernel's bias cache is different from normal INT8
  // bias cache. The bias calculation not only consists of scaling but also
  // complex compensation for asymmetric zero-point. Therefore, we cannot reuse
  // the BiasCacheManager. Here we implement a limited functionality bias
  // manager.
  mutex bias_cache_mutex_;
  PersistentTensor bias_cached_data_ TF_GUARDED_BY(bias_cache_mutex_);

  const int kInputIndex_Src = 0;
  const int kInputIndex_Filter = 1;
  const int kInputIndex_Bias = 2;
  const int kOutputIndex_Dst = 0;

  int kSrcMinRangeIndex;
  int kSrcMaxRangeIndex;
  int kFilterMinRangeIndex;
  int kFilterMaxRangeIndex;
  int kMinFreezedIndex;
  int kMaxFreezedIndex;
  int kDstMinRangeIndex;
  int kDstMaxRangeIndex;

  /* Quantization mode */
  QuantizeMode mode_;
  /* Fused MatMul */
  PostOpUtil post_op_util_;
  // Weight cache manager
  WeightCacheManager<Tweight> weight_cache_manager;

  float saved_min_input_ = -std::numeric_limits<float>::infinity();
  float saved_max_input_ = std::numeric_limits<float>::infinity();

  // Cache oneDNN object and TF memory
  mutex mu_compute_;

  bool enable_cache_ = false;
  bool is_init_ = false;
  bool is_input_zero_ = false;
  bool is_weight_reorder_ = false;

  dnnl::memory src_mem_;
  dnnl::memory bias_mem_;
  // Original native weight memory
  dnnl::memory weight_mem_;
  // Target block weight memory.
  dnnl::memory weight_mem_opt_;
  dnnl::memory dst_mem_;
  dnnl::memory scratchpad_mem_;

  std::vector<int64> input_dims_;
  TensorShape dst_shape_;

  memory::dims dst_dims_onednn_;

  Tensor* dst_tensor_ = nullptr;
  Tensor tmp_weight_;
  std::shared_ptr<Tensor> scratchpad_tensor_;
  int64_t scratchpad_size_ = 0;

  dnnl::stream onednn_stream_;
  dnnl::engine onednn_engine_;

  dnnl::primitive fwd_primitive_;
  dnnl::inner_product_forward::primitive_desc fwd_pd_;
  std::unordered_map<int, memory> fwd_primitive_args_;
};

template <typename Device, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class QuantizedMatMulOp
    : public LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias,
                                         Toutput> {
 public:
  explicit QuantizedMatMulOp(OpKernelConstruction* context)
      : LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias, Toutput>(
            context) {
    // Quantize mode assignment
    string mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("input_quant_mode", &mode_string));
    if (mode_string == "MIN_FIRST") {
      this->mode_ = QuantizeMode::MIN_FIRST;
    } else if (mode_string == "SCALED") {
      this->mode_ = QuantizeMode::SCALED;
    } else {
      context->CtxFailure(errors::InvalidArgument(
          "Quantization mode must be either MIN_FIRST or SCALED, but received ",
          mode_string));
    }

    // weight/bias const flag set
    if (context->HasAttr("is_weight_const")) {
      OP_REQUIRES_OK(context, context->GetAttr("is_weight_const",
                                               &(this->is_weight_const_)));
    }
    this->is_bias_const_ = true;

    // PostOpUtil set
    std::vector<string> fused_ops;
    fused_ops.push_back("Quantized");
    fused_ops.push_back("BiasAdd");
    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops),
                errors::InvalidArgument(
                    "Found unsupported fusion in QuantizedMatMul."));

    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_a", &this->transpose_a_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_b", &this->transpose_b_));

    // Set input/output tensor index
    this->kSrcMinRangeIndex = 3;
    this->kSrcMaxRangeIndex = 4;
    this->kFilterMinRangeIndex = 5;
    this->kFilterMaxRangeIndex = 6;
    this->kMinFreezedIndex = 7;
    this->kMaxFreezedIndex = 8;
    this->kDstMinRangeIndex = 1;
    this->kDstMaxRangeIndex = 2;
  }

  void Compute(OpKernelContext* context) override {
    LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias,
                                Toutput>::Compute(context);
  }

  void ExtendInt8PostOps(OpKernelContext* context) override {
    // When the output type is quint8, the output data is requantized into
    // quint8. A post_op "output_scale" is added to do the conversion.
    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value ||
        std::is_same<Toutput, float>::value ||
        std::is_same<Toutput, Eigen::bfloat16>::value ||
        std::is_same<Toutput, Eigen::half>::value) {
      float min_output_value;
      float max_output_value;
      this->ComputeOutputRangeForInt32(context, &min_output_value,
                                       &max_output_value);
      float scale_int32 =
          std::max(std::abs(min_output_value), std::abs(max_output_value));
      const float min_freezed_output =
          context->input(this->kMinFreezedIndex).template flat<float>()(0);
      const float max_freezed_output =
          context->input(this->kMaxFreezedIndex).template flat<float>()(0);
      float scale_eightbit =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      float scale = 1.0;
      if (std::is_same<Toutput, quint8>::value) {
        scale = scale_int32 / scale_eightbit / static_cast<float>(1u << 23);
      } else if (std::is_same<Toutput, qint8>::value) {
        scale = scale_int32 / scale_eightbit / static_cast<float>(1u << 24);
      } else if (std::is_same<Toutput, float>::value ||
                 std::is_same<Toutput, Eigen::bfloat16>::value ||
                 std::is_same<Toutput, Eigen::half>::value) {
        scale = scale_int32 / static_cast<float>(1u << 31);
      } else {
        // TODO(itex): keeping the default qint8 as before. Change to error
        // later.
        scale = scale_int32 / scale_eightbit / static_cast<float>(1u << 24);
      }
      this->post_op_util_.SetOutputScale({scale});
    }
  }
};

template <typename Device, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class QuantizedMatMulReluOp
    : public QuantizedMatMulOp<Device, Tinput, Tweight, Tbias, Toutput> {
 public:
  explicit QuantizedMatMulReluOp(OpKernelConstruction* context)
      : QuantizedMatMulOp<Device, Tinput, Tweight, Tbias, Toutput>(context) {
    std::vector<string> fused_ops;
    fused_ops.push_back("Relu");
    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops),
                errors::InvalidArgument(
                    "Found unsupported fusion in QuantizedMatMulRelu."));
  }

 protected:
  void ExtendInt8PostOps(OpKernelContext* context) override {
    QuantizedMatMulOp<Device, quint8, qint8, Tbias, Toutput>::ExtendInt8PostOps(
        context);
    this->post_op_util_.SetPostOpScale("Relu", 1.0);
  }
};

// Currently, Targs = Tbias. We may improve such design
// QuantizedFusedMatMulOp is for previous MatMul INT8 V1 new API, it seems the
// V1 API is not used by Intel-TF

// TODO(itex): Add typename U for additional args to align with Intel-TF code.
// Currently, the additional arg type is always equal to Toutput
template <typename Device, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class QuantizedFusedMatMulOp
    : public LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias,
                                         Toutput> {
 public:
  explicit QuantizedFusedMatMulOp(OpKernelConstruction* context)
      : LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias, Toutput>(
            context) {
    // Quantize mode assignment
    string mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("input_quant_mode", &mode_string));
    if (mode_string == "MIN_FIRST") {
      this->mode_ = QuantizeMode::MIN_FIRST;
    } else if (mode_string == "SCALED") {
      this->mode_ = QuantizeMode::SCALED;
    } else {
      context->CtxFailure(errors::InvalidArgument(
          "Quantization mode must be either MIN_FIRST or SCALED, but received ",
          mode_string));
    }

    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_a", &this->transpose_a_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_b", &this->transpose_b_));

    // weight/bias const flag set
    OP_REQUIRES_OK(context, context->GetAttr("is_filter_const",
                                             &(this->is_weight_const_)));
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_bias_const", &(this->is_bias_const_)));
    // Set alpha if get `LeakyRelu` after adding ops.

    // PostOpUtil set
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES(context, fused_ops_.size() <= 2,
                errors::InvalidArgument(
                    "_QuantizedFusedMatMul supports maximum 2 post ops"));
    OP_REQUIRES(context, fused_ops_.size() == 0 || fused_ops_[0] == "BiasAdd",
                errors::InvalidArgument(
                    "_QuantizedFusedMatMul first post op must be BiasAdd"));
    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops_),
                errors::InvalidArgument(
                    "Found unsupported fusion in _QuantizedFusedMatMul."));

    // Set alpha if get `LeakyRelu` after adding ops.
    if (this->post_op_util_.HasLeakyRelu()) {
      float alpha;
      OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha", &alpha));
      this->post_op_util_.SetLeakyReluAlpha(alpha);
    }

    // Set input/output tensor index
    bool has_postop_add = std::find(fused_ops_.begin(), fused_ops_.end(),
                                    "Add") != fused_ops_.end();
    this->kSrcMinRangeIndex = has_postop_add + 3;
    this->kSrcMaxRangeIndex = has_postop_add + 4;
    this->kFilterMinRangeIndex = has_postop_add + 5;
    this->kFilterMaxRangeIndex = has_postop_add + 6;
    this->kMinFreezedIndex = has_postop_add + 7;
    this->kMaxFreezedIndex = has_postop_add + 8;
    this->kDstMinRangeIndex = 1;
    this->kDstMaxRangeIndex = 2;
  }

  void Compute(OpKernelContext* context) override {
    LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias,
                                Toutput>::Compute(context);
  }

  void ExtendInt8PostOps(OpKernelContext* context) override {
    if (std::is_same<Toutput, qint32>::value) {
      // If output is qint32, the scale for MatMul is 1.0f and the scales for
      // activations is 1.0f (default).
      this->post_op_util_.SetOutputScale({1.0f});
    } else if (std::is_same<Toutput, qint8>::value ||
               std::is_same<Toutput, quint8>::value ||
               std::is_same<Toutput, float>::value ||
               std::is_same<Toutput, Eigen::bfloat16>::value ||
               std::is_same<Toutput, Eigen::half>::value) {
      // When Toutput is float, the fusion semantic has its output dequantized,
      // and when Toutput is q{u}int8 the fusion semantic has its output
      // requantized.
      const float min_input =
          context->input(this->kSrcMinRangeIndex).template flat<float>()(0);
      const float max_input =
          context->input(this->kSrcMaxRangeIndex).template flat<float>()(0);
      const Tensor& min_weight_tensor =
          context->input(this->kFilterMinRangeIndex);
      const Tensor& max_weight_tensor =
          context->input(this->kFilterMaxRangeIndex);
      const float* min_weight = min_weight_tensor.flat<float>().data();
      const float* max_weight = max_weight_tensor.flat<float>().data();
      const size_t num_output_channels = min_weight_tensor.NumElements();

      const float max_int8_input =
          (std::is_same<Tinput, quint8>::value) ? 255.0f : 127.0f;
      const float max_int8_weight =
          (std::is_same<Tweight, quint8>::value) ? 255.0f : 127.0f;
      const float range_input =
          (this->mode_ == QuantizeMode::MIN_FIRST)
              ? max_input - min_input
              : std::max(std::abs(min_input), std::abs(max_input));

      std::vector<float> scale_output(num_output_channels);
      for (size_t i = 0; i < num_output_channels; ++i) {
        float range_weight =
            std::max(std::abs(min_weight[i]), std::abs(max_weight[i]));
        scale_output[i] =
            (range_input * range_weight) / (max_int8_input * max_int8_weight);
      }

      float scale_post_op = 1.0;

      // Note: When Toutput is u8/s8, in other words Requantize mode, the scale
      // calculation needs to take min/max_freezed_output into account.
      if (std::is_same<Toutput, qint8>::value ||
          std::is_same<Toutput, quint8>::value) {
        // Requantize condition
        const float min_output =
            context->input(this->kMinFreezedIndex).template flat<float>()(0);
        const float max_output =
            context->input(this->kMaxFreezedIndex).template flat<float>()(0);
        const float range_output =
            std::max(std::abs(min_output), std::abs(max_output));

        // Note: INT8 primitive needs different scale factor calculation with or
        // without post op activations, when output is qint8 or quint8.
        if (fused_ops_.size() == 1) {
          // No post op activations
          if (std::is_same<Toutput, qint8>::value) {
            for (size_t i = 0; i < scale_output.size(); ++i) {
              scale_output[i] = scale_output[i] * (127.0f / range_output);
            }
          } else if (std::is_same<Toutput, quint8>::value) {
            for (size_t i = 0; i < scale_output.size(); ++i) {
              scale_output[i] = scale_output[i] * (255.0f / range_output);
            }
          }
        } else {
          // Has post op activations
          if (std::is_same<Toutput, qint8>::value) {
            scale_post_op = 127.0f / range_output;
          } else if (std::is_same<Toutput, quint8>::value) {
            scale_post_op = 255.0f / range_output;
          }
        }
      }

      // Set output scale
      this->post_op_util_.SetOutputScale(scale_output);

      // Set postop scale if needed
      if (fused_ops_.size() == 2) {
        string postop = fused_ops_[1];
        if (postop != "Add") {
          this->post_op_util_.SetPostOpScale(postop, scale_post_op);
        } else {
          this->post_op_util_.SetPostOpScale("Add", 1.0f);
        }
      }
    }
  }

 protected:
  // QuantizedFusedMatMul needs the fused ops information during runtime. This
  // information is necessary in calculating scales.
  std::vector<string> fused_ops_;
};

// QuantizedFusedMatMulV2Op is for latest MatMul INT8 V2 new API by Intel-TF
template <typename Device, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class QuantizedFusedMatMulV2Op
    : public LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias,
                                         Toutput> {
 protected:
  string input_quant_mode_;   // 0-th input
  string output_quant_mode_;  // 0-th output
  string activation_type_;    // Activation op type

  // Unlike fused_ops is a local variable within construction function, class
  // member fused_ops_ contains post op name supported by ITEX PostOpUtil
  std::vector<string> fused_ops_;

  void Initialize(OpKernelConstruction* context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("input_quant_mode", &input_quant_mode_));
    // TODO(itex): merge input_quant_mode_ and
    // LegacyQuantizedMatMulOpBase::mode_
    // into single variable
    if (input_quant_mode_ == "MIN_FIRST") {
      this->mode_ = QuantizeMode::MIN_FIRST;
    } else if (input_quant_mode_ == "SCALED") {
      this->mode_ = QuantizeMode::SCALED;
    } else {
      context->CtxFailure(errors::InvalidArgument(
          "Quantization mode must be either MIN_FIRST or SCALED, but received ",
          input_quant_mode_));
    }
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_quant_mode", &output_quant_mode_));
    OP_REQUIRES(
        context, output_quant_mode_ == "SCALED",
        errors::Unimplemented("Requantize is supported for SCALED mode only."));
    OP_REQUIRES_OK(
        context, context->GetAttr("is_weight_const", &this->is_weight_const_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_bias_const", &this->is_bias_const_));

    // Extract activation info and canonicalize activation types to
    // common name "Activation" in the fused_ops attribute.
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops_),
                errors::InvalidArgument(
                    "Found unsupported fusion in _QuantizedMatMul."));

    this->kSrcMinRangeIndex = 3;
    this->kSrcMaxRangeIndex = 4;
    this->kFilterMinRangeIndex = 5;
    this->kFilterMaxRangeIndex = 6;

    // Configure oneDNN post ops
    if (this->post_op_util_.HasAdd()) {
      OP_REQUIRES(context,
                  (std::is_same<Toutput, float>::value ||
                   std::is_same<Toutput, Eigen::bfloat16>::value),
                  errors::Unimplemented(
                      "Quantized addend tensor is not implemented yet."));
      // Addend tensor precedes all minmax tensors. Shift the indices from
      // default initilized values.
      this->kSrcMinRangeIndex += 1;
      this->kSrcMaxRangeIndex += 1;
      this->kFilterMinRangeIndex += 1;
      this->kFilterMaxRangeIndex += 1;
    }

    // Currently, these index are hardcoded, due to the fusion currently
    // supported
    this->kMinFreezedIndex = 7;
    this->kMaxFreezedIndex = 8;
    this->kDstMinRangeIndex = 1;
    this->kDstMaxRangeIndex = 2;

    // Set alpha if get `LeakyRelu` after adding ops.
    if (this->post_op_util_.HasLeakyRelu()) {
      float alpha;
      OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha", &alpha));
      this->post_op_util_.SetLeakyReluAlpha(alpha);
    }
  }

 public:
  explicit QuantizedFusedMatMulV2Op(OpKernelConstruction* context)
      : LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias, Toutput>(
            context) {
    Initialize(context);
  }

  void Compute(OpKernelContext* context) override {
    LegacyQuantizedMatMulOpBase<Device, Tinput, Tweight, Tbias,
                                Toutput>::Compute(context);
  }

  void ExtendInt8PostOps(OpKernelContext* context) override {
    if (!fused_ops_.empty()) {
      // Hard code output range here since it's never changed.
      const int kOutputMinIdx = 7;
      const int kOutputMaxIdx = 8;

      if (this->post_op_util_.HasOutputScales()) {
        const float min_input =
            context->input(this->kSrcMinRangeIndex).template flat<float>()(0);
        const float max_input =
            context->input(this->kSrcMaxRangeIndex).template flat<float>()(0);
        const Tensor& min_weight_tensor =
            context->input(this->kFilterMinRangeIndex);
        const Tensor& max_weight_tensor =
            context->input(this->kFilterMaxRangeIndex);
        const float* min_weight = min_weight_tensor.flat<float>().data();
        const float* max_weight = max_weight_tensor.flat<float>().data();
        const size_t num_output_channels = min_weight_tensor.NumElements();

        const float max_int8_input =
            (std::is_same<Tinput, quint8>::value) ? 255.0f : 127.0f;
        const float max_int8_weight =
            (std::is_same<Tweight, quint8>::value) ? 255.0f : 127.0f;
        const float range_input =
            (input_quant_mode_ == "MIN_FIRST")
                ? max_input - min_input
                : std::max(std::abs(min_input), std::abs(max_input));

        std::vector<float> output_scale(num_output_channels);
        for (size_t i = 0; i < num_output_channels; ++i) {
          float range_weight =
              std::max(std::abs(min_weight[i]), std::abs(max_weight[i]));
          output_scale[i] =
              (range_input * range_weight) / (max_int8_input * max_int8_weight);
        }

        // Update output_scale for Requantize fusion without Activation.
        // Activation scale will be handled later.
        if (this->post_op_util_.HasRequantize() &&
            !this->post_op_util_.HasActivation()) {
          const float min_output =
              context->input(kOutputMinIdx).template flat<float>()(0);
          const float max_output =
              context->input(kOutputMaxIdx).template flat<float>()(0);
          const float range_output =
              std::max(std::abs(min_output), std::abs(max_output));
          const float max_int8_output =
              (std::is_same<Toutput, quint8>::value) ? 255.0f : 127.0f;
          for (size_t i = 0; i < output_scale.size(); ++i) {
            output_scale[i] =
                output_scale[i] * (max_int8_output / range_output);
          }
        }

        this->post_op_util_.SetOutputScale(output_scale);
      }

      if (this->post_op_util_.HasActivation()) {
        if (this->post_op_util_.HasRequantize()) {
          // Update scale for requantize fusion.
          const float min_output =
              context->input(kOutputMinIdx).template flat<float>()(0);
          const float max_output =
              context->input(kOutputMaxIdx).template flat<float>()(0);
          const float range_output =
              std::max(std::abs(min_output), std::abs(max_output));
          const float max_int8_output =
              (std::is_same<Toutput, quint8>::value) ? 255.0f : 127.0f;
          float scale = max_int8_output / range_output;

          // Current supported fusion, activation is always 2nd post op
          string activation_type = fused_ops_[1];
          this->post_op_util_.SetPostOpScale(activation_type, scale);
        }
      }

      if (this->post_op_util_.HasAdd()) {
        // TODD(itex): Set input index for add input if required. Currently,
        // the input index is always 3.
        this->post_op_util_.SetPostOpScale("Add", 1.0f);
      }
    }
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_QUANTIZED_MATMUL_COMMON_H_
