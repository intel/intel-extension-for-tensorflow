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

#ifndef ITEX_CORE_KERNELS_LEGACY_MATMUL_COMMON_H_
#define ITEX_CORE_KERNELS_LEGACY_MATMUL_COMMON_H_

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/devices/xpu_device_util.h"
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
class LegacyOneDnnQuantizedMatMulOpBase : public OpKernel {
 public:
  explicit LegacyOneDnnQuantizedMatMulOpBase(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_a", &this->transpose_a_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_b", &this->transpose_b_));
  }
  void Compute(OpKernelContext* context) override {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      // Input tensors
      const Tensor& src_tensor = context->input(this->kInputIndex_Src);
      const Tensor& weight_tensor = context->input(this->kInputIndex_Filter);
      const Tensor& bias_tensor = context->input(this->kInputIndex_Bias);

      // Get shapes of input & filter tensors
      OneDnnShape src_onednn_shape, filter_onednn_shape;
      GetOneDnnShape(context, this->kInputIndex_Src, &src_onednn_shape);
      TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                     ? src_onednn_shape.GetTfShape()
                                     : src_tensor.shape();
      TensorShape weight_tf_shape = weight_tensor.shape();

      memory::dims src_dims, weight_dims;
      memory::dims dst_dims_tf_order, dst_dims_onednn_order;

      const int batch = this->transpose_a_ ? src_tf_shape.dim_size(1)
                                           : src_tf_shape.dim_size(0);
      const int k = this->transpose_a_ ? src_tf_shape.dim_size(0)
                                       : src_tf_shape.dim_size(1);
      const int channel = this->transpose_b_ ? weight_tf_shape.dim_size(0)
                                             : weight_tf_shape.dim_size(1);

      src_dims = {batch, k};
      weight_dims = {channel, k};
      dst_dims_onednn_order = {batch, channel};

      // Create memory for user data.
      // Describe how the inputs and outputs of inner-product look like. Also
      // specify buffers containing actual input and output data.
      auto src_md = src_onednn_shape.IsOneDnnTensor()
                        ? src_onednn_shape.GetOneDnnLayout()
                        : memory::desc(src_dims, OneDnnType<Tinput>(),
                                       memory::format_tag::nc);

      auto weight_md = memory::desc(
          weight_dims, OneDnnType<Tweight>(),
          this->transpose_b_ ? memory::format_tag::oi : memory::format_tag::io);

      auto src_exec_md =
          memory::desc(src_dims, OneDnnType<Tinput>(), memory::format_tag::any);

      auto weight_exec_md = memory::desc(weight_dims, OneDnnType<Tweight>(),
                                         memory::format_tag::any);

      dnnl::memory::dims bias_dims = {
          static_cast<int>(bias_tensor.dim_size(0))};

      auto bias_exec_md =
          memory::desc(bias_dims, OneDnnType<Tbias>(), memory::format_tag::any);

      auto dst_exec_md =
          memory::desc(dst_dims_onednn_order, OneDnnType<Toutput>(),
                       memory::format_tag::any);

      // Note: Extend the basic parameters for data types and fusions.
      this->ExtendInt8PostOps(context);

      auto fwd_desc = dnnl::inner_product_forward::desc(
          dnnl::prop_kind::forward_inference, src_exec_md, weight_exec_md,
          bias_exec_md, dst_exec_md);

      // Set post op attribution.
      dnnl::primitive_attr post_ops_attr;
      this->post_op_util_.SetPostOpAttr(&post_ops_attr);
      post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      auto fwd_pd = dnnl::inner_product_forward::primitive_desc(
          fwd_desc, post_ops_attr, onednn_engine);
      auto fwd_primitive = dnnl::inner_product_forward(fwd_pd);

      // Allocate output Tensor.
      OneDnnShape dst_onednn_shape;
      int64 dst_data_size = fwd_pd.dst_desc().get_size() / sizeof(Toutput);
      TensorShape dst_shape = TensorShape({dst_data_size});

      Tensor* dst_tensor = nullptr;
      this->AllocateOutputTensor(context, fwd_pd, dst_dims_onednn_order,
                                 OneDnnTensorFormat::FORMAT_NC,
                                 &dst_onednn_shape, dst_shape, &dst_tensor);

      // Create src memory, check if src needs to be reordered
      dnnl::memory src_mem = CreateDnnlMemory(
          src_md, onednn_engine, GetTensorBuffer<Tinput>(&src_tensor));
      dnnl::memory src_reorder_mem;
      Tensor src_reorder_tensor;
      bool is_src_reordered = (src_md != fwd_pd.src_desc());
      if (is_src_reordered) {
        int64 src_reorder_size = fwd_pd.src_desc().get_size() / sizeof(Tinput);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));
        src_reorder_mem =
            CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                             GetTensorBuffer<Tinput>(&src_reorder_tensor));
        ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
      }

      dnnl::memory weight_mem, weight_reorder_mem;
      Tensor weight_reorder_tensor;
      const Tweight* weight_data = weight_tensor.flat<Tweight>().data();
      memory::desc expected_md = fwd_pd.weights_desc();

      bool is_weight_reordered = (weight_md != expected_md);
      if (is_weight_reordered) {
        if (this->weight_cache_manager.IsEmpty()) {
          // Cache weight in first time executing this node
          this->weight_cache_manager.SetCache(
              context, weight_md, expected_md,
              static_cast<void*>(const_cast<Tweight*>(weight_data)),
              onednn_engine);
        }
        Tweight* weight_cached_data =
            this->weight_cache_manager.GetCache(context, expected_md);
        weight_reorder_mem =
            CreateDnnlMemory(expected_md, onednn_engine, weight_cached_data);
      } else {
        // No reorder needed
        weight_mem = CreateDnnlMemory(
            weight_md, onednn_engine,
            static_cast<void*>(const_cast<Tweight*>(weight_data)));
      }

      // Create dst memory
      Toutput* dst_data = dst_tensor->flat<Toutput>().data();
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      static_cast<void*>(dst_data));

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          fwd_pd.scratchpad_desc().get_size() / sizeof(Tinput);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<Tinput>(&scratchpad_tensor));

      // Execute MatMul INT8
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> fwd_primitive_args = {
          {DNNL_ARG_SRC, is_src_reordered ? src_reorder_mem : src_mem},
          {DNNL_ARG_WEIGHTS,
           is_weight_reordered ? weight_reorder_mem : weight_mem},
          {DNNL_ARG_DST, dst_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};

      // Note: The asymmetric compensation is calculate in bias handle
      Tensor scaled_bias_tensor;
      Tbias* scaled_bias_data;
      if (std::is_same<Tweight, qint8>::value) {
        scaled_bias_data = this->GetScaledBias(context, fwd_pd, bias_tensor,
                                               &scaled_bias_tensor);
      }

      Tbias* bias_data =
          std::is_same<Tweight, qint8>::value
              ? scaled_bias_data
              : const_cast<Tbias*>(bias_tensor.flat<Tbias>().data());
      // Create bias memory, since it is 1-dimension, no reordered needed
      memory bias_mem =
          CreateDnnlMemory(fwd_pd.bias_desc(), onednn_engine, bias_data);

      fwd_primitive_args.emplace(DNNL_ARG_BIAS, bias_mem);

      fwd_primitive.execute(onednn_stream, fwd_primitive_args);
    } catch (dnnl::error& e) {
      string error_msg = itex::strings::StrCat(
          "Status: ", e.status, ", message: ", string(e.message), ", in file ",
          __FILE__, ":", __LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }

    const float min_input = context->input(kSrcMinRangeIndex).flat<float>()(0);
    const float max_input = context->input(kSrcMaxRangeIndex).flat<float>()(0);

    AllocateBlockOutputMinMax<Tinput, Tweight, Toutput>(
        context, min_input, max_input, kFilterMinRangeIndex,
        kFilterMaxRangeIndex, kMinFreezedIndex, kMaxFreezedIndex,
        kDstMinRangeIndex, kDstMaxRangeIndex);
  }

  // MatMul + Bias + Add handling
  void SumPostopHandling(
      OpKernelContext* context,
      const dnnl::inner_product_forward::primitive_desc& matmul_pd,
      const dnnl::memory::dims& dst_dims_onednn,
      OneDnnTensorFormat dst_tf_format, OneDnnShape* dst_onednn_shape,
      TensorShape tensor_shape, Tensor** dst_tensor) {
    if (!(std::is_same<Toutput, float>::value ||
          std::is_same<Toutput, Eigen::bfloat16>::value ||
          std::is_same<Toutput, Eigen::half>::value)) {
      ITEX_LOG(FATAL) << "Currently, we only support MatMul + Bias + Add INT8 "
                         "fusion with float/bfloat16/half output";
    }

    auto dst_md = matmul_pd.dst_desc();
    const int kInputIndex_Add = 3;
    const Tensor& add_tensor = context->input(kInputIndex_Add);
    OneDnnShape add_onednn_shape;
    GetOneDnnShape(context, kInputIndex_Add, &add_onednn_shape);

    // Check if reorder is needed.
    if (add_onednn_shape == *dst_onednn_shape) {
      // TODO(itex): Add inplace check
      if (true) {
        context->set_output(kOutputIndex_Dst, add_tensor);
        ForwardMetaData(context, kInputIndex_Add, kOutputIndex_Dst,
                        *dst_onednn_shape);
        *dst_tensor = context->mutable_output(kOutputIndex_Dst);
        return;
      }
      const int kUnsuccess = -1;
      int is_forward_success = kUnsuccess;
      ForwardOrAllocateOutputSetOneDnnShape(
          context, kInputIndex_Add, kOutputIndex_Dst, dst_tensor, tensor_shape,
          *dst_onednn_shape, &is_forward_success);

      // Everything is done if forward succeed.
      if (is_forward_success != kUnsuccess) return;
    }

    // Reorder is needed. Check `*dst_tensor` first:
    //   1) nullptr, add shape is different with dst shape;
    //   2) not nullptr, forward is failed but dst has been allocated;
    if (*dst_tensor == nullptr) {
      AllocateOutputSetOneDnnShape(context, kOutputIndex_Dst, dst_tensor,
                                   tensor_shape, *dst_onednn_shape);
    }

    auto dst_layout =
        OneDnnTensorFormatToTag(dst_onednn_shape->GetTfDataFormat());
    auto onednn_engine = CreateDnnlEngine<Device>(*context);
    auto add_md =
        add_onednn_shape.IsOneDnnTensor()
            ? add_onednn_shape.GetOneDnnLayout()
            : memory::desc(dst_dims_onednn, OneDnnType<Toutput>(), dst_layout);
    memory fuse_add_src =
        memory(add_md, onednn_engine, GetTensorBuffer<Toutput>(&add_tensor));
    memory fuse_add_dst =
        memory(dst_md, onednn_engine, GetTensorBuffer<Toutput>(*dst_tensor));
    ReorderMemory(*context, &fuse_add_src, &fuse_add_dst, onednn_engine);
  }

  // Allocate output tensor.
  virtual void AllocateOutputTensor(
      OpKernelContext* context,
      const dnnl::inner_product_forward::primitive_desc& matmul_pd,
      const dnnl::memory::dims& dst_dims_onednn,
      OneDnnTensorFormat dst_tf_format, OneDnnShape* dst_onednn_shape,
      TensorShape tensor_shape, Tensor** dst_tensor) {
    ITEX_DCHECK(dst_tensor);
    auto dst_md = matmul_pd.dst_desc();

    SetOutputTensorShape(dst_md, dst_tf_format, &tensor_shape, dst_onednn_shape,
                         true /*is_onednn*/);

    if (this->post_op_util_.HasAdd()) {
      SumPostopHandling(context, matmul_pd, dst_dims_onednn, dst_tf_format,
                        dst_onednn_shape, tensor_shape, dst_tensor);
    } else {
      AllocateOutputSetOneDnnShape(context, kOutputIndex_Dst, dst_tensor,
                                   tensor_shape, *dst_onednn_shape);
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

        auto onednn_engine = CreateDnnlEngine<Device>(*context);

        auto input_bias_mem =
            dnnl::memory(input_bias_md, onednn_engine, input_bias_buf);

        auto scaled_bias_mem =
            dnnl::memory(scaled_bias_md, onednn_engine, scaled_bias_buf);

        auto reorder_prim =
            dnnl::reorder(input_bias_mem, scaled_bias_mem, bias_attr);
        std::unordered_map<int, memory> reorder_net_args = {
            {DNNL_ARG_SRC, input_bias_mem}, {DNNL_ARG_DST, scaled_bias_mem}};
        auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
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
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_LEGACY_MATMUL_COMMON_H_
