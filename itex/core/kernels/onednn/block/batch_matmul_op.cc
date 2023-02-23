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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/kernels/common/host_data_cache.h"
#include "itex/core/kernels/common/matmul_op.h"
#include "itex/core/kernels/common/no_ops.h"
#include "itex/core/kernels/onednn/block/matmul_op.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

using dnnl::matmul;
using dnnl::memory;

template <typename Device, typename Tlhs, typename Trhs, typename Toutput>
class OneDnnBatchMatMulV2Op : public OneDnnMatMulBaseOp<Device, Trhs> {
 public:
  explicit OneDnnBatchMatMulV2Op(OpKernelConstruction* context)
      : OneDnnMatMulBaseOp<Device, Trhs>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &this->transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &this->transpose_b_));
    OP_REQUIRES_OK(
        context, context->GetAttr("is_filter_const", &this->is_filter_const_));

    if (context->HasAttr("fused_ops")) {
      std::vector<string> fused_ops;
      OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));

      // TODO(itex): Replace Add(Sum)/Mul(output_scale) fusion to Binary post
      //             op fusion manually. Will refine related fusion to binary
      //             fusion in future.
      for (int i = 0; i < fused_ops.size(); ++i) {
        if (fused_ops[i] == "Add") fused_ops[i] = "BinaryAdd";
        if (fused_ops[i] == "Mul") fused_ops[i] = "BinaryMul";
      }
      OP_REQUIRES(context, this->post_op_util_.AddOps(fused_ops),
                  errors::InvalidArgument(
                      "Found unsupported fusion in Fused BatchMatMul."));

      if (this->post_op_util_.HasBinary()) {
        const int binary_num = this->post_op_util_.GetBinaryNum();
        OP_REQUIRES(
            context, binary_num <= kMaxBinaryNum_,
            errors::Unimplemented(
                "The number of Binary op fusion in BatchMatMul is: ",
                binary_num, ", which is greater than ", kMaxBinaryNum_));
      }

      // Set alpha if get `LeakyRelu` after adding ops.
      if (this->post_op_util_.HasLeakyRelu()) {
        float alpha;
        OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha", &alpha));
        this->post_op_util_.SetLeakyReluAlpha(alpha);
      }
    }
  }

  void Compute(OpKernelContext* context) override {
    try {
      const Tensor& src_tensor = context->input(kSrcIndex_);
      const Tensor& wei_tensor = context->input(kWeightIndex_);

      OneDnnShape src_onednn_shape;
      OneDnnShape wei_onednn_shape;
      GetOneDnnShape(context, kSrcIndex_, &src_onednn_shape);
      GetOneDnnShape(context, kWeightIndex_, &wei_onednn_shape);

      TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                     ? src_onednn_shape.GetTfShape()
                                     : src_tensor.shape();

      TensorShape wei_tf_shape = wei_onednn_shape.IsOneDnnTensor()
                                     ? wei_onednn_shape.GetTfShape()
                                     : wei_tensor.shape();

      MatMulBCast bcast(src_tf_shape.dim_sizes(), wei_tf_shape.dim_sizes());

      OP_REQUIRES(
          context, bcast.IsValid(),
          errors::InvalidArgument(
              "Input tensors must have compatible batch dimensions: ",
              src_tf_shape.DebugString(), " vs. ", wei_tf_shape.DebugString()));

      int64 d0 = src_tf_shape.dim_size(src_tf_shape.dims() - 2);
      int64 d1 = src_tf_shape.dim_size(src_tf_shape.dims() - 1);

      int64 d2 = wei_tf_shape.dim_size(wei_tf_shape.dims() - 2);
      int64 d3 = wei_tf_shape.dim_size(wei_tf_shape.dims() - 1);

      if (this->transpose_a_) std::swap(d0, d1);
      if (this->transpose_b_) std::swap(d2, d3);

      OP_REQUIRES(
          context, d1 == d2,
          errors::InvalidArgument("Input[0] mismatch Input[1] shape :", d1,
                                  " vs. ", d2, ": ", src_tf_shape.DebugString(),
                                  " ", wei_tf_shape.DebugString(), " ",
                                  this->transpose_a_, " ", this->transpose_b_));

      // BatchMatMul primitive can only be executed in **plain format** because
      // it has indefinite dims and can't be expressed by layout propagation.
      // Follow below steps to construct valid oneDNN primitive params if
      // broadcast is required:
      //   1. Figure out real output tf shape in plain format
      //   2. Broadcast input tf shapes to get corresponding plain md, and use
      //      them for primitive execution
      //   3. Reorder real input memory to plain memory if needed
      TensorShape dst_tf_shape = bcast.output_batch_shape();
      dst_tf_shape.AddDim(d0);
      dst_tf_shape.AddDim(d3);

      if (!this->post_op_util_.HasBias() && dst_tf_shape.num_elements() == 0) {
        Tensor* dst_tensor = nullptr;
        OneDnnShape dst_onednn_shape;

        dst_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kDstIndex_, &dst_tensor,
                                     dst_tf_shape, dst_onednn_shape);
        return;
      }

      // Calculate dims and memory desc
      // `src_fwd_md` and `wei_fwd_md`: plain md for BatchMatMul primitive
      // execution, which are broadcasted and expressed by dims/strides.
      auto params = MatMulBaseUtil::CreateMatMulParams(
          src_tf_shape, wei_tf_shape, dst_tf_shape, this->transpose_a_,
          this->transpose_b_);
      auto src_fwd_md =
          memory::desc(params->a_dims, OneDnnType<Tlhs>(), params->a_strides);
      auto wei_fwd_md =
          memory::desc(params->b_dims, OneDnnType<Trhs>(), params->b_strides);
      auto dst_fwd_md = memory::desc(params->c_dims, OneDnnType<Toutput>(),
                                     params->c_strides);

      // `src_md` and `wei_md`: real input md in plain or block format
      memory::desc src_md = src_onednn_shape.IsOneDnnTensor()
                                ? src_onednn_shape.GetOneDnnLayout()
                                : src_fwd_md;

      memory::desc wei_md = wei_onednn_shape.IsOneDnnTensor()
                                ? wei_onednn_shape.GetOneDnnLayout()
                                : wei_fwd_md;

      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      // Create matmul forward primitive
      std::unordered_map<int, memory> fwd_primitive_args;
      auto fwd_desc = matmul::desc(src_fwd_md, wei_fwd_md, dst_fwd_md);
      memory bias_mem;
      if (this->post_op_util_.HasBias()) {
        auto bias_md = memory::desc(params->bias_dims, OneDnnType<Toutput>(),
                                    params->bias_strides);
        // create bias memory
        const Tensor& bias_tensor = context->input(kBiasIndex_);
        memory bias_mem = CreateDnnlMemory(
            bias_md, onednn_engine, GetTensorBuffer<Toutput>(&bias_tensor));
        // Reassigin desc if it has bias.
        fwd_desc = matmul::desc(src_fwd_md, wei_fwd_md, bias_md, dst_fwd_md);
      }
      auto fwd_pd = GetPrimitiveDesc(context, fwd_desc, &fwd_primitive_args,
                                     onednn_engine);
      auto fwd_primitive = matmul(fwd_pd);

      // Create src memory, check if src needs to be reordered
      memory src_mem = CreateDnnlMemory(src_md, onednn_engine,
                                        GetTensorBuffer<Tlhs>(&src_tensor));

      memory src_reorder_mem;
      Tensor src_reorder_tensor;

      // `src_tf_md` and `wei_tf_md`: dst md for Reorder primitives,
      // which are created with format_tag. For some complex formats,
      // such as NHWC, the memory dims of blocked input tensors with NHWC
      // are maintained as NCHW. However, `src_fwd_md` and `wei_fwd_md`
      // express in NHWC since they are created with real TF shapes and strides
      // for plain MatMul primitive execution. Different shapes between `src_md`
      // and `src_fwd_md` will cause crash when creating Reorder primitive.
      // Ditto for `wei_md` and `wei_fwd_md`.
      memory::desc src_tf_md = src_onednn_shape.IsOneDnnTensor()
                                   ? src_onednn_shape.GetTfLayout()
                                   : src_md;
      memory::desc wei_tf_md = wei_onednn_shape.IsOneDnnTensor()
                                   ? wei_onednn_shape.GetTfLayout()
                                   : wei_md;

      // Reorder `src_md` -> `src_tf_md` and `wei_md` -> `wei_tf_md` if needed.
      bool is_src_reordered = (src_md != src_tf_md);

      if (is_src_reordered) {
        int64 src_reorder_size = fwd_pd.src_desc().get_size() / sizeof(Tlhs);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<Tlhs>::v(),
                                              TensorShape({src_reorder_size}),
                                              &src_reorder_tensor));

        src_reorder_mem =
            CreateDnnlMemory(src_tf_md, onednn_engine,
                             GetTensorBuffer<Tlhs>(&src_reorder_tensor));

        ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
      }

      memory wei_mem, wei_reorder_mem;
      Tensor wei_reorder_tensor;

      bool is_wei_reordered = (wei_md != wei_tf_md);

      if (is_wei_reordered) {
        Trhs* wei_cached_data = nullptr;
        if (this->is_filter_const_) {
          if (this->weight_cache_manager_.IsEmpty()) {
            // Cache weight in first time executing this node
            this->weight_cache_manager_.SetCache(
                context, wei_md, wei_tf_md, GetTensorBuffer<Trhs>(&wei_tensor),
                onednn_engine);
          }

          wei_cached_data =
              this->weight_cache_manager_.GetCache(context, wei_tf_md);
        }
        // Weight cache may be failed, need to check it here.
        if (wei_cached_data != nullptr) {
          wei_reorder_mem =
              CreateDnnlMemory(wei_tf_md, onednn_engine, wei_cached_data);
        } else {
          // During training, reorder weight in each iteration
          wei_mem = CreateDnnlMemory(wei_md, onednn_engine,
                                     GetTensorBuffer<Trhs>(&wei_tensor));

          int64 wei_reorder_size =
              fwd_pd.weights_desc().get_size() / sizeof(Trhs);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<Trhs>::v(),
                                                TensorShape{wei_reorder_size},
                                                &wei_reorder_tensor));
          wei_reorder_mem =
              CreateDnnlMemory(wei_tf_md, onednn_engine,
                               GetTensorBuffer<Trhs>(&wei_reorder_tensor));
          ReorderMemory(*context, &wei_mem, &wei_reorder_mem, onednn_engine);
        }
      } else {
        // No reorder needed
        wei_mem = CreateDnnlMemory(wei_md, onednn_engine,
                                   GetTensorBuffer<Trhs>(&wei_tensor));
      }

      OneDnnShape dst_onednn_shape;
      dst_onednn_shape.SetOneDnnTensor(false);
      Tensor* dst_tensor = nullptr;
      AllocateOutputSetOneDnnShape(context, kDstIndex_, &dst_tensor,
                                   dst_tf_shape, dst_onednn_shape);

      // Create dst memory
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      GetTensorBuffer<Toutput>(dst_tensor));

      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          fwd_pd.scratchpad_desc().get_size() / sizeof(Tlhs);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Tlhs>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<Tlhs>(&scratchpad_tensor));

      // Execute BatchMatMul
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      fwd_primitive_args.emplace(DNNL_ARG_SRC,
                                 is_src_reordered ? src_reorder_mem : src_mem);
      fwd_primitive_args.emplace(DNNL_ARG_WEIGHTS,
                                 is_wei_reordered ? wei_reorder_mem : wei_mem);
      fwd_primitive_args.emplace(DNNL_ARG_DST, dst_mem);
      fwd_primitive_args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem);
      if (this->post_op_util_.HasBias()) {
        fwd_primitive_args.emplace(DNNL_ARG_BIAS, bias_mem);
      }
      fwd_primitive.execute(onednn_stream, fwd_primitive_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  virtual void AccumulateMulAndInt8Scale(OpKernelContext* context,
                                         float* mul_value) {
    return;
  }

  matmul::primitive_desc GetPrimitiveDesc(
      OpKernelContext* context, const matmul::desc& fwd_desc,
      std::unordered_map<int, memory>* fwd_args,
      const dnnl::engine& onednn_engine) {
    dnnl::primitive_attr post_ops_attr;
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (this->post_op_util_.HasOutputScales()) {
      // mul_value = INT8 scale
      float mul_value = 1.0;
      AccumulateMulAndInt8Scale(context, &mul_value);

      std::vector<float> scales = {mul_value};
      this->post_op_util_.SetOutputScale(scales);
    }

    std::vector<memory::desc> md_list;
    const int kPostOpStartIdx =
        1 + (this->post_op_util_.HasBias() ? kBiasIndex_ : kWeightIndex_);
    for (int i = 0; i < this->post_op_util_.GetBinaryNum(); ++i) {
      // FusedBatchMatMul need to set binary input md in node execution.
      const Tensor& binary_tensor = context->input(kPostOpStartIdx + i);
      OneDnnShape onednn_shape;
      GetOneDnnShape(context, kPostOpStartIdx + i, &onednn_shape);

      // Same as input and weight of BatchMatMul, add tensor also needs to:
      //   1. Get original block/plain md
      //   2. Figure out the extended md for primitive execution
      //   3. Reorder original md to extended md if needed
      TensorShape tf_shape = onednn_shape.IsOneDnnTensor()
                                 ? onednn_shape.GetTfShape()
                                 : binary_tensor.shape();
      ITEX_CHECK(binary_tensor.NumElements() == 1 || tf_shape.dims() >= 3)
          << "Binary input of FusedBatchMatMul must be scalar or have 3 dims "
          << "at least, but got " << tf_shape.dims();

      auto binary_dims = TFShapeToOneDnnDims(tf_shape);
      auto binary_strides = CalculateTFStrides(binary_dims);
      auto binary_md =
          memory::desc(binary_dims, OneDnnType<Toutput>(), binary_strides);

      bool is_reordered = false;
      if (onednn_shape.IsOneDnnTensor()) {
        is_reordered = (onednn_shape.GetOneDnnLayout() == binary_md);
      }
      // FIXME(itex): Simply ingnore reorder this time, will fix it soon.
      ITEX_CHECK(!is_reordered)
          << "Need to reorder binary input of FusedBatchMatMul";

      md_list.push_back(binary_md);
      auto binary_mem = CreateDnnlMemory(
          binary_md, onednn_engine, GetTensorBuffer<Toutput>(&binary_tensor));
      fwd_args->insert(
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, binary_mem});
    }

    this->post_op_util_.SetPostOpAttr(&post_ops_attr, md_list);
    return matmul::primitive_desc(fwd_desc, post_ops_attr, onednn_engine);
  }

 private:
  static const int kSrcIndex_ = 0, kWeightIndex_ = 1, kBiasIndex_ = 2,
                   kDstIndex_ = 0, kMaxBinaryNum_ = 2;
#ifdef ITEX_ONEDNN_3_0
  HostDataCache<Device, float> output_scale_cache_;
#endif
};

template <typename Device, typename Tlhs, typename Trhs, typename Toutput>
class OneDnnQuantizedBatchMatMulV2Op
    : public OneDnnBatchMatMulV2Op<Device, Tlhs, Trhs, Toutput> {
 public:
  explicit OneDnnQuantizedBatchMatMulV2Op(OpKernelConstruction* context)
      : OneDnnBatchMatMulV2Op<Device, Tlhs, Trhs, Toutput>(context) {
    int num_args = 0;
    std::vector<string> fused_ops;

    this->post_op_util_.AddOps({"Quantized"});

    if (context->HasAttr("fused_ops")) {
      // Fusion will be handled in fathter class, only get info for check here.
      OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    }

    if (context->HasAttr("num_args")) {
      OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));

      if (context->HasAttr("fused_ops")) {
        OP_REQUIRES(context, num_args == fused_ops.size(),
                    errors::InvalidArgument(
                        "_QuantizedFusedBatchMatMulV2AndDequantize should "
                        "have same number of additional "
                        "inputs as the number of fusions"));
      }
    }

    kInputIndexMinLhs = num_args + 2;
    kInputIndexMaxLhs = num_args + 3;
    kInputIndexMinRhs = num_args + 4;
    kInputIndexMaxRhs = num_args + 5;
  }

  void AccumulateMulAndInt8Scale(OpKernelContext* context,
                                 float* mul_value) override {
    const float min_lhs =
        context->input(kInputIndexMinLhs).template flat<float>()(0);
    const float max_lhs =
        context->input(kInputIndexMaxLhs).template flat<float>()(0);
    const float min_rhs =
        context->input(kInputIndexMinRhs).template flat<float>()(0);
    const float max_rhs =
        context->input(kInputIndexMaxRhs).template flat<float>()(0);
    const float range_lhs = std::max(std::abs(min_lhs), std::abs(max_lhs));
    const float range_rhs = std::max(std::abs(min_rhs), std::abs(max_rhs));
    const float max_int8_lhs =
        (std::is_same<Tlhs, quint8>::value) ? 255.0f : 127.0f;
    const float max_int8_rhs =
        (std::is_same<Trhs, quint8>::value) ? 255.0f : 127.0f;
    float scale_output = range_lhs * range_rhs / (max_int8_lhs * max_int8_rhs);
    *mul_value = *mul_value * scale_output;
  }

 private:
  int kInputIndexMinLhs;
  int kInputIndexMaxLhs;
  int kInputIndexMinRhs;
  int kInputIndexMaxRhs;
};

// BatchMatMul FP32 kernel registration
#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                                 \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnBatchMatMulV2")                        \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .HostMemory("a_meta")                           \
                              .HostMemory("b_meta")                           \
                              .HostMemory("product_meta"),                    \
                          OneDnnBatchMatMulV2Op<GPUDevice, TYPE, TYPE, TYPE>) \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedBatchMatMulV2")                   \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .HostMemory("a_meta")                           \
                              .HostMemory("b_meta")                           \
                              .HostMemory("args_meta")                        \
                              .HostMemory("product_meta"),                    \
                          OneDnnBatchMatMulV2Op<GPUDevice, TYPE, TYPE, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#else
#define REGISTER_KERNEL(TYPE)                                                 \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnBatchMatMulV2")                        \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T"),                     \
                          OneDnnBatchMatMulV2Op<CPUDevice, TYPE, TYPE, TYPE>) \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedBatchMatMulV2")                   \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T"),                     \
                          OneDnnBatchMatMulV2Op<CPUDevice, TYPE, TYPE, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

#endif  // INTEL_CPU_ONLY

// BatchMatMul INT8 kernel registration
#ifdef INTEL_CPU_ONLY
#define REGISTER_ONEDNN_KERNEL(op, kernel, lhs_type, rhs_type, output_type) \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(op)                                                              \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<lhs_type>("T1")                                   \
          .TypeConstraint<rhs_type>("T2")                                   \
          .TypeConstraint<output_type>("Toutput"),                          \
      kernel TEMPLATE_ARGS(CPUDevice, lhs_type, rhs_type, output_type));
#else
#define REGISTER_ONEDNN_KERNEL(op, kernel, lhs_type, rhs_type, output_type) \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(op)                                                              \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<lhs_type>("T1")                                   \
          .TypeConstraint<rhs_type>("T2")                                   \
          .TypeConstraint<output_type>("Toutput") HOSTMEMORYLIST,           \
      kernel TEMPLATE_ARGS(GPUDevice, lhs_type, rhs_type, output_type));
#endif  // INTEL_CPU_ONLY

#define REGISTER_ONEDNN_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, output_type) \
  REGISTER_ONEDNN_KERNEL(op, kernel, qint8, qint8, output_type);

#ifdef INTEL_CPU_ONLY
#define REGISTER_ONEDNN_KERNEL_ALL_OUTPUT_TYPES(op, kernel)    \
  REGISTER_ONEDNN_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, float); \
  REGISTER_ONEDNN_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, Eigen::bfloat16);
#else
#define REGISTER_ONEDNN_KERNEL_ALL_OUTPUT_TYPES(op, kernel)              \
  REGISTER_ONEDNN_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, float);           \
  REGISTER_ONEDNN_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, Eigen::bfloat16); \
  REGISTER_ONEDNN_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, Eigen::half);
#endif  // INTEL_CPU_ONLY

// Concrete OneDnn BatchnMatMul INT8 kernel implementation
#define TEMPLATE_ARGS(Device, lhs_type, rhs_type, output_type) \
<Device, lhs_type, rhs_type, output_type>
#define HOSTMEMORYLIST                                                 \
  .HostMemoryList4("min_x", "max_x", "min_y", "max_y")                 \
      .HostMemoryList6("x_meta", "y_meta", "min_x_meta", "max_x_meta", \
                       "min_y_meta", "max_y_meta")                     \
      .HostMemoryList1("output_meta")
REGISTER_ONEDNN_KERNEL_ALL_OUTPUT_TYPES(
    "_OneDnnQuantizedBatchMatMulV2AndDequantize",
    OneDnnQuantizedBatchMatMulV2Op);
#undef HOSTMEMORYLIST

#define HOSTMEMORYLIST                                                \
  .HostMemoryList4("min_x", "max_x", "min_y", "max_y")                \
      .HostMemoryList7("x_meta", "y_meta", "args_meta", "min_x_meta", \
                       "max_x_meta", "min_y_meta", "max_y_meta")      \
      .HostMemoryList1("output_meta")
REGISTER_ONEDNN_KERNEL_ALL_OUTPUT_TYPES(
    "_OneDnnQuantizedFusedBatchMatMulV2AndDequantize",
    OneDnnQuantizedBatchMatMulV2Op);
#undef HOSTMEMORYLIST
#undef TEMPLATE_ARGS

}  // namespace itex
