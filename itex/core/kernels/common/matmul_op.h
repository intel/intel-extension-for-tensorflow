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

#ifndef ITEX_CORE_KERNELS_COMMON_MATMUL_OP_H_
#define ITEX_CORE_KERNELS_COMMON_MATMUL_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/bcast.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {

using dnnl::memory;

// Simple wrapper over BCast specialized for MatMul.
// Provides utilities for broadcasting across batch dimensions for binary
// MatMul-like operations.
class MatMulBCast {
 public:
  using Vec = BCast::Vec;

  MatMulBCast(Vec x, Vec y) {
    if (x.size() < 2 || y.size() < 2) return;
    x.resize(x.size() - 2);
    y.resize(y.size() - 2);

    batch_bcast_ = absl::make_unique<BCast>(std::move(x), std::move(y));
    if (!batch_bcast_->IsValid()) return;

    x_batch_size_ = TensorShape(batch_bcast_->x_reshape()).num_elements();
    y_batch_size_ = TensorShape(batch_bcast_->y_reshape()).num_elements();
    output_shape_ = TensorShape(batch_bcast_->output_shape());
    output_batch_size_ = output_shape_.num_elements();
    broadcasting_required_ =
        std::min(x_batch_size_, y_batch_size_) != output_batch_size_;

    if (broadcasting_required_) {
      ComputeBatchIndices(output_batch_size_, batch_bcast_->x_reshape(),
                          batch_bcast_->x_bcast(), &x_batch_indices_);
      ComputeBatchIndices(output_batch_size_, batch_bcast_->y_reshape(),
                          batch_bcast_->y_bcast(), &y_batch_indices_);
    }
  }

  bool IsValid() const { return batch_bcast_ && batch_bcast_->IsValid(); }
  bool IsBroadcastingRequired() const { return broadcasting_required_; }

  const int64 output_batch_size() const { return output_batch_size_; }
  const int64 x_batch_size() const { return x_batch_size_; }
  const int64 y_batch_size() const { return y_batch_size_; }
  const TensorShape& output_batch_shape() const { return output_shape_; }

  // Returns the mapping from the flattened output batch indices to x's
  // flattened batch indices. The result is a vector of length
  // output_batch_size(). To compute the i'th batch output, a binary matmul-like
  // operation should use the `x_batch_indices()[i]`th batch index of `x`.
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64>& x_batch_indices() const { return x_batch_indices_; }
  // Returns the mapping from the flattened output batch indices to y's
  // flattened batch indices. Similar to x_batch_indices().
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64>& y_batch_indices() const { return y_batch_indices_; }

 private:
  std::unique_ptr<BCast> batch_bcast_;
  bool broadcasting_required_ = false;
  int64 x_batch_size_;
  int64 y_batch_size_;
  TensorShape output_shape_;
  int64 output_batch_size_;
  std::vector<int64> x_batch_indices_;
  std::vector<int64> y_batch_indices_;
};

struct OneDnnMatMulParams {
  memory::dims a_dims;
  memory::dims b_dims;
  memory::dims c_dims;
  memory::dims bias_dims;
  memory::dims a_strides;
  memory::dims b_strides;
  memory::dims c_strides;
  memory::dims bias_strides;

  OneDnnMatMulParams(memory::dims a_dims, memory::dims b_dims,
                     memory::dims c_dims, memory::dims bias_dims,
                     memory::dims a_strides, memory::dims b_strides,
                     memory::dims c_strides, memory::dims bias_strides)
      : a_dims(std::move(a_dims)),
        b_dims(std::move(b_dims)),
        c_dims(std::move(c_dims)),
        bias_dims(std::move(bias_dims)),
        a_strides(std::move(a_strides)),
        b_strides(std::move(b_strides)),
        c_strides(std::move(c_strides)),
        bias_strides(std::move(bias_strides)) {}
};

class MatMulBaseUtil {
 public:
  // This method makes the rank (ndims) of input same as the output by adding
  // new axes to the input. For example, if input shape is [a, b, c, d] and
  // output shape is [e, f, g, h, i, j], then the reshaped input would have a
  // shape of [1, 1, a, b, c, d].
  static void ExpandInputDimsToOutputShape(const TensorShape& input_shape,
                                           const TensorShape& output_shape,
                                           dnnl::memory::dims* reshaped_dims) {
    auto ndims_input = input_shape.dims();
    auto ndims_output = output_shape.dims();
    auto dim_offset = ndims_output - ndims_input;
    ITEX_DCHECK(dim_offset > 0);
    reshaped_dims->clear();
    reshaped_dims->resize(ndims_output, 1);
    auto input_dims = input_shape.dim_sizes();
    for (int dim_idx = 0; dim_idx < ndims_input; ++dim_idx)
      reshaped_dims->at(dim_idx + dim_offset) = input_dims[dim_idx];
  }

  static std::unique_ptr<OneDnnMatMulParams> CreateMatMulParams(
      const TensorShape& lhs_shape, const TensorShape& rhs_shape,
      const TensorShape& out_shape, bool adj_x_, bool adj_y_) {
    const auto ndims_lhs = lhs_shape.dims();
    const auto ndims_rhs = rhs_shape.dims();
    const auto ndims_out = out_shape.dims();
    auto lhs_dims = TFShapeToOneDnnDims(lhs_shape);
    auto rhs_dims = TFShapeToOneDnnDims(rhs_shape);
    auto out_dims = TFShapeToOneDnnDims(out_shape);

    // DNNL matmul_primitive requires ranks of inputs and output to be same.
    // Create dnnl::memory::dims for inputs and output of same rank.
    // It is assumed here that MatMulBCast object creates output_batch_shape as
    // a conforming superset of input batch shapes, i.e., ndims_out >=
    // ndims_lhs and ndims_out >= ndims_rhs.
    if (ndims_lhs < ndims_out) {
      ExpandInputDimsToOutputShape(lhs_shape, out_shape, &lhs_dims);
    }
    if (ndims_rhs < ndims_out) {
      ExpandInputDimsToOutputShape(rhs_shape, out_shape, &rhs_dims);
    }

    auto lhs_strides = CalculateTFStrides(lhs_dims);
    auto rhs_strides = CalculateTFStrides(rhs_dims);
    auto out_strides = CalculateTFStrides(out_dims);
    int idx_last = ndims_out - 1;
    int idx_2nd_last = ndims_out - 2;

    // dst(m,n) = \sigma{src(m,k) * weights(k, n)}
    // lhs_strides holds the strides for each dim, say {24, 12, 4, 1} for
    // src_tensor {1, 2, 3, 4} if adj_x_ is false.
    // If adj_x_ is true, swap the innermost two dims of lhs_strides
    // to {24, 12, 1, 4}, just like set memory::format_tag::abdc
    if (adj_x_) {
      std::swap(lhs_dims[idx_last], lhs_dims[idx_2nd_last]);
      std::swap(lhs_strides[idx_last], lhs_strides[idx_2nd_last]);
    }
    if (adj_y_) {
      std::swap(rhs_dims[idx_last], rhs_dims[idx_2nd_last]);
      std::swap(rhs_strides[idx_last], rhs_strides[idx_2nd_last]);
    }

    dnnl::memory::dims bias_dims(rhs_dims.size(), 1);
    bias_dims[rhs_dims.size() - 1] = rhs_dims[rhs_dims.size() - 1];
    auto bias_strides = CalculateTFStrides(bias_dims);

    return absl::make_unique<OneDnnMatMulParams>(
        lhs_dims, rhs_dims, out_dims, bias_dims, lhs_strides, rhs_strides,
        out_strides, bias_strides);
  }
};

template <typename Device, typename T, typename Tout, typename Tpost,
          bool allow_bcast = true>
class MatMulOp : public OpKernel {
 public:
  explicit MatMulOp(OpKernelConstruction* context) : OpKernel(context) {
    if (context->HasAttr("transpose_a")) {
      OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &adj_x_));
    }
    if (context->HasAttr("transpose_b")) {
      OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &adj_y_));
    }
    if (context->HasAttr("is_filter_const")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_filter_const", &is_filter_const_));
    }

    if (context->HasAttr("fused_ops")) {
      std::vector<string> fused_ops;
      OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
      OP_REQUIRES(
          context, post_op_util_.AddOps(fused_ops),
          errors::InvalidArgument("Found unsupported fusion in Fused MatMul."));

      // Set alpha if get `LeakyRelu` after adding ops.
      if (post_op_util_.HasLeakyRelu()) {
        float alpha;
        OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha", &alpha));
        post_op_util_.SetLeakyReluAlpha(alpha);
      }
    }

    if (context->HasAttr("inplace_sum")) {
      OP_REQUIRES_OK(context, context->GetAttr("inplace_sum", &inplace_sum_));
    }

    fp32_math_mode_ = GetFP32MathMode<Device>();
    bool is_bf16_math_mode = false;
    if (context->HasAttr("is_bf16_math_mode")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_bf16_math_mode", &is_bf16_math_mode));
    }
    if (is_bf16_math_mode && std::is_same<Device, CPUDevice>::value) {
      fp32_math_mode_ = dnnl::fpmath_mode::bf16;
    }

    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));
  }

  void InitOrSetMemory(OpKernelContext* context) {
    if (!(enable_cache_ && is_init_ && context->is_input_same(0, input_dims_) &&
          context->is_input_same(1, weights_dims_))) {
      Init(context);
      return;
    }

    if (is_input_zero_) {
      functor::SetZeroFunctor<Device, Tout> f;
      OP_REQUIRES_OK(context, context->allocate_output(kDstIndex_, dst_shape_,
                                                       &dst_tensor_));
      f(context->eigen_device<Device>(), dst_tensor_->flat<Tout>());
      return;
    }

    src_mem_.set_data_handle(context->tensor_data(kSrcIndex_));
    if (is_weight_reorder_) {
      if (!is_filter_const_) {
        weights_mem_input_.set_data_handle(context->tensor_data(kWeightIndex_));
        weights_mem_.set_data_handle(GetTensorBuffer<T>(&tmp_weight_));
        ReorderMemory(*context, &weights_mem_input_, &weights_mem_,
                      dnnl_engine_);
      }
    } else {
      weights_mem_.set_data_handle(context->tensor_data(kWeightIndex_));
    }

    if (post_op_util_.HasBias()) {
      bias_mem_.set_data_handle(context->tensor_data(kBiasIndex_));
    }

    if (post_op_util_.HasAdd() || post_op_util_.HasBinary()) {
      add_tensor_ = &context->input(kAddIndex_);
    }

    if (post_op_util_.HasAdd()) {
      int is_forward_success = kUnsuccess_;
      // Try to do in-place.
      // TODO(itex): Remove this workaround when inplace works.
      if (inplace_sum_) {
        context->set_output(kDstIndex_, *add_tensor_);
        dst_tensor_ = context->mutable_output(kDstIndex_);
        is_forward_success = kAddIndex_;
      } else {
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {kAddIndex_}, kDstIndex_, dst_shape_,
                                    &dst_tensor_, &is_forward_success));
      }
      // Reorder is needed, forward is failed but dst has been allocated;
      if (is_forward_success == kUnsuccess_) {
        // In-place do not success, need reorder.
        fuse_add_src_mem_.set_data_handle(GetTensorBuffer<Tpost>(add_tensor_));
        fuse_add_dst_mem_.set_data_handle(GetTensorBuffer<Tout>(dst_tensor_));
        ReorderMemory(*context, &fuse_add_src_mem_, &fuse_add_dst_mem_,
                      dnnl_engine_);
      }

    } else {
      OP_REQUIRES_OK(context, context->allocate_output(kDstIndex_, dst_shape_,
                                                       &dst_tensor_));
    }
    if (post_op_util_.HasBinary()) {
      // BatchMatMul + Add needs to set add input md in node execution.
      add_mem_.set_data_handle(GetTensorBuffer<Tpost>(add_tensor_));
    }
    dst_mem_.set_data_handle(GetTensorBuffer<Tout>(dst_tensor_));
  }

  void Init(OpKernelContext* context) {
    const Tensor& src_tensor = context->input(0);
    const Tensor& weights_tensor = context->input(1);
    fwd_primitive_args_.clear();
    auto input_shape = src_tensor.shape();
    input_dims_.clear();
    for (int i = 0; i < input_shape.dims(); ++i) {
      input_dims_.push_back(input_shape.dim_size(i));
    }
    auto weights_tensor_shape = weights_tensor.shape();
    weights_dims_.clear();
    for (int i = 0; i < weights_tensor_shape.dims(); ++i) {
      weights_dims_.push_back(weights_tensor_shape.dim_size(i));
    }

    OP_REQUIRES(context, src_tensor.dims() >= 2,
                errors::InvalidArgument("In[0] ndims must be >= 2: ",
                                        src_tensor.dims()));

    if (!allow_bcast) {
      // Using V1, so check to make sure lhs and rhs dimensions are correct and
      // no broadcasting is needed.
      OP_REQUIRES(
          context, src_tensor.dims() == weights_tensor.dims(),
          errors::InvalidArgument("lhs and rhs has different ndims: ",
                                  src_tensor.shape().DebugString(), " vs. ",
                                  weights_tensor.shape().DebugString()));
      const int ndims = src_tensor.dims();
      OP_REQUIRES(
          context, ndims >= 2,
          errors::InvalidArgument("lhs and rhs ndims must be >= 2: ", ndims));
      for (int i = 0; i < ndims - 2; ++i) {
        OP_REQUIRES(
            context, src_tensor.dim_size(i) == weights_tensor.dim_size(i),
            errors::InvalidArgument(
                "lhs.dim(", i, ") and rhs.dim(", i,
                ") must be the same: ", src_tensor.shape().DebugString(),
                " vs ", weights_tensor.shape().DebugString()));
      }
    }

    MatMulBCast bcast(src_tensor.shape().dim_sizes(),
                      weights_tensor.shape().dim_sizes());
    OP_REQUIRES(context, bcast.IsValid(),
                errors::InvalidArgument(
                    "In[0] and In[1] must have compatible batch dimensions: ",
                    src_tensor.shape().DebugString(), " vs. ",
                    weights_tensor.shape().DebugString()));

    // dst(bs, m,n) = \sigma{src(bs, m,k) * weights(bs, k, n)} + bias(bs, m,n)
    // Get the actual m & n to set dst_shape, and MatMulBCast will calculate the
    // shape of batches for us
    const int kSrcDims = src_tensor.dims();
    const auto m = adj_x_ ? src_tensor.dim_size(kSrcDims - 1)
                          : src_tensor.dim_size(kSrcDims - 2);
    const auto k = adj_x_ ? src_tensor.dim_size(kSrcDims - 2)
                          : src_tensor.dim_size(kSrcDims - 1);
    const int kWeightsDims = weights_tensor.dims();
    const auto k_weights = adj_y_ ? weights_tensor.dim_size(kWeightsDims - 1)
                                  : weights_tensor.dim_size(kWeightsDims - 2);
    const auto n = adj_y_ ? weights_tensor.dim_size(kWeightsDims - 2)
                          : weights_tensor.dim_size(kWeightsDims - 1);
    OP_REQUIRES(context, k == k_weights,
                errors::InvalidArgument(
                    "Matrix size-incompatible: In[0]: ",
                    src_tensor.shape().DebugString(),
                    ", In[1]: ", weights_tensor.shape().DebugString()));

    dst_shape_ = bcast.output_batch_shape();
    dst_shape_.AddDim(m);
    dst_shape_.AddDim(n);
    // The maximum number of dimensions for a tensor in DNNL is 6 on GPU.
    OP_REQUIRES(
        context, dst_shape_.dims() <= 6,
        errors::InvalidArgument(
            "Rank of output tensor must be <= 6, but is ", dst_shape_.dims(),
            ". Current implementation supports up to rank 6 tensors."));

    if (dst_shape_.num_elements() == 0) {
      is_input_zero_ = true;
      functor::SetZeroFunctor<Device, Tout> f;
      OP_REQUIRES_OK(context, context->allocate_output(kDstIndex_, dst_shape_,
                                                       &dst_tensor_));
      f(context->eigen_device<Device>(), dst_tensor_->flat<Tout>());
      is_init_ = true;
      return;
    }

    // Direct return if either input has 0 elements, but take care of fused ops
    // because they will change default value.
    if (!post_op_util_.HasBias() && !post_op_util_.HasAdd() &&
        (src_tensor.NumElements() == 0 || weights_tensor.NumElements() == 0)) {
      is_input_zero_ = true;
      functor::SetZeroFunctor<Device, Tout> f;
      OP_REQUIRES_OK(context, context->allocate_output(kDstIndex_, dst_shape_,
                                                       &dst_tensor_));
      f(context->eigen_device<Device>(), dst_tensor_->flat<Tout>());
      is_init_ = true;
      return;
    }

    try {
      // Compute parameters for DNNL matmul primitive.
      auto params = MatMulBaseUtil::CreateMatMulParams(
          src_tensor.shape(), weights_tensor.shape(), dst_shape_, adj_x_,
          adj_y_);
      auto src_md =
          memory::desc(params->a_dims, OneDnnType<T>(), params->a_strides);
      auto weights_md =
          memory::desc(params->b_dims, OneDnnType<T>(), params->b_strides);
      // Let oneDNN choose weight format if:
      //   1. Weight is const and can be cached
      //   2. Kernel is on CPU and weight is not tranposed
      bool is_any =
          is_filter_const_ ||
          (Eigen::internal::is_same<Device, CPUDevice>::value && !adj_y_);
      auto weights_md_prefer =
          is_any ? memory::desc(params->b_dims, OneDnnType<T>(),
                                memory::format_tag::any)
                 : weights_md;
      auto dst_md =
          memory::desc(params->c_dims, OneDnnType<Tout>(), params->c_strides);

      std::shared_ptr<dnnl::matmul::desc> matmul_desc_;
      if (post_op_util_.HasBias()) {
        // bias use same dims as dst
        auto bias_md = memory::desc(params->bias_dims, OneDnnType<Tpost>(),
                                    params->bias_strides);
        matmul_desc_.reset(
            new dnnl::matmul::desc(src_md, weights_md_prefer, bias_md, dst_md));
        // create bias memory
        const Tensor& bias_tensor = context->input(kBiasIndex_);
        bias_mem_ = CreateDnnlMemory(bias_md, dnnl_engine_,
                                     GetTensorBuffer<Tpost>(&bias_tensor));
      } else {
        matmul_desc_.reset(
            new dnnl::matmul::desc(src_md, weights_md_prefer, dst_md));
      }

      dnnl::primitive_attr post_ops_attr;
      post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<T, float>::value) {
        post_ops_attr.set_fpmath_mode(fp32_math_mode_);
      }
      std::shared_ptr<dnnl::matmul::primitive_desc> matmul_pd_ =
          std::make_shared<dnnl::matmul::primitive_desc>(*matmul_desc_,
                                                         dnnl_engine_);
      if (post_op_util_.HasAdd() || post_op_util_.HasBinary()) {
        add_tensor_ = &context->input(kAddIndex_);
      }

      // Handle Add fusion and decide output tensor buffer.
      if (post_op_util_.HasAdd()) {
        // const Tensor& add_tensor = context->input(kAddIndex_);
        int is_forward_success = kUnsuccess_;

        // Try to do in-place.
        // TODO(itex): Remove this workaround when inplace works.
        if (inplace_sum_) {
          context->set_output(kDstIndex_, *add_tensor_);
          dst_tensor_ = context->mutable_output(kDstIndex_);
          is_forward_success = kAddIndex_;
        } else {
          OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                      {kAddIndex_}, kDstIndex_, dst_shape_,
                                      &dst_tensor_, &is_forward_success));
        }
        // Reorder is needed, forward is failed but dst has been allocated;
        if (is_forward_success == kUnsuccess_) {
          // In-place do not success, need reorder.
          fuse_add_src_mem_ = CreateDnnlMemory(
              memory::desc(params->c_dims, OneDnnType<Tpost>(),
                           params->c_strides),
              dnnl_engine_, GetTensorBuffer<Tpost>(add_tensor_));
          fuse_add_dst_mem_ =
              CreateDnnlMemory(matmul_pd_->dst_desc(), dnnl_engine_,
                               GetTensorBuffer<Tout>(dst_tensor_));
          ReorderMemory(*context, &fuse_add_src_mem_, &fuse_add_dst_mem_,
                        dnnl_engine_);
        }
      } else {
        OP_REQUIRES_OK(context, context->allocate_output(kDstIndex_, dst_shape_,
                                                         &dst_tensor_));
      }

      // Handle Mul fusion.
      if (post_op_util_.HasOutputScales()) {
        const Tensor& scale_tensor = context->input(kMulIndex_);
        OP_REQUIRES(context, scale_tensor.NumElements() == 1,
                    errors::InvalidArgument("Mul Tensor must be a scalar"));

#ifndef INTEL_CPU_ONLY
        if (IsMulCacheEmpty()) {
          // Cache weight
          const T* mul_device_data = scale_tensor.flat<T>().data();
          CacheMul(context, mul_device_data);
        }
        T* mul_host_data = GetCachedMul(context);
        float mul_value = static_cast<float>(mul_host_data[0]);
#else
        float mul_value =
            static_cast<float>(scale_tensor.flat<Tpost>().data()[0]);
#endif  // INTEL_CPU_ONLY
        std::vector<float> scales = {mul_value};

        post_op_util_.SetOutputScale(scales);
      }
      if (this->post_op_util_.HasBinary()) {
        // BatchMatMul + Add needs to set add input md in node execution.
        // const Tensor& add_tensor = context->input(kAddIndex_);

        // Figure out the extended md for primitive execution
        TensorShape tf_shape = add_tensor_->shape();

        ITEX_CHECK(tf_shape.dims() >= 3)
            << "Add input of FusedBatchMatMul must have 3 dims at least";

        auto add_dims = TFShapeToOneDnnDims(tf_shape);
        auto add_strides = CalculateTFStrides(add_dims);
        auto add_md = memory::desc(add_dims, OneDnnType<Tpost>(), add_strides);

        this->post_op_util_.SetBinaryInput(add_md);
        add_mem_ = CreateDnnlMemory(add_md, dnnl_engine_,
                                    GetTensorBuffer<Tpost>(add_tensor_));

        fwd_primitive_args_.emplace(
            DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, add_mem_);
      }
      // Set post ops attr after handling all fusions.
      post_op_util_.SetPostOpAttr(&post_ops_attr);
      matmul_pd_.reset(new dnnl::matmul::primitive_desc(
          *matmul_desc_, post_ops_attr, dnnl_engine_));

      // Do weight cache only if Reorder is needed and weight is const.
      weights_mem_input_ = CreateDnnlMemory(
          weights_md, dnnl_engine_, GetTensorBuffer<T>(&weights_tensor));
      weights_md_prefer = matmul_pd_->weights_desc();
      is_weight_reorder_ = (weights_md != weights_md_prefer);
      if (is_weight_reorder_) {
        T* weight_cached_data = nullptr;

        // Check weight cache
        if (is_filter_const_) {
          if (weight_cache_manager_.IsEmpty()) {
            // Cache weight in first time executing this node.
            weight_cache_manager_.SetCache(
                context, weights_md, weights_md_prefer,
                GetTensorBuffer<T>(&weights_tensor), dnnl_engine_);
          }
          weight_cached_data =
              this->weight_cache_manager_.GetCache(context, weights_md_prefer);
        }

        if (weight_cached_data != nullptr) {
          weights_mem_ = CreateDnnlMemory(weights_md_prefer, dnnl_engine_,
                                          weight_cached_data);
        } else {
          // Reorder if cache is failed since pd has already used any format.
          int64_t reorder_size = weights_md_prefer.get_size() / sizeof(T);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                TensorShape({reorder_size}),
                                                &tmp_weight_));
          void* data_handle = GetTensorBuffer<T>(&tmp_weight_);
          weights_mem_ =
              CreateDnnlMemory(weights_md_prefer, dnnl_engine_, data_handle);
          ReorderMemory(*context, &weights_mem_input_, &weights_mem_,
                        dnnl_engine_);
        }
      } else {
        weights_mem_ = weights_mem_input_;
      }
      int64 scratchpad_size =
          matmul_pd_->scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor_));
      scratchpad_mem_ =
          dnnl::memory(matmul_pd_->scratchpad_desc(), dnnl_engine_,
                       GetTensorBuffer<T>(&scratchpad_tensor_));

      matmul_primitive_ = dnnl::matmul(*matmul_pd_);
      src_mem_ = CreateDnnlMemory(src_md, dnnl_engine_,
                                  GetTensorBuffer<T>(&src_tensor));
      dst_mem_ = CreateDnnlMemory(dst_md, dnnl_engine_,
                                  GetTensorBuffer<Tout>(dst_tensor_));
      fwd_primitive_args_.emplace(DNNL_ARG_SRC, src_mem_);
      fwd_primitive_args_.emplace(DNNL_ARG_WEIGHTS, weights_mem_);
      fwd_primitive_args_.emplace(DNNL_ARG_DST, dst_mem_);
      fwd_primitive_args_.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem_);

      if (post_op_util_.HasBias()) {
        fwd_primitive_args_.emplace(DNNL_ARG_BIAS, bias_mem_);
      }
      is_init_ = true;
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);
    dst_tensor_ = nullptr;
    dnnl_engine_ = CreateDnnlEngine<Device>(*context);
    // onednn_stream has thread safety issue, need create a new one in
    // every compute.
    dnnl_stream_ = CreateDnnlStream(*context, dnnl_engine_);
    InitOrSetMemory(context);

    // Skip primitive execution if the calculation is meaningless.
    if (is_input_zero_) return;

    matmul_primitive_.execute(dnnl_stream_, fwd_primitive_args_);
  }

  // TODO(itex): Wrap all cache related code to a module, reuse this module
  inline bool IsMulCacheEmpty() TF_LOCKS_EXCLUDED(mul_cache_mu_) {
    tf_shared_lock lock(&mul_cache_mu_);
    return (!mul_cached_tensor_.IsInitialized());
  }

  void AllocatePersistentTensor(OpKernelContext* context, Tensor** mul_tensor) {
    ITEX_DCHECK(mul_tensor);
    TensorShape mul_tf_shape;
    // Only one number is stored
    mul_tf_shape.AddDim(1);
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(context, context->allocate_persistent(
                                DataTypeToEnum<T>::value, mul_tf_shape,
                                &mul_cached_tensor_, mul_tensor, alloc_attr));
  }

#ifndef INTEL_CPU_ONLY
  void CacheMul(OpKernelContext* context, const T* mul_device_data)
      TF_LOCKS_EXCLUDED(mul_cache_mu_) {
    mutex_lock lock(&mul_cache_mu_);

    // If mul is already cached, there's nothing to do.
    if (mul_cached_tensor_.IsInitialized()) {
      return;
    }

    // Create cached mul buffer
    Tensor* mul_tensor_ptr = nullptr;
    AllocatePersistentTensor(context, &mul_tensor_ptr);
    T* mul_host_data = const_cast<T*>(mul_tensor_ptr->flat<T>().data());

    // TODO(itex): refactor the memcpy code
    auto* dpcpp_stream = context->GetDeviceStream();
    auto event =
        dpcpp_stream->memcpy(mul_host_data, mul_device_data, 1 * sizeof(T));
    event.wait();
  }

  T* GetCachedMul(OpKernelContext* context) TF_LOCKS_EXCLUDED(mul_cache_mu_) {
    tf_shared_lock lock(&mul_cache_mu_);
    const Tensor& mul_cached_data = *mul_cached_tensor_.AccessTensor(context);

    return static_cast<T*>(const_cast<T*>(mul_cached_data.flat<T>().data()));
  }
#endif  // INTEL_CPU_ONLY

 protected:
  bool adj_x_ = false;
  bool adj_y_ = false;
  bool inplace_sum_ = false;
  bool is_filter_const_ = false;
  bool is_weight_reorder_ = false;
  bool enable_cache_ = false;
  bool is_init_ = false;
  bool is_input_zero_ = false;
  const int kSrcIndex_ = 0, kDstIndex_ = 0, kWeightIndex_ = 1, kBiasIndex_ = 2,
            kAddIndex_ = 3, kMulIndex_ = 2, kUnsuccess_ = -1;

  // Fusion util.
  PostOpUtil post_op_util_;

  // Weight cache manager
  WeightCacheManager<T> weight_cache_manager_;

 private:
  mutex mul_cache_mu_, mu_compute_;
  std::unordered_map<int, memory> fwd_primitive_args_;
  memory src_mem_, weights_mem_, weights_mem_input_, dst_mem_, bias_mem_,
      add_mem_, fuse_add_src_mem_, fuse_add_dst_mem_, scratchpad_mem_;
  dnnl::matmul matmul_primitive_;
  Tensor* dst_tensor_;
  const Tensor* add_tensor_;
  Tensor tmp_weight_, scratchpad_tensor_;
  std::vector<int64> input_dims_, weights_dims_;
  TensorShape dst_shape_;
  PersistentTensor mul_cached_tensor_ TF_GUARDED_BY(mul_cache_mu_);
  dnnl::fpmath_mode fp32_math_mode_ = dnnl::fpmath_mode::strict;
  dnnl::stream dnnl_stream_;
  dnnl::engine dnnl_engine_;
};

template <typename Device, typename T, typename Tgrad>
class FusedMatMulGradOp : public OpKernel {
 public:
  explicit FusedMatMulGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));

    OP_REQUIRES(context, fused_ops_.size() == 1,
                errors::InvalidArgument(
                    "_FusedMatMulGrad must have 1 post-arguments at most."));
    OP_REQUIRES(
        context, fused_ops_[0] == "BiasAddGrad",
        errors::InvalidArgument(
            "The 1st post-argument of _FusedMatMulGrad must be BiasAddGrad."));
    fp32_math_mode_ = GetFP32MathMode<Device>();
    bool is_bf16_math_mode = false;
    if (context->HasAttr("is_bf16_math_mode")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_bf16_math_mode", &is_bf16_math_mode));
    }
    if (is_bf16_math_mode && std::is_same<Device, CPUDevice>::value) {
      fp32_math_mode_ = dnnl::fpmath_mode::bf16;
    }

    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));
  }

  void InitOrSetMemory(OpKernelContext* context) {
    diff_weight_tensor_ = nullptr;
    diff_bias_tensor_ = nullptr;

    if (enable_cache_ && is_init_ &&
        context->is_input_same(kSrcIndex_, input_dims_) &&
        context->is_input_same(kDiffDstIndex_, diff_dst_dims_)) {
      src_mem_.set_data_handle(context->tensor_data(kSrcIndex_));
      diff_dst_mem_.set_data_handle(context->tensor_data(kDiffDstIndex_));

      OP_REQUIRES_OK(context, context->allocate_output(kDiffWeightIndex_,
                                                       diff_weight_tf_shape_,
                                                       &diff_weight_tensor_));
      OP_REQUIRES_OK(context, context->allocate_output(kDiffBiasIndex_,
                                                       diff_bias_tf_shape_,
                                                       &diff_bias_tensor_));
      diff_weight_mem_.set_data_handle(GetTensorBuffer<T>(diff_weight_tensor_));
      diff_bias_mem_.set_data_handle(GetTensorBuffer<Tgrad>(diff_bias_tensor_));
    } else {
      Init(context);
    }
  }

  void Init(OpKernelContext* context) {
    fwd_primitive_args_.clear();
    const Tensor& src_tensor = context->input(kSrcIndex_);
    const Tensor& diff_dst_tensor = context->input(kDiffDstIndex_);
    auto src_tf_shape = src_tensor.shape();
    auto diff_dst_tf_shape = diff_dst_tensor.shape();
    input_dims_.clear();
    for (int i = 0; i < src_tf_shape.dims(); ++i) {
      input_dims_.push_back(src_tf_shape.dim_size(i));
    }
    diff_dst_dims_.clear();
    for (int i = 0; i < diff_dst_tf_shape.dims(); ++i) {
      diff_dst_dims_.push_back(diff_dst_tf_shape.dim_size(i));
    }
    try {
      const int dim_pair[] = {transpose_a_ ? 0 : 1, transpose_a_ ? 1 : 0};
      const int batch = src_tf_shape.dim_size(1 - dim_pair[0]);
      const int k = src_tf_shape.dim_size(dim_pair[0]);
      const int channel = diff_dst_tf_shape.dim_size(1);

      OP_REQUIRES(
          context, batch == diff_dst_tf_shape.dim_size(0),
          errors::InvalidArgument(
              "Matrix size-incompatible: In[0]: ", src_tf_shape.DebugString(),
              ", In[1]: ", diff_dst_tf_shape.DebugString()));

      if (batch == 0 || channel == 0) {
        return;
      }
      // Create primitive.
      dnnl::memory::dims src_dims = dnnl::memory::dims({batch, k});
      dnnl::memory::dims diff_dst_dims = dnnl::memory::dims({batch, channel});
      dnnl::memory::dims diff_weight_dims = dnnl::memory::dims({channel, k});
      dnnl::memory::dims diff_bias_dims = dnnl::memory::dims({channel});
      dnnl::memory::format_tag src_format = transpose_a_
                                                ? dnnl::memory::format_tag::cn
                                                : dnnl::memory::format_tag::nc;
      dnnl::memory::format_tag diff_weight_format =
          transpose_b_ ? dnnl::memory::format_tag::oi
                       : dnnl::memory::format_tag::io;
      dnnl::memory::format_tag diff_dst_format = dnnl::memory::format_tag::nc;

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      if (std::is_same<T, float>::value) {
        attr.set_fpmath_mode(fp32_math_mode_);
      }
      auto src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(), src_format);
      auto diff_dst_md =
          dnnl::memory::desc(diff_dst_dims, OneDnnType<T>(), diff_dst_format);
      auto diff_weight_md = dnnl::memory::desc(
          diff_weight_dims, OneDnnType<T>(), diff_weight_format);
      auto diff_bias_md = dnnl::memory::desc(
          {diff_bias_dims}, OneDnnType<Tgrad>(), dnnl::memory::format_tag::x);
      auto fwd_desc = dnnl::inner_product_forward::desc(
          dnnl::prop_kind::forward, src_md, diff_weight_md, diff_bias_md,
          diff_dst_md);
      auto fwd_pd = dnnl::inner_product_forward::primitive_desc(fwd_desc, attr,
                                                                onednn_engine_);
      auto bwd_desc = dnnl::inner_product_backward_weights::desc(
          src_md, diff_weight_md, diff_bias_md, diff_dst_md);
      auto matmul_bwd_pd = dnnl::inner_product_backward_weights::primitive_desc(
          bwd_desc, attr, onednn_engine_, fwd_pd);
      matmul_bwd_primitive_ =
          dnnl::inner_product_backward_weights(matmul_bwd_pd);
      // Allocate output tensors.
      diff_weight_tf_shape_ =
          transpose_b_ ? TensorShape({channel, k}) : TensorShape({k, channel});
      OP_REQUIRES_OK(context, context->allocate_output(kDiffWeightIndex_,
                                                       diff_weight_tf_shape_,
                                                       &diff_weight_tensor_));
      diff_bias_tf_shape_ = TensorShape({channel});
      OP_REQUIRES_OK(context, context->allocate_output(kDiffBiasIndex_,
                                                       diff_bias_tf_shape_,
                                                       &diff_bias_tensor_));
      // Create memory primitive.
      src_mem_ = CreateDnnlMemory(src_md, onednn_engine_,
                                  GetTensorBuffer<T>(&src_tensor));
      diff_dst_mem_ = CreateDnnlMemory(diff_dst_md, onednn_engine_,
                                       GetTensorBuffer<T>(&diff_dst_tensor));
      diff_bias_mem_ =
          CreateDnnlMemory(diff_bias_md, onednn_engine_,
                           GetTensorBuffer<Tgrad>(diff_bias_tensor_));
      diff_weight_mem_ =
          CreateDnnlMemory(matmul_bwd_pd.diff_weights_desc(), onednn_engine_,
                           GetTensorBuffer<T>(diff_weight_tensor_));
      int64 scratchpad_size =
          matmul_bwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor_));
      scratchpad_mem_ =
          dnnl::memory(matmul_bwd_pd.scratchpad_desc(), onednn_engine_,
                       GetTensorBuffer<T>(&scratchpad_tensor_));
      // Execute.
      fwd_primitive_args_ = {{DNNL_ARG_SRC, src_mem_},
                             {DNNL_ARG_DIFF_DST, diff_dst_mem_},
                             {DNNL_ARG_DIFF_WEIGHTS, diff_weight_mem_},
                             {DNNL_ARG_DIFF_BIAS, diff_bias_mem_},
                             {DNNL_ARG_SCRATCHPAD, scratchpad_mem_}};
      is_init_ = true;
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);
    onednn_engine_ = CreateDnnlEngine<Device>(*context);
    onednn_stream_ = CreateDnnlStream(*context, onednn_engine_);
    InitOrSetMemory(context);
    matmul_bwd_primitive_.execute(onednn_stream_, fwd_primitive_args_);
  }

 protected:
  const int kSrcIndex_ = 0, kDiffDstIndex_ = 1, kDiffWeightIndex_ = 0,
            kDiffBiasIndex_ = 1;
  bool is_init_ = false;
  bool enable_cache_ = false;

 private:
  mutex mul_cache_mu_, mu_compute_;
  std::unordered_map<int, memory> fwd_primitive_args_;  // ?
  dnnl::stream onednn_stream_;
  dnnl::engine onednn_engine_;
  dnnl::inner_product_backward_weights matmul_bwd_primitive_;
  dnnl::memory src_mem_, diff_dst_mem_, diff_bias_mem_, diff_weight_mem_,
      scratchpad_mem_;
  Tensor scratchpad_tensor_;
  Tensor* diff_weight_tensor_ = nullptr;
  Tensor* diff_bias_tensor_ = nullptr;
  TensorShape diff_weight_tf_shape_, diff_bias_tf_shape_;
  std::vector<int64> input_dims_, diff_dst_dims_;
  bool transpose_a_;
  bool transpose_b_;
  std::vector<string> fused_ops_;
  dnnl::fpmath_mode fp32_math_mode_ = dnnl::fpmath_mode::strict;
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_MATMUL_OP_H_
