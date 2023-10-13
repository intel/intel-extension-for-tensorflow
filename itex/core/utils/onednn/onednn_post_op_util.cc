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

#include "itex/core/utils/onednn/onednn_post_op_util.h"

namespace itex {

using memory = dnnl::memory;

const std::vector<PostOpInfo>& PostOpUtil::GetAllPostOpInfo() {
  // Here's the const parameters for oneDNN post op, consider the simplest
  // expression:
  //   (alpha * x + beta) * scale
  // Set `alpha` = 1, `beta` = 0 by default. `scale` will be passed as runtime
  // value.
  const float kAlphaZero = 0;
  const float kAlphaOne = 1;
  const float kBetaZero = 0;
  const float kBetaSix = 6;

  // TODO(itex): Try map container to replace vector here.
  static std::vector<PostOpInfo> info_vec = {
      /* Kind: sum */
      {"Add", kind::sum, algorithm::undef, kAlphaOne, kBetaZero},

      /* Kind: eltwise */
      {"Elu", kind::eltwise, algorithm::eltwise_elu, kAlphaOne, kBetaZero},
      // Here `Gelu` is a placeholder for activation check, it will be
      // converted to `"GeluExact` or `"GeluApproximate` after remapper.
      {"Gelu", kind::eltwise, algorithm::undef, kAlphaOne, kBetaZero},
      {"ITEXGelu", kind::eltwise, algorithm::undef, kAlphaOne, kBetaZero},
      {"GeluExact", kind::eltwise, algorithm::eltwise_gelu_erf, kAlphaZero,
       kBetaZero},
      {"GeluApproximate", kind::eltwise, algorithm::eltwise_gelu_tanh,
       kAlphaZero, kBetaZero},
      {"HardSwish", kind::eltwise, algorithm::eltwise_hardswish, kAlphaZero,
       kBetaZero},
      {"LeakyRelu", kind::eltwise, algorithm::eltwise_relu, kAlphaZero,
       kBetaZero},
      {"Linear", kind::eltwise, algorithm::eltwise_linear, kAlphaOne,
       kBetaZero},
      {"_ITEXMish", kind::eltwise, algorithm::eltwise_mish, kAlphaZero,
       kBetaZero},
      {"Relu", kind::eltwise, algorithm::eltwise_relu, kAlphaZero, kBetaZero},
      {"Relu6", kind::eltwise, algorithm::eltwise_clip_v2, kAlphaZero,
       kBetaSix},
      {"Sigmoid", kind::eltwise, algorithm::eltwise_logistic, kAlphaOne,
       kBetaZero},
      {"_ITEXSwish", kind::eltwise, algorithm::eltwise_swish, kAlphaOne,
       kBetaZero},
      {"Tanh", kind::eltwise, algorithm::eltwise_tanh, kAlphaZero, kBetaZero},

      /* Kind: binary */
      {"BinaryAdd", kind::binary, algorithm::binary_add, kAlphaOne, kBetaZero},
      {"BinaryMul", kind::binary, algorithm::binary_mul, kAlphaOne, kBetaZero},
  };

  return info_vec;
}

bool PostOpUtil::AddOps(const std::vector<string>& fused_ops) {
  for (string name : fused_ops) {
    const PostOpInfo* info = GetPostOpInfoByName(name);
    if (info != nullptr) {
      kind op_kind = info->kind;
      // Default `scale` is 1 if no runtime value is passed.
      const float scale_default = 1;

      // Record post op info to internal oneDNN struct, it will be used for
      // creating pritimive desc.
      if (op_kind == kind::eltwise) {
        this->has_activation_ = true;
        if (name == "LeakyRelu") this->has_leaky_relu_ = true;
        if (name == "Linear") this->has_linear_ = true;
        postop_scale_list_.push_back(std::make_pair(name, scale_default));
      } else if (op_kind == kind::sum) {
        this->has_add_ = true;
        postop_scale_list_.push_back(std::make_pair(name, scale_default));
      } else if (op_kind == kind::binary) {
        this->binary_num_++;
        // TODO(itex): Scale for binary is useless now, but it can be supported
        //             in future once oneDNN supports it.
        postop_scale_list_.push_back(std::make_pair(name, scale_default));
      } else {
        // TODO(itex): Support `depthwise` in future.
        ITEX_VLOG(3) << "PostOpUtil: unsupported post op fusion: " << name;

        return false;
      }
    } else {
      // Handle special case `BiasAdd`, it will be fused in primitive directly.
      // Simply record status of `BiasAdd` instead of putting it to table.
      if (name == "BiasAdd") {
        this->has_bias_ = true;
      } else if (name == "FusedBatchNorm") {
        // BatchNorm will also be fused in primitive construction.
        // Just record it.
        this->has_bn_ = true;
      } else if (name == "Quantized" || name == "Requantize" ||
                 name == "Dequantize") {
        // Handle Quantized kernel.
        this->has_output_scales_ = true;

        if (name == "Requantize") {
          this->has_requantize_ = true;
        }
      } else {
        ITEX_VLOG(3) << "PostOpUtil: unsupported post op fusion: " << name;

        return false;
      }
    }
  }

  return true;
}

void PostOpUtil::BuildBNContext(memory::format_tag* data_layout,
                                memory::dims* fuse_bn_dims,
                                dnnl::engine* engine) {
  auto reset_md = [&fuse_bn_dims,
                   &data_layout](std::shared_ptr<dnnl::memory::desc>& ptr) {
    ptr.reset(
        new memory::desc({*fuse_bn_dims}, OneDnnType<float>(), *data_layout));
  };
  reset_md(bn_context_.bn_scale_md);
  reset_md(bn_context_.bn_mean_md);
  reset_md(bn_context_.bn_rsqrt_md);
  reset_md(bn_context_.bn_offset_md);

  auto reset_mem = [&engine](std::shared_ptr<dnnl::memory>& mem_ptr,
                             std::shared_ptr<dnnl::memory::desc>& md_ptr) {
    mem_ptr.reset(new memory(*md_ptr, *engine, nullptr));
  };
  reset_mem(bn_context_.bn_scale_mem, bn_context_.bn_scale_md);
  reset_mem(bn_context_.bn_mean_mem, bn_context_.bn_mean_md);
  reset_mem(bn_context_.bn_offset_mem, bn_context_.bn_offset_md);
  reset_mem(bn_context_.bn_rsqrt_mem, bn_context_.bn_rsqrt_md);
}

void PostOpUtil::SetBNMemory(const Tensor& bn_scale_tensor,
                             const Tensor& bn_mean_tensor,
                             const Tensor& bn_offset_tensor,
                             const Tensor& bn_rsqrt_tensor) {
  bn_context_.bn_scale_mem->set_data_handle(
      GetTensorBuffer<float>(&bn_scale_tensor));
  bn_context_.bn_mean_mem->set_data_handle(
      GetTensorBuffer<float>(&bn_mean_tensor));
  bn_context_.bn_rsqrt_mem->set_data_handle(
      GetTensorBuffer<float>(&bn_rsqrt_tensor));
  bn_context_.bn_offset_mem->set_data_handle(
      GetTensorBuffer<float>(&bn_offset_tensor));
}

void PostOpUtil::AddBNPrimArgs(
    std::unordered_map<int, memory>* fwd_primitives_args_) {
  fwd_primitives_args_->insert(
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
       *bn_context_.bn_mean_mem});
  fwd_primitives_args_->insert(
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
       *bn_context_.bn_rsqrt_mem});
  fwd_primitives_args_->insert(
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
       *bn_context_.bn_scale_mem});
  fwd_primitives_args_->insert(
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(3) | DNNL_ARG_SRC_1,
       *bn_context_.bn_offset_mem});
}

void PostOpUtil::SetLeakyReluAlpha(float alpha) {
  ITEX_CHECK(this->has_leaky_relu_)
      << "PostOpUtil: can't find LeakyRelu when set alpha";
  this->leaky_relu_alpha_ = alpha;
}

void PostOpUtil::SetLinearAlphaBeta(float alpha, float beta) {
  ITEX_CHECK(this->has_linear_)
      << "PostOpUtil: can't find Linear when set alpha/beta";
  this->linear_alpha_ = alpha;
  this->linear_beta_ = beta;
}

void PostOpUtil::SetPostOpScale(const absl::string_view name, float scale) {
  bool is_find = false;
  for (auto& postop_data : postop_scale_list_) {
    const absl::string_view postop_name = postop_data.first;
    if (name == postop_name) {
      postop_data.second = scale;
      is_find = true;
      break;
    }
  }
  ITEX_CHECK(is_find) << "Not find post op: " << name;
}

void PostOpUtil::SetOutputScale(const std::vector<float>& scales) {
  if (scales.size() > 1) {
    int mask = 1;
    output_scale_param_.mask = mask;
    output_scale_param_.scales = scales;
    this->has_output_scales_ = true;
  } else if (scales.size() == 1) {
    int mask = 0;
    // TODO(ITEX): Do not use output_scale_param_ = {mask, scales}. It will
    // cause error in Bert weight sharing case, where multiple threads excute
    // the same kernel. Need further investigate.
    output_scale_param_.mask = mask;
    output_scale_param_.scales = scales;
    this->has_output_scales_ = true;
  } else {
    ITEX_CHECK(scales.size() >= 1) << "invalid output scales";
  }
}

bool PostOpUtil::IsSupportedActivation(const absl::string_view op_name) {
  const std::vector<PostOpInfo>& info_vec = PostOpUtil::GetAllPostOpInfo();
  for (PostOpInfo info : info_vec) {
    if (info.name == op_name && info.kind == kind::eltwise) return true;
  }

  return false;
}

void PostOpUtil::SetPostOp(dnnl::post_ops* post_ops,
                           const std::vector<memory::desc>& md_list) {
  auto it = md_list.begin();
  for (const auto& postop_data : postop_scale_list_) {
    const absl::string_view name = postop_data.first;
    float scale = postop_data.second;

    const PostOpInfo* info = GetPostOpInfoByName(name);
    ITEX_CHECK(info != nullptr);

    kind op_kind = info->kind;
    if (op_kind == kind::eltwise) {
      float alpha = info->alpha;
      float beta = info->beta;
      if (this->has_leaky_relu_) {
        alpha = this->leaky_relu_alpha_;
        ITEX_CHECK(!std::isnan(alpha))
            << "PostOpUtil: LeakyRelu alpha is never set";
      }
      if (this->has_linear_) {
        alpha = this->linear_alpha_;
        ITEX_CHECK(!std::isnan(alpha))
            << "PostOpUtil: Linear alpha is never set";
        beta = this->linear_beta_;
        ITEX_CHECK(!std::isnan(beta)) << "PostOpUtil: Linear beta is never set";
      }
      post_ops->append_eltwise(info->alg, alpha, beta);
    } else if (op_kind == kind::sum) {
      post_ops->append_sum(scale);
    } else if (op_kind == kind::binary) {
      post_ops->append_binary(info->alg, *it);
      ++it;
    } else {
      // TODO(itex): Support `depthwise` and in future.
      ITEX_LOG(FATAL) << "PostOpUtil: unsupported post op fusion: " << name;
    }
  }
}

void PostOpUtil::SetPostOpAttr(dnnl::primitive_attr* attr,
                               const std::vector<memory::desc>& md_list) {
  ITEX_CHECK(md_list.size() == this->binary_num_)
      << "PostOpUtil: missing binary input md, required " << this->binary_num_
      << ", but got " << md_list.size();
  ITEX_DCHECK(attr);

  if (postop_scale_list_.size() != 0) {
    dnnl::post_ops post_ops = dnnl::post_ops();
    SetPostOp(&post_ops, md_list);
    attr->set_post_ops(post_ops);
  }

  if (has_output_scales_) {
    if (output_scale_param_.scales.size()) {
      // if use DNNL_ARG_DST, dst scales should be 1/scale
      attr->set_scales_mask(DNNL_ARG_WEIGHTS, output_scale_param_.mask);
    }
  }
}

// Since batchnorm do the following for each input x:
// scale * (x - mean) / sqrt(\sigma + \epsilon) + offset
// BatchNorm can be decomposed into the following post ops:
// 1. A sub op `x - mean`
// 2. A mul op `1 / sqrt(\sigma + \epsilon) * x`
// 3. A mul op `scale * x`,
// 4. An add op `x + offset`
// where x means the output of previous step.
void PostOpUtil::SetBNPostOpAttr(dnnl::primitive_attr* attr,
                                 const std::vector<memory::desc>& md_list) {
  ITEX_DCHECK(attr);
  dnnl::post_ops post_ops = dnnl::post_ops();
  post_ops.append_binary(algorithm::binary_sub, *bn_context_.bn_mean_md);
  post_ops.append_binary(algorithm::binary_mul, *bn_context_.bn_rsqrt_md);
  post_ops.append_binary(algorithm::binary_mul, *bn_context_.bn_scale_md);
  post_ops.append_binary(algorithm::binary_add, *bn_context_.bn_offset_md);
  if (postop_scale_list_.size() != 0) {
    SetPostOp(&post_ops, md_list);
  }
  attr->set_post_ops(post_ops);
  if (has_output_scales_) {
    if (output_scale_param_.scales.size()) {
      attr->set_scales_mask(DNNL_ARG_WEIGHTS, output_scale_param_.mask);
    }
  }
}

const PostOpInfo* PostOpUtil::GetPostOpInfoByName(
    const absl::string_view op_name) {
  const std::vector<PostOpInfo>& info_vec = PostOpUtil::GetAllPostOpInfo();
  std::vector<PostOpInfo>::const_iterator iter;

  for (iter = info_vec.begin(); iter != info_vec.end(); ++iter) {
    if ((*iter).name == op_name) {
      return &(*iter);
    }
  }

  return nullptr;
}

}  // namespace itex
