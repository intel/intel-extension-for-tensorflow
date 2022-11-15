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

using kind = dnnl::primitive::kind;
using algorithm = dnnl::algorithm;

const std::vector<PostOpInfo>& PostOpUtil::GetAllPostOpInfo() {
  // Here's the const parameters for oneDNN post op, consider the simplest
  // expression:
  //   (alpha * x + beta) * scale
  // Set alpha = 1, beta = 0 and scale = 1 by default.
  const float kAlphaZero = 0;
  const float kAlphaOne = 1;
  const float kBetaZero = 0;
  const float kBetaSix = 6;
  const float kScaleOne = 1;

  // TODO(itex): Try map container to replace vector here.
  static std::vector<PostOpInfo> info_vec = {
      /* Kind: sum */
      {"Add", kind::sum, algorithm::undef, kAlphaOne, kBetaZero, kScaleOne},

      /* Kind: eltwise */
      {"Elu", kind::eltwise, algorithm::eltwise_elu, kAlphaOne, kBetaZero,
       kScaleOne},
      // Here `Gelu` is a placeholder for activation check, it will be
      // converted to `"GeluExact` or `"GeluApproximate` after remapper.
      {"Gelu", kind::eltwise, algorithm::undef, kAlphaOne, kBetaZero,
       kScaleOne},
      {"GeluExact", kind::eltwise, algorithm::eltwise_gelu_erf, kAlphaZero,
       kBetaZero, kScaleOne},
      {"GeluApproximate", kind::eltwise, algorithm::eltwise_gelu_tanh,
       kAlphaZero, kBetaZero, kScaleOne},
      {"HardSwish", kind::eltwise, algorithm::eltwise_hardswish, kAlphaZero,
       kBetaZero, kScaleOne},
      {"LeakyRelu", kind::eltwise, algorithm::eltwise_relu, kAlphaZero,
       kBetaZero, kScaleOne},
      {"Mish", kind::eltwise, algorithm::eltwise_mish, kAlphaZero, kBetaZero,
       kScaleOne},
      {"Relu", kind::eltwise, algorithm::eltwise_relu, kAlphaZero, kBetaZero,
       kScaleOne},
      {"Relu6", kind::eltwise, algorithm::eltwise_clip_v2, kAlphaZero, kBetaSix,
       kScaleOne},
      {"Sigmoid", kind::eltwise, algorithm::eltwise_logistic, kAlphaOne,
       kBetaZero, kScaleOne},
      {"Swish", kind::eltwise, algorithm::eltwise_swish, kAlphaOne, kBetaZero,
       kScaleOne},
      {"Tanh", kind::eltwise, algorithm::eltwise_tanh, kAlphaZero, kBetaZero,
       kScaleOne},

      /* Kind: binary */
      {"BinaryAdd", kind::binary, algorithm::binary_add, kAlphaOne, kBetaZero,
       kScaleOne},
  };

  return info_vec;
}

bool PostOpUtil::AddOps(const std::vector<string>& fused_ops) {
  for (string name : fused_ops) {
    const PostOpInfo* info = GetPostOpInfoByName(name);
    if (info != nullptr) {
      kind op_kind = info->kind;
      const float scale_default = info->scale;

      // Record post op info to internal oneDNN struct, it will be used for
      // creating pritimive desc.
      if (op_kind == kind::eltwise) {
        this->has_activation_ = true;
        if (name == "LeakyRelu") this->has_leaky_relu_ = true;
        postop_scale_list_.push_back(std::make_pair(name, scale_default));
      } else if (op_kind == kind::sum) {
        this->has_add_ = true;
        postop_scale_list_.push_back(std::make_pair(name, scale_default));
      } else if (op_kind == kind::binary) {
        this->has_binary_ = true;
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
      } else if (name == "Mul" || name == "Quantized" || name == "Requantize" ||
                 name == "Dequantize") {
        // Handle special case BatchMatMul + Mul fusion or Quantized kernel
        this->has_output_scales_ = true;
        // TODO(itex): Remove "has_mul_", once we use binary::mul instead of
        // output_scale to implement Mul fusion.
        if (name == "Mul") {
          this->has_mul_ = true;
        }

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

void PostOpUtil::SetBinaryInput(const dnnl::memory::desc& binary_md) {
  ITEX_CHECK(this->has_binary_)
      << "PostOpUtil: can't find binary op when set input md";
  // TODO(ITEX): Do not .clear() and .push_back(). It is thread unsafe, and it
  // will cause error in Bert weight sharing case, where multiple threads excute
  // the same kernel.
  this->binary_md_list_ = {binary_md};
}

void PostOpUtil::SetLeakyReluAlpha(float alpha) {
  ITEX_CHECK(this->has_leaky_relu_)
      << "PostOpUtil: can't find LeakyRelu when set alpha";
  this->leaky_relu_alpha_ = alpha;
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
    int mask = 2;
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

void PostOpUtil::SetPostOp(dnnl::post_ops* post_ops) {
  for (const auto& postop_data : postop_scale_list_) {
    const absl::string_view name = postop_data.first;
    float scale = postop_data.second;

    const PostOpInfo* info = GetPostOpInfoByName(name);
    ITEX_CHECK(info != nullptr);

    kind op_kind = info->kind;
    if (op_kind == kind::eltwise) {
      float alpha = info->alpha;
      if (this->has_leaky_relu_) {
        alpha = this->leaky_relu_alpha_;
        ITEX_CHECK(!std::isnan(alpha))
            << "PostOpUtil: LeakyRelu alpha is never set";
      }
      post_ops->append_eltwise(scale, info->alg, alpha, info->beta);
    } else if (op_kind == kind::sum) {
      post_ops->append_sum(scale);
    } else if (op_kind == kind::binary) {
      ITEX_CHECK(this->binary_md_list_.size() == 1)
          << "PostOpUtil: binary input md is set wrongly";
      post_ops->append_binary(info->alg, this->binary_md_list_.front());
    } else {
      // TODO(itex): Support `depthwise` and in future.
      ITEX_LOG(FATAL) << "PostOpUtil: unsupported post op fusion: " << name;
    }
  }
}

void PostOpUtil::SetPostOpAttr(dnnl::primitive_attr* attr) {
  ITEX_DCHECK(attr);
  if (postop_scale_list_.size() != 0) {
    dnnl::post_ops post_ops = dnnl::post_ops();
    SetPostOp(&post_ops);
    attr->set_post_ops(post_ops);
  }
  if (has_output_scales_) {
    if (output_scale_param_.scales.size()) {
      attr->set_output_scales(output_scale_param_.mask,
                              output_scale_param_.scales);
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
