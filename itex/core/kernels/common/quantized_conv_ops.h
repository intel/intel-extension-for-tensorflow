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

#ifndef ITEX_CORE_KERNELS_COMMON_QUANTIZED_CONV_OPS_H_
#define ITEX_CORE_KERNELS_COMMON_QUANTIZED_CONV_OPS_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/kernels/common/cast_op.h"
#include "itex/core/kernels/common/conv_ops.h"
#include "itex/core/kernels/common/host_data_cache.h"
#include "itex/core/kernels/onednn/block/quantized_ops.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;

namespace itex {

// TODO(intel-tf) Remove this once old API of quantized ops is abandoned
namespace quantized_fusions {
const char* const none[] = {};
const char* const bias[] = {"BiasAdd"};
const char* const relu[] = {"Relu"};
const char* const requantize[] = {"Requantize"};
const char* const bias_relu[] = {"BiasAdd", "Relu"};
const char* const bias_requantize[] = {"BiasAdd", "Requantize"};
const char* const relu_requantize[] = {"Relu", "Requantize"};
const char* const bias_relu_requantize[] = {"BiasAdd", "Relu", "Requantize"};
const char* const bias_sum_relu[] = {"BiasAdd", "Sum", "Relu"};
const char* const bias_sum_relu_requantize[] = {"BiasAdd", "Sum", "Relu",
                                                "Requantize"};
}  // namespace quantized_fusions

template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Tsummand, bool is_depthwise,
          const char* const legacy_fused_ops[], int num_fused_ops>
class LegacyQuantizedConvOpBase
    // Currently, Tfilter only supports qint8
    : public ConvOpBase<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
                        is_depthwise> {
 public:
  explicit LegacyQuantizedConvOpBase(OpKernelConstruction* context)
      : ConvOpBase<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
                   is_depthwise>(context) {
    std::vector<string> fused_ops;
    fused_ops.push_back("Quantized");
    std::vector<std::vector<string>> supported_fusions = {
        {"BiasAdd"},
        {"Relu"},
        {"Requantize"},
        {"Dequantize"},
        {"BiasAdd", "Relu"},
        {"BiasAdd", "LeakyRelu"},
        {"BiasAdd", "Elu"},
        {"BiasAdd", "_FusedSwish"},
        {"BiasAdd", "_FusedHardSwish"},
        {"BiasAdd", "Sigmoid"},
        {"BiasAdd", "Sum", "LeakyRelu"},
        {"BiasAdd", "Requantize"},
        {"BiasAdd", "Dequantize"},
        {"Relu", "Requantize"},
        {"BiasAdd", "Relu", "Requantize"},
        {"BiasAdd", "Sum", "LeakyRelu", "Requantize"},
        {"BiasAdd", "Relu", "Sum"},
        {"BiasAdd", "Relu", "Sum", "Requantize"},
        {"BiasAdd", "Sum"},
        {"BiasAdd", "Sum", "Requantize"},
        {"BiasAdd", "LeakyRelu", "Requantize"},
        {"BiasAdd", "LeakyRelu", "Sum"},
        {"BiasAdd", "LeakyRelu", "Sum", "Requantize"},
        {"BiasAdd", "Elu", "Requantize"},
        {"BiasAdd", "_FusedSwish", "Requantize"},
        {"BiasAdd", "_FusedHardSwish", "Requantize"},
        {"BiasAdd", "Sigmoid", "Requantize"},
        {"BiasAdd", "Sum", "Relu"},
        {"BiasAdd", "Sum", "Relu", "Requantize"}};

    std::vector<string> fused_ops_attr;
    // Old quantized ops don't have fused_ops attribute
    if (context->HasAttr("fused_ops")) {
      OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops_attr));
    }

    // Number of fused ops for new API is determined by size of fused_ops_attr.
    // For old API, num_fused_ops is used to determine number of fused ops.
    // TODO(itex): num_fused_ops and legacy_fused_ops should go away once
    // old API is abandoned.
    OP_REQUIRES(context, !(fused_ops_attr.size() > 0 && num_fused_ops > 0),
                errors::InvalidArgument(
                    "QuantizedConv fused ops should be only availabe through "
                    "either new API or old API, got both."));

    // Code to deal with some legacy int8 pb
    if (context->HasAttr("padding_list")) {
      OP_REQUIRES_OK(
          context, context->GetAttr("padding_list", &this->explicit_paddings_));
    }

    if (fused_ops_attr.size() > 0) {
      for (int i = 0; i < fused_ops_attr.size(); ++i) {
        if (fused_ops_attr[i] == "Requantize" ||
            fused_ops_attr[i] == "Dequantize") {
          continue;
        } else if (fused_ops_attr[i] == "Sum") {
          fused_ops_.push_back("Add");
        } else if (fused_ops_attr[i] == "_FusedSwish") {
          fused_ops_.push_back("Swish");
        } else if (fused_ops_attr[i] == "_FusedHardSwish") {
          fused_ops_.push_back("HardSwish");
        } else {
          fused_ops_.push_back(fused_ops_attr[i]);
        }
      }
    } else if (num_fused_ops > 0) {
      for (int i = 0; i < num_fused_ops; ++i) {
        if (strcmp(legacy_fused_ops[i], "Requantize") == 0 ||
            strcmp(legacy_fused_ops[i], "Dequantize") == 0) {
          continue;
        } else if (strcmp(legacy_fused_ops[i], "Sum") == 0) {
          fused_ops_.push_back("Add");
        } else {
          fused_ops_.push_back(legacy_fused_ops[i]);
        }
      }
    }

    if (fused_ops_attr.size() > 0) {
      fused_ops_intel_tf_ = fused_ops_attr;
    } else if (num_fused_ops > 0) {
      for (int i = 0; i < num_fused_ops; ++i) {
        fused_ops_intel_tf_.push_back(legacy_fused_ops[i]);
      }
    }

    fuse_bias_ = std::find(fused_ops_.begin(), fused_ops_.end(), "BiasAdd") !=
                 fused_ops_.end();
    DataType bias_dt, summand_dt, out_dt;
    if (fuse_bias_) {
      // TF raw op Conv INT8 old API, doesn't have bias_const attribute
      if (context->HasAttr("is_bias_const")) {
        OP_REQUIRES_OK(context,
                       context->GetAttr("is_bias_const", &is_bias_const_));
      } else {
        is_bias_const_ = true;
      }
      if (context->HasAttr("Tbias")) {
        OP_REQUIRES_OK(context, context->GetAttr("Tbias", &bias_dt));
      }
    }

    fuse_sum_ = std::find(fused_ops_.begin(), fused_ops_.end(), "Add") !=
                fused_ops_.end();

    fuse_requantize_ =
        std::find(fused_ops_intel_tf_.begin(), fused_ops_intel_tf_.end(),
                  "Requantize") != fused_ops_intel_tf_.end();
    fuse_dequantize_ =
        std::find(fused_ops_intel_tf_.begin(), fused_ops_intel_tf_.end(),
                  "Dequantize") != fused_ops_intel_tf_.end();

    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_dt));
    if (fuse_requantize_) {
      OP_REQUIRES(context, out_dt == DT_QINT8 || out_dt == DT_QUINT8,
                  errors::InvalidArgument("QuantizedConv: unsupported output "
                                          "type when Requantize is fused."));
    }
    if (fuse_dequantize_) {
      OP_REQUIRES(context, out_dt == DT_FLOAT || out_dt == DT_BFLOAT16,
                  errors::InvalidArgument("QuantizedConv: unsupported output "
                                          "type when Dequantize is fused."));
    }

    if (context->HasAttr("Tsummand")) {
      OP_REQUIRES_OK(context, context->GetAttr("Tsummand", &summand_dt));
      if (!fuse_sum_) {
        OP_REQUIRES(
            context, summand_dt == out_dt,
            errors::InvalidArgument(
                "QuantizedConv: incorrect summand data type. When Sum is not "
                "fused, Tsummand attribute must have same value as out_type."));
      }
      // TODO(intel-tf): Remove this restriction when u8 summand + s8 output
      // is supported.
      if (summand_dt == DT_QUINT8 && out_dt == DT_QINT8) {
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "Current fusion requires summand has same dtype as "
                        "output if output is qint8"));
      }
    }

    if (context->HasAttr("alpha")) {
      OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    }

    this->post_op_util_.AddOps(fused_ops_);

    // Set alpha if get `LeakyRelu` after adding ops.
    if (this->post_op_util_.HasLeakyRelu()) {
      OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
      this->post_op_util_.SetLeakyReluAlpha(alpha_);
    }

    if (num_fused_ops == -1) {
      // If num_fused_ops is -1 then the new API (ops) are being used.
      // Expected inputs order for new API is as follows. {} means optional
      // input needed by certain fusion.
      // (0)  input
      // (1)  filter
      // (2)  {bias}
      // (3)  {summand}
      // (5)  min_input
      // (6)  max_input
      // (7)  min_filter
      // (8)  max_filter
      // (9)  {min_bias}
      // (10) {max_bias}
      // (11) {min_summand}
      // (12) {max_summand}
      // (15) {min_freezed_output}
      // (16) {max_freezed_output}
      int non_minmax_arg_idx_base = 2;
      int minmax_arg_idx_base = 6;
      int bias_idx_offset = fuse_bias_ ? 1 : 0;
      int summand_idx_offset = fuse_sum_ ? 1 : 0;
      // Currently min and max for bias are not expected if bias data type is
      // DT_QINT32.
      int bias_min_max_idx_offset =
          fuse_bias_ && !(bias_dt == DT_FLOAT || bias_dt == DT_QINT32) ? 2 : 0;
      int summand_min_max_idx_offset =
          fuse_sum_ && summand_dt != DT_FLOAT ? 2 : 0;
      // TODO(itex): recheck the calculation of kSummandDataIndex here
      kSrcMinRangeIndex =
          non_minmax_arg_idx_base + bias_idx_offset + summand_idx_offset;
      kSrcMaxRangeIndex = kSrcMinRangeIndex + 1;
      kFilterMinRangeIndex = kSrcMaxRangeIndex + 1;
      kFilterMaxRangeIndex = kFilterMinRangeIndex + 1;
      if (fuse_bias_) {
        kBiasMinRangeIndex =
            minmax_arg_idx_base + bias_idx_offset + summand_idx_offset;
        kBiasMaxRangeIndex = kBiasMinRangeIndex + 1;
      }
      if (fuse_sum_) {
        //    this->set_input_add_idx(non_minmax_arg_idx_base +
        //    bias_idx_offset);
        // TODO(itex): recheck the calculation of kSummandDataIndex here
        kSummandDataIndex = non_minmax_arg_idx_base + bias_idx_offset;
        if (summand_dt == DT_QINT8 || summand_dt == DT_QUINT8) {
          kSummandMinRangeIndex = minmax_arg_idx_base + bias_idx_offset +
                                  summand_idx_offset + bias_min_max_idx_offset;
          kSummandMaxRangeIndex = kSummandMinRangeIndex + 1;
        }
      }

      if (fuse_requantize_) {
        kMinFreezedIndex = minmax_arg_idx_base + bias_idx_offset +
                           summand_idx_offset + bias_min_max_idx_offset +
                           summand_min_max_idx_offset;
        kMaxFreezedIndex = kMinFreezedIndex + 1;
      }
    } else {
      int bias_idx_offset = fuse_bias_ ? 1 : 0;
      kSrcMinRangeIndex = 2 + bias_idx_offset;
      kSrcMaxRangeIndex = 3 + bias_idx_offset;
      kFilterMinRangeIndex = 4 + bias_idx_offset;
      kFilterMaxRangeIndex = 5 + bias_idx_offset;
      if (fuse_requantize_) {
        kMinFreezedIndex = 6 + bias_idx_offset;
        kMaxFreezedIndex = 7 + bias_idx_offset;
      }
      if (fuse_sum_) {
        int min_max_freezed_offset = (std::is_same<Toutput, quint8>::value ||
                                      std::is_same<Toutput, qint8>::value)
                                         ? 2
                                         : 0;
        kSummandDataIndex = 6 + bias_idx_offset + min_max_freezed_offset;
        if (summand_dt == DT_QINT8 || summand_dt == DT_QUINT8) {
          kSummandMinRangeIndex = 9 + bias_idx_offset;
          kSummandMaxRangeIndex = 10 + bias_idx_offset;
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override {
    // Compute int32 output tensor
    ConvOpBase<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
               is_depthwise>::Compute(context);

    const float min_input = context->input(kSrcMinRangeIndex).flat<float>()(0);
    const float max_input = context->input(kSrcMaxRangeIndex).flat<float>()(0);

    AllocateNativeOutputMinMax<Tinput, qint8, Toutput>(
        context, min_input, max_input, kFilterMinRangeIndex,
        kFilterMaxRangeIndex, kMinFreezedIndex, kMaxFreezedIndex,
        kDstMinRangeIndex, kDstMaxRangeIndex);
  }

 protected:
  void ExtendInt8PostOps(OpKernelContext* context) override {
    // When the output type is quint8, the output data is requantized
    // into quint8. A post_op "output_scale" is added to do the conversion.
    // Otherwise the output_scale will be 1.f
    const Tensor& min_filter_vector = context->input(kFilterMinRangeIndex);
    const Tensor& max_filter_vector = context->input(kFilterMaxRangeIndex);
    size_t depth = min_filter_vector.NumElements();

    const float* min_filter = min_filter_vector.flat<float>().data();
    const float* max_filter = max_filter_vector.flat<float>().data();

    std::vector<float> scales(depth, 1.f);

    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value ||
        std::is_same<Toutput, float>::value ||
        std::is_same<Toutput, Eigen::half>::value ||
        std::is_same<Toutput, Eigen::bfloat16>::value) {
      const float min_input =
          context->input(kSrcMinRangeIndex).flat<float>()(0);
      const float max_input =
          context->input(kSrcMaxRangeIndex).flat<float>()(0);

      if (std::is_same<Toutput, float>::value ||
          std::is_same<Toutput, Eigen::half>::value ||
          std::is_same<Toutput, Eigen::bfloat16>::value) {
        const float int_input_limit =
            (std::is_same<Tinput, quint8>::value) ? 255.0 : 127.0;
        const float int_filter_limit = 127.0;
        float src_qscale_f32 =
            std::max(std::abs(min_input), std::abs(max_input)) /
            int_input_limit;
        for (size_t i = 0; i < depth; ++i) {
          float wei_qscale_f32 =
              std::max(std::abs(min_filter[i]), std::abs(max_filter[i])) /
              int_filter_limit;
          scales[i] = src_qscale_f32 * wei_qscale_f32;
        }
      } else {
        // min_freezed_output and max_freezed_output are the actual range
        // for the output.
        const float min_freezed_output =
            context->input(kMinFreezedIndex).flat<float>()(0);
        const float max_freezed_output =
            context->input(kMaxFreezedIndex).flat<float>()(0);

        float int_output_limit =
            std::is_same<Toutput, quint8>::value ? 255.0f : 127.0f;

        float float_input_range =
            std::max(std::abs(min_input), std::abs(max_input));
        float float_output_range = std::max(std::abs(min_freezed_output),
                                            std::abs(max_freezed_output));
        const float int_const_scale_limit =
            (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0
                                                  : 127.0 * 127.0;
        for (size_t i = 0; i < depth; ++i) {
          // For simplicity and symmetry, we set filter range to be outer
          // bounds of min_filter and max_filter.
          float float_filter_range =
              std::max(std::abs(min_filter[i]), std::abs(max_filter[i]));
          scales[i] = int_output_limit * float_input_range *
                      float_filter_range /
                      (int_const_scale_limit * float_output_range);
        }
      }
    }
    this->post_op_util_.SetOutputScale(scales);

    if (fuse_sum_) {
      // Calculate the scale (beta in OneDnn api term) for sum
      float sum_post_op_scale;
      if (!std::is_same<Toutput, qint32>::value) {
        const Tensor& summand = context->input(kSummandDataIndex);
        // TODO(itex): investigate OpKernel::input_type
        DataType summand_type = summand.dtype();
        ITEX_CHECK((summand_type == DT_QINT8) || (summand_type == DT_QUINT8));

        const float min_freezed_output =
            context->input(this->kMinFreezedIndex).template flat<float>()(0);
        const float max_freezed_output =
            context->input(this->kMaxFreezedIndex).template flat<float>()(0);
        const float min_freezed_summand =
            context->input(kSummandMinRangeIndex).flat<float>()(0);
        const float max_freezed_summand =
            context->input(kSummandMaxRangeIndex).flat<float>()(0);

        float scale_output = std::max(std::abs(min_freezed_output),
                                      std::abs(max_freezed_output));
        float scale_summand = std::max(std::abs(min_freezed_summand),
                                       std::abs(max_freezed_summand));
        // if summand_type is also DT_QUINT8 as the scale_output,
        // the scaling factor of 255.0f cancels each other and thus is avoided.
        // If it is not then  it is DT_INT8 and is scaled appropriately.

        if (std::is_same<Toutput, quint8>::value && summand_type == DT_QINT8) {
          sum_post_op_scale = 255.0f * scale_summand / (scale_output * 127.0f);
        } else {
          sum_post_op_scale = scale_summand / scale_output;
        }
      } else {
        sum_post_op_scale = 1.0;
      }

      this->post_op_util_.SetPostOpScale("Add", sum_post_op_scale);
    }

    // TODO(ITEX): move this candidate_element_ops to a public place
    std::vector<string> candidate_element_ops = {
        "Elu", "HardSwish", "LeakyRelu", "Relu", "Sigmoid", "Swish"};

    for (auto element_op : candidate_element_ops) {
      if (std::find(fused_ops_.begin(), fused_ops_.end(), element_op) !=
          fused_ops_.end()) {
        this->post_op_util_.SetPostOpScale(element_op, 1.0);
      }
    }
  }

  Tbias* GetBiasHandle(OpKernelContext* context,
                       const Tensor& bias_tensor) override {
    if (std::is_same<Tbias, qint32>::value) {
      return static_cast<Tbias*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
    }

    const float min_input = context->input(kSrcMinRangeIndex).flat<float>()(0);
    const float max_input = context->input(kSrcMaxRangeIndex).flat<float>()(0);
    const Tensor& min_filter_vector = context->input(kFilterMinRangeIndex);
    const Tensor& max_filter_vector = context->input(kFilterMaxRangeIndex);
    const float* min_filter = min_filter_vector.flat<float>().data();
    const float* max_filter = max_filter_vector.flat<float>().data();

    const float int_const_scale_limit =
        (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
    // Re-scale bias if either of following 2 conditions are met:
    // 1. Bias is not const;
    // 2. Bias is const, but bias cache is empty (first iteration).

    // TODO(itex): avoid to use new memory
    size_t depth = min_filter_vector.NumElements();
    scales_.resize(depth);
    for (size_t i = 0; i < depth; ++i) {
      float tmp_scale =
          int_const_scale_limit /
          (std::max(std::abs(max_input), std::abs(min_input)) *
           std::max(std::abs(max_filter[i]), std::abs(min_filter[i])));
      // TODO(itex): Check whether delete some instuctions about
      // scales_are_valid is correct
      scales_[i] = tmp_scale;
    }
    // TODO(itex): is_bias_const_ is useless, delete it
    if (!is_bias_const_ || bias_cache_manager.IsEmpty()) {
      dnnl::primitive_attr bias_attr;
      if (depth == 1) {
        bias_attr.set_output_scales(0, scales_);
      } else {
        bias_attr.set_output_scales(1, scales_);
      }

      auto bias_md = memory::desc({static_cast<int>(bias_tensor.NumElements())},
                                  OneDnnType<Tbias>(), memory::format_tag::x);
      void* bias_data = static_cast<void*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));

      // TODO(itex): Check whether the bias_md is always equals to
      // conv_pd.bias_desc()
      bias_cache_manager.SetCache(context, bias_md, bias_attr, bias_data,
                                  this->onednn_engine_);
    }
    return bias_cache_manager.GetCache(context);
  }

  void AllocateOutputTensor(
      OpKernelContext* context,
      const dnnl::convolution_forward::primitive_desc& conv_prim_desc,
      const memory::dims& dst_dims_onednn, TensorShape dst_tensor_shape,
      Tensor** dst_tensor, Tensor* dst_tensor_opt) override {
    if (!fuse_sum_) {
      ConvOpBase<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
                 is_depthwise>::AllocateOutputTensor(context, conv_prim_desc,
                                                     dst_dims_onednn,
                                                     dst_tensor_shape,
                                                     dst_tensor,
                                                     dst_tensor_opt);
      return;
    }

    if (!std::is_same<Toutput, qint32>::value) {
      Tensor& summand = const_cast<Tensor&>(context->input(kSummandDataIndex));

      // TODO(itex): We could try to use Tsummand here
      DataType summand_type = summand.dtype();
      ITEX_CHECK((summand_type == DT_QINT8) || (summand_type == DT_QUINT8));

      // TODO(itex): Handle both block and plain layout tensors
      if (std::is_same<Toutput, quint8>::value && summand_type == DT_QINT8) {
        // TODO(itex): TF proper uses bitcastfrom, check whether there is
        // problem here.
        OP_REQUIRES_OK(
            context, summand.BitcastFrom(summand, DT_QUINT8, summand.shape()));
      }

      // Here is workaround to always forward add tensor in conv + bias + add +
      // relu int8 fusion
      // FIXME(itex): Implement code for "inplace_sum = False" and discuss with
      // LPOT about new design.
      // JIRA: https://jira.devtools.intel.com/browse/TFDO-5059
      if (std::is_same<Toutput, qint8>::value &&
          std::is_same<Tsummand, qint8>::value &&
          context->input(kSummandDataIndex).dtype() == DT_QUINT8) {
        // To bypass the INC pb generation bug. INC may wrongly set Tsummand
        // attr qint8 when the actual input is quint8. Intel-TF can avoid the
        // issue by internal type check in forward_input_to_output_with_shape.
        // Since ITEX have to use set_output here, it will always inplace, and
        // cause crash.
        // TODO(itex): Discuss with INC to fix incorrect pb.
        OP_REQUIRES_OK(context,
                       context->allocate_output(this->kDstIndex_,
                                                dst_tensor_shape, dst_tensor));
      } else {
        context->set_output(this->kDstIndex_,
                            context->input(kSummandDataIndex));
      }

      *dst_tensor = context->mutable_output(this->kDstIndex_);
      return;
    }
    // TODO(itex): investigate the influence of additional attr tensor_shape
    ConvOpBase<Device, Tinput, qint8, Tbias, Toutput, Tsummand, false,
               is_depthwise>::AllocateOutputTensor(context, conv_prim_desc,
                                                   dst_dims_onednn,
                                                   dst_tensor_shape, dst_tensor,
                                                   dst_tensor_opt);
    const Tensor& summand = context->input(kSummandDataIndex);
    if (summand.dtype() != DT_FLOAT) {
      ITEX_LOG(FATAL) << "Current fusion requires summand to be float";
    }
    // We need to compute scale for the summand
    const float min_input =
        context->input(this->kSrcMinRangeIndex).template flat<float>()(0);
    const float max_input =
        context->input(this->kSrcMaxRangeIndex).template flat<float>()(0);
    const Tensor& min_filter_vector =
        context->input(this->kFilterMinRangeIndex);
    const Tensor& max_filter_vector =
        context->input(this->kFilterMaxRangeIndex);
    const float* min_filter = min_filter_vector.flat<float>().data();
    const float* max_filter = max_filter_vector.flat<float>().data();

    const float int_const_scale_limit =
        (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
    size_t depth = min_filter_vector.NumElements();
    std::vector<float> scales(depth);
    for (size_t i = 0; i < depth; ++i) {
      // TODO(itex): scale factors for UINT8(inputs) & INT8(weights) are
      // done regularly. A Cleaner design to address all mapping in one
      // function needs to be implemented in future which also supports other
      // quantized type mapping in future.
      scales[i] = int_const_scale_limit /
                  (std::max(std::abs(max_input), std::abs(min_input)) *
                   std::max(std::abs(max_filter[i]), std::abs(min_filter[i])));
    }
    dnnl::primitive_attr reorder_attr;
    if (depth == 1) {
      reorder_attr.set_output_scales(0, scales);
    } else {
      reorder_attr.set_output_scales(2, scales);
    }

    // TODO(itex) Remove this hard code.
    auto summand_md =
        memory::desc(dst_dims_onednn, OneDnnType<Tbias>(),
                     this->is_conv2d_ ? memory::format_tag::nhwc
                                      : memory::format_tag::ndhwc);

    // Reasons for using Tbias: if code here is executed before the requantize
    // op is fused with int8 conv op. At that time, both bias tensor and summand
    // tensor are fp32.
    void* summand_buf =
        static_cast<void*>(const_cast<Tbias*>(summand.flat<Tbias>().data()));
    void* dst_buf = static_cast<void*>((*dst_tensor)->flat<Tsummand>().data());

    memory summand_mem =
        CreateDnnlMemory(summand_md, this->onednn_engine_, summand_buf);
    memory dst_mem = CreateDnnlMemory(conv_prim_desc.dst_desc(),
                                      this->onednn_engine_, dst_buf);

    dnnl::reorder summand_scaled_primitive =
        dnnl::reorder(summand_mem, dst_mem, reorder_attr);
    std::unordered_map<int, dnnl::memory> reorder_args = {
        {DNNL_ARG_SRC, summand_mem}, {DNNL_ARG_DST, dst_mem}};
    auto onednn_stream = CreateDnnlStream(*context, this->onednn_engine_);
    summand_scaled_primitive.execute(onednn_stream, reorder_args);
  }

 protected:
  bool is_bias_const_;
  bool fuse_sum_ = false;
  float alpha_ = 0.0;
  std::vector<string> fused_ops_;
  std::vector<string> fused_ops_intel_tf_;
  std::map<string, int> post_op_to_idx_;
  bool fuse_bias_;
  bool fuse_requantize_, fuse_dequantize_;
  std::shared_ptr<dnnl::memory> summand_;
  std::shared_ptr<dnnl::memory> dst_;

  // input and output tensor index
  int kSrcMinRangeIndex;
  int kSrcMaxRangeIndex;
  int kFilterMinRangeIndex;
  int kFilterMaxRangeIndex;
  int kBiasMinRangeIndex;
  int kBiasMaxRangeIndex;
  int kSummandDataIndex;
  int kSummandMinRangeIndex;
  int kSummandMaxRangeIndex;
  int kMinFreezedIndex;
  int kMaxFreezedIndex;
  const int kDstMinRangeIndex = 1;
  const int kDstMaxRangeIndex = 2;

 private:
  std::vector<float> scales_;
  // Bias cache manager
  BiasCacheManager<Tbias> bias_cache_manager;
#ifdef ITEX_ONEDNN_3_0
  HostDataCache<Device, float> output_scale_cache_;
  HostDataCache<Device, float> bias_scale_cache_;
#endif
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_QUANTIZED_CONV_OPS_H_
