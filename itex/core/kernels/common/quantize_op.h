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

#ifndef ITEX_CORE_KERNELS_COMMON_QUANTIZE_OP_H_
#define ITEX_CORE_KERNELS_COMMON_QUANTIZE_OP_H_

// This file contains the normal implementation QuantizeV2 op with ""SCALED"
// mode and legacy Intel-TF "MIN_FIRST" mode.

// Actually, Intel-TF MIN_FIRST result is different from public TF. However,
// some INT8 pb are generated with Intel-TF. To be backward compatible with this
// pb, ITEX has this implementation here.

/////////////////////////////////////////////////////////////////////////////
// SCALED MODE
// Formula:
// A_u8 = (int8)(A_f32 * scale)

// MIN_FIRST MODE
// Legacy formula:
// A_u8 = (uint8)(A_f32 * scale + (-min_range) * scale)
// Here we actually uses OneDnn Binary Add op in this kernel. Min_range is
// called shift memory in our implementation. Normal formula: A_u8 =
// (uint8)(A_f32 * scale) + (int32)(-min_range * scale) The different round
// strategy causing slightly different result
/////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/quantization_util.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// TODO(itex): For intel-tf proper, the Quantize and Dequantize op only have
// narrow_range implementation, regardless the `narrow_range` flag is True or
// False. Currently, ITEX also follow such logic.

using dnnl::memory;
using dnnl::primitive;
using dnnl::primitive_attr;
using dnnl::reorder;

namespace itex {
template <typename Device, typename T, typename S = float>
class QuantizeV2Op : public OpKernel {
 public:
  explicit QuantizeV2Op(OpKernelConstruction* context) : OpKernel(context) {
    string mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_string));
    OP_REQUIRES(context,
                (mode_string == "MIN_COMBINED" || mode_string == "MIN_FIRST" ||
                 mode_string == "SCALED"),
                errors::InvalidArgument("Mode string must be 'MIN_COMBINED',"
                                        " 'MIN_FIRST', or 'SCALED', is '" +
                                        mode_string + "'"));
    if (mode_string == "MIN_COMBINED") {
      mode_ = QuantizeMode::MIN_COMBINED;
    } else if (mode_string == "MIN_FIRST") {
      mode_ = QuantizeMode::MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QuantizeMode::SCALED;
    }
    OP_REQUIRES(context,
                (mode_string == "SCALED" || mode_string == "MIN_FIRST"),
                errors::InvalidArgument(
                    "_ITEXQuantizeV2 only supports SCALED and MIN_FIRST MODE"));

    string round_mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("round_mode", &round_mode_string));
    OP_REQUIRES(context,
                (round_mode_string == "HALF_AWAY_FROM_ZERO" ||
                 round_mode_string == "HALF_TO_EVEN"),
                errors::InvalidArgument("Round mode string must be "
                                        "'HALF_AWAY_FROM_ZERO' or "
                                        "'HALF_TO_EVEN', is '" +
                                        round_mode_string + "'"));
    if (round_mode_string == "HALF_AWAY_FROM_ZERO") {
      round_mode_ = QuantizeRoundMode::ROUND_HALF_AWAY_FROM_ZERO;
    } else if (round_mode_string == "HALF_TO_EVEN") {
      OP_REQUIRES(context, mode_string == "SCALED",
                  errors::InvalidArgument("Round mode 'HALF_TO_EVEN' "
                                          "only supported for mode 'SCALED', "
                                          "but mode is '" +
                                          mode_string + "'."));
      round_mode_ = QuantizeRoundMode::ROUND_HALF_TO_EVEN;
    }
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("ensure_minimum_range",
                                             &ensure_minimum_range_));

    if (context->HasAttr("dtype")) {
      OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    } else {
      dtype_ = DT_FLOAT;
    }
  }

  void AdjustInputMinMaxRange(OpKernelContext* context, float input_min_range,
                              float input_max_range, float* adjust_min_range,
                              float* adjust_max_range) {
    OP_REQUIRES(context, (input_max_range >= input_min_range),
                errors::InvalidArgument(
                    "input_max_range must be larger than input_min_range."));

    *adjust_min_range = std::min(0.0f, input_min_range);
    // When the minimum and maximum ranges are too close together, nudge them
    // apart by a small value so that they are slightly different. This helps
    // us avoid creating ill-formed buffers where all quantized values map to
    // the same float number. These kinds of buffers cause problems for
    // downstream ops when they need to do calculations on them.
    // We pick the value by making sure that zero is not more than 100x the
    // overall range from the maximum, so that the value can be easily
    // represented when we promote the quantized value to a higher
    // intermediate bit depth, since that's a common requirement.
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                  fabsf(input_max_range))) *
                          ensure_minimum_range_;
    *adjust_max_range =
        std::max(0.0f, std::max(input_max_range, *adjust_min_range + epsilon));
  }

  void Compute(OpKernelContext* context) override {
    const int kSrcDataIndex = 0;
    const int kSrcMinRangeIndex = 1;
    const int kSrcMaxRangeIndex = 2;
    const int kDstDataIndex = 0;
    const int kDstMinRangeIndex = 1;
    const int kDstMaxRangeIndex = 2;

    const Tensor& src_tensor = context->input(kSrcDataIndex);
    const Tensor& input_min_range = context->input(kSrcMinRangeIndex);
    const Tensor& input_max_range = context->input(kSrcMaxRangeIndex);

    int num_slices = 1;
    if (axis_ > -1) {
      num_slices = input_min_range.NumElements();
    }

    std::vector<float> min_range(num_slices);
    std::vector<float> max_range(num_slices);

    if (num_slices == 1) {
      const float min_range_before_adjust =
          input_min_range.template flat<float>()(0);
      const float max_range_before_adjust =
          input_max_range.template flat<float>()(0);
      AdjustInputMinMaxRange(context, min_range_before_adjust,
                             max_range_before_adjust, &min_range[0],
                             &max_range[0]);
    } else {
      auto min_ranges_before_adjust = input_min_range.template vec<float>();
      auto max_ranges_before_adjust = input_max_range.template vec<float>();
      for (int i = 0; i < num_slices; ++i) {
        AdjustInputMinMaxRange(context, min_ranges_before_adjust(i),
                               max_ranges_before_adjust(i), &min_range[i],
                               &max_range[i]);
      }
    }

    // Calculating scales and zeropoints for quantization.
    std::vector<float> scale_factor(num_slices, 0);
    // Zeropoint not used currently. Because we use legacy MIN_FIRST
    // implemenation.
    std::vector<int32> zero_points(num_slices, 0);

    if (mode_ == QuantizeMode::SCALED) {
      GetScaleAndZeropointAndAlignMinMax<T>(
          min_range.data(), max_range.data(), mode_, QuantDequantFlag::Quantize,
          num_slices, scale_factor.data(), zero_points.data());

    } else if (mode_ == QuantizeMode::MIN_FIRST) {
      // Estimate scale for qunatization
      const int number_of_bits = sizeof(T) * 8;
      const int64 number_of_steps = static_cast<int64>(1) << number_of_bits;

      for (int i = 0; i < num_slices; ++i) {
        scale_factor[i] =
            (number_of_steps - 1.0) / (max_range[i] - min_range[i]);
      }

    } else {
      ITEX_LOG(FATAL) << "Unsupported quantize mode";
    }

    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      // Get src_md
      TensorShape src_tf_shape = src_tensor.shape();
      memory::dims src_dims = TFShapeToOneDnnDims(src_tensor.shape());
      memory::desc src_md = CreatePlainMemDescWithFormatTag<S>(src_dims);

      // Get dst_md.
      memory::dims dst_dims = src_dims;
      memory::desc dst_md = CreatePlainMemDescWithFormatTag<T>(dst_dims);

      // Set the scale factor for quantize
      dnnl::primitive_attr post_ops_attr;
      // Base class pointers are used here, since we may use either Reorder or
      // Binary primitive.
      std::unique_ptr<dnnl::primitive_desc_base> fwd_pd;
      std::unique_ptr<dnnl::primitive> fwd_primitive;

      // Here is a bit tricky. Min range is actually always passed with float
      // datatype. However, intel-tf cast its datatype to input data tensor
      // datatype. Here we just follow the implementation in Intel-TF.
      std::vector<S> shift_vec(min_range.size());  // cpu shift vector
      Tensor shift_device_tensor;                  // gpu shift vector
      void* shift_data = nullptr;                  // unified shift data
      std::transform(min_range.begin(), min_range.end(), shift_vec.begin(),
                     [](float v) -> S { return static_cast<S>(-v); });

      if (mode_ == QuantizeMode::SCALED) {
        if (num_slices == 1) {
          post_ops_attr.set_output_scales(0, scale_factor);
          // For MIN_FIRST, we use legacy implementation
          // if (mode_ == QuantizeMode::MIN_FIRST) {
          //   post_ops_attr.set_zero_points(DNNL_ARG_DST, 0, zero_points);
          // }
        } else {
          int mask = static_cast<int>(std::pow(2, axis_));
          post_ops_attr.set_output_scales(mask, scale_factor);
          // if (mode_ == QuantizeMode::MIN_FIRST) {
          //   post_ops_attr.set_zero_points(DNNL_ARG_DST, mask, zero_points);
          // }
        }
        // Create Reorder primitive
        std::unique_ptr<dnnl::reorder::primitive_desc> reorder_pd =
            std::make_unique<dnnl::reorder::primitive_desc>(
                onednn_engine, src_md, onednn_engine, dst_md, post_ops_attr);
        fwd_primitive = std::make_unique<dnnl::reorder>(*reorder_pd);
        fwd_pd = std::move(reorder_pd);
      } else if (mode_ == QuantizeMode::MIN_FIRST) {
        if (num_slices == 1) {
          post_ops_attr.set_scales(DNNL_ARG_SRC_0, 0, scale_factor);
          post_ops_attr.set_scales(DNNL_ARG_SRC_1, 0, scale_factor);
        } else {
          int mask = static_cast<int>(std::pow(2, axis_));
          post_ops_attr.set_scales(DNNL_ARG_SRC_0, mask, scale_factor);
          post_ops_attr.set_scales(DNNL_ARG_SRC_1, mask, scale_factor);
        }

#ifdef INTEL_CPU_ONLY
        shift_data = shift_vec.data();
#else
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<S>::v(),
                                                       src_tf_shape,
                                                       &shift_device_tensor));
        auto* stream = context->GetDeviceStream();
        DeviceMemcpy<Device>(
            const_cast<char*>(shift_device_tensor.tensor_data().data()),
            shift_vec.data(), shift_vec.size() * sizeof(S), stream);
        shift_data = GetTensorBuffer<S>(&shift_device_tensor);
#endif

        memory::dims shift_dims(src_tf_shape.dims(), 1);
        memory::desc shift_md = CreatePlainMemDescWithFormatTag<S>(shift_dims);
        // Create Binary primitive
        auto fwd_desc = dnnl::binary::desc(dnnl::algorithm::binary_add, src_md,
                                           shift_md, dst_md);
        std::unique_ptr<dnnl::binary::primitive_desc> binary_pd =
            std::make_unique<dnnl::binary::primitive_desc>(
                fwd_desc, post_ops_attr, onednn_engine);
        fwd_primitive = std::make_unique<dnnl::binary>(*binary_pd);
        fwd_pd = std::move(binary_pd);
      }

      TensorShape dst_tf_shape;
      dst_tf_shape = OneDnnDimsToTFShape(dst_dims);

      // Allocate output's data tensor and meta tensor
      Tensor* dst_tensor = nullptr;

      OP_REQUIRES_OK(context, context->allocate_output(
                                  kDstDataIndex, dst_tf_shape, &dst_tensor));

      const TensorShape& minmax_shape =
          context->input(kSrcMinRangeIndex).shape();

      // Allocate output_min's data tensor and meta tensor
      TensorShape output_min_tf_shape;
      if (num_slices == 1) {
        output_min_tf_shape = {};
      } else {
        output_min_tf_shape = minmax_shape;
      }

      Tensor* output_min_tensor = nullptr;

      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {kSrcMinRangeIndex}, kDstMinRangeIndex,
                                  output_min_tf_shape, &output_min_tensor));

      // Allocate output_max's data tensor and meta tensor
      TensorShape output_max_tf_shape;
      if (num_slices == 1) {
        output_max_tf_shape = {};
      } else {
        output_max_tf_shape = minmax_shape;
      }
      Tensor* output_max_tensor = nullptr;

      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {kSrcMaxRangeIndex}, kDstMaxRangeIndex,
                                  output_max_tf_shape, &output_max_tensor));

      // Create src and dst memory
      auto src_mem = CreateDnnlMemory(fwd_pd->src_desc(), onednn_engine,
                                      GetTensorBuffer<S>(&src_tensor));
      dnnl::memory shift_mem;
      if (mode_ == QuantizeMode::MIN_FIRST) {
        shift_mem =
            CreateDnnlMemory(fwd_pd->src_desc(1), onednn_engine, shift_data);
      }
      auto dst_mem = CreateDnnlMemory(fwd_pd->dst_desc(), onednn_engine,
                                      GetTensorBuffer<T>(dst_tensor));

      // Execute Reorder primitive
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> fwd_primitive_args;
      if (mode_ == QuantizeMode::SCALED) {
        fwd_primitive_args = {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}};

      } else if (mode_ == QuantizeMode::MIN_FIRST) {
        fwd_primitive_args = {{DNNL_ARG_SRC_0, src_mem},
                              {DNNL_ARG_SRC_1, shift_mem},
                              {DNNL_ARG_DST, dst_mem}};
      }

      fwd_primitive->execute(onednn_stream, fwd_primitive_args);

      // Set data for output_min and output_max tensor
      if (std::is_same<T, quint8>::value && mode_ == QuantizeMode::SCALED) {
        // Align with Intel-TF implmentation
        for (int i = 0; i < num_slices; ++i) {
          output_min_tensor->flat<float>()(i) = 0;
          output_max_tensor->flat<float>()(i) = max_range[i];
        }
      } else {
        for (int i = 0; i < num_slices; ++i) {
          output_min_tensor->flat<float>()(i) = min_range[i];
          output_max_tensor->flat<float>()(i) = max_range[i];
        }
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  float ensure_minimum_range_;
  QuantizeMode mode_;
  QuantizeRoundMode round_mode_;
  int axis_;
  bool narrow_range_;
  DataType dtype_;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_QUANTIZE_OP_H_
