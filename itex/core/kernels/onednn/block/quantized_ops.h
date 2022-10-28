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

#ifndef ITEX_CORE_KERNELS_ONEDNN_BLOCK_QUANTIZED_OPS_H_
#define ITEX_CORE_KERNELS_ONEDNN_BLOCK_QUANTIZED_OPS_H_

#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/plugin_tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
template <typename T>
float OneDnnFloatForOneQuantizedLevel(float range_min, float range_max) {
  int64 highest = static_cast<int64>(Eigen::NumTraits<T>::highest());
  int64 lowest = static_cast<int64>(Eigen::NumTraits<T>::lowest());

  // Adjusting for having a symmetric range.
  // for example: for 8-bit [-127, 127] as opposed to [-128, 127].
  if (lowest < -highest) ++lowest;

  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <typename T1, typename T2, typename T3>
void OneDnnQuantizationRangeForMultiplication(float min_a, float max_a,
                                              float min_b, float max_b,
                                              float* min_c, float* max_c) {
  const float a_float_for_one_quant_level =
      OneDnnFloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      OneDnnFloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int64 c_highest = static_cast<int64>(Eigen::NumTraits<T3>::highest());
  const int64 c_lowest = static_cast<int64>(Eigen::NumTraits<T3>::lowest());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

template <typename T1, typename T2, typename T3>
void OneDnnQuantizationRangeForMultiplication(float min_a, float max_a,
                                              const Tensor& min_b_vector,
                                              const Tensor& max_b_vector,
                                              Tensor** min_c_vector,
                                              Tensor** max_c_vector) {
  ITEX_DCHECK(min_b_vector.NumElements() == (*min_c_vector)->NumElements());
  ITEX_DCHECK(max_b_vector.NumElements() == (*max_c_vector)->NumElements());
  size_t n_channel = min_b_vector.NumElements();
  const int64 c_highest = static_cast<int64>(Eigen::NumTraits<T3>::highest());
  const int64 c_lowest = static_cast<int64>(Eigen::NumTraits<T3>::lowest());
  const float* min_b = min_b_vector.flat<float>().data();
  const float* max_b = max_b_vector.flat<float>().data();
  float* min_c = (*min_c_vector)->flat<float>().data();
  float* max_c = (*max_c_vector)->flat<float>().data();

  // TODO(itex): Parallel for
  for (size_t n = 0; n < n_channel; ++n) {
    float a_float_for_one_quant_level =
        OneDnnFloatForOneQuantizedLevel<T1>(min_a, max_a);
    float b_float_for_one_quant_level =
        OneDnnFloatForOneQuantizedLevel<T2>(min_b[n], max_b[n]);
    float c_float_for_one_quant_level =
        a_float_for_one_quant_level * b_float_for_one_quant_level;
    min_c[n] = c_float_for_one_quant_level * c_lowest;
    max_c[n] = c_float_for_one_quant_level * c_highest;
  }
}

template <typename Tinput, typename Tfilter, typename Toutput>
void AllocateBlockOutputMinMax(OpKernelContext* context, float min_input,
                               float max_input, int kFilterMinRangeIndex,
                               int kFilterMaxRangeIndex, int kMinFreezedIndex,
                               int kMaxFreezedIndex, int kDstMinRangeIndex,
                               int kDstMaxRangeIndex) {
  OneDnnShape output_min_onednn_shape, output_max_onednn_shape;
  output_min_onednn_shape.SetOneDnnTensor(false);
  output_max_onednn_shape.SetOneDnnTensor(false);

  Tensor* output_min = nullptr;
  Tensor* output_max = nullptr;
  if (std::is_same<Toutput, quint8>::value ||
      std::is_same<Toutput, qint8>::value) {
    // Toutput = qint32, requantize fused
    AllocateOutputSetOneDnnShape(context, kDstMinRangeIndex, &output_min, {},
                                 output_min_onednn_shape);
    AllocateOutputSetOneDnnShape(context, kDstMaxRangeIndex, &output_max, {},
                                 output_max_onednn_shape);
    // This is the case the convolution and requantization are fused.
    output_min->flat<float>()(0) =
        context->input(kMinFreezedIndex).flat<float>()(0);
    output_max->flat<float>()(0) =
        context->input(kMaxFreezedIndex).flat<float>()(0);
  } else if (std::is_same<Toutput, qint32>::value) {
    const Tensor& min_filter = context->input(kFilterMinRangeIndex);
    const Tensor& max_filter = context->input(kFilterMaxRangeIndex);
    if (min_filter.dims() == 0) {
      float min_output_value;
      float max_output_value;
      OneDnnQuantizationRangeForMultiplication<Tinput, qint8, Toutput>(
          min_input, max_input, min_filter.flat<float>()(0),
          max_filter.flat<float>()(0), &min_output_value, &max_output_value);
      AllocateOutputSetOneDnnShape(context, kDstMinRangeIndex, &output_min, {},
                                   output_min_onednn_shape);
      AllocateOutputSetOneDnnShape(context, kDstMaxRangeIndex, &output_max, {},
                                   output_max_onednn_shape);
      output_min->flat<float>()(0) = min_output_value;
      output_max->flat<float>()(0) = max_output_value;
    } else {
      size_t depth = min_filter.NumElements();
      AllocateOutputSetOneDnnShape(context, kDstMinRangeIndex, &output_min,
                                   {static_cast<ptrdiff_t>(depth)},
                                   output_min_onednn_shape);
      AllocateOutputSetOneDnnShape(context, kDstMaxRangeIndex, &output_max,
                                   {static_cast<ptrdiff_t>(depth)},
                                   output_max_onednn_shape);
      OneDnnQuantizationRangeForMultiplication<Tinput, Tfilter, Toutput>(
          min_input, max_input, min_filter, max_filter, &output_min,
          &output_max);
    }
  } else {
    ITEX_VLOG(FATAL)
        << "Output datatype should be within float, uint8 or int8.";
  }
}

template <typename Tinput, typename Tfilter, typename Toutput>
void AllocateNativeOutputMinMax(OpKernelContext* context, float min_input,
                                float max_input, int kFilterMinRangeIndex,
                                int kFilterMaxRangeIndex, int kMinFreezedIndex,
                                int kMaxFreezedIndex, int kDstMinRangeIndex,
                                int kDstMaxRangeIndex) {
  Tensor* output_min = nullptr;
  Tensor* output_max = nullptr;
  if (std::is_same<Toutput, quint8>::value ||
      std::is_same<Toutput, qint8>::value) {
    // Toutput = qint32, requantize fused
    OP_REQUIRES_OK(
        context, context->allocate_output(kDstMinRangeIndex, {}, &output_min));
    OP_REQUIRES_OK(
        context, context->allocate_output(kDstMaxRangeIndex, {}, &output_max));
    // This is the case the convolution and requantization are fused.

    // TODO(itex): Follow intel-tf design, future we may need to differentiate
    // scaled/min_first output_quant_mode, with different output min/max
    // setting.
    output_min->flat<float>()(0) =
        context->input(kMinFreezedIndex).flat<float>()(0);
    output_max->flat<float>()(0) =
        context->input(kMaxFreezedIndex).flat<float>()(0);
  } else if (std::is_same<Toutput, qint32>::value) {
    const Tensor& min_filter = context->input(kFilterMinRangeIndex);
    const Tensor& max_filter = context->input(kFilterMaxRangeIndex);
    if (min_filter.dims() == 0) {
      float min_output_value;
      float max_output_value;
      OneDnnQuantizationRangeForMultiplication<Tinput, qint8, Toutput>(
          min_input, max_input, min_filter.flat<float>()(0),
          max_filter.flat<float>()(0), &min_output_value, &max_output_value);

      OP_REQUIRES_OK(context, context->allocate_output(kDstMinRangeIndex, {},
                                                       &output_min));
      OP_REQUIRES_OK(context, context->allocate_output(kDstMaxRangeIndex, {},
                                                       &output_max));
      output_min->flat<float>()(0) = min_output_value;
      output_max->flat<float>()(0) = max_output_value;
    } else {
      size_t depth = min_filter.NumElements();
      OP_REQUIRES_OK(
          context,
          context->allocate_output(
              kDstMinRangeIndex, {static_cast<ptrdiff_t>(depth)}, &output_min));
      OP_REQUIRES_OK(
          context,
          context->allocate_output(
              kDstMaxRangeIndex, {static_cast<ptrdiff_t>(depth)}, &output_max));
      OneDnnQuantizationRangeForMultiplication<Tinput, Tfilter, Toutput>(
          min_input, max_input, min_filter, max_filter, &output_min,
          &output_max);
    }
  } else if (std::is_same<Toutput, float>::value ||
             std::is_same<Toutput, Eigen::bfloat16>::value) {
    // Kernel is registered for Dequantization fusion. Nothing to do.
  } else {
    ITEX_VLOG(FATAL)
        << "Output datatype should be within float, uint8 or int8.";
  }
}

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_ONEDNN_BLOCK_QUANTIZED_OPS_H_
