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

#ifndef ITEX_CORE_UTILS_QUANTIZATION_UTIL_H_
#define ITEX_CORE_UTILS_QUANTIZATION_UTIL_H_

#include <algorithm>

#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/plugin_tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

enum class QuantizeMode {
  MIN_COMBINED,
  MIN_FIRST,
  SCALED,
};

enum class QuantizeRoundMode {
  // Round half away from zero: if the fraction of y is exactly 0.5, then
  // round(y) = y + 0.5 if y > 0
  // round(y) = y - 0.5 if y < 0
  // E.g., -5.5 gets rounded to -6, -5.4 goes to -5,
  // 5.4 goes to 5, and 5.5 goes to 6.
  ROUND_HALF_AWAY_FROM_ZERO,
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};

enum class QuantDequantFlag {
  Quantize,
  Dequantize,
};

// TODO(itex): separate symmetric and asymmetric into 2 functions
template <typename T>
void GetScaleAndZeropointAndAlignMinMax(float* min_data, float* max_data,
                                        QuantizeMode mode,
                                        QuantDequantFlag flag, int length,
                                        float* scales, int32* zero_points) {
  int64 max_int = static_cast<int64>(Eigen::NumTraits<T>::highest());
  int64 min_int = static_cast<int64>(Eigen::NumTraits<T>::lowest());

  if (mode == QuantizeMode::SCALED) {
    for (int i = 0; i < length; ++i) {
      float abs_range = std::max(std::abs(min_data[i]), std::abs(max_data[i]));
      // symmetric quantization needs to adjust the output min/max range
      max_data[i] = abs_range;
      min_data[i] = -abs_range;
      scales[i] = (flag == QuantDequantFlag::Quantize) ? (max_int / abs_range)
                                                       : (abs_range / max_int);
      // set zeropoint 0 for symmetric quantization
      zero_points[i] = 0;
    }
  } else if (mode == QuantizeMode::MIN_FIRST) {
    for (int i = 0; i < length; ++i) {
      scales[i] = (flag == QuantDequantFlag::Quantize)
                      ? ((max_int - min_int) / (max_data[i] - min_data[i]))
                      : ((max_data[i] - min_data[i]) / (max_int - min_int));
      zero_points[i] = (flag == QuantDequantFlag::Quantize)
                           ? static_cast<int32_t>(
                                 std::round(max_int - max_data[i] * scales[i]))
                           : static_cast<int32_t>(
                                 std::round(max_int - max_data[i] / scales[i]));
    }
  }
}

}  // namespace itex

#endif  // ITEX_CORE_UTILS_QUANTIZATION_UTIL_H_
