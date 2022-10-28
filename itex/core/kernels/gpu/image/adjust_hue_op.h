/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_IMAGE_ADJUST_HUE_OP_H_
#define ITEX_CORE_KERNELS_GPU_IMAGE_ADJUST_HUE_OP_H_

#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace internal {

typedef struct RgbTuple {
  float r;
  float g;
  float b;
} RgbTuple;

typedef struct HsvTuple {
  float h;
  float s;
  float v;
} HsvTuple;

inline HsvTuple rgb2hsv_dpcpp(const float r, const float g, const float b) {
  HsvTuple tuple;
  const float M = sycl::fmax(r, sycl::fmax(g, b));
  const float m = sycl::fmin(r, sycl::fmin(g, b));
  const float chroma = M - m;
  float h = 0.0f, s = 0.0f;
  // hue
  if (chroma > 0.0f) {
    if (M == r) {
      const float num = (g - b) / chroma;
      const float sign = sycl::copysign(1.0f, num);
      h = ((sign < 0.0f) * 6.0f + sign * sycl::fmod(sign * num, 6.0f)) / 6.0f;
    } else if (M == g) {
      h = ((b - r) / chroma + 2.0f) / 6.0f;
    } else {
      h = ((r - g) / chroma + 4.0f) / 6.0f;
    }
  } else {
    h = 0.0f;
  }
  // saturation
  if (M > 0.0) {
    // Maozhou: accuracy issue if s = chroma / M
    s = 1 - m / M;
  } else {
    s = 0.0f;
  }
  tuple.h = h;
  tuple.s = s;
  tuple.v = M;
  return tuple;
}

inline RgbTuple hsv2rgb_dpcpp(const float h, const float s, const float v) {
  RgbTuple tuple;
  const float new_h = h * 6.0f;
  const float chroma = v * s;
  const float x = chroma * (1.0f - sycl::fabs(sycl::fmod(new_h, 2.0f) - 1.0f));
  const float new_m = v - chroma;
  const bool between_0_and_1 = new_h >= 0.0f && new_h < 1.0f;
  const bool between_1_and_2 = new_h >= 1.0f && new_h < 2.0f;
  const bool between_2_and_3 = new_h >= 2.0f && new_h < 3.0f;
  const bool between_3_and_4 = new_h >= 3.0f && new_h < 4.0f;
  const bool between_4_and_5 = new_h >= 4.0f && new_h < 5.0f;
  const bool between_5_and_6 = new_h >= 5.0f && new_h < 6.0f;
  tuple.r = chroma * (between_0_and_1 || between_5_and_6) +
            x * (between_1_and_2 || between_4_and_5) + new_m;
  tuple.g = chroma * (between_1_and_2 || between_2_and_3) +
            x * (between_0_and_1 || between_3_and_4) + new_m;
  tuple.b = chroma * (between_3_and_4 || between_4_and_5) +
            x * (between_2_and_3 || between_5_and_6) + new_m;
  return tuple;
}

template <bool AdjustHue, bool AdjustSaturation, bool AdjustV, typename T>
struct AdjustHsvNHWC {
  AdjustHsvNHWC(const int64 number_elements, const T* const input,
                T* const output, const float* const hue_delta,
                const float* const saturation_scale,
                const float* const value_scale)
      : number_elements_(number_elements),
        input_(input),
        output_(output),
        hue_delta_(hue_delta),
        saturation_scale_(saturation_scale),
        value_scale_(value_scale) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    // multiply by 3 since we're dealing with contiguous RGB bytes for each
    // pixel (NHWC)
    const int64 idx = id * 3;
    if (idx > number_elements_ - 3) {
      return;
    }

    if (!AdjustHue && !AdjustSaturation && !AdjustV) {
      output_[idx] = input_[idx];
      output_[idx + 1] = input_[idx + 1];
      output_[idx + 2] = input_[idx + 2];
      return;
    }

    const HsvTuple hsv = rgb2hsv_dpcpp(static_cast<float>(input_[idx]),
                                       static_cast<float>(input_[idx + 1]),
                                       static_cast<float>(input_[idx + 2]));
    float new_h = hsv.h;
    float new_s = hsv.s;
    float new_v = hsv.v;

    // hue adjustment
    if (AdjustHue) {
      const float delta = *hue_delta_;
      new_h = sycl::fmod(hsv.h + delta, 1.0f);
      if (new_h < 0.0f) {
        new_h = sycl::fmod(1.0f + new_h, 1.0f);
      }
    }
    // saturation adjustment
    if (AdjustSaturation && saturation_scale_ != nullptr) {
      const float scale = *saturation_scale_;
      new_s = sycl::fmin(1.0f, sycl::fmax(0.0f, hsv.s * scale));
    }
    // value adjustment
    if (AdjustV && value_scale_ != nullptr) {
      const float scale = *value_scale_;
      new_v = hsv.v * scale;
    }

    const RgbTuple rgb = hsv2rgb_dpcpp(new_h, new_s, new_v);
    output_[idx] = static_cast<T>(rgb.r);
    output_[idx + 1] = static_cast<T>(rgb.g);
    output_[idx + 2] = static_cast<T>(rgb.b);
  }

 private:
  const int64 number_elements_;
  const T* const input_;
  T* const output_;
  const float* const hue_delta_;
  const float* const saturation_scale_;
  const float* const value_scale_;
};
}  // namespace internal

namespace functor {

template <typename T>
struct AdjustHueGPU {
  void operator()(GPUDevice* device, const int64 number_of_elements,
                  const T* const input, const float* const delta,
                  T* const output);
};

template <typename T>
void AdjustHueGPU<T>::operator()(GPUDevice* device,
                                 const int64 number_of_elements,
                                 const T* const input, const float* const delta,
                                 T* const output) {
  auto stream = device->stream();
  auto work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_items = number_of_elements;
  auto num_work_groups =
      (num_work_items + work_group_size - 1) / work_group_size;
  stream->submit([&](sycl::handler& cgh) {
    internal::AdjustHsvNHWC<true, false, false, T> task(
        number_of_elements, input, output, delta, nullptr, nullptr);
    cgh.parallel_for<internal::AdjustHsvNHWC<true, false, false, T> >(
        sycl::nd_range<1>(sycl::range<1>(num_work_groups * work_group_size),
                          sycl::range<1>(work_group_size)),
        task);
  });
}

template struct AdjustHueGPU<float>;
template struct AdjustHueGPU<Eigen::half>;
template struct AdjustHueGPU<Eigen::bfloat16>;
}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_IMAGE_ADJUST_HUE_OP_H_
