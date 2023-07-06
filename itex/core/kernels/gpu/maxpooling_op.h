/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_MAXPOOLING_OP_H_
#define ITEX_CORE_KERNELS_GPU_MAXPOOLING_OP_H_

#include "itex/core/kernels/common/pooling_ops_common.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/integral_types.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {
template <typename T>
struct MaxPoolForwardWithArgmax {
  void operator()(TensorFormat data_format, const T* bottom_data,
                  const int batch, const int height, const int width,
                  const int channels, const int pooled_height,
                  const int pooled_width, const int kernel_h,
                  const int kernel_w, const int stride_h, const int stride_w,
                  const int pad_t, const int pad_l, T* top_data, int64* mask,
                  const Eigen::GpuDevice& d, bool propagate_nans,
                  const bool include_batch_in_index);
};

struct MaxPoolBackwardWithArgmax {
  void operator()(const int output_size, const int input_size,
                  const float* top_diff, const int64* mask,
                  const int top_offset, const int bottom_offset,
                  float* bottom_diff, const Eigen::GpuDevice& d,
                  const bool include_batch_in_index);
};

template <typename T>
struct MaxPoolGradBackwardWithArgmax {
  void operator()(const int output_size, const int input_size,
                  const T* top_diff, const int64* mask, const int top_offset,
                  const int bottom_offset, T* bottom_diff,
                  const Eigen::GpuDevice& d, const bool include_batch_in_index);
};

template <typename T>
struct MaxPoolGradBackwardNoMask {
  void operator()(TensorFormat data_format, const T* bottom_data,
                  const T* output_data, const int batch,
                  const int pooled_height, const int pooled_width,
                  const int channels, const int height, const int width,
                  const int kernel_h, const int kernel_w, const int stride_h,
                  const int stride_w, const int pad_t, const int pad_l,
                  const T* top_diff, T* bottom_diff, const Eigen::GpuDevice& d);
};

template <typename T>
struct MaxPool3dGradBackward {
  void operator()(TensorFormat data_format, const T* bottom_data,
                  const T* output_data, const int batch,
                  const int64 pooled_plane, const int64 pooled_height,
                  const int64 pooled_width, const int channels, const int plane,
                  const int height, const int width, const int kernel_p,
                  const int kernel_h, const int kernel_w, const int stride_p,
                  const int stride_h, const int stride_w, const int64 pad_p,
                  const int64 pad_t, const int64 pad_l, const T* top_diff,
                  T* bottom_diff, const Eigen::GpuDevice& d);
};

}  // namespace functor

template <bool, typename T>
struct MaxPoolFwdArgmaxNHWC;

template <typename T>
struct MaxPoolBwdArgMax;

template <typename T>
struct MaxPoolGradNCHWBwd;

template <typename T>
struct MaxPoolGradNHWCBwd;

template <typename T>
struct MaxPoolGradBwdArgMax;

template <typename Device, typename T>
struct LaunchMaxPooling3dGradGradOp;

template <bool propagate_nans, typename dtype>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool IsGreaterThan(dtype a, dtype b) {
  if (propagate_nans) {
    return !(a <= b);
  } else {
    return a > b;
  }
}

template <bool propagate_nans, typename T>
struct MaxPoolWithArgmaxKernel {
  MaxPoolWithArgmaxKernel(const int output_size, const T* bottom_data,
                          const int height, const int width, const int channels,
                          const int pooled_height, const int pooled_width,
                          const int kernel_h, const int kernel_w,
                          const int stride_h, const int stride_w,
                          const int pad_t, const int pad_l, T* top_data,
                          int64* mask, const bool include_batch_in_index)
      : output_size(output_size),
        bottom_data(bottom_data),
        height(height),
        width(width),
        channels(channels),
        pooled_height(pooled_height),
        pooled_width(pooled_width),
        kernel_h(kernel_h),
        kernel_w(kernel_w),
        stride_h(stride_h),
        stride_w(stride_w),
        pad_t(pad_t),
        pad_l(pad_l),
        top_data(top_data),
        mask(mask),
        include_batch_in_index(include_batch_in_index) {}
  void operator()(sycl::nd_item<1> item) const {
    int64_t index = item.get_global_linear_id();
    if (index < output_size) {
      int n = index;
      int c = n % channels;
      n /= channels;
      int wstart = (n % pooled_width) * stride_w - pad_l;
      n /= pooled_width;
      int hstart = (n % pooled_height) * stride_h - pad_t;
      n /= pooled_height;
      int hend = Eigen::numext::mini(hstart + kernel_h, height);
      int wend = Eigen::numext::mini(wstart + kernel_w, width);
      hstart = Eigen::numext::maxi(hstart, 0);
      wstart = Eigen::numext::maxi(wstart, 0);
      T maxval = Eigen::NumTraits<T>::lowest();
      int maxidx = -1;
      const int offset = n * height * width * channels;
      const T* bottom_data_n = bottom_data + offset;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int idx = (h * width + w) * channels + c;
          if (IsGreaterThan<propagate_nans>(bottom_data_n[idx], maxval)) {
            maxidx = include_batch_in_index ? idx + offset : idx;
            maxval = bottom_data_n[idx];
          }
        }
      }
      top_data[index] = maxval;
      mask[index] = maxidx;
    }
  }

 private:
  const int output_size;
  const T* bottom_data;
  const int height;
  const int width;
  const int channels;
  const int pooled_height;
  const int pooled_width;
  const int kernel_h;
  const int kernel_w;
  const int stride_h;
  const int stride_w;
  const int pad_t;
  const int pad_l;
  T* top_data;
  int64* mask;
  const bool include_batch_in_index;
};

struct MaxPoolBackwardKernel {
  MaxPoolBackwardKernel(const int nthreads, const float* top_diff,
                        const int64* mask, const int top_offset,
                        const int bottom_offset, float* bottom_diff,
                        const bool include_batch_in_index)
      : nthreads(nthreads),
        top_diff(top_diff),
        mask(mask),
        top_offset(top_offset),
        bottom_offset(bottom_offset),
        bottom_diff(bottom_diff),
        include_batch_in_index(include_batch_in_index) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= nthreads) return;
    const int offset =
        include_batch_in_index ? 0 : (index / top_offset) * bottom_offset;
    ItexAtomicAdd<float, float>(bottom_diff + offset + mask[index],
                                top_diff[index]);
  }

 private:
  const int nthreads;
  const float* top_diff;
  const int64* mask;
  const int top_offset;
  const int bottom_offset;
  float* bottom_diff;
  const bool include_batch_in_index;
};

template <typename T>
struct MaxPoolGradBackwardKernel {
  MaxPoolGradBackwardKernel(const int nthreads, const T* top_diff,
                            const int64* mask, const int top_offset,
                            const int bottom_offset, T* bottom_diff,
                            const bool include_batch_in_index)
      : nthreads(nthreads),
        top_diff(top_diff),
        mask(mask),
        top_offset(top_offset),
        bottom_offset(bottom_offset),
        bottom_diff(bottom_diff),
        include_batch_in_index(include_batch_in_index) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= nthreads) return;
    const int offset =
        include_batch_in_index ? 0 : (index / bottom_offset) * top_offset;
    bottom_diff[index] = top_diff[offset + mask[index]];
  }

 private:
  const int nthreads;
  const T* top_diff;
  const int64* mask;
  const int top_offset;
  const int bottom_offset;
  T* bottom_diff;
  const bool include_batch_in_index;
};

template <typename T>
struct MaxPoolGradBackwardNoMaskNHWCKernel {
  MaxPoolGradBackwardNoMaskNHWCKernel(
      const int nitems, const T* bottom_data, const T* output_data,
      const int pooled_height, const int pooled_width, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      const int stride_h, const int stride_w, const int pad_t, const int pad_l,
      const T* top_diff, T* bottom_diff)
      : nitems_(nitems),
        bottom_data_(bottom_data),
        output_data_(output_data),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        channels_(channels),
        height_(height),
        width_(width),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_t_(pad_t),
        pad_l_(pad_l),
        top_diff_(top_diff),
        bottom_diff_(bottom_diff) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t index = item.get_global_linear_id();
    if (index >= nitems_) return;
    int n = index;
    int c = n % channels_;
    n /= channels_;
    int wstart = (n % pooled_width_) * stride_w_ - pad_l_;
    n /= pooled_width_;
    int hstart = (n % pooled_height_) * stride_h_ - pad_t_;
    n /= pooled_height_;
    int hend = Eigen::numext::mini(hstart + kernel_h_, height_);
    int wend = Eigen::numext::mini(wstart + kernel_w_, width_);
    hstart = Eigen::numext::maxi(hstart, 0);
    wstart = Eigen::numext::maxi(wstart, 0);
    bool should_stop = false;
    int maxidx = -1;
    const T* bottom_data_n = bottom_data_ + n * height_ * width_ * channels_;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int h = hstart; h < hend && !should_stop; ++h) {
      for (int w = wstart; w < wend && !should_stop; ++w) {
        int idx = (h * width_ + w) * channels_ + c;
        if (output_data_[index] == bottom_data_n[idx]) {
          maxidx = idx;
          should_stop = true;
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff_[index] =
          top_diff_[n * height_ * width_ * channels_ + maxidx];
    }
  }

 private:
  const int nitems_;
  const T* bottom_data_;
  const T* output_data_;
  const int pooled_height_;
  const int pooled_width_;
  const int channels_;
  const int height_;
  const int width_;
  const int kernel_h_;
  const int kernel_w_;
  const int stride_h_;
  const int stride_w_;
  const int pad_t_;
  const int pad_l_;
  const T* top_diff_;
  T* bottom_diff_;
};

template <typename T>
struct MaxPoolGradBackwardNoMaskNCHWKernel {
  MaxPoolGradBackwardNoMaskNCHWKernel(
      const int nitems, const T* bottom_data, const T* output_data,
      const int pooled_height, const int pooled_width, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      const int stride_h, const int stride_w, const int pad_t, const int pad_l,
      const T* top_diff, T* bottom_diff)
      : nitems_(nitems),
        bottom_data_(bottom_data),
        output_data_(output_data),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        channels_(channels),
        height_(height),
        width_(width),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_t_(pad_t),
        pad_l_(pad_l),
        top_diff_(top_diff),
        bottom_diff_(bottom_diff) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t index = item.get_global_linear_id();
    if (index >= nitems_) return;
    int pw = index % pooled_width_;
    int ph = (index / pooled_width_) % pooled_height_;
    int c = (index / pooled_width_ / pooled_height_) % channels_;
    int n = index / pooled_width_ / pooled_height_ / channels_;
    int hstart = ph * stride_h_ - pad_t_;
    int wstart = pw * stride_w_ - pad_l_;
    const int hend = Eigen::numext::mini(hstart + kernel_h_, height_);
    const int wend = Eigen::numext::mini(wstart + kernel_w_, width_);
    hstart = Eigen::numext::maxi(hstart, 0);
    wstart = Eigen::numext::maxi(wstart, 0);
    bool should_stop = false;
    int maxidx = -1;
    const T* bottom_data_n = bottom_data_ + n * channels_ * height_ * width_;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int h = hstart; h < hend && !should_stop; ++h) {
      for (int w = wstart; w < wend && !should_stop; ++w) {
        int idx = c * height_ * width_ + h * width_ + w;
        if (output_data_[index] == bottom_data_n[idx]) {
          maxidx = idx;
          should_stop = true;
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff_[index] =
          top_diff_[n * channels_ * height_ * width_ + maxidx];
    }
  }

 private:
  const int nitems_;
  const T* bottom_data_;
  const T* output_data_;
  const int pooled_height_;
  const int pooled_width_;
  const int channels_;
  const int height_;
  const int width_;
  const int kernel_h_;
  const int kernel_w_;
  const int stride_h_;
  const int stride_w_;
  const int pad_t_;
  const int pad_l_;
  const T* top_diff_;
  T* bottom_diff_;
};

template <typename T>
struct MaxPoolGradBackwardNoMaskNDHWC {
  MaxPoolGradBackwardNoMaskNDHWC(
      const int nitems, const T* bottom_data, const T* output_data,
      const int64 pooled_plane, const int64 pooled_height,
      const int64 pooled_width, const int channels, const int plane,
      const int height, const int width, const int kernel_p, const int kernel_h,
      const int kernel_w, const int stride_p, const int stride_h,
      const int stride_w, const int64 pad_p, const int64 pad_t,
      const int64 pad_l, const T* top_diff, T* bottom_diff)
      : nitems_(nitems),
        bottom_data_(bottom_data),
        output_data_(output_data),
        pooled_plane_(pooled_plane),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        channels_(channels),
        plane_(plane),
        height_(height),
        width_(width),
        kernel_p_(kernel_p),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_p_(stride_p),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_p_(pad_p),
        pad_t_(pad_t),
        pad_l_{pad_l},
        top_diff_(top_diff),
        bottom_diff_(bottom_diff) {}

  void operator()(sycl::item<1> item) const {
    int64_t index = item.get_linear_id();
    if (index >= nitems_) return;

    int n = index;
    int c = n % channels_;
    n /= channels_;
    int wstart = (n % pooled_width_) * stride_w_ - pad_l_;
    int wend = Eigen::numext::mini(wstart + kernel_w_, width_);
    wstart = Eigen::numext::maxi(wstart, 0);
    n /= pooled_width_;
    int hstart = (n % pooled_height_) * stride_h_ - pad_t_;
    int hend = Eigen::numext::mini(hstart + kernel_h_, height_);
    hstart = Eigen::numext::maxi(hstart, 0);
    n /= pooled_height_;
    int pstart = (n % pooled_plane_) * stride_p_ - pad_p_;
    int pend = Eigen::numext::mini(pstart + kernel_p_, plane_);
    pstart = Eigen::numext::maxi(pstart, 0);
    n /= pooled_plane_;
    bool should_stop = false;
    int maxidx = -1;
    const T* bottom_data_n =
        bottom_data_ + n * plane_ * height_ * width_ * channels_;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int h = hstart; h < hend && !should_stop; ++h) {
        for (int w = wstart; w < wend && !should_stop; ++w) {
          int idx = ((p * height_ + h) * width_ + w) * channels_ + c;
          if (output_data_[index] == bottom_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }

    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff_[index] =
          top_diff_[n * plane_ * height_ * width_ * channels_ + maxidx];
    }
  }

 private:
  const int nitems_;
  const T* bottom_data_;
  const T* output_data_;
  const int64 pooled_plane_;
  const int64 pooled_height_;
  const int64 pooled_width_;
  const int channels_;
  const int plane_;
  const int height_;
  const int width_;
  const int kernel_p_;
  const int kernel_h_;
  const int kernel_w_;
  const int stride_p_;
  const int stride_h_;
  const int stride_w_;
  const int64 pad_p_;
  const int64 pad_t_;
  const int64 pad_l_;
  const T* top_diff_;
  T* bottom_diff_;
};

template <typename T>
struct MaxPoolGradBackwardNoMaskNCDHW {
  MaxPoolGradBackwardNoMaskNCDHW(
      const int nitems, const T* bottom_data, const T* output_data,
      const int64 pooled_plane, const int64 pooled_height,
      const int64 pooled_width, const int channels, const int plane,
      const int height, const int width, const int kernel_p, const int kernel_h,
      const int kernel_w, const int stride_p, const int stride_h,
      const int stride_w, const int64 pad_p, const int64 pad_t,
      const int64 pad_l, const T* top_diff, T* bottom_diff)
      : nitems_(nitems),
        bottom_data_(bottom_data),
        output_data_(output_data),
        pooled_plane_(pooled_plane),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        channels_(channels),
        plane_(plane),
        height_(height),
        width_(width),
        kernel_p_(kernel_p),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_p_(stride_p),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_p_(pad_p),
        pad_t_(pad_t),
        pad_l_{pad_l},
        top_diff_(top_diff),
        bottom_diff_(bottom_diff) {}

  void operator()(sycl::item<1> item) const {
    int64_t index = item.get_linear_id();
    if (index >= nitems_) return;

    int pw = index % pooled_width_;
    int ph = (index / pooled_width_) % pooled_height_;
    int pp = (index / pooled_width_ / pooled_height_) % pooled_plane_;
    int c =
        (index / pooled_width_ / pooled_height_ / pooled_plane_) % channels_;
    int n =
        (index / pooled_width_ / pooled_height_ / pooled_plane_ / channels_);
    int pstart = pp * stride_p_ - pad_p_;
    int hstart = ph * stride_h_ - pad_t_;
    int wstart = pw * stride_w_ - pad_l_;
    const int pend = Eigen::numext::mini(pstart + kernel_p_, plane_);
    const int hend = Eigen::numext::mini(hstart + kernel_h_, height_);
    const int wend = Eigen::numext::mini(wstart + kernel_w_, width_);
    pstart = Eigen::numext::maxi(pstart, 0);
    hstart = Eigen::numext::maxi(hstart, 0);
    wstart = Eigen::numext::maxi(wstart, 0);
    bool should_stop = false;
    int maxidx = -1;
    const T* bottom_data_n =
        bottom_data_ + n * channels_ * plane_ * height_ * width_;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int h = hstart; h < hend && !should_stop; ++h) {
        for (int w = wstart; w < wend && !should_stop; ++w) {
          int idx =
              c * plane_ * height_ * width_ + (p * height_ + h) * width_ + w;
          if (output_data_[index] == bottom_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff_[index] =
          top_diff_[n * channels_ * plane_ * height_ * width_ + maxidx];
    }
  }

 private:
  const int nitems_;
  const T* bottom_data_;
  const T* output_data_;
  const int64 pooled_plane_;
  const int64 pooled_height_;
  const int64 pooled_width_;
  const int channels_;
  const int plane_;
  const int height_;
  const int width_;
  const int kernel_p_;
  const int kernel_h_;
  const int kernel_w_;
  const int stride_p_;
  const int stride_h_;
  const int stride_w_;
  const int64 pad_p_;
  const int64 pad_t_;
  const int64 pad_l_;
  const T* top_diff_;
  T* bottom_diff_;
};

template <typename T>
struct LaunchMaxPoolingWithArgmax {
  static void launch(OpKernelContext* context, TensorFormat data_format,
                     const OneDnnPoolParameters& params, const Tensor& input,
                     Tensor* output, Tensor* argmax, bool propagate_nans,
                     bool include_batch_in_index) {
    functor::MaxPoolForwardWithArgmax<T>()(
        data_format, input.flat<T>().data(), params.tensor_in_batch,
        params.tensor_in_rows, params.tensor_in_cols, params.depth,
        params.out_height, params.out_width, params.window_rows,
        params.window_cols, params.row_stride, params.col_stride,
        params.pad_top, params.pad_left, output->flat<T>().data(),
        reinterpret_cast<int64*>(argmax->flat<int64>().data()),
        context->eigen_gpu_device(), propagate_nans, include_batch_in_index);
  }
};

template <typename T>
struct LaunchMaxPoolingGradWithArgmax {
  static void launch(OpKernelContext* context,
                     const OneDnnPoolParameters& params, const Tensor& grad_in,
                     const Tensor& argmax, Tensor* grad_out,
                     const bool include_batch_in_index) {
    const int input_size = params.tensor_in_batch * params.tensor_in_rows *
                           params.tensor_in_cols * params.depth;
    const int output_size = params.tensor_in_batch * params.out_height *
                            params.out_width * params.depth;
    const int top_offset = params.out_height * params.out_width * params.depth;
    const int bottom_offset =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    auto dev = context->eigen_gpu_device();

    Tensor grad_in_fp32;
    TF_ABORT_IF_ERROR(context->allocate_temp(DataTypeToEnum<float>::v(),
                                             grad_in.shape(), &grad_in_fp32));
    grad_in_fp32.flat<float>().device(dev) =
        grad_in.flat<T>().template cast<float>();

    Tensor grad_out_fp32;
    TF_ABORT_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<float>::v(), grad_out->shape(), &grad_out_fp32));
    grad_out_fp32.flat<float>().device(dev) =
        grad_out_fp32.flat<float>().constant(0.0f);

    functor::MaxPoolBackwardWithArgmax()(
        output_size, input_size, grad_in_fp32.flat<float>().data(),
        reinterpret_cast<const int64*>(argmax.flat<int64>().data()), top_offset,
        bottom_offset, grad_out_fp32.flat<float>().data(), dev,
        include_batch_in_index);
    grad_out->flat<T>().device(dev) =
        grad_out_fp32.flat<float>().template cast<T>();
  }
};

template <>
struct LaunchMaxPoolingGradWithArgmax<float> {
  static void launch(OpKernelContext* context,
                     const OneDnnPoolParameters& params, const Tensor& grad_in,
                     const Tensor& argmax, Tensor* grad_out,
                     const bool include_batch_in_index) {
    const int input_size = params.tensor_in_batch * params.tensor_in_rows *
                           params.tensor_in_cols * params.depth;
    const int output_size = params.tensor_in_batch * params.out_height *
                            params.out_width * params.depth;
    const int top_offset = params.out_height * params.out_width * params.depth;
    const int bottom_offset =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;

    auto dev = context->eigen_gpu_device();
    grad_out->flat<float>().device(dev) =
        grad_out->flat<float>().constant(0.0f);

    functor::MaxPoolBackwardWithArgmax()(
        output_size, input_size, grad_in.flat<float>().data(),
        reinterpret_cast<const int64*>(argmax.flat<int64>().data()), top_offset,
        bottom_offset, grad_out->flat<float>().data(), dev,
        include_batch_in_index);
  }
};

template <typename T>
struct LaunchMaxPoolingGradGradWithArgmax {
  static void launch(OpKernelContext* context,
                     const OneDnnPoolParameters& params, const Tensor& grad_in,
                     const Tensor& argmax, Tensor* grad_out,
                     const bool include_batch_in_index) {
    const int input_size = params.tensor_in_batch * params.tensor_in_rows *
                           params.tensor_in_cols * params.depth;
    const int output_size = params.tensor_in_batch * params.out_height *
                            params.out_width * params.depth;
    const int top_offset =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    const int bottom_offset =
        params.out_width * params.out_height * params.depth;
    functor::MaxPoolGradBackwardWithArgmax<T>()(
        output_size, input_size, grad_in.flat<T>().data(),
        reinterpret_cast<const int64*>(argmax.flat<int64>().data()), top_offset,
        bottom_offset, grad_out->flat<T>().data(), context->eigen_gpu_device(),
        include_batch_in_index);
  }
};

template <typename T>
struct LaunchMaxPooling3dGradGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context,
                     const OneDnnPoolParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& tensor_top_diff,
                     Tensor* tensor_bottom_diff) {
    functor::MaxPool3dGradBackward<T>()(
        params.data_format, tensor_in.flat<T>().data(),
        tensor_out.flat<T>().data(), params.tensor_in_batch, params.out_planes,
        params.out_height, params.out_width, params.depth,
        params.tensor_in_planes, params.tensor_in_rows, params.tensor_in_cols,
        params.window_planes, params.window_rows, params.window_cols,
        params.planes_stride, params.row_stride, params.col_stride,
        params.pad_P1, params.pad_top, params.pad_left,
        tensor_top_diff.flat<T>().data(), tensor_bottom_diff->flat<T>().data(),
        context->eigen_gpu_device());
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_MAXPOOLING_OP_H_
