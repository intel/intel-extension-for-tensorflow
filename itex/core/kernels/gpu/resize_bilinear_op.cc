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

#include "itex/core/kernels/gpu/resize_bilinear_op.h"

#include <memory>

#include "itex/core/kernels/gpu/cast_op.h"
#include "itex/core/kernels/gpu/image_resizer_state.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::internal::TensorIntDivisor<int> FastDivisor;

template <typename Device, typename T>
class ResizeBilinearOp : public OpKernel {
 public:
  explicit ResizeBilinearOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor image_data(input.tensor<T, 4>());
    TTypes<float, 4>::Tensor output_data = st.output->tensor<float, 4>();

    functor::ResizeBilinear<Device, T>()(
        context->eigen_gpu_device(), image_data, st.height_scale,
        st.width_scale, half_pixel_centers_, output_data);
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

namespace functor {

namespace impl {
template <typename T>
struct ResizeBilinearKernel {
  ResizeBilinearKernel(const T* images, int total_count, float height_scale,
                       float width_scale, int batch, int in_height,
                       int in_width, int channels, int out_height,
                       int out_width, FastDivisor channels_fast_divisor,
                       FastDivisor out_height_fast_divisor,
                       FastDivisor out_width_fast_divisor, float* output)
      : images(images),
        total_count(total_count),
        height_scale(height_scale),
        width_scale(width_scale),
        batch(batch),
        in_height(in_height),
        in_width(in_width),
        channels(channels),
        out_height(out_height),
        out_width(out_width),
        channels_fast_divisor(channels_fast_divisor),
        out_height_fast_divisor(out_height_fast_divisor),
        out_width_fast_divisor(out_width_fast_divisor),
        output(output) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();

    if (id >= total_count) return;

    int prev_id = id;
    int n = id;
    n = n / channels_fast_divisor;
    const int c = prev_id - n * channels;
    prev_id = n;
    n = n / out_width_fast_divisor;
    const int x = prev_id - n * out_width;
    prev_id = n;
    n = n / out_height_fast_divisor;
    const int b = n;
    const int y = prev_id - n * out_height;

    const float in_y = (static_cast<float>(y) + 0.5f) * height_scale - 0.5f;

    const int top_y_index = in_y > 0.0 ? sycl::floor(in_y) : 0;
    const int bottom_y_index =
        (in_y < in_height - 1) ? sycl::ceil(in_y) : in_height - 1;
    const float y_lerp = in_y - sycl::floor(in_y);

    const float in_x = (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;
    const int left_x_index = in_x > 0.0 ? sycl::floor(in_x) : 0;
    const int right_x_index =
        (in_x < in_width - 1) ? sycl::ceil(in_x) : in_width - 1;
    const float x_lerp = in_x - left_x_index;

    const float top_left(
        images[((b * in_height + top_y_index) * in_width + left_x_index) *
                   channels +
               c]);
    const float top_right(
        images[((b * in_height + top_y_index) * in_width + right_x_index) *
                   channels +
               c]);
    const float bottom_left(
        images[((b * in_height + bottom_y_index) * in_width + left_x_index) *
                   channels +
               c]);
    const float bottom_right(
        images[((b * in_height + bottom_y_index) * in_width + right_x_index) *
                   channels +
               c]);

    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

    // write output
    output[id] = static_cast<float>(top + (bottom - top) * y_lerp);
  }

 private:
  const T* images;
  int total_count;
  float height_scale;
  float width_scale;
  int batch;
  int in_height;
  int in_width;
  int channels;
  int out_height;
  int out_width;
  FastDivisor channels_fast_divisor;
  FastDivisor out_height_fast_divisor;
  FastDivisor out_width_fast_divisor;
  float* output;
};

template <typename T>
struct LegacyResizeBilinearKernel {
  LegacyResizeBilinearKernel(const T* images, int total_count,
                             float height_scale, float width_scale, int batch,
                             int in_height, int in_width, int channels,
                             int out_height, int out_width,
                             FastDivisor channels_fast_divisor,
                             FastDivisor out_height_fast_divisor,
                             FastDivisor out_width_fast_divisor, float* output)
      : images(images),
        total_count(total_count),
        height_scale(height_scale),
        width_scale(width_scale),
        batch(batch),
        in_height(in_height),
        in_width(in_width),
        channels(channels),
        out_height(out_height),
        out_width(out_width),
        channels_fast_divisor(channels_fast_divisor),
        out_height_fast_divisor(out_height_fast_divisor),
        out_width_fast_divisor(out_width_fast_divisor),
        output(output) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();

    if (id >= total_count) return;

    int prev_id = id;
    int n = id;
    n = n / channels_fast_divisor;
    const int c = prev_id - n * channels;
    prev_id = n;
    n = n / out_width_fast_divisor;
    const int x = prev_id - n * out_width;
    prev_id = n;
    n = n / out_height_fast_divisor;
    const int b = n;
    const int y = prev_id - n * out_height;

    const float in_y = y * height_scale;
    const int top_y_index = sycl::floor(in_y);
    const int bottom_y_index =
        (in_y < in_height - 1) ? sycl::ceil(in_y) : in_height - 1;
    const float y_lerp = in_y - top_y_index;

    const float in_x = x * width_scale;
    const int left_x_index = sycl::floor(in_x);
    const int right_x_index =
        (in_x < in_width - 1) ? sycl::ceil(in_x) : in_width - 1;
    const float x_lerp = in_x - left_x_index;

    const float top_left(
        images[((b * in_height + top_y_index) * in_width + left_x_index) *
                   channels +
               c]);
    const float top_right(
        images[((b * in_height + top_y_index) * in_width + right_x_index) *
                   channels +
               c]);
    const float bottom_left(
        images[((b * in_height + bottom_y_index) * in_width + left_x_index) *
                   channels +
               c]);
    const float bottom_right(
        images[((b * in_height + bottom_y_index) * in_width + right_x_index) *
                   channels +
               c]);

    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

    // write output
    output[id] = static_cast<float>(top + (bottom - top) * y_lerp);
  }

 private:
  const T* images;
  int total_count;
  float height_scale;
  float width_scale;
  int batch;
  int in_height;
  int in_width;
  int channels;
  int out_height;
  int out_width;
  FastDivisor channels_fast_divisor;
  FastDivisor out_height_fast_divisor;
  FastDivisor out_width_fast_divisor;
  float* output;
};

}  // namespace impl

template <typename T>
class ResizeBilinearTask;
template <typename T>
class LegacyResizeBilinearTask;

// Partial specialization of ResizeBilinear functor for a GPUDevice.
template <typename T>
struct ResizeBilinear<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor images,
                  const float height_scale, const float width_scale,
                  const bool half_pixel_centers,
                  typename TTypes<float, 4>::Tensor output) {
    const int batch = images.dimension(0);
    const int in_height = images.dimension(1);
    const int in_width = images.dimension(2);
    const int channels = images.dimension(3);

    const int out_height = output.dimension(1);
    const int out_width = output.dimension(2);

    const int total_count = batch * out_height * out_width * channels;
    if (total_count == 0) return;

#define EigenFastDivisor(divisor, num)                     \
  Eigen::internal::TensorIntDivisor<int> divisor;          \
  if (num != 0) {                                          \
    divisor = Eigen::internal::TensorIntDivisor<int>(num); \
  }

    EigenFastDivisor(channels_fast_divisor, channels);
    EigenFastDivisor(out_width_fast_divisor, out_width);
    EigenFastDivisor(out_height_fast_divisor, out_height);
#undef EigenFastDivisor

    auto stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (total_count + group_size - 1) / group_size;

    if (half_pixel_centers) {
      stream->submit([&](sycl::handler& cgh) {
        impl::ResizeBilinearKernel<T> task(
            images.data(), total_count, height_scale, width_scale, batch,
            in_height, in_width, channels, out_height, out_width,
            channels_fast_divisor, out_height_fast_divisor,
            out_width_fast_divisor, output.data());
        cgh.parallel_for<ResizeBilinearTask<T>>(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            task);
      });
    } else {
      stream->submit([&](sycl::handler& cgh) {
        impl::LegacyResizeBilinearKernel<T> task(
            images.data(), total_count, height_scale, width_scale, batch,
            in_height, in_width, channels, out_height, out_width,
            channels_fast_divisor, out_height_fast_divisor,
            out_width_fast_divisor, output.data());
        cgh.parallel_for<LegacyResizeBilinearTask<T>>(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            task);
      });
    }
    return;
  }
};
#define DEFINE_GPU_SPECS(T) template struct ResizeBilinear<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

}  // namespace functor

namespace {

template <typename Device, typename SrcType, typename DstType>
struct CastDataType {
  void operator()(const Device& d, typename TTypes<SrcType>::ConstFlat input,
                  typename TTypes<DstType>::Flat output) {
    output.device(d) = input.template cast<DstType>();
  }
};

template <typename SrcType, typename DstType>
struct CastDataType<GPUDevice, SrcType, DstType> {
  void operator()(const GPUDevice& d, typename TTypes<SrcType>::ConstFlat input,
                  typename TTypes<DstType>::Flat output) {
    // Use existing cast functor instead of directly casting Eigen tensor, as
    // otherwise we need to instantiate the cast function in a .cu.cc file
    functor::CastFunctor<GPUDevice, DstType, SrcType> cast;
    cast(d, output, input);
  }
};
}  // namespace

template <typename T>
class ResizeBilinearGradKernelTask;
template <typename T>
class LegacyResizeBilinearGradKernelTask;

template <typename Device, typename T>
class ResizeBilinearOpGrad : public OpKernel {
 public:
  explicit ResizeBilinearOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    // Validate input.
    // First argument is gradient with respect to resized image, a tensor of
    // type `float32`
    const Tensor& input = context->input(0);
    const Tensor& original_image = context->input(1);

    ImageResizerGradientState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context, input, original_image);

    if (!context->status().ok()) return;

    TTypes<float, 4>::ConstTensor input_grad = input.tensor<float, 4>();

    if (std::is_same<T, float>::value) {
      typename TTypes<float, 4>::Tensor output_grad(
          st.output->tensor<float, 4>());
      functor::ResizeBilinearGrad<Device, float>()(
          context->eigen_gpu_device(), input_grad, st.height_scale,
          st.width_scale, half_pixel_centers_, output_grad);
    } else {
      // Accumulate output to float instead of half tensor, since float
      // accumulation is more numerically stable and GPU half implementation is
      // slow.
      // TODO(itex): Create optimized and numerically stable half
      // implementation
      Tensor output_grad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DT_FLOAT, st.output->shape(), &output_grad));
      functor::ResizeBilinearGrad<Device, float>()(
          context->eigen_gpu_device(), input_grad, st.height_scale,
          st.width_scale, half_pixel_centers_, output_grad.tensor<float, 4>());
      const Tensor& output_grad_const = output_grad;
      CastDataType<Device, float, T>{}(context->eigen_gpu_device(),
                                       output_grad_const.template flat<float>(),
                                       st.output->template flat<T>());
    }
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

namespace functor {

namespace impl {
template <typename T>
struct ResizeBilinearGradKernel {
  ResizeBilinearGradKernel(const int total_count, const float* input_grad,
                           const float height_scale, const float width_scale,
                           int batch, int original_height, int original_width,
                           int channels, int resized_height, int resized_width,
                           FastDivisor channels_fast_divisor,
                           FastDivisor resized_height_fast_divisor,
                           FastDivisor resized_width_fast_divisor,
                           T* output_grad)
      : total_count(total_count),
        input_grad(input_grad),
        height_scale(height_scale),
        width_scale(width_scale),
        batch(batch),
        original_height(original_height),
        original_width(original_width),
        channels(channels),
        resized_height(resized_height),
        resized_width(resized_width),
        channels_fast_divisor(channels_fast_divisor),
        resized_height_fast_divisor(resized_height_fast_divisor),
        resized_width_fast_divisor(resized_width_fast_divisor),
        output_grad(output_grad) {}
  void operator()(sycl::nd_item<1> item) const {
    auto in_idx = item.get_global_linear_id();
    if (in_idx >= total_count) return;

    // in_idx = c + channels * (x + resized_width * (y + resized_height *
    // b))
    int prev_id = in_idx;
    int n = in_idx;
    n = n / channels_fast_divisor;
    const int c = prev_id - n * channels;
    prev_id = n;
    n = n / resized_width_fast_divisor;
    const int x = prev_id - n * resized_width;
    prev_id = n;
    n = n / resized_height_fast_divisor;
    const int b = n;
    const int y = prev_id - n * resized_height;

    const float original_y =
        (static_cast<float>(y) + 0.5f) * height_scale - 0.5f;
    const int top_y_index = original_y > 0.0 ? sycl::floor(original_y) : 0;
    const int bottom_y_index = (original_y < original_height - 1)
                                   ? sycl::ceil(original_y)
                                   : original_height - 1;
    const float y_lerp = original_y - sycl::floor(original_y);

    const float original_x =
        (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;

    const int left_x_index = original_x > 0.0 ? sycl::floor(original_x) : 0;
    const int right_x_index = (original_x < original_width - 1)
                                  ? sycl::ceil(original_x)
                                  : original_width - 1;
    const float x_lerp = original_x - sycl::floor(original_x);

    const float dtop = (1 - y_lerp) * input_grad[in_idx];
    ItexAtomicAdd(output_grad +
                      ((b * original_height + top_y_index) * original_width +
                       left_x_index) *
                          channels +
                      c,
                  static_cast<T>((1 - x_lerp) * dtop));
    ItexAtomicAdd(output_grad +
                      ((b * original_height + top_y_index) * original_width +
                       right_x_index) *
                          channels +
                      c,
                  static_cast<T>(x_lerp * dtop));

    const float dbottom = y_lerp * input_grad[in_idx];
    ItexAtomicAdd(output_grad +
                      ((b * original_height + bottom_y_index) * original_width +
                       left_x_index) *
                          channels +
                      c,
                  static_cast<T>((1 - x_lerp) * dbottom));
    ItexAtomicAdd(output_grad +
                      ((b * original_height + bottom_y_index) * original_width +
                       right_x_index) *
                          channels +
                      c,
                  static_cast<T>(x_lerp * dbottom));
  }

 private:
  const int total_count;
  const float* input_grad;
  const float height_scale;
  const float width_scale;
  int batch;
  int original_height;
  int original_width;
  int channels;
  int resized_height;
  int resized_width;
  FastDivisor channels_fast_divisor;
  FastDivisor resized_height_fast_divisor;
  FastDivisor resized_width_fast_divisor;
  T* output_grad;
};

template <typename T>
struct LegacyResizeBilinearGradKernel {
  LegacyResizeBilinearGradKernel(
      const int total_count, const float* input_grad, const float height_scale,
      const float width_scale, int batch, int original_height,
      int original_width, int channels, int resized_height, int resized_width,
      FastDivisor channels_fast_divisor,
      FastDivisor resized_height_fast_divisor,
      FastDivisor resized_width_fast_divisor, T* output_grad)
      : total_count(total_count),
        input_grad(input_grad),
        height_scale(height_scale),
        width_scale(width_scale),
        batch(batch),
        original_height(original_height),
        original_width(original_width),
        channels(channels),
        resized_height(resized_height),
        resized_width(resized_width),
        channels_fast_divisor(channels_fast_divisor),
        resized_height_fast_divisor(resized_height_fast_divisor),
        resized_width_fast_divisor(resized_width_fast_divisor),
        output_grad(output_grad) {}

  void operator()(sycl::nd_item<1> item) const {
    auto in_idx = item.get_global_linear_id();
    if (in_idx >= total_count) return;
    // in_idx = c + channels * (x + resized_width * (y + resized_height *
    // b))
    int prev_id = in_idx;
    int n = in_idx;
    n = n / channels_fast_divisor;
    const int c = prev_id - n * channels;
    prev_id = n;
    n = n / resized_width_fast_divisor;
    const int x = prev_id - n * resized_width;
    prev_id = n;
    n = n / resized_height_fast_divisor;
    const int b = n;
    const int y = prev_id - n * resized_height;

    const float original_y = y * height_scale;
    const int top_y_index = sycl::floor(original_y);
    const int bottom_y_index = (original_y < original_height - 1)
                                   ? sycl::ceil(original_y)
                                   : original_height - 1;
    const float y_lerp = original_y - top_y_index;

    const float original_x = x * width_scale;
    const int left_x_index = sycl::floor(original_x);
    const int right_x_index = (original_x < original_width - 1)
                                  ? sycl::ceil(original_x)
                                  : original_width - 1;
    const float x_lerp = original_x - left_x_index;

    const float dtop = (1 - y_lerp) * input_grad[in_idx];
    ItexAtomicAdd(output_grad +
                      ((b * original_height + top_y_index) * original_width +
                       left_x_index) *
                          channels +
                      c,
                  static_cast<T>((1 - x_lerp) * dtop));
    ItexAtomicAdd(output_grad +
                      ((b * original_height + top_y_index) * original_width +
                       right_x_index) *
                          channels +
                      c,
                  static_cast<T>(x_lerp * dtop));

    const float dbottom = y_lerp * input_grad[in_idx];
    ItexAtomicAdd(output_grad +
                      ((b * original_height + bottom_y_index) * original_width +
                       left_x_index) *
                          channels +
                      c,
                  static_cast<T>((1 - x_lerp) * dbottom));
    ItexAtomicAdd(output_grad +
                      ((b * original_height + bottom_y_index) * original_width +
                       right_x_index) *
                          channels +
                      c,
                  static_cast<T>(x_lerp * dbottom));
  }

 private:
  const int total_count;
  const float* input_grad;
  const float height_scale;
  const float width_scale;
  int batch;
  int original_height;
  int original_width;
  int channels;
  int resized_height;
  int resized_width;
  FastDivisor channels_fast_divisor;
  FastDivisor resized_height_fast_divisor;
  FastDivisor resized_width_fast_divisor;
  T* output_grad;
};

}  // namespace impl

// Partial specialization of ResizeBilinearGrad functor for a GPUDevice.
template <typename T>
struct ResizeBilinearGrad<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor input_grad,
                  const float height_scale, const float width_scale,
                  const bool half_pixel_centers,
                  typename TTypes<T, 4>::Tensor output_grad) {
    const int batch = output_grad.dimension(0);
    const int original_height = output_grad.dimension(1);
    const int original_width = output_grad.dimension(2);
    const int channels = output_grad.dimension(3);

    const int resized_height = input_grad.dimension(1);
    const int resized_width = input_grad.dimension(2);

    int total_count;
    total_count = batch * resized_height * resized_width * channels;
    if (total_count == 0) return;

#define EigenFastDivisor(divisor, num)                     \
  Eigen::internal::TensorIntDivisor<int> divisor;          \
  if (num != 0) {                                          \
    divisor = Eigen::internal::TensorIntDivisor<int>(num); \
  }

    EigenFastDivisor(channels_fast_divisor, channels);
    EigenFastDivisor(resized_width_fast_divisor, resized_width);
    EigenFastDivisor(resized_height_fast_divisor, resized_height);
#undef EigenFastDivisor

    // Initialize output_grad with all zeros.
    output_grad.device(d) = output_grad.constant(T(0));

    auto stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (total_count + group_size - 1) / group_size;

    if (half_pixel_centers) {
      stream->submit([&](sycl::handler& cgh) {
        impl::ResizeBilinearGradKernel<T> task(
            total_count, input_grad.data(), height_scale, width_scale, batch,
            original_height, original_width, channels, resized_height,
            resized_width, channels_fast_divisor, resized_height_fast_divisor,
            resized_width_fast_divisor, output_grad.data());

        cgh.parallel_for<ResizeBilinearGradKernelTask<T>>(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            task);
      });
    } else {
      stream->submit([&](sycl::handler& cgh) {
        impl::LegacyResizeBilinearGradKernel<T> task(
            total_count, input_grad.data(), height_scale, width_scale, batch,
            original_height, original_width, channels, resized_height,
            resized_width, channels_fast_divisor, resized_height_fast_divisor,
            resized_width_fast_divisor, output_grad.data());

        cgh.parallel_for<LegacyResizeBilinearGradKernelTask<T>>(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            task);
      });
    }
    return;
  }
};

#define DEFINE_GPU_GRAD(T) template struct ResizeBilinearGrad<GPUDevice, T>;

TF_CALL_float(DEFINE_GPU_GRAD);
#undef DEFINE_GPU_GRAD

}  // namespace functor

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear")      \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBilinearOp<GPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_bfloat16(REGISTER_KERNEL)
#ifdef ITEX_ENABLE_DOUBLE
    TF_CALL_double(REGISTER_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE

#define REGISTER_GRAD_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ResizeBilinearGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ResizeBilinearOpGrad<GPUDevice, T>);

TF_CALL_float(REGISTER_GRAD_KERNEL);
TF_CALL_half(REGISTER_GRAD_KERNEL);
TF_CALL_bfloat16(REGISTER_GRAD_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GRAD_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_KERNEL
#undef REGISTER_GRAD_KERNEL

}  // namespace itex
