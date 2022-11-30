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

#include "itex/core/kernels/gpu/resize_nearest_neighbor_op.h"

#include <algorithm>

#include "itex/core/kernels/gpu/image_resizer_state.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::internal::TensorIntDivisor<int> FastDivisor;

template <typename Device, typename T>
class ResizeNearestNeighborOp : public OpKernel {
 public:
  explicit ResizeNearestNeighborOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    OP_REQUIRES(context, st.in_height < (1 << 24) && st.in_width < (1 << 24),
                errors::InvalidArgument("nearest neighbor requires max height "
                                        "& width of 2^24"));

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor input_data(input.tensor<T, 4>());
    typename TTypes<T, 4>::Tensor output_data(st.output->tensor<T, 4>());

    bool status;
    if (half_pixel_centers_) {
      if (align_corners_) {
        status = functor::ResizeNearestNeighbor<Device, T,
                                                /*half_pixe_centers=*/true,
                                                /*align_corners=*/true>()(
            context->eigen_device<Device>(), input_data, st.height_scale,
            st.width_scale, output_data);
      } else {
        status = functor::ResizeNearestNeighbor<Device, T,
                                                /*half_pixe_centers=*/true,
                                                /*align_corners=*/false>()(
            context->eigen_device<Device>(), input_data, st.height_scale,
            st.width_scale, output_data);
      }
    } else {
      if (align_corners_) {
        status = functor::ResizeNearestNeighbor<Device, T,
                                                /*half_pixe_centers=*/false,
                                                /*align_corners=*/true>()(
            context->eigen_device<Device>(), input_data, st.height_scale,
            st.width_scale, output_data);
      } else {
        status = functor::ResizeNearestNeighbor<Device, T,
                                                /*half_pixe_centers=*/false,
                                                /*align_corners=*/false>()(
            context->eigen_device<Device>(), input_data, st.height_scale,
            st.width_scale, output_data);
      }
    }
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResizeNearestNeighbor"));
    }
  }

 private:
  bool align_corners_ = false;
  bool half_pixel_centers_ = false;
};

template <typename Device, typename T>
class ResizeNearestNeighborOpGrad : public OpKernel {
 public:
  explicit ResizeNearestNeighborOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab and validate the input:
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));

    // Grab and validate the output shape:
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));

    auto sizes = shape_t.vec<int32>();
    OP_REQUIRES(context, sizes(0) > 0 && sizes(1) > 0,
                errors::InvalidArgument("shape_t's elements must be positive"));

    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);

    const int64 out_height = sizes(0);
    const int64 out_width = sizes(1);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, out_height, out_width, channels}),
            &output));

    // Return if the output is empty.
    if (output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor input_data(input.tensor<T, 4>());
    typename TTypes<T, 4>::Tensor output_data(output->tensor<T, 4>());

    const float height_scale =
        CalculateResizeScale(out_height, in_height, align_corners_);
    const float width_scale =
        CalculateResizeScale(out_width, in_width, align_corners_);

    bool status;
    if (half_pixel_centers_) {
      if (align_corners_) {
        status = functor::ResizeNearestNeighborGrad<Device, T,
                                                    /*half_pixel_centers=*/true,
                                                    /*align_corners=*/true>()(
            context, context->eigen_device<Device>(), input_data, height_scale,
            width_scale, output_data);
      } else {
        status = functor::ResizeNearestNeighborGrad<Device, T,
                                                    /*half_pixel_centers=*/true,
                                                    /*align_corners=*/false>()(
            context, context->eigen_device<Device>(), input_data, height_scale,
            width_scale, output_data);
      }
    } else {
      if (align_corners_) {
        status =
            functor::ResizeNearestNeighborGrad<Device, T,
                                               /*half_pixel_centers=*/false,
                                               /*align_corners=*/true>()(
                context, context->eigen_device<Device>(), input_data,
                height_scale, width_scale, output_data);
      } else {
        status =
            functor::ResizeNearestNeighborGrad<Device, T,
                                               /*half_pixel_centers=*/false,
                                               /*align_corners=*/false>()(
                context, context->eigen_device<Device>(), input_data,
                height_scale, width_scale, output_data);
      }
    }
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResizeNearestNeighborGrad"));
    }
  }

 private:
  bool align_corners_ = false;
  bool half_pixel_centers_ = false;
};

template <typename T>
struct ResizeNearestNeighborNHWCTask {
  ResizeNearestNeighborNHWCTask(int total_count, const T* bottom_data,
                                int in_height, int in_width, int channels,
                                int out_height, int out_width,
                                FastDivisor channels_fast_divisor,
                                FastDivisor out_height_fast_divisor,
                                FastDivisor out_width_fast_divisor,
                                float height_scale, float width_scale,
                                int image_size, T* top_data)
      : total_count(total_count),
        bottom_data(bottom_data),
        in_height(in_height),
        in_width(in_width),
        channels(channels),
        out_height(out_height),
        out_width(out_width),
        channels_fast_divisor(channels_fast_divisor),
        out_height_fast_divisor(out_height_fast_divisor),
        out_width_fast_divisor(out_width_fast_divisor),
        height_scale(height_scale),
        width_scale(width_scale),
        image_size(image_size),
        top_data(top_data) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= total_count) return;
    int n = index;
    int prev_id = index;
    n = n / channels_fast_divisor;
    int c = prev_id - n * channels;
    prev_id = n;
    n = n / out_width_fast_divisor;
    int out_x = prev_id - n * out_width;
    prev_id = n;
    n = n / out_height_fast_divisor;
    int out_y = prev_id - n * out_height;

    const T* bottom_data_n = bottom_data + n * image_size;
    const int in_y = sycl::max(
        sycl::min(static_cast<int>(sycl::floor(
                      (static_cast<float>(out_y) + 0.5f) * height_scale)),
                  in_height - 1),
        0);
    const int in_x = sycl::max(
        sycl::min(static_cast<int>(sycl::floor(
                      (static_cast<float>(out_x) + 0.5f) * width_scale)),
                  in_width - 1),
        0);
    const int idx = (in_y * in_width + in_x) * channels + c;
    top_data[index] = *(bottom_data_n + idx);
  }

 private:
  int total_count;
  const T* bottom_data;
  int in_height;
  int in_width;
  int channels;
  int out_height;
  int out_width;
  FastDivisor channels_fast_divisor;
  FastDivisor out_height_fast_divisor;
  FastDivisor out_width_fast_divisor;
  float height_scale;
  float width_scale;
  int image_size;
  T* top_data;
};

template <typename T, bool align_corners>
struct LegacyResizeNearestNeighborNHWCTask {
  LegacyResizeNearestNeighborNHWCTask(int total_count, const T* bottom_data,
                                      int in_height, int in_width, int channels,
                                      int out_height, int out_width,
                                      FastDivisor channels_fast_divisor,
                                      FastDivisor out_height_fast_divisor,
                                      FastDivisor out_width_fast_divisor,
                                      float height_scale, float width_scale,
                                      int image_size, T* top_data)
      : total_count(total_count),
        bottom_data(bottom_data),
        in_height(in_height),
        in_width(in_width),
        channels(channels),
        out_height(out_height),
        out_width(out_width),
        channels_fast_divisor(channels_fast_divisor),
        out_height_fast_divisor(out_height_fast_divisor),
        out_width_fast_divisor(out_width_fast_divisor),
        height_scale(height_scale),
        width_scale(width_scale),
        image_size(image_size),
        top_data(top_data) {}

  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= total_count) return;
    int n = index;
    int prev_id = index;
    n = n / channels_fast_divisor;
    int c = prev_id - n * channels;
    prev_id = n;
    n = n / out_width_fast_divisor;
    int out_x = prev_id - n * out_width;
    prev_id = n;
    n = n / out_height_fast_divisor;
    int out_y = prev_id - n * out_height;

    const T* bottom_data_n = bottom_data + n * image_size;
    const int in_y = sycl::min(
        (align_corners) ? static_cast<int>(sycl::round(out_y * height_scale))
                        : static_cast<int>(sycl::floor(out_y * height_scale)),
        in_height - 1);
    const int in_x = sycl::min(
        (align_corners) ? static_cast<int>(sycl::round(out_x * width_scale))
                        : static_cast<int>(sycl::floor(out_x * width_scale)),
        in_width - 1);
    const int idx = (in_y * in_width + in_x) * channels + c;
    top_data[index] = *(bottom_data_n + idx);
  }

 private:
  int total_count;
  const T* bottom_data;
  int in_height;
  int in_width;
  int channels;
  int out_height;
  int out_width;
  FastDivisor channels_fast_divisor;
  FastDivisor out_height_fast_divisor;
  FastDivisor out_width_fast_divisor;
  float height_scale;
  float width_scale;
  int image_size;
  T* top_data;
};

template <typename T>
Status ResizeNearestNeighborNHWC(
    const GPUDevice& d, int total_count, const T* bottom_data, int in_height,
    int in_width, int channels, int out_height, int out_width,
    FastDivisor channels_fast_divisor, FastDivisor out_height_fast_divisor,
    FastDivisor out_width_fast_divisor, float height_scale, float width_scale,
    T* top_data) {
  auto stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (total_count + group_size - 1) / group_size;
  auto image_size = in_height * in_width * channels;
  stream->submit([&](sycl::handler& cgh) {
    ResizeNearestNeighborNHWCTask<T> task(
        total_count, bottom_data, in_height, in_width, channels, out_height,
        out_width, channels_fast_divisor, out_height_fast_divisor,
        out_width_fast_divisor, height_scale, width_scale, image_size,
        top_data);
    cgh.parallel_for<ResizeNearestNeighborNHWCTask<T>>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        task);
  });
  return Status::OK();
}

template <typename T, bool align_corners>
Status LegacyResizeNearestNeighborNHWC(
    const GPUDevice& d, int total_count, const T* bottom_data, int in_height,
    int in_width, int channels, int out_height, int out_width,
    FastDivisor channels_fast_divisor, FastDivisor out_height_fast_divisor,
    FastDivisor out_width_fast_divisor, float height_scale, float width_scale,
    T* top_data) {
  auto stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (total_count + group_size - 1) / group_size;
  auto image_size = in_height * in_width * channels;
  stream->submit([&](sycl::handler& cgh) {
    LegacyResizeNearestNeighborNHWCTask<T, align_corners> task(
        total_count, bottom_data, in_height, in_width, channels, out_height,
        out_width, channels_fast_divisor, out_height_fast_divisor,
        out_width_fast_divisor, height_scale, width_scale, image_size,
        top_data);
    cgh.parallel_for<LegacyResizeNearestNeighborNHWCTask<T, align_corners>>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        task);
  });
  return Status::OK();
}

template <typename T>
struct ResizeNearestNeighborGradKernelTask {
  ResizeNearestNeighborGradKernelTask(int input_size, const T* top_diff,
                                      int in_height, int in_width, int channels,
                                      int out_height, int out_width,
                                      FastDivisor channels_fast_divisor,
                                      FastDivisor in_height_fast_divisor,
                                      FastDivisor in_width_fast_divisor,
                                      float height_scale, float width_scale,
                                      int image_size, T* bottom_diff)
      : input_size(input_size),
        top_diff(top_diff),
        in_height(in_height),
        in_width(in_width),
        channels(channels),
        out_height(out_height),
        out_width(out_width),
        channels_fast_divisor(channels_fast_divisor),
        in_height_fast_divisor(in_height_fast_divisor),
        in_width_fast_divisor(in_width_fast_divisor),
        height_scale(height_scale),
        width_scale(width_scale),
        image_size(image_size),
        bottom_diff(bottom_diff) {}
  void operator()(sycl::nd_item<1> nd_item) const {
    auto id = nd_item.get_global_linear_id();
    if (id >= input_size) return;

    int n = id;
    int prev_id = id;
    n = n / channels_fast_divisor;
    int c = prev_id - n * channels;
    prev_id = n;
    n = n / in_width_fast_divisor;
    int in_x = prev_id - n * in_width;
    prev_id = n;
    n = n / in_height_fast_divisor;
    int in_y = prev_id - n * in_height;

    T* bottom_diff_n = bottom_diff + n * image_size;
    const int out_y = sycl::max(
        sycl::min(static_cast<int>(sycl::floor(
                      (static_cast<float>(in_y) + 0.5f) * height_scale)),
                  out_height - 1),
        0);
    const int out_x = sycl::max(
        sycl::min(static_cast<int>(sycl::floor(
                      (static_cast<float>(in_x) + 0.5f) * width_scale)),
                  out_width - 1),
        0);
    const int idx = (out_y * out_width + out_x) * channels + c;
    DpcppAtomicAdd(bottom_diff_n + idx, top_diff[id]);
  }

 private:
  int input_size;
  const T* top_diff;
  int in_height;
  int in_width;
  int channels;
  int out_height;
  int out_width;
  FastDivisor channels_fast_divisor;
  FastDivisor in_height_fast_divisor;
  FastDivisor in_width_fast_divisor;
  float height_scale;
  float width_scale;
  int image_size;
  T* bottom_diff;
};
template <typename T, bool align_corners>
struct LegacyResizeNearestNeighborGradKernelTask {
  LegacyResizeNearestNeighborGradKernelTask(
      int input_size, const T* top_diff, int in_height, int in_width,
      int channels, int out_height, int out_width,
      FastDivisor channels_fast_divisor, FastDivisor in_height_fast_divisor,
      FastDivisor in_width_fast_divisor, float height_scale, float width_scale,
      int image_size, T* bottom_diff)
      : input_size(input_size),
        top_diff(top_diff),
        in_height(in_height),
        in_width(in_width),
        channels(channels),
        out_height(out_height),
        out_width(out_width),
        channels_fast_divisor(channels_fast_divisor),
        in_height_fast_divisor(in_height_fast_divisor),
        in_width_fast_divisor(in_width_fast_divisor),
        height_scale(height_scale),
        width_scale(width_scale),
        image_size(image_size),
        bottom_diff(bottom_diff) {}
  void operator()(sycl::nd_item<1> nd_item) const {
    auto id = nd_item.get_global_linear_id();
    if (id >= input_size) return;

    int n = id;
    int prev_id = id;
    n = n / channels_fast_divisor;
    int c = prev_id - n * channels;
    prev_id = n;
    n = n / in_width_fast_divisor;
    int in_x = prev_id - n * in_width;
    prev_id = n;
    n = n / in_height_fast_divisor;
    int in_y = prev_id - n * in_height;

    T* bottom_diff_n = bottom_diff + n * image_size;
    const int out_y = sycl::min(
        (align_corners) ? static_cast<int>(sycl::round(in_y * height_scale))
                        : static_cast<int>(sycl::floor(in_y * height_scale)),
        out_height - 1);
    const int out_x = sycl::min(
        (align_corners) ? static_cast<int>(sycl::round(in_x * width_scale))
                        : static_cast<int>(sycl::floor(in_x * width_scale)),
        out_width - 1);
    const int idx = (out_y * out_width + out_x) * channels + c;
    DpcppAtomicAdd(bottom_diff_n + idx, top_diff[id]);
  }

 private:
  int input_size;
  const T* top_diff;
  int in_height;
  int in_width;
  int channels;
  int out_height;
  int out_width;
  FastDivisor channels_fast_divisor;
  FastDivisor in_height_fast_divisor;
  FastDivisor in_width_fast_divisor;
  float height_scale;
  float width_scale;
  int image_size;
  T* bottom_diff;
};

template <typename T>
Status ResizeNearestNeighborGradNHWCKernel(
    const GPUDevice& d, int input_size, const T* top_diff, int in_height,
    int in_width, int channels, int out_height, int out_width,
    FastDivisor channels_fast_divisor, FastDivisor in_height_fast_divisor,
    FastDivisor in_width_fast_divisor, float height_scale, float width_scale,
    T* bottom_diff) {
  auto& stream = d.stream();
  auto wg_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_group = (input_size + wg_size - 1) / wg_size;
  auto image_size = out_height * out_width * channels;
  stream->submit([&](sycl::handler& cgh) {
    ResizeNearestNeighborGradKernelTask<T> task(
        input_size, top_diff, in_height, in_width, channels, out_height,
        out_width, channels_fast_divisor, in_height_fast_divisor,
        in_width_fast_divisor, height_scale, width_scale, image_size,
        bottom_diff);

    cgh.parallel_for<ResizeNearestNeighborGradKernelTask<T>>(
        sycl::nd_range<1>(sycl::range<1>(wg_size * num_work_group),
                          sycl::range<1>(wg_size)),
        task);
  });
  return Status::OK();
}

template <typename T, bool align_corners>
Status LegacyResizeNearestNeighborGradNHWCKernel(
    const GPUDevice& d, int input_size, const T* top_diff, int in_height,
    int in_width, int channels, int out_height, int out_width,
    FastDivisor channels_fast_divisor, FastDivisor in_height_fast_divisor,
    FastDivisor in_width_fast_divisor, float height_scale, float width_scale,
    T* bottom_diff) {
  auto& stream = d.stream();
  auto wg_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_work_group = (input_size + wg_size - 1) / wg_size;
  auto image_size = out_height * out_width * channels;
  stream->submit([&](sycl::handler& cgh) {
    LegacyResizeNearestNeighborGradKernelTask<T, align_corners> task(
        input_size, top_diff, in_height, in_width, channels, out_height,
        out_width, channels_fast_divisor, in_height_fast_divisor,
        in_width_fast_divisor, height_scale, width_scale, image_size,
        bottom_diff);

    cgh.parallel_for<
        LegacyResizeNearestNeighborGradKernelTask<T, align_corners>>(
        sycl::nd_range<1>(sycl::range<1>(wg_size * num_work_group),
                          sycl::range<1>(wg_size)),
        task);
  });
  return Status::OK();
}

namespace functor {
template <typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighbor<GPUDevice, T, half_pixel_centers, align_corners> {
  bool operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int64 in_height = input.dimension(1);
    const int64 in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    const int output_size = batch_size * out_height * out_width * channels;
    if (output_size == 0) return true;

#define EigenFastDivisor(divisor, num)                     \
  Eigen::internal::TensorIntDivisor<int> divisor;          \
  if (num != 0) {                                          \
    divisor = Eigen::internal::TensorIntDivisor<int>(num); \
  }

    EigenFastDivisor(channels_fast_divisor, channels);
    EigenFastDivisor(out_width_fast_divisor, out_width);
    EigenFastDivisor(out_height_fast_divisor, out_height);
#undef EigenFastDivisor

    if (half_pixel_centers) {
      auto status = ResizeNearestNeighborNHWC<T>(
          d, output_size, input.data(), in_height, in_width, channels,
          out_height, out_width, channels_fast_divisor, out_height_fast_divisor,
          out_width_fast_divisor, height_scale, width_scale, output.data());
      return true;
    } else {
      auto status = LegacyResizeNearestNeighborNHWC<T, align_corners>(
          d, output_size, input.data(), in_height, in_width, channels,
          out_height, out_width, channels_fast_divisor, out_height_fast_divisor,
          out_width_fast_divisor, height_scale, width_scale, output.data());
      return true;
    }
  }
};

#define DECLARE_GPU_SPEC(T)                                          \
  template struct ResizeNearestNeighbor<GPUDevice, T, false, false>; \
  template struct ResizeNearestNeighbor<GPUDevice, T, false, true>;  \
  template struct ResizeNearestNeighbor<GPUDevice, T, true, false>;  \
  template struct ResizeNearestNeighbor<GPUDevice, T, true, true>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DECLARE_GPU_SPEC);
#endif  // ITEX_ENABLE_DOUBLE
#undef DECLARE_GPU_SPEC

template <typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighborGrad<
    GPUDevice, T, half_pixel_centers, align_corners,
    typename std::enable_if<std::is_same<T, Eigen::bfloat16>::value ||
                            std::is_same<T, Eigen::half>::value>::type> {
  bool operator()(OpKernelContext* ctx, const GPUDevice& d,
                  typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int64 in_height = input.dimension(1);
    const int64 in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    const int input_size = batch_size * channels * in_height * in_width;
    if (input_size == 0) return true;

#define EigenFastDivisor(divisor, num)                     \
  Eigen::internal::TensorIntDivisor<int> divisor;          \
  if (num != 0) {                                          \
    divisor = Eigen::internal::TensorIntDivisor<int>(num); \
  }

    EigenFastDivisor(channels_fast_divisor, channels);
    EigenFastDivisor(in_width_fast_divisor, in_width);
    EigenFastDivisor(in_height_fast_divisor, in_height);
#undef EigenFastDivisor

    // SYCL does not support atomic operations for bf16 and half data type
    Tensor input_fp32, output_fp32;
    TF_ABORT_IF_ERROR(ctx->allocate_temp(
        DataTypeToEnum<float>::v(),
        TensorShape({batch_size, in_height, in_width, channels}), &input_fp32));
    input_fp32.tensor<float, 4>().device(d) = input.template cast<float>();
    TF_ABORT_IF_ERROR(ctx->allocate_temp(
        DataTypeToEnum<float>::v(),
        TensorShape({batch_size, out_height, out_width, channels}),
        &output_fp32));
    output_fp32.tensor<float, 4>().device(d) =
        output_fp32.tensor<float, 4>().constant(0.0f);
    if (half_pixel_centers) {
      auto status = ResizeNearestNeighborGradNHWCKernel<float>(
          d, input_size, static_cast<float*>(input_fp32.data()), in_height,
          in_width, channels, out_height, out_width, channels_fast_divisor,
          in_height_fast_divisor, in_width_fast_divisor, height_scale,
          width_scale, static_cast<float*>(output_fp32.data()));
    } else {
      auto status =
          LegacyResizeNearestNeighborGradNHWCKernel<float, align_corners>(
              d, input_size, static_cast<float*>(input_fp32.data()), in_height,
              in_width, channels, out_height, out_width, channels_fast_divisor,
              in_height_fast_divisor, in_width_fast_divisor, height_scale,
              width_scale, static_cast<float*>(output_fp32.data()));
    }

    output.device(d) = output_fp32.tensor<float, 4>().template cast<T>();

    return true;
  }
};

template <typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighborGrad<
    GPUDevice, T, half_pixel_centers, align_corners,
    typename std::enable_if<!std::is_same<T, Eigen::bfloat16>::value &&
                            !std::is_same<T, Eigen::half>::value>::type> {
  bool operator()(OpKernelContext* ctx, const GPUDevice& d,
                  typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int64 in_height = input.dimension(1);
    const int64 in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    output.device(d) = output.constant(T(0));

    const int input_size = batch_size * channels * in_height * in_width;
    if (input_size == 0) return true;

#define EigenFastDivisor(divisor, num)                     \
  Eigen::internal::TensorIntDivisor<int> divisor;          \
  if (num != 0) {                                          \
    divisor = Eigen::internal::TensorIntDivisor<int>(num); \
  }

    EigenFastDivisor(channels_fast_divisor, channels);
    EigenFastDivisor(in_width_fast_divisor, in_width);
    EigenFastDivisor(in_height_fast_divisor, in_height);
#undef EigenFastDivisor

    if (half_pixel_centers) {
      auto status = ResizeNearestNeighborGradNHWCKernel<T>(
          d, input_size, input.data(), in_height, in_width, channels,
          out_height, out_width, channels_fast_divisor, in_height_fast_divisor,
          in_width_fast_divisor, height_scale, width_scale, output.data());
    } else {
      auto status = LegacyResizeNearestNeighborGradNHWCKernel<T, align_corners>(
          d, input_size, input.data(), in_height, in_width, channels,
          out_height, out_width, channels_fast_divisor, in_height_fast_divisor,
          in_width_fast_divisor, height_scale, width_scale, output.data());
    }

    return true;
  }
};

#define DECLARE_GPU_SPEC(T)                                              \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, false, false>; \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, false, true>;  \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, true, false>;  \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, true, true>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DECLARE_GPU_SPEC);
#endif  // ITEX_ENABLE_DOUBLE
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_KERNEL(T)                              \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighbor") \
                              .Device(DEVICE_GPU)       \
                              .TypeConstraint<T>("T")   \
                              .HostMemory("size"),      \
                          ResizeNearestNeighborOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                             \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighborGrad") \
                              .Device(DEVICE_GPU)           \
                              .TypeConstraint<T>("T")       \
                              .HostMemory("size"),          \
                          ResizeNearestNeighborOpGrad<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GRAD_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GRAD_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GRAD_KERNEL
}  // namespace itex
