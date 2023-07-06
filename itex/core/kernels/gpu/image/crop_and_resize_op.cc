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

#include "itex/core/kernels/gpu/image/crop_and_resize_op.h"

#include <string>
#include <utility>

#include "itex/core/kernels/common/cast_op.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;
using Callback = std::function<void()>;

namespace {
enum InterpolationMethod {
  BILINEAR = 0,
  NEAREST = 1,
};

static inline Status ParseAndCheckBoxSizes(const Tensor& boxes,
                                           const Tensor& box_index,
                                           int* num_boxes) {
  if (boxes.NumElements() == 0 && box_index.NumElements() == 0) {
    *num_boxes = 0;
    return Status::OK();
  }
  // The shape of 'boxes' is [num_boxes, 4].
  if (boxes.dims() != 2) {
    return errors::InvalidArgument("boxes must be 2-D",
                                   boxes.shape().DebugString());
  }
  *num_boxes = boxes.dim_size(0);
  if (boxes.dim_size(1) != 4) {
    return errors::InvalidArgument("boxes must have 4 columns");
  }
  // The shape of 'box_index' is [num_boxes].
  if (box_index.dims() != 1) {
    return errors::InvalidArgument("box_index must be 1-D",
                                   box_index.shape().DebugString());
  }
  if (box_index.dim_size(0) != *num_boxes) {
    return errors::InvalidArgument("box_index has incompatible shape");
  }
  return Status::OK();
}

// Conditionally calls the compute callback if all values in box_index are in
// [0, batch_size) then calls done.
template <typename Device>
inline void RunIfBoxIndexIsValid(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute);

// Specialization of CheckValidBoxIndex for a GPUDevice.
template <>
inline void RunIfBoxIndexIsValid<GPUDevice>(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute) {
  const int num_boxes = box_index.dimension(0);
  if (num_boxes == 0) {
    compute();
    return;
  }

  Tensor isvalid_dev_tensor;
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<bool>::value,
                                        TensorShape({}), &isvalid_dev_tensor));
  typename TTypes<bool, 0>::Tensor isvalid_dev =
      isvalid_dev_tensor.tensor<bool, 0>();

  const GPUDevice& d = context->eigen_device<GPUDevice>();
  // Checks if all values in box_index are in [0, batch).
  isvalid_dev.device(d) = ((box_index >= 0) && (box_index < batch_size)).all();
  // Copy the result back to the host.
  auto stream = d.stream();

  OP_REQUIRES(context, &stream, errors::Internal("No GPU stream available."));
  Tensor isvalid_host_tensor;

  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  OP_REQUIRES_OK(context, context->allocate_temp(
                              DataTypeToEnum<bool>::value, TensorShape({}),
                              &isvalid_host_tensor, alloc_attr));

  d.memcpyDeviceToHost(isvalid_host_tensor.scalar<bool>().data(),
                       isvalid_dev.data(), sizeof(bool));

  compute();
}
}  // namespace

template <typename Device, typename T>
class CropAndResizeOp : public OpKernel {
 public:
  explicit CropAndResizeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
    OP_REQUIRES(context, method_ == "bilinear" || method_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'bilinear' or 'nearest'", method_));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolation_value",
                                             &extrapolation_value_));
  }

  void Compute(OpKernelContext* context) override {
    // The shape of 'image' is [batch_size, image_height, image_width,
    // channels].
    const Tensor& image = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(1);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(2);
    // The shape of 'crop_size' is [2].
    const Tensor& crop_size = context->input(3);

    // Validate inputs dimensions.
    OP_REQUIRES(context, image.dims() == 4,
                errors::InvalidArgument("input image must be 4-D",
                                        image.shape().DebugString()));
    const int batch_size = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    const int depth = image.dim_size(3);
    OP_REQUIRES(context, image_height > 0 && image_width > 0,
                errors::InvalidArgument("image dimensions must be positive"));
    int num_boxes = 0;
    OP_REQUIRES_OK(context,
                   ParseAndCheckBoxSizes(boxes, box_index, &num_boxes));

    OP_REQUIRES(context, crop_size.dims() == 1,
                errors::InvalidArgument("crop_size must be 1-D",
                                        crop_size.shape().DebugString()));
    OP_REQUIRES(context, crop_size.dim_size(0) == 2,
                errors::InvalidArgument("crop_size must have two elements",
                                        crop_size.shape().DebugString()));

    // Copy and validate crop sizes.
    auto crop_size_vec = crop_size.vec<int32>();
    const int crop_height = internal::SubtleMustCopy(crop_size_vec(0));
    const int crop_width = internal::SubtleMustCopy(crop_size_vec(1));
    OP_REQUIRES(context, crop_height > 0 && crop_width > 0,
                errors::InvalidArgument("crop dimensions must be positive"));

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({num_boxes, crop_height, crop_width, depth}),
            &output));

    auto compute_callback = [this, context, output]() {
      const Tensor& image = context->input(0);
      const Tensor& boxes = context->input(1);
      const Tensor& box_index = context->input(2);
      const bool status = functor::CropAndResize<Device, T>()(
          context, image.tensor<T, 4>(), boxes.tensor<float, 2>(),
          box_index.tensor<int32, 1>(), method_, extrapolation_value_,
          output->tensor<float, 4>());

      if (!status) {
        context->SetStatus(
            errors::Internal("Failed to launch CropAndResizeKernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback));
  }

 private:
  float extrapolation_value_;
  string method_;
};

template <typename Device, typename T>
class CropAndResizeGradImageOp : public OpKernel {
 public:
  explicit CropAndResizeGradImageOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
    OP_REQUIRES(context, method_ == "bilinear" || method_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'bilinear' or 'nearest'", method_));
  }

  void Compute(OpKernelContext* context) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(1);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(2);
    // The shape of 'image_size' is [4].
    const Tensor& image_size = context->input(3);

    // Validate input shapes.
    OP_REQUIRES(context, grads.dims() == 4,
                errors::InvalidArgument("grads image must be 4-D",
                                        grads.shape().DebugString()));
    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    OP_REQUIRES(context, crop_height > 0 && crop_width > 0,
                errors::InvalidArgument("grads dimensions must be positive"));
    int num_boxes = 0;
    OP_REQUIRES_OK(context,
                   ParseAndCheckBoxSizes(boxes, box_index, &num_boxes));
    OP_REQUIRES(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"));

    OP_REQUIRES(context, image_size.dims() == 1,
                errors::InvalidArgument("image_size must be 1-D",
                                        image_size.shape().DebugString()));
    OP_REQUIRES(context, image_size.dim_size(0) == 4,
                errors::InvalidArgument("image_size must have 4 elements",
                                        image_size.shape().DebugString()));
    auto image_size_vec = image_size.vec<int32>();
    const int batch_size = internal::SubtleMustCopy(image_size_vec(0));
    const int image_height = internal::SubtleMustCopy(image_size_vec(1));
    const int image_width = internal::SubtleMustCopy(image_size_vec(2));
    const int depth = internal::SubtleMustCopy(image_size_vec(3));
    OP_REQUIRES(context, image_height > 0 && image_width > 0,
                errors::InvalidArgument("image dimensions must be positive"));
    OP_REQUIRES(
        context, grads.dim_size(3) == depth,
        errors::InvalidArgument("image_size and grads are incompatible"));

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, image_height, image_width, depth}),
            &output));

    auto compute_callback = [this, context, output]() {
      const Tensor& grads = context->input(0);
      const Tensor& boxes = context->input(1);
      const Tensor& box_index = context->input(2);
      const Tensor& image_size = context->input(3);

      auto image_size_vec = image_size.vec<int32>();
      const int batch_size = internal::SubtleMustCopy(image_size_vec(0));
      const int image_height = internal::SubtleMustCopy(image_size_vec(1));
      const int image_width = internal::SubtleMustCopy(image_size_vec(2));
      const int depth = internal::SubtleMustCopy(image_size_vec(3));

      bool status = false;

      if (std::is_same<T, float>::value) {
        status = functor::CropAndResizeBackpropImage<Device, float>()(
            context, grads.tensor<float, 4>(), boxes.tensor<float, 2>(),
            box_index.tensor<int32, 1>(), output->tensor<float, 4>(), method_);

      } else if (std::is_same<T, Eigen::half>::value) {
        Tensor output_tmp;
        OP_REQUIRES_OK(
            context,
            context->allocate_temp(
                DT_FLOAT,
                TensorShape({batch_size, image_height, image_width, depth}),
                &output_tmp));

        status = functor::CropAndResizeBackpropImage<Device, float>()(
            context, grads.tensor<float, 4>(), boxes.tensor<float, 2>(),
            box_index.tensor<int32, 1>(), output_tmp.tensor<float, 4>(),
            method_);

        const Tensor& output_tmp_const = output_tmp;
        functor::CastFunctor<GPUDevice, Eigen::half, float> cast;
        cast(context->eigen_gpu_device(), output->template flat<Eigen::half>(),
             output_tmp_const.template flat<float>());
      } else if (std::is_same<T, Eigen::bfloat16>::value) {
        Tensor output_tmp;
        OP_REQUIRES_OK(
            context,
            context->allocate_temp(
                DT_FLOAT,
                TensorShape({batch_size, image_height, image_width, depth}),
                &output_tmp));

        status = functor::CropAndResizeBackpropImage<Device, float>()(
            context, grads.tensor<float, 4>(), boxes.tensor<float, 2>(),
            box_index.tensor<int32, 1>(), output_tmp.tensor<float, 4>(),
            method_);

        const Tensor& output_tmp_const = output_tmp;
        functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast;
        cast(context->eigen_gpu_device(),
             output->template flat<Eigen::bfloat16>(),
             output_tmp_const.template flat<float>());
      }

      if (!status) {
        context->SetStatus(errors::Internal(
            "Failed to launch CropAndResizeBackpropImage kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback));
  }

 private:
  string method_;
};

template <typename Device, typename T>
class CropAndResizeGradBoxesOp : public OpKernel {
 public:
  explicit CropAndResizeGradBoxesOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "bilinear",
                errors::InvalidArgument("method must be 'bilinear'", method));
  }

  void Compute(OpKernelContext* context) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(2);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(3);
    // The shape of 'image' is [batch_size, image_height, image_width, depth].
    const Tensor& image = context->input(1);

    // Validate input shapes.
    OP_REQUIRES(context, grads.dims() == 4,
                errors::InvalidArgument("grads image must be 4-D",
                                        grads.shape().DebugString()));
    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    const int depth = grads.dim_size(3);
    OP_REQUIRES(context, crop_height > 0 && crop_width > 0,
                errors::InvalidArgument("grads dimensions must be positive"));

    OP_REQUIRES(context, image.dims() == 4,
                errors::InvalidArgument("input image must be 4-D",
                                        image.shape().DebugString()));
    const int batch_size = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    OP_REQUIRES(context, image_height > 0 && image_width > 0,
                errors::InvalidArgument("image dimensions must be positive"));
    OP_REQUIRES(context, image.dim_size(3) == depth,
                errors::InvalidArgument("image, grads depth differ"));

    int num_boxes = 0;
    OP_REQUIRES_OK(context,
                   ParseAndCheckBoxSizes(boxes, box_index, &num_boxes));

    OP_REQUIRES(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"));

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_boxes, 4}), &output));

    auto compute_callback = [context, output]() {
      const Tensor& grads = context->input(0);
      const Tensor& image = context->input(1);
      const Tensor& boxes = context->input(2);
      const Tensor& box_index = context->input(3);
      const bool status = functor::CropAndResizeBackpropBoxes<Device, T>()(
          context->eigen_device<Device>(), grads.tensor<float, 4>(),
          image.tensor<T, 4>(), boxes.tensor<float, 2>(),
          box_index.tensor<int32, 1>(), output->tensor<float, 2>());
      if (!status) {
        context->SetStatus(errors::Internal(
            "Failed to launch CropAndResizeBackpropBoxes kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback));
  }
};

template <typename T>
struct CropAndResizeKernelTask {
  CropAndResizeKernelTask(int total_count, const T* image_ptr,
                          const float* boxes_ptr, const int32* box_ind_ptr,
                          int num_boxes, int batch, int image_height,
                          int image_width, int crop_height, int crop_width,
                          int depth, int method_id, float extrapolation_value,
                          float* crops_ptr)
      : total_count(total_count),
        image_ptr(image_ptr),
        boxes_ptr(boxes_ptr),
        box_ind_ptr(box_ind_ptr),
        num_boxes(num_boxes),
        batch(batch),
        image_height(image_height),
        image_width(image_width),
        crop_height(crop_height),
        crop_width(crop_width),
        depth(depth),
        method_id(method_id),
        extrapolation_value(extrapolation_value),
        crops_ptr(crops_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    // out_idx = d + depth * (w + crop_width * (h + crop_height * b))
    auto out_idx = item.get_global_linear_id();
    if (out_idx >= total_count) {
      return;
    }
    int idx = out_idx;
    const int d = idx % depth;
    idx /= depth;
    const int x = idx % crop_width;
    idx /= crop_width;
    const int y = idx % crop_height;
    const int b = idx / crop_height;

    const float y1 = boxes_ptr[b * 4];
    const float x1 = boxes_ptr[b * 4 + 1];
    const float y2 = boxes_ptr[b * 4 + 2];
    const float x2 = boxes_ptr[b * 4 + 3];

    const int32 b_in = box_ind_ptr[b];
    if (b_in < 0 || b_in >= batch) {
      return;
    }

    const float height_scale =
        (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                          : 0;
    const float width_scale =
        (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

    const float in_y = (crop_height > 1)
                           ? y1 * (image_height - 1) + y * height_scale
                           : 0.5f * (y1 + y2) * (image_height - 1);
    if (in_y < 0 || in_y > image_height - 1) {
      crops_ptr[out_idx] = extrapolation_value;
      return;
    }

    const float in_x = (crop_width > 1)
                           ? x1 * (image_width - 1) + x * width_scale
                           : 0.5f * (x1 + x2) * (image_width - 1);
    if (in_x < 0 || in_x > image_width - 1) {
      crops_ptr[out_idx] = extrapolation_value;
      return;
    }

    if (method_id == BILINEAR) {
      const int top_y_index = sycl::floor(in_y);
      const int bottom_y_index = sycl::ceil(in_y);
      const float y_lerp = in_y - top_y_index;

      const int left_x_index = sycl::floor(in_x);
      const int right_x_index = sycl::ceil(in_x);
      const float x_lerp = in_x - left_x_index;

      const float top_left(static_cast<float>(
          image_ptr[((b_in * image_height + top_y_index) * image_width +
                     left_x_index) *
                        depth +
                    d]));
      const float top_right(static_cast<float>(
          image_ptr[((b_in * image_height + top_y_index) * image_width +
                     right_x_index) *
                        depth +
                    d]));
      const float bottom_left(static_cast<float>(
          image_ptr[((b_in * image_height + bottom_y_index) * image_width +
                     left_x_index) *
                        depth +
                    d]));
      const float bottom_right(static_cast<float>(
          image_ptr[((b_in * image_height + bottom_y_index) * image_width +
                     right_x_index) *
                        depth +
                    d]));
      const float top = top_left + (top_right - top_left) * x_lerp;
      const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
      crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    } else {  // method_id == kMethodNearestId
      const int closest_x_index = sycl::round(in_x);
      const int closest_y_index = sycl::round(in_y);
      crops_ptr[out_idx] = static_cast<float>(
          image_ptr[((b_in * image_height + closest_y_index) * image_width +
                     closest_x_index) *
                        depth +
                    d]);
    }
  }

 private:
  int total_count;
  const T* image_ptr;
  const float* boxes_ptr;
  const int32* box_ind_ptr;
  int num_boxes;
  int batch;
  int image_height;
  int image_width;
  int crop_height;
  int crop_width;
  int depth;
  int method_id;
  float extrapolation_value;
  float* crops_ptr;
};
template <typename T>
struct CropAndResizeBackpropImageKernelTask {
  CropAndResizeBackpropImageKernelTask(int total_count, const float* grads_ptr,
                                       const float* boxes_ptr,
                                       const int32* box_ind_ptr, int num_boxes,
                                       int batch, int image_height,
                                       int image_width, int crop_height,
                                       int crop_width, int depth,
                                       T* grads_image_ptr, int method_id)
      : total_count(total_count),
        grads_ptr(grads_ptr),
        boxes_ptr(boxes_ptr),
        box_ind_ptr(box_ind_ptr),
        num_boxes(num_boxes),
        batch(batch),
        image_height(image_height),
        image_width(image_width),
        crop_height(crop_height),
        crop_width(crop_width),
        depth(depth),
        grads_image_ptr(grads_image_ptr),
        method_id(method_id) {}
  void operator()(sycl::nd_item<1> item) const {
    auto out_idx = item.get_global_linear_id();

    if (out_idx >= total_count) {
      return;
    }

    // out_idx = d + depth * (w + crop_width * (h + crop_height * b))
    int idx = out_idx;
    const int d = idx % depth;
    idx /= depth;
    const int x = idx % crop_width;
    idx /= crop_width;
    const int y = idx % crop_height;
    const int b = idx / crop_height;

    const float y1 = boxes_ptr[b * 4];
    const float x1 = boxes_ptr[b * 4 + 1];
    const float y2 = boxes_ptr[b * 4 + 2];
    const float x2 = boxes_ptr[b * 4 + 3];

    const int32 b_in = box_ind_ptr[b];
    if (b_in < 0 || b_in >= batch) {
      return;
    }

    const float height_scale =
        (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                          : 0;
    const float width_scale =
        (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

    const float in_y = (crop_height > 1)
                           ? y1 * (image_height - 1) + y * height_scale
                           : 0.5 * (y1 + y2) * (image_height - 1);
    if (in_y < 0 || in_y > image_height - 1) {
      return;
    }

    const float in_x = (crop_width > 1)
                           ? x1 * (image_width - 1) + x * width_scale
                           : 0.5 * (x1 + x2) * (image_width - 1);
    if (in_x < 0 || in_x > image_width - 1) {
      return;
    }

    if (method_id == BILINEAR) {
      const int top_y_index = sycl::floor(in_y);
      const int bottom_y_index = sycl::ceil(in_y);
      const float y_lerp = in_y - top_y_index;

      const int left_x_index = sycl::floor(in_x);
      const int right_x_index = sycl::ceil(in_x);
      const float x_lerp = in_x - left_x_index;

      const float dtop = (1 - y_lerp) * grads_ptr[out_idx];
      ItexAtomicAdd(grads_image_ptr +
                        ((b_in * image_height + top_y_index) * image_width +
                         left_x_index) *
                            depth +
                        d,
                    static_cast<T>((1 - x_lerp) * dtop));
      ItexAtomicAdd(grads_image_ptr +
                        ((b_in * image_height + top_y_index) * image_width +
                         right_x_index) *
                            depth +
                        d,
                    static_cast<T>(x_lerp * dtop));

      const float dbottom = y_lerp * grads_ptr[out_idx];
      ItexAtomicAdd(grads_image_ptr +
                        ((b_in * image_height + bottom_y_index) * image_width +
                         left_x_index) *
                            depth +
                        d,
                    static_cast<T>((1 - x_lerp) * dbottom));
      ItexAtomicAdd(grads_image_ptr +
                        ((b_in * image_height + bottom_y_index) * image_width +
                         right_x_index) *
                            depth +
                        d,
                    static_cast<T>(x_lerp * dbottom));
    } else {  // method_id == NEAREST
      const int closest_x_index = sycl::round(in_x);
      const int closest_y_index = sycl::round(in_y);
      ItexAtomicAdd(grads_image_ptr +
                        ((b_in * image_height + closest_y_index) * image_width +
                         closest_x_index) *
                            depth +
                        d,
                    static_cast<T>(grads_ptr[out_idx]));
    }
  }

 private:
  int total_count;
  const float* grads_ptr;
  const float* boxes_ptr;
  const int32* box_ind_ptr;
  int num_boxes;
  int batch;
  int image_height;
  int image_width;
  int crop_height;
  int crop_width;
  int depth;
  T* grads_image_ptr;
  int method_id;
};

template <typename T>
struct CropAndResizeBackpropBoxesKernelTask {
  CropAndResizeBackpropBoxesKernelTask(
      int total_count, const float* grads_ptr, const T* image_ptr,
      const float* boxes_ptr, const int32* box_ind_ptr, int num_boxes,
      int batch, int image_height, int image_width, int crop_height,
      int crop_width, int depth, float* grads_boxes_ptr)
      : total_count(total_count),
        grads_ptr(grads_ptr),
        image_ptr(image_ptr),
        boxes_ptr(boxes_ptr),
        box_ind_ptr(box_ind_ptr),
        num_boxes(num_boxes),
        batch(batch),
        image_height(image_height),
        image_width(image_width),
        crop_height(crop_height),
        crop_width(crop_width),
        depth(depth),
        grads_boxes_ptr(grads_boxes_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto out_idx = item.get_global_linear_id();

    if (out_idx >= total_count) {
      return;
    }
    // out_idx = d + depth * (w + crop_width * (h + crop_height * b))
    int idx = out_idx;
    const int d = idx % depth;
    idx /= depth;
    const int x = idx % crop_width;
    idx /= crop_width;
    const int y = idx % crop_height;
    const int b = idx / crop_height;

    const float y1 = boxes_ptr[b * 4];
    const float x1 = boxes_ptr[b * 4 + 1];
    const float y2 = boxes_ptr[b * 4 + 2];
    const float x2 = boxes_ptr[b * 4 + 3];

    const int32 b_in = box_ind_ptr[b];
    if (b_in < 0 || b_in >= batch) {
      return;
    }

    const float height_ratio =
        (crop_height > 1)
            ? static_cast<float>(image_height - 1) / (crop_height - 1)
            : 0;
    const float width_ratio =
        (crop_width > 1)
            ? static_cast<float>(image_width - 1) / (crop_width - 1)
            : 0;

    const float height_scale = (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;

    const float in_y = (crop_height > 1)
                           ? y1 * (image_height - 1) + y * height_scale
                           : 0.5 * (y1 + y2) * (image_height - 1);
    if (in_y < 0 || in_y > image_height - 1) {
      return;
    }

    const float in_x = (crop_width > 1)
                           ? x1 * (image_width - 1) + x * width_scale
                           : 0.5 * (x1 + x2) * (image_width - 1);
    if (in_x < 0 || in_x > image_width - 1) {
      return;
    }

    const int top_y_index = sycl::floor(in_y);
    const int bottom_y_index = sycl::ceil(in_y);
    const float y_lerp = in_y - top_y_index;

    const int left_x_index = sycl::floor(in_x);
    const int right_x_index = sycl::ceil(in_x);
    const float x_lerp = in_x - left_x_index;

    const float top_left(static_cast<float>(
        image_ptr[((b_in * image_height + top_y_index) * image_width +
                   left_x_index) *
                      depth +
                  d]));
    const float top_right(static_cast<float>(
        image_ptr[((b_in * image_height + top_y_index) * image_width +
                   right_x_index) *
                      depth +
                  d]));
    const float bottom_left(static_cast<float>(
        image_ptr[((b_in * image_height + bottom_y_index) * image_width +
                   left_x_index) *
                      depth +
                  d]));
    const float bottom_right(static_cast<float>(
        image_ptr[((b_in * image_height + bottom_y_index) * image_width +
                   right_x_index) *
                      depth +
                  d]));

    // Compute the image gradient.
    float image_grad_y = (1 - x_lerp) * (bottom_left - top_left) +
                         x_lerp * (bottom_right - top_right);
    float image_grad_x = (1 - y_lerp) * (top_right - top_left) +
                         y_lerp * (bottom_right - bottom_left);
    // Modulate the image gradient with the incoming gradient.
    const float top_grad = grads_ptr[out_idx];
    image_grad_y *= top_grad;
    image_grad_x *= top_grad;

    float dy1, dy2;
    if (crop_height > 1) {
      dy1 = image_grad_y * (image_height - 1 - y * height_ratio);
      dy2 = image_grad_y * (y * height_ratio);
    } else {
      dy1 = image_grad_y * 0.5 * (image_height - 1);
      dy2 = image_grad_y * 0.5 * (image_height - 1);
    }

    float dx1, dx2;
    if (crop_width > 1) {
      dx1 = image_grad_x * (image_width - 1 - x * width_ratio);
      dx2 = image_grad_x * (x * width_ratio);
    } else {
      dx1 = image_grad_x * 0.5 * (image_width - 1);
      dx2 = image_grad_x * 0.5 * (image_width - 1);
    }

    ItexAtomicAdd(grads_boxes_ptr + b * 4 + 0, dy1);
    ItexAtomicAdd(grads_boxes_ptr + b * 4 + 1, dx1);
    ItexAtomicAdd(grads_boxes_ptr + b * 4 + 2, dy2);
    ItexAtomicAdd(grads_boxes_ptr + b * 4 + 3, dx2);
  }

 private:
  int total_count;
  const float* grads_ptr;
  const T* image_ptr;
  const float* boxes_ptr;
  const int32* box_ind_ptr;
  int num_boxes;
  int batch;
  int image_height;
  int image_width;
  int crop_height;
  int crop_width;
  int depth;
  float* grads_boxes_ptr;
};

template <typename T>
Status CropAndResizeKernel(const GPUDevice& device, int total_count,
                           const T* image_ptr, const float* boxes_ptr,
                           const int32* box_ind_ptr, int num_boxes, int batch,
                           int image_height, int image_width, int crop_height,
                           int crop_width, int depth, int method_id,
                           float extrapolation_value, float* crops_ptr) {
  auto stream = device.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (total_count + group_size - 1) / group_size;
  stream->submit([&](sycl::handler& cfg) {
    CropAndResizeKernelTask<T> task(total_count, image_ptr, boxes_ptr,
                                    box_ind_ptr, num_boxes, batch, image_height,
                                    image_width, crop_height, crop_width, depth,
                                    method_id, extrapolation_value, crops_ptr);
    cfg.parallel_for<CropAndResizeKernelTask<T>>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        task);
  });
  return Status::OK();
}

template <typename T>
Status CropAndResizeBackpropImageKernel(
    const GPUDevice& device, int total_count, const float* grads_ptr,
    const float* boxes_ptr, const int32* box_ind_ptr, int num_boxes, int batch,
    int image_height, int image_width, int crop_height, int crop_width,
    int depth, T* grads_image_ptr, int method_id) {
  auto stream = device.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (total_count + group_size - 1) / group_size;
  stream->submit([&](sycl::handler& cfg) {
    CropAndResizeBackpropImageKernelTask task(
        total_count, grads_ptr, boxes_ptr, box_ind_ptr, num_boxes, batch,
        image_height, image_width, crop_height, crop_width, depth,
        grads_image_ptr, method_id);
    cfg.parallel_for<CropAndResizeBackpropImageKernelTask<T>>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        task);
  });
  return Status::OK();
}

template <typename T>
Status CropAndResizeBackpropBoxesKernel(
    const GPUDevice& device, int total_count, const float* grads_ptr,
    const T* image_ptr, const float* boxes_ptr, const int32* box_ind_ptr,
    int num_boxes, int batch, int image_height, int image_width,
    int crop_height, int crop_width, int depth, float* grads_boxes_ptr) {
  auto stream = device.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (total_count + group_size - 1) / group_size;
  stream->submit([&](sycl::handler& cfg) {
    CropAndResizeBackpropBoxesKernelTask<T> task(
        total_count, grads_ptr, image_ptr, boxes_ptr, box_ind_ptr, num_boxes,
        batch, image_height, image_width, crop_height, crop_width, depth,
        grads_boxes_ptr);
    cfg.parallel_for<CropAndResizeBackpropBoxesKernelTask<T>>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        task);
  });
  return Status::OK();
}

namespace functor {
template <typename T>
struct CropAndResize<GPUDevice, T> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  const string& method_name, float extrapolation_value,
                  typename TTypes<float, 4>::Tensor crops) {
    const int batch = image.dimension(0);
    const int image_height = image.dimension(1);
    const int image_width = image.dimension(2);

    const int num_boxes = crops.dimension(0);
    const int crop_height = crops.dimension(1);
    const int crop_width = crops.dimension(2);
    const int depth = crops.dimension(3);

    const int total_count = num_boxes * crop_height * crop_width * depth;
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    InterpolationMethod method = BILINEAR;
    if (method_name == "nearest") {
      method = NEAREST;
    }

    if (total_count > 0) {
      auto status = CropAndResizeKernel<T>(
          d, total_count, image.data(), boxes.data(), box_ind.data(), num_boxes,
          batch, image_height, image_width, crop_height, crop_width, depth,
          method, extrapolation_value, crops.data());
    }
    return true;
  }
};

template <typename T>
struct CropAndResizeBackpropImage<GPUDevice, T> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<T, 4>::Tensor grads_image,
                  const std::string& method_name) {
    const int batch = grads_image.dimension(0);
    const int image_height = grads_image.dimension(1);
    const int image_width = grads_image.dimension(2);

    const int num_boxes = grads.dimension(0);
    const int crop_height = grads.dimension(1);
    const int crop_width = grads.dimension(2);
    const int depth = grads.dimension(3);
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    int total_count;

    // Initialize grads_image with all zeros.
    total_count = batch * image_height * image_width * depth;
    if (total_count > 0) {
      auto stream = d.stream();
      stream->fill<T>(grads_image.data(), T(0), grads_image.size());
    }

    // Configure interpolation method.
    InterpolationMethod method = BILINEAR;
    if (method_name == "nearest") {
      method = NEAREST;
    }

    // Accumulate.
    total_count = num_boxes * crop_height * crop_width * depth;
    if (total_count > 0) {
      auto status = CropAndResizeBackpropImageKernel<T>(
          d, total_count, grads.data(), boxes.data(), box_ind.data(), num_boxes,
          batch, image_height, image_width, crop_height, crop_width, depth,
          grads_image.data(), method);
    }
    return true;
  }
};

template <typename T>
struct CropAndResizeBackpropBoxes<GPUDevice, T> {
  bool operator()(const GPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<float, 2>::Tensor grads_boxes) {
    const int batch = image.dimension(0);
    const int image_height = image.dimension(1);
    const int image_width = image.dimension(2);

    const int num_boxes = grads.dimension(0);
    const int crop_height = grads.dimension(1);
    const int crop_width = grads.dimension(2);
    const int depth = grads.dimension(3);

    int total_count;

    // Initialize grads_boxes with all zeros.
    total_count = num_boxes * 4;
    if (total_count > 0) {
      auto stream = d.stream();
      stream->fill<float>(grads_boxes.data(), 0, grads_boxes.size());
    }

    // Accumulate.
    total_count = num_boxes * crop_height * crop_width * depth;
    if (total_count > 0) {
      auto status = CropAndResizeBackpropBoxesKernel<T>(
          d, total_count, grads.data(), image.data(), boxes.data(),
          box_ind.data(), num_boxes, batch, image_height, image_width,
          crop_height, crop_width, depth, grads_boxes.data());
    }
    return true;
  }
};
}  // namespace functor

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("CropAndResize")                    \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .HostMemory("crop_size"),            \
                          CropAndResizeOp<GPUDevice, T>);          \
                                                                   \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeGradImage")           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .HostMemory("image_size"),           \
                          CropAndResizeGradImageOp<GPUDevice, T>); \
                                                                   \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeGradBoxes")           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          CropAndResizeGradBoxesOp<GPUDevice, T>);

TF_CALL_float(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_bfloat16(REGISTER_KERNEL);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_KERNEL

}  // namespace itex
