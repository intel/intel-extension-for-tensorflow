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

#include "itex/core/kernels/gpu/maxpooling_op.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"
#include "itex/core/kernels/common/maxpooling_op.h"
#include "itex/core/kernels/common/pooling_ops_common.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
void functor::MaxPoolForwardWithArgmax<T>::operator()(
    TensorFormat data_format, const T* bottom_data, const int batch,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_t,
    const int pad_l, T* top_data, int64* mask, const Eigen::GpuDevice& d,
    bool propagate_nans, const bool include_batch_in_index) {
  const int output_size = batch * channels * pooled_height * pooled_width;
  if (output_size == 0) return;
  auto stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (output_size + group_size - 1) / group_size;
  if (propagate_nans) {
    stream->submit([&](sycl::handler& cgh) {
      MaxPoolWithArgmaxKernel<true, T> task(
          output_size, bottom_data, height, width, channels, pooled_height,
          pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_l,
          top_data, mask, include_batch_in_index);
      cgh.parallel_for<MaxPoolFwdArgmaxNHWC<true, T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
  } else {
    // NHWC && propagate_nans==false
    stream->submit([&](sycl::handler& cgh) {
      MaxPoolWithArgmaxKernel<false, T> task(
          output_size, bottom_data, height, width, channels, pooled_height,
          pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_l,
          top_data, mask, include_batch_in_index);
      cgh.parallel_for<MaxPoolFwdArgmaxNHWC<false, T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
  }

  return;
}

template <typename Device, typename T>
class MaxPoolingWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    data_format_ = FORMAT_NHWC;
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    string padding;
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    if (padding == "VALID")
      this->padding_ = Padding::VALID;
    else if (padding == "SAME")
      this->padding_ = Padding::SAME;
    else
      this->padding_ = Padding::EXPLICIT;

    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES_OK(context, context->GetAttr("include_batch_in_index",
                                             &include_batch_in_index_));
    ITEX_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_MAXPOOL_NANPROP", false,
                                     &propagate_nans_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    OneDnnPoolParameters params;
    params.Init(context, ksize_, stride_, padding_, {}, FORMAT_NHWC,
                tensor_in.shape());

    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    Tensor* argmax = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, out_shape, &argmax));

    LaunchMaxPoolingWithArgmax<T>::launch(
        context, data_format_, params, tensor_in, output, argmax,
        propagate_nans_, include_batch_in_index_);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool propagate_nans_;
  bool include_batch_in_index_;
};

void functor::MaxPoolBackwardWithArgmax::operator()(
    const int output_size, const int input_size, const float* top_diff,
    const int64* mask, const int top_offset, const int bottom_offset,
    float* bottom_diff, const Eigen::GpuDevice& d,
    const bool include_batch_in_index) {
  if (input_size == 0) return;
  auto stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (output_size + group_size - 1) / group_size;

  stream->submit([&](sycl::handler& cgh) {
    MaxPoolBackwardKernel task(output_size, top_diff, mask, top_offset,
                               bottom_offset, bottom_diff,
                               include_batch_in_index);
    cgh.parallel_for<MaxPoolBackwardKernel>(
        sycl::nd_range<1>(num_wg * group_size, group_size), task);
  });
}

template <typename Device, typename T>
class MaxPoolingGradWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingGradWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format_str;

    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));

    string padding;
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    if (padding == "VALID")
      this->padding_ = Padding::VALID;
    else if (padding == "SAME")
      this->padding_ = Padding::SAME;
    else
      this->padding_ = Padding::EXPLICIT;

    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES_OK(context, context->GetAttr("include_batch_in_index",
                                             &include_batch_in_index_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& grad_in = context->input(1);
    const Tensor& argmax = context->input(2);
    auto& d = context->eigen_gpu_device();

    OneDnnPoolParameters params;
    params.Init(context, ksize_, stride_, padding_, {}, FORMAT_NHWC,
                tensor_in.shape());

    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.tensor_in_rows,
                           params.tensor_in_cols, params.depth});
    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, out_shape, &grad_out));

    typename TTypes<T, 4>::Tensor eigen_grad_out = grad_out->tensor<T, 4>();
    eigen_grad_out.device(d) = eigen_grad_out.constant(static_cast<T>(0));

    LaunchMaxPoolingGradWithArgmax<T>::launch(
        context, params, grad_in, argmax, grad_out, include_batch_in_index_);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool include_batch_in_index_;
};

template <typename T>
void functor::MaxPoolGradBackwardWithArgmax<T>::operator()(
    const int output_size, const int input_size, const T* top_diff,
    const int64* mask, const int top_offset, const int bottom_offset,
    T* bottom_diff, const Eigen::GpuDevice& d,
    const bool include_batch_in_index) {
  if (input_size == 0) return;
  auto stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (output_size + group_size - 1) / group_size;

  stream->submit([&](sycl::handler& cgh) {
    MaxPoolGradBackwardKernel<T> task(output_size, top_diff, mask, top_offset,
                                      bottom_offset, bottom_diff,
                                      include_batch_in_index);
    cgh.parallel_for<MaxPoolGradBwdArgMax<T>>(
        sycl::nd_range<1>(num_wg * group_size, group_size), task);
  });
}

template <typename Device, typename T>
class MaxPoolingGradGradWithArgmaxOp : public PoolingOpBase<T> {
 public:
  explicit MaxPoolingGradGradWithArgmaxOp(OpKernelConstruction* context)
      : PoolingOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& grad_in = context->input(1);
    const Tensor& argmax = context->input(2);

    OneDnnPoolParameters params;
    params.Init(context, this->ksize_, this->stride_, this->padding_,
                /*explicit_paddings=*/{}, FORMAT_NHWC, tensor_in.shape());
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});

    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, out_shape, &grad_out));

    LaunchMaxPoolingGradGradWithArgmax<T>::launch(
        context, params, grad_in, argmax, grad_out,
        this->include_batch_in_index_);
  }
};

template <typename T>
void functor::MaxPoolGradBackwardNoMask<T>::operator()(
    TensorFormat data_format, const T* bottom_data, const T* output_data,
    const int batch, const int pooled_height, const int pooled_width,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_t,
    const int pad_l, const T* top_diff, T* bottom_diff,
    const Eigen::GpuDevice& d) {
  const int num_items = batch * channels * pooled_height * pooled_width;
  if (num_items == 0) return;
  auto stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (num_items + group_size - 1) / group_size;

  if (data_format == FORMAT_NHWC) {
    stream->submit([&](sycl::handler& cgh) {
      MaxPoolGradBackwardNoMaskNHWCKernel<T> task(
          num_items, bottom_data, output_data, pooled_height, pooled_width,
          channels, height, width, kernel_h, kernel_w, stride_h, stride_w,
          pad_t, pad_l, top_diff, bottom_diff);
      cgh.parallel_for<MaxPoolGradNHWCBwd<T>>(
          sycl::nd_range<1>(num_wg * group_size, group_size), task);
    });

  } else {  // data_format == FORMAT_NCHW
    stream->submit([&](sycl::handler& cgh) {
      MaxPoolGradBackwardNoMaskNCHWKernel<T> task(
          num_items, bottom_data, output_data, pooled_height, pooled_width,
          channels, height, width, kernel_h, kernel_w, stride_h, stride_w,
          pad_t, pad_l, top_diff, bottom_diff);
      cgh.parallel_for<MaxPoolGradNCHWBwd<T>>(
          sycl::nd_range<1>(num_wg * group_size, group_size), task);
    });
  }

  return;
}

template <typename Device, typename T>
class MaxPoolingGradGradOp : public PoolingOpBase<T> {
 public:
  explicit MaxPoolingGradGradOp(OpKernelConstruction* context)
      : PoolingOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_grad_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional 4"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_grad_backprop should have 4 dimensions.
    OP_REQUIRES(
        context, out_grad_backprop.dims() == 4,
        errors::InvalidArgument("out_grad_backprop must be 4-dimensional"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, tensor_out.shape(), &output));

    std::vector<int32> ksize = this->ksize_;
    std::vector<int32> stride = this->stride_;
    if (context->num_inputs() == 5) {
      const Tensor& tensor_ksize = context->input(3);
      auto value_ksize = tensor_ksize.flat<int32>();
      ksize.resize(tensor_ksize.shape().num_elements());
      std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

      const Tensor& tensor_stride = context->input(4);
      auto value_stride = tensor_stride.flat<int32>();
      stride.resize(tensor_stride.shape().num_elements());
      std::copy_n(&value_stride(0), stride.size(), stride.begin());
    }
    this->ksize_ = ksize;
    this->stride_ = stride;

    OP_REQUIRES(context, ksize.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, stride.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int32 ksize_n = GetTensorDim(ksize, this->data_format_tf_, 'N');
    const int32 stride_n = GetTensorDim(stride, this->data_format_tf_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    OneDnnPoolParameters params;
    params.Init(context, this->ksize_, this->stride_, this->padding_, {},
                this->data_format_tf_, tensor_in.shape());

    functor::MaxPoolGradBackwardNoMask<T>()(
        this->data_format_tf_, tensor_in.flat<T>().data(),
        tensor_out.flat<T>().data(), params.tensor_in_batch, params.out_height,
        params.out_width, params.depth, params.tensor_in_rows,
        params.tensor_in_cols, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_top, params.pad_left,
        out_grad_backprop.flat<T>().data(), output->flat<T>().data(),
        context->eigen_device<Eigen::GpuDevice>());
  }
};

template <typename T>
void functor::MaxPool3dGradBackward<T>::operator()(
    TensorFormat data_format, const T* bottom_data, const T* output_data,
    const int batch, const int64 pooled_plane, const int64 pooled_height,
    const int64 pooled_width, const int channels, const int plane,
    const int height, const int width, const int kernel_p, const int kernel_h,
    const int kernel_w, const int stride_p, const int stride_h,
    const int stride_w, const int64 pad_p, const int64 pad_t, const int64 pad_l,
    const T* top_diff, T* bottom_diff, const Eigen::GpuDevice& d) {
  int num_items =
      batch * channels * pooled_plane * pooled_height * pooled_width;
  if (num_items == 0) return;
  auto stream = d.stream();

  if (data_format == FORMAT_NHWC) {
    stream->submit([&](sycl::handler& cgh) {
      MaxPoolGradBackwardNoMaskNDHWC<T> task(
          num_items, bottom_data, output_data, pooled_plane, pooled_height,
          pooled_width, channels, plane, height, width, kernel_p, kernel_h,
          kernel_w, stride_p, stride_h, stride_w, pad_p, pad_t, pad_l, top_diff,
          bottom_diff);
      cgh.parallel_for<MaxPoolGradBackwardNoMaskNDHWC<T>>(
          sycl::range<1>(num_items), task);
    });
  } else {
    stream->submit([&](sycl::handler& cgh) {
      MaxPoolGradBackwardNoMaskNCDHW<T> task(
          num_items, bottom_data, output_data, pooled_plane, pooled_height,
          pooled_width, channels, plane, height, width, kernel_p, kernel_h,
          kernel_w, stride_p, stride_h, stride_w, pad_p, pad_t, pad_l, top_diff,
          bottom_diff);
      cgh.parallel_for<MaxPoolGradBackwardNoMaskNCDHW<T>>(
          sycl::range<1>(num_items), task);
    });
  }

  return;
}

template <class Device, class T>
class MaxPooling3dGradGradOp : public OpKernel {
 public:
  explicit MaxPooling3dGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    string padding;
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    if (padding == "VALID")
      this->padding_ = Padding::VALID;
    else if (padding == "SAME")
      this->padding_ = Padding::SAME;
    else
      this->padding_ = Padding::EXPLICIT;
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    const int32 ksize_c = GetTensorDim(ksize_, data_format_, 'C');
    const int32 stride_c = GetTensorDim(stride_, data_format_, 'C');
    OP_REQUIRES(context, ksize_c == 1 && stride_c == 1,
                errors::Unimplemented("MaxPooling3dGradGrad is not yet "
                                      "supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_grad_backprop = context->input(2);

    // For maxpooling3d, tensor_in should have 5 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 5,
                errors::InvalidArgument("tensor_in must be 5-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 5,
                errors::InvalidArgument("tensor_out must be 5-dimensional"));
    // For maxpooling3d, out_grad_backprop should have 5 dimensions.
    OP_REQUIRES(
        context, out_grad_backprop.dims() == 5,
        errors::InvalidArgument("out_grad_backprop must be 5-dimensional"));

    OneDnnPoolParameters params;
    params.Init(context, ksize_, stride_, padding_, {}, data_format_,
                tensor_in.shape());

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {2}, 0, tensor_out.shape(), &output));

    LaunchMaxPooling3dGradGradOp<Device, T>::launch(
        context, params, tensor_in, tensor_out, out_grad_backprop, output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

#define REGISTER_GPU_POOL_KERNELS(T)                                    \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MaxPool").Device(DEVICE_GPU).TypeConstraint<T>("T"),        \
      PoolingOp<GPUDevice, T, dnnl::algorithm::pooling_max>);           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MaxPoolV2")                                                 \
          .Device(DEVICE_GPU)                                           \
          .HostMemory("ksize")                                          \
          .HostMemory("strides")                                        \
          .TypeConstraint<T>("T"),                                      \
      PoolingOp<GPUDevice, T, dnnl::algorithm::pooling_max>);           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ITEXMaxPoolV2")                                            \
          .Device(DEVICE_GPU)                                           \
          .HostMemory("ksize")                                          \
          .HostMemory("strides")                                        \
          .TypeConstraint<T>("T"),                                      \
      PoolingOp<GPUDevice, T, dnnl::algorithm::pooling_max>);           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MaxPool3D").Device(DEVICE_GPU).TypeConstraint<T>("T"),      \
      PoolingOp<GPUDevice, T, dnnl::algorithm::pooling_max>);           \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolWithArgmax")                     \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<int64>("Targmax")         \
                              .TypeConstraint<T>("T"),                  \
                          MaxPoolingWithArgmaxOp<GPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ITEXMaxPool").Device(DEVICE_GPU).TypeConstraint<T>("T"),   \
      PoolingOp<GPUDevice, T, dnnl::algorithm::pooling_max>);           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ITEXMaxPool3D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      PoolingOp<GPUDevice, T, dnnl::algorithm::pooling_max>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_POOL_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(Name("MaxPoolWithArgmax")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("Targmax")
                            .TypeConstraint<double>("T"),
                        MaxPoolingWithArgmaxOp<GPUDevice, double>);
TF_CALL_double(REGISTER_GPU_POOL_KERNELS);
#endif
#undef REGISTER_GPU_POOL_KERNELS

#define REGISTER_GPU_POOL_GRAD_KERNELS(T)                                 \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MaxPoolGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),      \
      MaxPoolGradOp<GPUDevice, T, dnnl::prop_kind::forward_training>);    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MaxPoolGradV2")                                               \
          .Device(DEVICE_GPU)                                             \
          .HostMemory("ksize")                                            \
          .HostMemory("strides")                                          \
          .TypeConstraint<T>("T"),                                        \
      MaxPoolGradOp<GPUDevice, T, dnnl::prop_kind::forward_training>);    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MaxPool3DGrad")                                               \
          .Device(DEVICE_GPU)                                             \
          .TypeConstraint<T>("T")                                         \
          .TypeConstraint<T>("TInput"),                                   \
      MaxPoolGradOp<GPUDevice, T, dnnl::prop_kind::forward_training>);    \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolGradWithArgmax")                   \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<int64>("Targmax"),          \
                          MaxPoolingGradWithArgmaxOp<GPUDevice, T>);      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ITEXMaxPoolGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MaxPoolGradOp<GPUDevice, T, dnnl::prop_kind::forward_training>);    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ITEXMaxPoolGradV2")                                          \
          .Device(DEVICE_GPU)                                             \
          .HostMemory("ksize")                                            \
          .HostMemory("strides")                                          \
          .TypeConstraint<T>("T"),                                        \
      MaxPoolGradOp<GPUDevice, T, dnnl::prop_kind::forward_training>);    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ITEXMaxPool3DGrad")                                          \
          .Device(DEVICE_GPU)                                             \
          .TypeConstraint<T>("T")                                         \
          .TypeConstraint<T>("TInput"),                                   \
      MaxPoolGradOp<GPUDevice, T, dnnl::prop_kind::forward_training>);

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_GPU_POOL_GRAD_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(Name("MaxPoolGradWithArgmax")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .TypeConstraint<int64>("Targmax"),
                        MaxPoolingGradWithArgmaxOp<GPUDevice, double>);
TF_CALL_double(REGISTER_GPU_POOL_GRAD_KERNELS);
#endif
#undef REGISTER_GPU_POOL_GRAD_KERNELS

#define REGISTER_GPU_POOL_GRADGRAD_KERNELS(T)                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MaxPoolGradGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),   \
      MaxPoolingGradGradOp<GPUDevice, T>);                                 \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolGradGradV2")                        \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("ksize")                         \
                              .HostMemory("strides")                       \
                              .TypeConstraint<T>("T"),                     \
                          MaxPoolingGradGradOp<GPUDevice, T>);             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MaxPool3DGradGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MaxPooling3dGradGradOp<GPUDevice, T>);                               \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolGradGradWithArgmax")                \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<int64>("Targmax"),           \
                          MaxPoolingGradGradWithArgmaxOp<GPUDevice, T>);

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_GPU_POOL_GRADGRAD_KERNELS);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_POOL_GRADGRAD_KERNELS);
#endif
#undef REGISTER_GPU_POOL_GRADGRAD_KERNELS

// Quantized Kernels
// TF INT8 kernel
#define REGISTER_KERNEL(TYPE)         \
  REGISTER_KERNEL_BUILDER(            \
      Name("QuantizedMaxPool")        \
          .Device(DEVICE_GPU)         \
          .HostMemory("min_input")    \
          .HostMemory("max_input")    \
          .HostMemory("min_output")   \
          .HostMemory("max_output")   \
          .TypeConstraint<TYPE>("T"), \
      PoolingOp<GPUDevice, TYPE, dnnl::algorithm::pooling_max>)

TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(TYPE)         \
  REGISTER_KERNEL_BUILDER(            \
      Name("_QuantizedMaxPool3D")     \
          .Device(DEVICE_GPU)         \
          .HostMemory("min_input")    \
          .HostMemory("max_input")    \
          .HostMemory("min_output")   \
          .HostMemory("max_output")   \
          .TypeConstraint<TYPE>("T"), \
      PoolingOp<GPUDevice, TYPE, dnnl::algorithm::pooling_max>)

TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace itex
