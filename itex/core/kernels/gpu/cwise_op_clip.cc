/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/cwise_op_clip.h"
#include "itex/core/kernels/common/cwise_ops_common.h"

namespace itex {

template <typename Device, typename T>
class ClipOp : public OpKernel {
 public:
  explicit ClipOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& in2 = ctx->input(2);
    OP_REQUIRES(ctx,
                (in0.shape() == in1.shape() ||
                 TensorShapeUtils::IsScalar(in1.shape())) &&
                    (in0.shape() == in2.shape() ||
                     TensorShapeUtils::IsScalar(in2.shape())),
                errors::InvalidArgument(
                    "clip_value_min and clip_value_max must be either of "
                    "the same shape as input, or a scalar. ",
                    "input shape: ", in0.shape().DebugString(),
                    "clip_value_min shape: ", in1.shape().DebugString(),
                    "clip_value_max shape: ", in2.shape().DebugString()));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output({0}, 0, in0.shape(), &out));
    if (out->NumElements() == 0) return;  // Nothing to do for empty output

    auto in0_flat = in0.flat<T>();
    auto in1_flat = in1.flat<T>();
    auto in2_flat = in2.flat<T>();
    auto out_flat = out->flat<T>();
    const Device& d = ctx->eigen_device<Device>();

    if (in1.shape() == in2.shape()) {
      if (in0.shape() == in1.shape()) {
        functor::TernaryClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                            out_flat);
      } else {
        functor::UnaryClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                          out_flat);
      }
    } else {
      if (in0.shape() == in1.shape()) {
        functor::BinaryLeftClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                               out_flat);
      } else {
        functor::BinaryRightClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                                out_flat);
      }
    }
  }
};

template <typename T>
struct UnaryClipCustomKernel {
  UnaryClipCustomKernel(const int32 size_in, const T* in0, const T* in1,
                        const T* in2, T* out)
      : size_in_(size_in),
        in0_ptr(in0),
        in1_ptr(in1),
        in2_ptr(in2),
        out_ptr(out) {}

  void operator()(sycl::nd_item<1> item) const {
    auto o_idx = item.get_global_linear_id();

    if (o_idx < size_in_) {
      T value;
      // do not use "A?B:C" because half and bfloat16 doesn't support it.
      if (in2_ptr[0] < in0_ptr[o_idx])
        value = in2_ptr[0];
      else
        value = in0_ptr[o_idx];

      if (value < in1_ptr[0])
        out_ptr[o_idx] = in1_ptr[0];
      else
        out_ptr[o_idx] = value;
    }
  }

 private:
  const int32 size_in_;
  const T* in0_ptr;
  const T* in1_ptr;
  const T* in2_ptr;
  T* out_ptr;
};

template <typename T>
struct BinaryRightClipCustomKernel {
  BinaryRightClipCustomKernel(const int32 size_in, const T* in0, const T* in1,
                              const T* in2, T* out)
      : size_in_(size_in),
        in0_ptr(in0),
        in1_ptr(in1),
        in2_ptr(in2),
        out_ptr(out) {}

  void operator()(sycl::nd_item<1> item) const {
    auto o_idx = item.get_global_linear_id();

    if (o_idx < size_in_) {
      T value;  // not using "A?B:C" as half/bf16 not support it
      if (in2_ptr[o_idx] < in0_ptr[o_idx])
        value = in2_ptr[o_idx];
      else
        value = in0_ptr[o_idx];

      if (value < in1_ptr[0])
        out_ptr[o_idx] = in1_ptr[0];
      else
        out_ptr[o_idx] = value;
    }
  }

 private:
  const int32 size_in_;
  const T* in0_ptr;
  const T* in1_ptr;
  const T* in2_ptr;
  T* out_ptr;
};

template <typename T>
struct BinaryLeftClipCustomKernel {
  BinaryLeftClipCustomKernel(const int32 size_in, const T* in0, const T* in1,
                             const T* in2, T* out)
      : size_in_(size_in),
        in0_ptr(in0),
        in1_ptr(in1),
        in2_ptr(in2),
        out_ptr(out) {}

  void operator()(sycl::nd_item<1> item) const {
    auto o_idx = item.get_global_linear_id();

    if (o_idx < size_in_) {
      T value;
      if (in2_ptr[0] < in0_ptr[o_idx])
        value = in2_ptr[0];
      else
        value = in0_ptr[o_idx];

      if (value < in1_ptr[o_idx])
        out_ptr[o_idx] = in1_ptr[o_idx];
      else
        out_ptr[o_idx] = value;
    }
  }

 private:
  const int32 size_in_;
  const T* in0_ptr;
  const T* in1_ptr;
  const T* in2_ptr;
  T* out_ptr;
};

namespace functor {

// Unary functor for clip [Tensor, Scalar, Scalar]
template <typename T>
struct UnaryClipOp<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  const typename TTypes<T>::ConstFlat& in0_flat,
                  const typename TTypes<T>::ConstFlat& in1_flat,
                  const typename TTypes<T>::ConstFlat& in2_flat,
                  const typename TTypes<T>::Flat& out_flat) const {
    auto& stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (in0_flat.size() + group_size - 1) / group_size;

    stream->submit([&](sycl::handler& cgh) {
      UnaryClipCustomKernel<T> task(in0_flat.size(), in0_flat.data(),
                                    in1_flat.data(), in2_flat.data(),
                                    out_flat.data());
      cgh.parallel_for<UnaryClipCustomKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
  }
};

// Binary functor for clip [Tensor, Scalar, Tensor]
template <typename T>
struct BinaryRightClipOp<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  const typename TTypes<T>::ConstFlat& in0_flat,
                  const typename TTypes<T>::ConstFlat& in1_flat,
                  const typename TTypes<T>::ConstFlat& in2_flat,
                  const typename TTypes<T>::Flat& out_flat) const {
    auto& stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (in0_flat.size() + group_size - 1) / group_size;

    stream->submit([&](sycl::handler& cgh) {
      BinaryRightClipCustomKernel<T> task(in0_flat.size(), in0_flat.data(),
                                          in1_flat.data(), in2_flat.data(),
                                          out_flat.data());
      cgh.parallel_for<BinaryRightClipCustomKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
  }
};

// Binary functor for clip [Tensor, Tensor, Scalar]
template <typename T>
struct BinaryLeftClipOp<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  const typename TTypes<T>::ConstFlat& in0_flat,
                  const typename TTypes<T>::ConstFlat& in1_flat,
                  const typename TTypes<T>::ConstFlat& in2_flat,
                  const typename TTypes<T>::Flat& out_flat) const {
    auto& stream = d.stream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (in0_flat.size() + group_size - 1) / group_size;

    stream->submit([&](sycl::handler& cgh) {
      BinaryLeftClipCustomKernel<T> task(in0_flat.size(), in0_flat.data(),
                                         in1_flat.data(), in2_flat.data(),
                                         out_flat.data());
      cgh.parallel_for<BinaryLeftClipCustomKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
  }
};

// Ternary functor for clip [Tensor, Tensor, Tensor]
template <typename T>
struct TernaryClipOp<GPUDevice, T> {
  void operator()(
      const GPUDevice& d, const typename TTypes<T>::ConstFlat& in0_flat,
      const typename TTypes<T>::ConstFlat& in1_flat,
      const typename TTypes<T>::ConstFlat& in2_flat,
      typename TTypes<T>::Flat& out_flat) const {  // NOLINT(runtime/references)
    out_flat.device(d) = in0_flat.cwiseMin(in2_flat).cwiseMax(in1_flat);
  }
};

#define INSTANTIATE_GPU(T)                         \
  template struct UnaryClipOp<GPUDevice, T>;       \
  template struct BinaryRightClipOp<GPUDevice, T>; \
  template struct BinaryLeftClipOp<GPUDevice, T>;  \
  template struct TernaryClipOp<GPUDevice, T>;
INSTANTIATE_GPU(Eigen::half);
INSTANTIATE_GPU(float);
INSTANTIATE_GPU(Eigen::bfloat16);
#undef INSTANTIATE_GPU

}  // namespace functor

#define REGISTER_GPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ClipByValue").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      ClipOp<GPUDevice, type>);
REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(Eigen::bfloat16);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_GPU_KERNEL(double);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNEL
}  // namespace itex
