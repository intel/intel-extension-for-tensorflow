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

#include "itex/core/kernels/gpu/data_format_ops.h"

#include <iostream>
#include <string>
#include <vector>

#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class DataFormatDimMapOp : public OpKernel {
 public:
  explicit DataFormatDimMapOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format_));
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format_));
    OP_REQUIRES(context, src_format_.size() == 4 || src_format_.size() == 5,
                errors::InvalidArgument(strings::StrCat(
                    "Source format must of length 4 or 5, received "
                    "src_format = ",
                    src_format_)));
    OP_REQUIRES(
        context, dst_format_.size() == 4 || dst_format_.size() == 5,
        errors::InvalidArgument(strings::StrCat(
            "Destination format must of length 4 or 5, received dst_format = ",
            dst_format_)));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    Tensor dst_idx;
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DT_INT32, {static_cast<int64>(src_format_.size())},
                       &dst_idx, alloc_attr));
    for (int i = 0; i < src_format_.size(); ++i) {
      for (int j = 0; j < dst_format_.size(); ++j) {
        if (dst_format_[j] == src_format_[i]) {
          dst_idx.vec<int>()(i) = j;
          break;
        }
      }
    }

    functor::DataFormatDimMap<GPUDevice, T> functor;
    functor(context->eigen_gpu_device(), input.flat<T>(), output->flat<T>(),
            dst_idx.vec<int>());
  }

 private:
  string src_format_;
  string dst_format_;
};

template <typename Device, typename T>
class DataFormatVecPermuteOp : public OpKernel {
 public:
  explicit DataFormatVecPermuteOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format_));
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 1 || input.dims() == 2,
                errors::InvalidArgument(
                    "input must be a vector or 2D tensor, but got shape ",
                    input.shape().DebugString()));
    if (input.dims() == 1) {
      OP_REQUIRES(context,
                  input.NumElements() == 2 || input.NumElements() == 4 ||
                      input.NumElements() == 5,
                  errors::InvalidArgument(
                      "1D input must be of size 2, 4 or 5, but got shape ",
                      input.shape().DebugString()));
    } else if (input.dims() == 2) {
      OP_REQUIRES(context, input.dim_size(0) == 2 || input.dim_size(0) == 4,
                  errors::InvalidArgument("First dimension of 2D input must be "
                                          "of size 2 or 4, but got shape ",
                                          input.shape().DebugString()));
      OP_REQUIRES(
          context, input.dim_size(1) == 2,
          errors::InvalidArgument(
              "Second dimension of 2D input must be of size 2, but got shape ",
              input.shape().DebugString()));
    }

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    // Support 1D and 2D cases.
    Eigen::DSizes<Eigen::DenseIndex, 8> dst_idx;
    string src_format_str = src_format_;
    string dst_format_str = dst_format_;
    if (input.dim_size(0) == 2) {
      // If the input is a vector of size 2, treat the two elements as spatial
      // dimensions.
      auto keep_only_spatial_dimensions = [](string* format_str) -> void {
        auto new_end = std::remove_if(
            format_str->begin(), format_str->end(),
            [](const char dim) { return dim != 'H' && dim != 'W'; });
        format_str->erase(new_end, format_str->end());
      };
      keep_only_spatial_dimensions(&src_format_str);
      keep_only_spatial_dimensions(&dst_format_str);
    }
    ComputeDstIndex(src_format_str, dst_format_str, input.dims(), &dst_idx);
    functor::DataFormatVecPermute<GPUDevice, T> functor;
    functor(context->eigen_gpu_device(), input.flat<T>(), output->flat<T>(),
            dst_idx);
  }

 private:
  // Finds out the destination index. Support 1D and 2D cases.
  // Example: HWNC --> NHWC
  // 1D: dst = [1, 2, 0, 3],
  // 2D: dst = [2, 3, 4, 5, 0, 1, 6, 7]
  void ComputeDstIndex(const string& src_format_str,
                       const string& dst_format_str, int num_dim,
                       Eigen::DSizes<Eigen::DenseIndex, 8>* dst) {
    for (int i = 0; i < src_format_str.size(); ++i) {
      for (int j = 0; j < dst_format_str.size(); ++j) {
        if (dst_format_str[j] != src_format_str[i]) continue;
        // Found the dst index. Set output based on the number of dims.
        for (int k = 0; k < num_dim; ++k) {
          (*dst)[i * num_dim + k] = j * num_dim + k;
        }
      }
    }
  }

  string src_format_;
  string dst_format_;
};

#define REGISTER_GPU_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DataFormatDimMap").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DataFormatDimMapOp<GPUDevice, T>);

TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#define REGISTER_GPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("DataFormatVecPermute").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DataFormatVecPermuteOp<GPUDevice, T>);

TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

}  // namespace itex
