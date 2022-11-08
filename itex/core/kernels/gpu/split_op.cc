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

#include "itex/core/kernels/gpu/split_lib.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SplitOpBase : public OpKernel {
 public:
  explicit SplitOpBase(OpKernelConstruction* c) : OpKernel(c) {}

  void ComputeEasyCases(OpKernelContext* context, bool* done) {
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const Tensor& split_dim_tensor = context->input(0);
    OP_REQUIRES(
        context, split_dim_tensor.shape().dims() == 0,
        errors::InvalidArgument("split_dim must be a scalar but has rank ",
                                split_dim_tensor.shape().dims()));
    const int32 split_dim_orig = split_dim_tensor.flat<int32>()(0);
    const int32 split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;
    const int32 num_split = context->num_outputs();

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input_shape.dims(),
        errors::InvalidArgument("-input rank(-", input.dims(),
                                ") <= split_dim < input rank (", input.dims(),
                                "), but got ", split_dim_orig));

    OP_REQUIRES(
        context, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    OP_REQUIRES(context, input_shape.dim_size(split_dim) % num_split == 0,
                errors::InvalidArgument(
                    "Number of ways to split should evenly divide the split "
                    "dimension, but got split_dim ",
                    split_dim, " (size = ", input_shape.dim_size(split_dim),
                    ") ", "and num_split ", num_split));
    // Special case 1: num_split == 1. Nothing to do.
    if (num_split == 1) {
      ITEX_VLOG(3) << "Split identity";
      context->set_output(0, context->input(1));
      *done = true;
      return;
    }

    // TODO(itex): Special case 2: split along the 1st dimension. We can share
    // the underlying buffer.
  }

  template <typename IndexType>
  std::tuple<IndexType, IndexType, IndexType> SetDims(
      const TensorShape& input_shape, int32 split_dim) const {
    static_assert(std::is_integral<IndexType>::value,
                  "IndexType must be an integer type");
    int32 prefix_dim_size = 1;
    for (int i = 0; i < split_dim; ++i) {
      prefix_dim_size *= input_shape.dim_size(i);
    }

    // Caller must ensure that dim_size and suffix_dim_size are <
    // std::numeric_limits<IndexType>::max()
    IndexType split_dim_size =
        static_cast<IndexType>(input_shape.dim_size(split_dim));

    IndexType suffix_dim_size = 1;
    for (int i = split_dim + 1; i < input_shape.dims(); ++i) {
      suffix_dim_size *= static_cast<IndexType>(input_shape.dim_size(i));
    }
    return std::make_tuple(prefix_dim_size, split_dim_size, suffix_dim_size);
  }
};

template <typename T>
class SplitOpGPU : public SplitOpBase<GPUDevice, T> {
 public:
  typedef SplitOpBase<GPUDevice, T> Base;
  explicit SplitOpGPU(OpKernelConstruction* c) : Base(c) {}

  void Compute(OpKernelContext* context) override {
    bool done = false;
    Base::ComputeEasyCases(context, &done);
    if (!context->status().ok() || done) {
      return;
    }
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const int32 split_dim_orig = context->input(0).flat<int32>()(0);
    const int32 split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;
    const int32 num_split = context->num_outputs();

    // Android also uses int32 indexing, so check here also.
    OP_REQUIRES(
        context,
        FastBoundsCheck(input.NumElements(),
                        std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("Split requires input size < ",
                                std::numeric_limits<Eigen::DenseIndex>::max()));

    int32 prefix_dim_size;
    int32 split_dim_size;
    int32 suffix_dim_size;

    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<int32>(input_shape, split_dim);

    const int32 split_dim_output_size = split_dim_size / num_split;
    TensorShape output_shape(input_shape);
    output_shape.set_dim(split_dim, split_dim_output_size);

    const int max_out_num_handle = 8;
    for (int i = 0; i < num_split; i += max_out_num_handle) {
      int out_num_handle = num_split - i < max_out_num_handle
                               ? num_split - i
                               : max_out_num_handle;
      GpuDeviceArrayOnHost<T*> ptrs(context, out_num_handle);
      OP_REQUIRES_OK(context, ptrs.Init());

      for (int j = 0; j < out_num_handle; ++j) {
        Tensor* result = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(i + j, output_shape, &result));
        ptrs.Set(j, result->flat<T>().data());
      }
      if (prefix_dim_size * split_dim_output_size * suffix_dim_size != 0) {
        OP_REQUIRES_OK(context, ptrs.Finalize());
        functor::SplitGpuFunctor<T>()(context->eigen_device<GPUDevice>(),
                                      input.flat<T>().data(), prefix_dim_size,
                                      split_dim_size, suffix_dim_size,
                                      split_dim_output_size, i, ptrs.data());
      }
    }
  }
};

#define REGISTER_SPLIT_GPU_KERNEL(type)                  \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOpGPU<type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SPLIT_GPU_KERNEL);
TF_CALL_complex64(REGISTER_SPLIT_GPU_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SPLIT_GPU_KERNEL);
TF_CALL_complex128(REGISTER_SPLIT_GPU_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_SPLIT_GPU_KERNEL

}  // namespace itex
