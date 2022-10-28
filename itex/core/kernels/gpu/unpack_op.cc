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
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class UnpackOp : public OpKernel {
 public:
  explicit UnpackOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* c) override {
    const int32 num = c->num_outputs();
    const Tensor& input = c->input(0);
    const TensorShape& input_shape = input.shape();

    if (axis_ < 0) axis_ += input_shape.dims();

    OP_REQUIRES(c, 0 <= axis_ && axis_ < input_shape.dims(),
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -input_shape.dims(), ", ",
                                        input_shape.dims(), ")"));

    OP_REQUIRES(
        c, input_shape.dims() > 0 && input_shape.dim_size(axis_) == num,
        errors::InvalidArgument("Input shape axis ", axis_, " must equal ", num,
                                ", got shape ", input_shape.DebugString()));

    auto output_shape = input_shape;
    output_shape.RemoveDim(axis_);
    const int64 output_size = output_shape.num_elements();
    OP_REQUIRES(
        c,
        FastBoundsCheck(output_size,
                        std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("output size must fit in Eigen DenseIndex"));

    Eigen::DenseIndex before_dim = 1;
    for (int i = 0; i < axis_; ++i) {
      before_dim *= input_shape.dim_size(i);
    }

    Eigen::DenseIndex after_dim = 1;
    for (int i = axis_ + 1; i < input_shape.dims(); ++i) {
      after_dim *= input_shape.dim_size(i);
    }
    const Eigen::DenseIndex axis_dim = input_shape.dim_size(axis_);

    // Except for shape, unpack is a special case of split, so we reuse the
    // same computational kernels.

    const int max_out_num_handle = 8;
    for (int i = 0; i < num; i += max_out_num_handle) {
      int out_num_handle =
          num - i < max_out_num_handle ? num - i : max_out_num_handle;
      GpuDeviceArrayOnHost<T*> ptrs(c, out_num_handle);
      OP_REQUIRES_OK(c, ptrs.Init());

      for (int j = 0; j < out_num_handle; ++j) {
        Tensor* output = nullptr;
        OP_REQUIRES_OK(c, c->allocate_output(i + j, output_shape, &output));
        ptrs.Set(j, output->flat<T>().data());
      }
      if (output_shape.num_elements() > 0) {
        OP_REQUIRES_OK(c, ptrs.Finalize());
        functor::SplitGpuFunctor<T>()(c->eigen_device<GPUDevice>(),
                                      input.flat<T>().data(), before_dim,
                                      axis_dim, after_dim, 1, i, ptrs.data());
      }
    }
  }

 private:
  int axis_;
};

#define REGISTER_GPU(type)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Unpack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      UnpackOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU

}  // namespace itex
