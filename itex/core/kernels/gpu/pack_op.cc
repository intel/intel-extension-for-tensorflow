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

#include <vector>

#include "itex/core/kernels/gpu/concat_lib.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class PackOp : public OpKernel {
 public:
  explicit PackOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* c) override {
    const int num = c->num_inputs();
    const Tensor& first_input = c->input(0);

    int expanded_num_dims = first_input.dims() + 1;
    if (axis_ < 0) axis_ += expanded_num_dims;

    OP_REQUIRES(c, 0 <= axis_ && axis_ < expanded_num_dims,
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -expanded_num_dims, ", ",
                                        expanded_num_dims, ")"));

    TensorShape output_shape(first_input.shape());
    output_shape.InsertDim(axis_, num);

    // In the num = 1 case, just reshape the input
    if (num == 1) {
      Tensor output;
      ITEX_CHECK(output.CopyFrom(first_input, output_shape));
      c->set_output(0, output);
      return;
    }

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));

    int64 before_dim = 1;
    for (int i = 0; i < axis_; ++i) {
      before_dim *= output_shape.dim_size(i);
    }

    int64 after_dim = 1;
    for (int i = axis_ + 1; i < output_shape.dims(); ++i) {
      after_dim *= output_shape.dim_size(i);
    }

    const int64 axis_dim = output_shape.dim_size(axis_);

    const int64 output_size = output->NumElements();
    if (output_size > 0) {
      auto output_flat =
          output->shaped<T, 2>({before_dim, after_dim * axis_dim});

      // Except for shapes, pack is a special case of concat, so we reuse the
      // same computational kernels.
      typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
          ConstMatrixVector;
      ConstMatrixVector inputs_flat;
      inputs_flat.reserve(num);
      for (int i = 0; i < num; ++i) {
        const Tensor& input = c->input(i);
        OP_REQUIRES(c, first_input.shape().IsSameSize(input.shape()),
                    errors::InvalidArgument(
                        "Shapes of all inputs must match: values[0].shape = ",
                        first_input.shape().DebugString(), " != values[", i,
                        "].shape = ", input.shape().DebugString()));

        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            input.shaped<T, 2>({before_dim, after_dim})));
      }
      Concat<T>(c, inputs_flat, &output_flat);
    }
  }

 private:
  int axis_;
};

#define REGISTER_GPU(type)                                       \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Pack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PackOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_bool(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU

}  // namespace itex
