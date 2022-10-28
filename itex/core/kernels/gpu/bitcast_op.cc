/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

class BitcastOp : public OpKernel {
 public:
  explicit BitcastOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("T", &input_data_type_));
    OP_REQUIRES_OK(context, context->GetAttr("type", &output_data_type_));
    in_size_ = DataTypeSize(input_data_type_);
    out_size_ = DataTypeSize(output_data_type_);
    int check_size =
        std::max(in_size_, out_size_) % std::min(in_size_, out_size_);
    OP_REQUIRES(
        context, check_size == 0,
        errors::InvalidArgument("cannot convert between datatype ",
                                input_data_type_, " and ", output_data_type_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    TensorShape adjusted_shape = input_tensor.shape();
    OP_REQUIRES(
        context,
        in_size_ >= out_size_ ||
            (input_tensor.dims() > 0 &&
             input_tensor.dim_size(input_tensor.dims() - 1) ==
                 out_size_ / in_size_),
        errors::InvalidArgument("Cannot bitcast from ", input_data_type_,
                                " to ", output_data_type_, ": shape ",
                                input_tensor.shape().DebugString()));

    if (out_size_ < in_size_) {
      adjusted_shape.AddDim(in_size_ / out_size_);
    } else if (out_size_ > in_size_) {
      adjusted_shape.RemoveDim(input_tensor.dims() - 1);
    }
    Tensor output_tensor;

    OP_REQUIRES_OK(context,
                   output_tensor.BitcastFrom(input_tensor, output_data_type_,
                                             adjusted_shape));
    context->set_output(0, output_tensor);
  }

 private:
  DataType input_data_type_;
  DataType output_data_type_;
  int in_size_;
  int out_size_;
};

REGISTER_KERNEL_BUILDER(Name("Bitcast").Device(DEVICE_GPU), BitcastOp);

}  // namespace itex
