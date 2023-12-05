/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/kernels/common/shape_ops.h"

namespace itex {

#ifdef USING_NEXTPLUGGABLE_DEVICE

void AllocateOutputAndReshapePjrtBuffer(OpKernelContext* context,
                                        const Tensor& input,
                                        TensorShape& shape) {  // NOLINT
  auto inp_tf_tensor = const_cast<TF_Tensor*>(input.GetTFTensor());
  if (pointer_is_pjrt_tensor(inp_tf_tensor) && shape != input.shape()) {
    TF_Status* status = TF_NewStatus();
    PJRT_Buffer* inp_buffer = TF_GetPjRtCBuffer(inp_tf_tensor, status);
    PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", status);
    auto tf_context = context->Get();
    int device_id = TF_GetDeviceId(tf_context);
    int rank = shape.dims();
    std::vector<int64_t> dimensions(rank);
    std::vector<int64_t> layout(rank);
    for (int d = 0; d < rank; ++d) {
      dimensions[d] = shape.dim_size(d);
    }
    std::iota(layout.rbegin(), layout.rend(), 0);
    TF_Tensor* out_tf_tensor = TF_AllocateOutput(
        tf_context, 0, static_cast<TF_DataType>(input.dtype()),
        shape.dim_sizes().data(), shape.dims(),
        shape.num_elements() * DataTypeSize(input.dtype()), status);
    TF_CreatePjRtBuffer(
        out_tf_tensor,
        ITEXCopyFromPjRtBuffer(inp_buffer, device_id,
                               DataTypeString(input.dtype()), dimensions,
                               layout, pjrt_c_client),
        "XPU", status);
  } else {
    Tensor output(input.dtype());
    ITEX_CHECK(output.CopyFrom(input, shape));
    context->set_output(0, output);
  }
}

#define REGISTER_GPU_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                           \
                              .Device(DEVICE_GPU)                   \
                              .HostMemory("shape")                  \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("Tshape"),     \
                          ReshapeOp);                               \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                           \
                              .Device(DEVICE_GPU)                   \
                              .HostMemory("shape")                  \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("Tshape"),   \
                          ReshapeOp);                               \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("Tdim")        \
                              .HostMemory("input")                  \
                              .HostMemory("dim")                    \
                              .HostMemory("output"),                \
                          ExpandDimsOp<int32>);                     \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("Tdim")      \
                              .HostMemory("input")                  \
                              .HostMemory("dim")                    \
                              .HostMemory("output"),                \
                          ExpandDimsOp<int64_t>);                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Squeeze").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SqueezeOp);

REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(Eigen::bfloat16);
#undef REGISTER_GPU_KERNEL

#endif  // USING_NEXTPLUGGABLE_DEVICE

}  // namespace itex
