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

#include "itex/core/kernels/common/avgpooling_op.h"

#include "itex/core/utils/register_types.h"

namespace itex {
#define REGISTER_GPU_POOL_KERNELS(T)                                         \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("AvgPool").Device(DEVICE_GPU).TypeConstraint<T>("T"),             \
      PoolingOp<GPUDevice, T, dnnl::algorithm::pooling_avg_exclude_padding>) \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("AvgPool3D").Device(DEVICE_GPU).TypeConstraint<T>("T"),           \
      PoolingOp<GPUDevice, T, dnnl::algorithm::pooling_avg_exclude_padding>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_POOL_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_POOL_KERNELS);
#endif
#undef REGISTER_GPU_POOL_KERNELS

#define REGISTER_GPU_POOL_GRAD_KERNELS(T)                             \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AvgPoolGrad")                                             \
          .Device(DEVICE_GPU)                                         \
          .TypeConstraint<T>("T")                                     \
          .HostMemory("orig_input_shape"),                            \
      AvgPoolGradOp<GPUDevice, T, dnnl::prop_kind::forward_training>) \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AvgPool3DGrad")                                           \
          .Device(DEVICE_GPU)                                         \
          .TypeConstraint<T>("T")                                     \
          .HostMemory("orig_input_shape"),                            \
      AvgPoolGradOp<GPUDevice, T, dnnl::prop_kind::forward_training>)

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_GPU_POOL_GRAD_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_POOL_GRAD_KERNELS);
#endif
#undef REGISTER_GPU_POOL_GRAD_KERNELS

// Quantized Kernels
// TF INT8 kernel
#define REGISTER_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("QuantizedAvgPool")                                 \
          .Device(DEVICE_GPU)                                  \
          .HostMemory("min_input")                             \
          .HostMemory("max_input")                             \
          .HostMemory("min_output")                            \
          .HostMemory("max_output")                            \
          .TypeConstraint<TYPE>("T"),                          \
      PoolingOp<GPUDevice, TYPE,                               \
                dnnl::algorithm::pooling_avg_exclude_padding>) \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("ITEXQuantizedAvgPool")                             \
          .Device(DEVICE_GPU)                                  \
          .HostMemory("min_input")                             \
          .HostMemory("max_input")                             \
          .HostMemory("min_output")                            \
          .HostMemory("max_output")                            \
          .TypeConstraint<TYPE>("T"),                          \
      PoolingOp<GPUDevice, TYPE,                               \
                dnnl::algorithm::pooling_avg_exclude_padding>)

TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace itex
