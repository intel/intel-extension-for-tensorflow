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

#include "itex/core/kernels/common/maxpooling_op.h"
#include "itex/core/utils/register_types.h"

namespace itex {

#define REGISTER_CPU_POOL_KERNELS(T)                                    \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ITEXMaxPool").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      PoolingOp<CPUDevice, T, dnnl::algorithm::pooling_max>);           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ITEXMaxPool3D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      PoolingOp<CPUDevice, T, dnnl::algorithm::pooling_max>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_CPU_POOL_KERNELS);
#undef REGISTER_CPU_POOL_KERNELS

#define REGISTER_CPU_POOL_GRAD_KERNELS(T)                                   \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_ITEXMaxPoolGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      MaxPoolGradOp<CPUDevice, T, dnnl::prop_kind::forward_training>);      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_ITEXMaxPool3DGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MaxPoolGradOp<CPUDevice, T, dnnl::prop_kind::forward_training>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_CPU_POOL_GRAD_KERNELS);
#undef REGISTER_CPU_POOL_GRAD_KERNELS

// ITEX native INT8 kernel
#define REGISTER_KERNEL(TYPE)         \
  REGISTER_KERNEL_BUILDER(            \
      Name("_ITEXQuantizedMaxPool")   \
          .Device(DEVICE_CPU)         \
          .TypeConstraint<TYPE>("T"), \
      PoolingOp<CPUDevice, TYPE, dnnl::algorithm::pooling_max>)

TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#undef REGISTER_KERNEL
}  // namespace itex
