/* Copyright (c) 2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/scan_ops.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int32>("Tidx")                                 \
          .HostMemory("axis"),                                           \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int64>("Tidx")                                 \
          .HostMemory("axis"),                                           \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>, int64>)

TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU_KERNELS

}  // namespace itex
