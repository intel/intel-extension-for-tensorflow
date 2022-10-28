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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "itex/core/kernels/gpu/debug_ops.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(in_type, out_type) \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DebugNumericSummaryV2")                              \
          .Device(DEVICE_GPU)                                    \
          .TypeConstraint<in_type>("T")                          \
          .TypeConstraint<out_type>("output_dtype"),             \
      DebugNumericSummaryV2Op<GPUDevice, in_type, out_type>);

REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(Eigen::half, float);
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(Eigen::bfloat16, float);
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(float, float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(double, float);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(int16, float);
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(int32, float);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(Eigen::half, double);
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(Eigen::bfloat16, double);
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(float, double);
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(double, double);
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(int16, double);
REGISTER_DEBUG_NUMERIC_SUMMARY_V2_GPU(int32, double);
#endif  // ITEX_ENABLE_DOUBLE

}  // namespace itex
