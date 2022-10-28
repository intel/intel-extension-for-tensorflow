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

#include "itex/core/kernels/onednn/block/resize_op.h"
#include "itex/core/utils/register_types.h"

namespace itex {
#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(T)                    \
  REGISTER_KERNEL_BUILDER(                    \
      Name("_OneDnnResizeBilinear")           \
          .Device(DEVICE_GPU)                 \
          .TypeConstraint<T>("T")             \
          .HostMemory("size")                 \
          .HostMemory("images_meta")          \
          .HostMemory("size_meta")            \
          .HostMemory("resized_images_meta"), \
      OneDnnResizeOp<GPUDevice, T, dnnl::algorithm::resampling_linear>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#define REGISTER_GRAD_KERNEL(T)              \
  REGISTER_KERNEL_BUILDER(                   \
      Name("_OneDnnResizeBilinearGrad")      \
          .Device(DEVICE_GPU)                \
          .TypeConstraint<T>("T")            \
          .HostMemory("grads_meta")          \
          .HostMemory("original_image_meta") \
          .HostMemory("output_meta"),        \
      OneDnnResizeGradOp<GPUDevice, T, dnnl::algorithm::resampling_linear>);

TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_GRAD_KERNEL);
#undef REGISTER_KERNEL
#undef REGISTER_GRAD_KERNEL
#else
#define REGISTER_KERNEL(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_OneDnnResizeBilinear").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OneDnnResizeOp<CPUDevice, T, dnnl::algorithm::resampling_linear>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

#define REGISTER_GRAD_KERNEL(T)         \
  REGISTER_KERNEL_BUILDER(              \
      Name("_OneDnnResizeBilinearGrad") \
          .Device(DEVICE_CPU)           \
          .TypeConstraint<T>("T"),      \
      OneDnnResizeGradOp<CPUDevice, T, dnnl::algorithm::resampling_linear>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_GRAD_KERNEL);
#undef REGISTER_KERNEL
#undef REGISTER_GRAD_KENREL
#endif
}  // namespace itex
