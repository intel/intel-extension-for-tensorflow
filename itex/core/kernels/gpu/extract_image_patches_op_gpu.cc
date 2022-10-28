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

#include "itex/core/kernels/gpu/extract_image_patches_op.h"
#include "itex/core/utils/register_types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

#define REGISTER(T) template struct ExtractImagePatchesForward<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(REGISTER);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER);
TF_CALL_complex128(REGISTER);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_complex64(REGISTER);

#undef REGISTER

}  // end namespace functor
}  // end namespace itex
