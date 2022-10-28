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

#if ITEX_USE_MKL
#include "itex/core/kernels/gpu/linalg/qr_op_impl.h"

#include "itex/core/utils/types.h"

namespace itex {

#define REGISTER_QR_OP_GPU(T) \
  REGISTER_KERNEL_BUILDER(    \
      Name("Qr").Device(DEVICE_GPU).TypeConstraint<T>("T"), QrOpGpu<T>)

REGISTER_QR_OP_GPU(float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_QR_OP_GPU(double);
#endif  // ITEX_ENABLE_DOUBLE
// complex64 and complex128 are not supported by oneMKL orgqr_batch routine
#undef REGISTER_QR_OP_GPU
}  // namespace itex
#endif  // ITEX_USE_MKL
