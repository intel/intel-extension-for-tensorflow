/* Copyright (c) 2023 Intel Corporation

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/service/gpu/mkl.h"

#include <string>

#include "itex/core/utils/integral_types.h"

namespace itex_xla {
namespace gpu {

bool IsCublasGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kGemmCallTarget;
}

const char* const kGemmCallTarget = "__cublas$gemm";

const char* const kTriangularSolveCallTarget = "__cublas$triangularSolve";
const char* const kCudnnConvForwardCallTarget = "__cudnn$convForward";
const char* const kCudnnConvBackwardInputCallTarget =
    "__cudnn$convBackwardInput";
const char* const kCudnnConvBackwardFilterCallTarget =
    "__cudnn$convBackwardFilter";
const char* const kCudnnConvBiasActivationForwardCallTarget =
    "__cudnn$convBiasActivationForward";

bool IsCustomCallToDnnConvolution(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnConvForwardCallTarget ||
         target == kCudnnConvBackwardInputCallTarget ||
         target == kCudnnConvBackwardFilterCallTarget ||
         target == kCudnnConvBiasActivationForwardCallTarget;
}

StatusOr<CudnnConvKind> GetCudnnConvKind(
    const HloCustomCallInstruction* instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kCudnnConvForwardCallTarget) {
    return CudnnConvKind::kForward;
  }
  if (target == kCudnnConvBackwardInputCallTarget) {
    return CudnnConvKind::kBackwardInput;
  }
  if (target == kCudnnConvBackwardFilterCallTarget) {
    return CudnnConvKind::kBackwardFilter;
  }
  if (target == kCudnnConvBiasActivationForwardCallTarget) {
    return CudnnConvKind::kForwardActivation;
  }
  return InternalError("Unexpected call target: %s", target);
}

std::string CudnnConvKindToString(CudnnConvKind kind) {
  switch (kind) {
    case CudnnConvKind::kForward:
      return "forward";
    case CudnnConvKind::kBackwardFilter:
      return "backward_filter";
    case CudnnConvKind::kBackwardInput:
      return "backward_input";
    case CudnnConvKind::kForwardActivation:
      return "forward with activation";
  }
}

#if ITEX_USE_MKL
std::string TransposeString(oneapi::mkl::transpose t) {
  switch (t) {
    case oneapi::mkl::transpose::N:
      return "NoTranspose";
    case oneapi::mkl::transpose::T:
      return "Transpose";
    case oneapi::mkl::transpose::C:
      return "ConjugateTranspose";
    default:
      ITEX_LOG(FATAL) << "Unknown transpose " << static_cast<itex::int32>(t);
  }
}

std::string UpperLowerString(oneapi::mkl::uplo ul) {
  switch (ul) {
    case oneapi::mkl::uplo::U:
      return "Upper";
    case oneapi::mkl::uplo::L:
      return "Lower";
    default:
      ITEX_LOG(FATAL) << "Unknown upperlower " << static_cast<itex::int32>(ul);
  }
}

std::string DiagonalString(oneapi::mkl::diag d) {
  switch (d) {
    case oneapi::mkl::diag::U:
      return "Unit";
    case oneapi::mkl::diag::N:
      return "NonUnit";
    default:
      ITEX_LOG(FATAL) << "Unknown diagonal " << static_cast<itex::int32>(d);
  }
}

std::string SideString(oneapi::mkl::side s) {
  switch (s) {
    case oneapi::mkl::side::L:
      return "Left";
    case oneapi::mkl::side::R:
      return "Right";
    default:
      ITEX_LOG(FATAL) << "Unknown side " << static_cast<itex::int32>(s);
  }
}

#endif  // ITEX_USE_MKL
}  // namespace gpu
}  // namespace itex_xla
