/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/service/gpu/cusolver_rewriter.h"

#include <cstdlib>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "itex/core/compiler/xla/literal.h"
#include "itex/core/compiler/xla/literal_util.h"
#include "itex/core/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "itex/core/compiler/xla/service/gpu/ir_emission_utils.h"
#include "itex/core/compiler/xla/service/gpu/mkl.h"
#include "itex/core/compiler/xla/service/hlo_computation.h"
#include "itex/core/compiler/xla/service/hlo_instruction.h"
#include "itex/core/compiler/xla/service/hlo_opcode.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/status.h"
#include "protos/xla_data.pb.h"
#include "third_party/build_option/dpcpp/runtime/eigen_itex_gpu_runtime.h"

namespace itex_xla {
namespace gpu {

namespace {

void SetFortranLayout(Shape* shape) {
  LayoutUtil::SetToDefaultLayout(shape);
  int n = shape->mutable_layout()->minor_to_major_size();
  ITEX_CHECK_GE(n, 2);
  std::swap(shape->mutable_layout()->mutable_minor_to_major()->at(0),
            shape->mutable_layout()->mutable_minor_to_major()->at(1));
}

StatusOr<HloInstruction*> CreateCholesky(HloInstruction* operand,
                                         const CholeskyOptions& options,
                                         const OpMetadata& metadata) {
  HloComputation* computation = operand->parent();

  Shape a_shape = operand->shape();
  int ndim = a_shape.dimensions_size();
  ITEX_CHECK_GE(ndim, 2);
  int64_t n = a_shape.dimensions(ndim - 1);

  std::vector<int64_t> batch_dims(a_shape.dimensions().begin(),
                                  a_shape.dimensions().end() - 2);
  std::vector<int64_t> batch_dim_ids(batch_dims.size());
  absl::c_iota(batch_dim_ids, 0);
  int64_t batch_size = absl::c_accumulate(batch_dims, 1, std::multiplies<>{});

  // workspace is allocated in Thunk execution.
  int64_t workspace_size = 0;

#if ITEX_USE_MKL
  // Find the workspace size.
  oneapi::mkl::uplo uplo =
      options.lower() ? oneapi::mkl::uplo::L : oneapi::mkl::uplo::U;
  sycl::property_list propList{sycl::property::queue::in_order()};
  sycl::queue queue(sycl::gpu_selector{}, propList);
  switch (a_shape.element_type()) {
    case F32:
      workspace_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<float>(
          queue, uplo, n, n, n * n, batch_size);
      break;
    case F64:
      workspace_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<double>(
          queue, uplo, n, n, n * n, batch_size);
      break;
    case C64:
      workspace_size =
          oneapi::mkl::lapack::potrf_batch_scratchpad_size<std::complex<float>>(
              queue, uplo, n, n, n * n, batch_size);
      break;
    case C128:
      workspace_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<
          std::complex<double>>(queue, uplo, n, n, n * n, batch_size);
      break;
    default:
      return InvalidArgument("Invalid type for cholesky %s",
                             PrimitiveType_Name(a_shape.element_type()));
  }
#endif

  // TODO(phawkins): Ideally we would relax this constraint. What we actually
  // want is that:
  // a) the batch dimensions are major, in no particular order.
  // b) the two minor dimensions are in fortran (column-major) order,

  SetFortranLayout(&a_shape);

  // This call returns a tuple of (cholesky_result, workspace, info) where:
  // * cholesky_result is the result of the Cholesky decomposition,
  // * workspace is temporary scratch memory used by cuSolver.
  // * info contains the Potrf success/failure status.
  // Currently we have no meaningful way to report an error, so we simply
  // discard the success/failure information. Obviously this is suboptimal.
  Shape info_shape = ShapeUtil::MakeShape(S32, batch_dims);
  Shape call_shape = ShapeUtil::MakeTupleShape(
      {a_shape,
       ShapeUtil::MakeShape(operand->shape().element_type(), {workspace_size}),
       info_shape});

  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, {operand}, kCusolverCholeskyCallTarget, {a_shape}));
  custom_call->set_metadata(metadata);
  TF_RETURN_IF_ERROR(custom_call->set_backend_config(options));
  HloInstruction* out = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(a_shape, custom_call, 0));
  return out;
}

// Tries to rewrite a single convolution into a call to cudnn.
StatusOr<bool> RunOnInstruction(HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCholesky) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * custom_call,
      CreateCholesky(instruction->mutable_operand(0),
                     instruction->cholesky_options(), instruction->metadata()));

  ITEX_VLOG(1) << "Replacing " << instruction->ToString() << " with "
               << custom_call->ToString();

  TF_RETURN_IF_ERROR(
      instruction->parent()->ReplaceInstruction(instruction, custom_call));
  return true;
}

}  // namespace

// Rewrites the convolutions in the given computation into calls to cudnn.
// Returns true if it made any changes.
StatusOr<bool> GpusolverRewriter::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloInstruction*> cusolver_calls;
  for (auto* hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kCholesky) {
      cusolver_calls.push_back(hlo);
    }
  }

  if (cusolver_calls.empty()) {
    return false;
  }

  bool changed = false;
  for (HloInstruction* instruction : cusolver_calls) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instruction));
    changed |= result;
  }
  return changed;
}

GpusolverRewriter::GpusolverRewriter() = default;

StatusOr<bool> GpusolverRewriter::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace itex_xla
