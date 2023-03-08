/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.mlir_hlo_to_hlo.h

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

#ifndef ITEX_CORE_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_
#define ITEX_CORE_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_

#include <string>
#include <vector>

#include "itex/core/compiler/mlir/tensorflow/utils/error_util.h"
#include "itex/core/compiler/mlir/xla/utils/layout_util.h"
#include "itex/core/compiler/mlir/xla/utils/xla_helpers.h"
#include "itex/core/compiler/xla/client/xla_builder.h"
#include "itex/core/compiler/xla/service/hlo_module.h"
#include "itex/core/utils/tensor_shape.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/IR/Location.h"              // from @llvm-project
#include "mlir/IR/Operation.h"             // from @llvm-project

namespace mlir {

struct MlirToHloConversionOptions {
  // Best-effort propagation of the layouts. These layouts serve as performance
  // hints to the backend.
  //
  // Note that non-array shapes are not carrying layouts, and users have to
  // figure out the proper layouts of them through context. This is one of the
  // reasons why the attribute-based solution is temporary.
  //
  // TODO(timshen): Investigate the necessity of having layouts in MHLO.
  bool propagate_layouts = false;

  // Propagate the source and result layouts from mhlo bitcast op into the
  // backend config for the bitcast. This is required for XLA:GPU backend to
  // use elemental IR emitters for fused bitcasts without propagating layouts.
  bool propagate_bitcast_layouts_to_backend_config = false;
};

// Converts a MLIR module in HLO dialect into a HloModuleProto. If
// use_tuple_args is set, then the entry computations's arguments are converted
// to a tuple and passed as a single parameter.
// Similarly, if return tuple is true, then the entry function's return values
// are converted to a tuple even when there is only a single return value.
// Multiple return values are always converted to a tuple and returned as a
// single value.
//
// TODO(timshen): move other options into `options`.
Status ConvertMlirHloToHlo(
    mlir::ModuleOp module, ::itex_xla::HloProto* hlo_proto, bool use_tuple_args,
    bool return_tuple,
    // const itex::XlaShapeLayoutHelpers::ShapeDeterminationFns
    //     shape_determination_fns = {},
    MlirToHloConversionOptions options = {});

// Transforms a Block into HLO, where the HLO is represented as calls into an
// XlaBuilder. Callee functions are allowed in the Block's ancestor ModuleOp.
// xla_params are inputs to block. returns are the returned XlaOps.
Status BuildHloFromMlirHlo(mlir::Block& block, itex_xla::XlaBuilder& builder,
                           llvm::ArrayRef<itex_xla::XlaOp> xla_params,
                           std::vector<itex_xla::XlaOp>& returns,
                           MlirToHloConversionOptions options = {});

// Converts a region to a computation. It returns a standalone module that
// contains the converted region as the entry computation.
Status ConvertRegionToComputation(mlir::Region* region,
                                  ::itex_xla::XlaComputation* func,
                                  MlirToHloConversionOptions options = {});

// Creates XlaOp equivalent of a given MLIR operation using the operand info
// from `value_lowering` map.
llvm::Optional<::itex_xla::XlaOp> CreateXlaOperator(
    mlir::Operation* op,
    llvm::DenseMap<mlir::Value, ::itex_xla::XlaOp>* value_lowering);

namespace mhlo {

// Returns a OpMetadata proto based on the location of the op. If the location
// is unknown, an empty proto is returned. `op_name` are populated with the op
// location (converted). FileLineColLoc locations are populated by taking the
// file name and line number, and populating `source_file` and `source_line`
// respectively.
itex_xla::OpMetadata CreateOpMetadataFromLocation(mlir::Operation* op);

// Returns a name that can be used for debugging purposes, e.g., naming
// variable names in generated IR or producing logging output.
std::string GetDebugNameFromLocation(mlir::Location location);

}  // namespace mhlo
}  // namespace mlir

#endif  // ITEX_CORE_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_
