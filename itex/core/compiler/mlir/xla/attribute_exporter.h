/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_COMPILER_MLIR_XLA_ATTRIBUTE_EXPORTER_H_
#define ITEX_CORE_COMPILER_MLIR_XLA_ATTRIBUTE_EXPORTER_H_

#include <utility>
#include <vector>

#include "itex/core/compiler/xla/shape_util.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/compiler/xla/types.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "protos/hlo.pb.h"
#include "protos/xla_data.pb.h"

namespace itex_xla {

// Converts the conv dimensions attribute to XLA HLO.
ConvolutionDimensionNumbers ConvertConvDimensionNumbers(
    mlir::mhlo::ConvDimensionNumbersAttr input);

// StatusOr<stream_executor::dnn::ActivationMode> ConvertConvActivationMode(
//     mlir::lmhlo_gpu::Activation input);

StatusOr<std::vector<ReplicaGroup>> ConvertReplicaGroups(
    mlir::DenseIntElementsAttr input);

// Convert a (N, 2) dense attribute to a list of tuples. This is the way padding
// and source-target pairs are defined in HLO.
StatusOr<std::vector<std::pair<int64_t, int64_t>>> ConvertNx2Attribute(
    llvm::Optional<mlir::DenseIntElementsAttr> optional_attr);

StatusOr<FftType> ConvertFftType(llvm::StringRef type_string);
StatusOr<TriangularSolveOptions::Transpose> ConvertTranspose(
    llvm::StringRef transpose_string);

StatusOr<itex_xla::CustomCallApiVersion> ConvertCustomCallApiVersion(
    mlir::mhlo::CustomCallApiVersion api_version);

StatusOr<std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>>
ConvertCustomCallOutputOperandAliasing(mlir::ArrayAttr aliasArrayAttr);
}  // namespace itex_xla
#endif  // ITEX_CORE_COMPILER_MLIR_XLA_ATTRIBUTE_EXPORTER_H_
