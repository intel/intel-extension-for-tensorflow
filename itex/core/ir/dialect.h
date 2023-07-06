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

#ifndef ITEX_CORE_IR_DIALECT_H_
#define ITEX_CORE_IR_DIALECT_H_

#include "itex/core/ir/types/dialect.h"
#include "mlir/IR/BuiltinTypes.h"      // from @llvm-project
#include "mlir/IR/Diagnostics.h"       // from @llvm-project
#include "mlir/IR/Dialect.h"           // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"     // from @llvm-project

namespace mlir {
namespace tfg {
// Include the relevant TensorFlow attrs/types directly in the TFG namespace.
using mlir::itex_type::Bfloat16RefType;    // NOLINT
using mlir::itex_type::BoolRefType;        // NOLINT
using mlir::itex_type::Complex128RefType;  // NOLINT
using mlir::itex_type::Complex64RefType;   // NOLINT
using mlir::itex_type::ControlType;        // NOLINT
using mlir::itex_type::DoubleRefType;      // NOLINT
using mlir::itex_type::FloatRefType;       // NOLINT
using mlir::itex_type::FuncAttr;           // NOLINT
using mlir::itex_type::HalfRefType;        // NOLINT
using mlir::itex_type::Int16RefType;       // NOLINT
using mlir::itex_type::Int32RefType;       // NOLINT
using mlir::itex_type::Int64RefType;       // NOLINT
using mlir::itex_type::Int8RefType;        // NOLINT
using mlir::itex_type::OpaqueTensorType;   // NOLINT
using mlir::itex_type::PlaceholderAttr;    // NOLINT
using mlir::itex_type::Qint16RefType;      // NOLINT
using mlir::itex_type::Qint16Type;         // NOLINT
using mlir::itex_type::Qint32RefType;      // NOLINT
using mlir::itex_type::Qint32Type;         // NOLINT
using mlir::itex_type::Qint8RefType;       // NOLINT
using mlir::itex_type::Qint8Type;          // NOLINT
using mlir::itex_type::Quint16RefType;     // NOLINT
using mlir::itex_type::Quint16Type;        // NOLINT
using mlir::itex_type::Quint8RefType;      // NOLINT
using mlir::itex_type::Quint8Type;         // NOLINT
using mlir::itex_type::ResourceRefType;    // NOLINT
using mlir::itex_type::ResourceType;       // NOLINT
using mlir::itex_type::ShapeAttr;          // NOLINT
using mlir::itex_type::StringRefType;      // NOLINT
using mlir::itex_type::StringType;         // NOLINT
using mlir::itex_type::Uint16RefType;      // NOLINT
using mlir::itex_type::Uint32RefType;      // NOLINT
using mlir::itex_type::Uint64RefType;      // NOLINT
using mlir::itex_type::Uint8RefType;       // NOLINT
using mlir::itex_type::VariantRefType;     // NOLINT
using mlir::itex_type::VariantType;        // NOLINT
using mlir::itex_type::VersionAttr;        // NOLINT

struct TFGraphOpAsmInterface;
class TFOp;
}  // namespace tfg
}  // namespace mlir

// Dialect main class is defined in ODS, we include it here.
#include "itex/core/ir/dialect.h.inc"
// ODS-generated attribute classes.
#define GET_ATTRDEF_CLASSES
#include "itex/core/ir/attributes.h.inc"

#endif  // ITEX_CORE_IR_DIALECT_H_
