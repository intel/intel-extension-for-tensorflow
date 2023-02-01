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

#ifndef ITEX_CORE_IR_IMPORTEXPORT_CONVERT_ATTRIBUTES_H_
#define ITEX_CORE_IR_IMPORTEXPORT_CONVERT_ATTRIBUTES_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "itex/core/ir/dialect.h"
#include "itex/core/utils/node_def_util.h"
#include "itex/core/utils/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Attributes.h"    // from @llvm-project
#include "mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project

namespace mlir {
namespace itex_type {
class FullTypeAttr;
}  // namespace itex_type

namespace tfg {

// Convert the list of MLIR Attributes `attrs` to the `itex::AttrValueMap`
// `values`.
itex::Status ConvertAttributes(ArrayRef<NamedAttribute> attrs,
                               ArrayRef<StringRef> attrs_to_ignore,
                               bool remove_ref_type,
                               itex::AttrValueMap* values);

// Convert the MLIR attribute `attr` and return a `itex::AttrValue`.
itex::StatusOr<itex::AttrValue> ConvertAttribute(Attribute attr);

itex::Status SetShapeAttribute(absl::string_view name, ShapedType shaped_type,
                               itex::AttrValueMap* values);

// Converts an MLIR shaped type to a TensorFlow shape attribute.
ShapeAttr ConvertTypeToTensorShapeAttr(const Type& type);

/// Import from TensorFlow to MLIR

// Converts non func AttrValue proto into an MLIR attribute. Func attribute is
// exclused in this function because the function might be renamed when the
// function definition is imported.
itex::StatusOr<Attribute> ConvertNonFuncAttributeValue(
    const itex::AttrValue& value, Builder& builder,  // NOLINT
    TFGraphDialect* tfgDialect);

// Converts all kinds of AttrValue proto into an MLIR attribute.
itex::StatusOr<Attribute> ConvertAttributeValue(const itex::AttrValue& value,
                                                Builder& builder,  // //NOLINT
                                                TFGraphDialect* tfgDialect);

// Convert the MLIR FullTyoe attribute `attr` and return a
// `itex::FullTypeDef`.
itex::StatusOr<itex::FullTypeDef> ConvertAttribute(
    itex_type::FullTypeAttr full_type);

// Converts fulltype proto to attribute.
itex::StatusOr<::mlir::itex_type::FullTypeAttr> ConvertAttribute(
    const itex::FullTypeDef& full_type, Builder& builder,  // NOLINT
    TFGraphDialect* tfgDialect);

}  // namespace tfg
}  // namespace mlir

#endif  // ITEX_CORE_IR_IMPORTEXPORT_CONVERT_ATTRIBUTES_H_
