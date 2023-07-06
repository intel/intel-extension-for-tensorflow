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

#include "itex/core/compiler/mlir/tensorflow/utils/convert_type.h"

#include "absl/strings/str_cat.h"
#include "itex/core/compiler/mlir/tensorflow/ir/tf_types.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/types.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/IR/Types.h"                   // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "protos/types.pb.h"

namespace itex {

using mlir::Builder;
using mlir::ShapedType;
using mlir::Type;

Status ConvertScalarTypeToDataType(Type type, DataType* dtype) {
  if (type.isF16()) {
    *dtype = DT_HALF;
    return Status::OK();
  } else if (type.isF32()) {
    *dtype = DT_FLOAT;
    return Status::OK();
  } else if (type.isF64()) {
    *dtype = DT_DOUBLE;
    return Status::OK();
  } else if (type.isBF16()) {
    *dtype = DT_BFLOAT16;
    return Status::OK();
  } else if (auto itype = type.dyn_cast<mlir::IntegerType>()) {
    switch (itype.getWidth()) {
      case 1:
        *dtype = DT_BOOL;
        return Status::OK();
      case 8:
        *dtype = itype.isUnsigned() ? DT_UINT8 : DT_INT8;
        return Status::OK();
      case 16:
        *dtype = itype.isUnsigned() ? DT_UINT16 : DT_INT16;
        return Status::OK();
      case 32:
        *dtype = itype.isUnsigned() ? DT_UINT32 : DT_INT32;
        return Status::OK();
      case 64:
        *dtype = itype.isUnsigned() ? DT_UINT64 : DT_INT64;
        return Status::OK();
      default:
        return errors::Unimplemented(
            absl::StrCat("Converting ", debugString(type), " to DataType"));
    }
  } else if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto etype = complex_type.getElementType();
    if (etype.isF32()) {
      *dtype = DT_COMPLEX64;
      return Status::OK();
    } else if (etype.isF64()) {
      *dtype = DT_COMPLEX128;
      return Status::OK();
    }
    return errors::Unimplemented(
        absl::StrCat("Converting ", debugString(type), " to DataType"));
  }

#define HANDLE_TF_TYPE(tftype, enumerant, name)    \
  if (type.isa<mlir::itex_type::tftype##Type>()) { \
    *dtype = DT_##enumerant;                       \
    return Status::OK();                           \
  }
// NOLINTNEXTLINE
#include "itex/core/compiler/mlir/tensorflow/ir/tf_types.def"

  return errors::Unimplemented(
      absl::StrCat("Converting ", debugString(type), " to DataType"));
}

Status ConvertToDataType(Type type, DataType* dtype) {
  if (auto stype = type.dyn_cast<ShapedType>()) {
    TF_RETURN_IF_ERROR(
        ConvertScalarTypeToDataType(stype.getElementType(), dtype));
  } else {
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, dtype));
  }
  return Status::OK();
}

}  // namespace itex
