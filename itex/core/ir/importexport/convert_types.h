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

#ifndef ITEX_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_
#define ITEX_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_

#include "itex/core/utils/statusor.h"
#include "itex/core/utils/tensor_shape.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Types.h"     // from @llvm-project
#include "protos/tensor_shape.pb.h"
#include "protos/types.pb.h"

namespace mlir {
namespace tfg {
// Converts the TensorFlow DataType 'dtype' into an MLIR (scalar) type.
itex::Status ConvertDataType(itex::DataType dtype, Builder& builder,  // NOLINT
                             Type* type);

// Converts a scalar MLIR type to a TensorFlow Datatype.
itex::Status ConvertScalarTypeToDataType(Type type, itex::DataType* dtype);

// Converts an MLIR type to TensorFlow DataType. If 'type' is a scalar type, it
// is converted directly. If it is a shaped type, the element type is converted.
itex::Status ConvertToDataType(Type type, itex::DataType* dtype);

// Converts an TensorFlow shape to the one used in MLIR.
void ConvertToMlirShape(const itex::TensorShape& input_shape,
                        SmallVectorImpl<int64_t>* shape);

// Converts an TensorFlow shape proto to the one used in MLIR.
itex::Status ConvertToMlirShape(const itex::TensorShapeProto& input_shape,
                                SmallVectorImpl<int64_t>* shape);

// Given a tensor shape and dtype, get the corresponding MLIR tensor type.
itex::StatusOr<Type> ConvertToMlirTensorType(
    const itex::TensorShapeProto& shape, itex::DataType dtype,
    Builder* builder);

}  // namespace tfg
}  // namespace mlir

#endif  // ITEX_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_
