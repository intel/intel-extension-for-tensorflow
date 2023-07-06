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

#ifndef ITEX_CORE_IR_IMPORTEXPORT_MANGLING_H_
#define ITEX_CORE_IR_IMPORTEXPORT_MANGLING_H_

#include <string>

#include "absl/strings/string_view.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/types.h"
#include "protos/tensor.pb.h"
#include "protos/tensor_shape.pb.h"

namespace mlir {
namespace tfg {
namespace mangling_util {
// The type of a mangled string.
enum class MangledKind { kUnknown, kDataType, kTensorShape, kTensor };

// Mangles an attribute name, marking the attribute as a TensorFlow attribute.
std::string MangleAttributeName(absl::string_view str);

// Returns true if 'str' was mangled with MangleAttributeName.
bool IsMangledAttributeName(absl::string_view str);

// Demangles an attribute name that was manged with MangleAttributeName.
// REQUIRES: IsMangledAttributeName returns true.
absl::string_view DemangleAttributeName(absl::string_view str);

// Returns the type of a mangled string, or kUnknown.
MangledKind GetMangledKind(absl::string_view str);

// Return a TensorShapeProto mangled as a string.
std::string MangleShape(const itex::TensorShapeProto& shape);
// Demangle a string mangled with MangleShape.
itex::Status DemangleShape(absl::string_view str,
                           itex::TensorShapeProto* proto);

// Return a TensorProto mangled as a string.
std::string MangleTensor(const itex::TensorProto& tensor);
// Demangle a string mangled with MangleTensor.
itex::Status DemangleTensor(absl::string_view str, itex::TensorProto* proto);

// Return a DataType mangled as a string.
std::string MangleDataType(const itex::DataType& dtype);
// Demangle a string mangled with MangleDataType.
itex::Status DemangleDataType(absl::string_view str, itex::DataType* proto);

}  // namespace mangling_util
}  // namespace tfg
}  // namespace mlir

#endif  // ITEX_CORE_IR_IMPORTEXPORT_MANGLING_H_
