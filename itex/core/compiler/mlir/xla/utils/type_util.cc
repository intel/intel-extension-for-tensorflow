/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/mlir/xla/utils/type_util.h"

#include "absl/container/flat_hash_map.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/types.h"

namespace itex {

Status DataTypeToPrimitiveType(DataType data_type,
                               itex_xla::PrimitiveType* type) {
  switch (data_type) {
    case itex::DT_BOOL:
      *type = itex_xla::PRED;
      return Status::OK();
    case itex::DT_INT8:
    case itex::DT_QINT8:
      *type = itex_xla::S8;
      return Status::OK();
    case itex::DT_INT16:
    case itex::DT_QINT16:
      *type = itex_xla::S16;
      return Status::OK();
    case itex::DT_INT32:
    case itex::DT_QINT32:
      *type = itex_xla::S32;
      return Status::OK();
    case itex::DT_INT64:
      *type = itex_xla::S64;
      return Status::OK();
    case itex::DT_UINT8:
    case itex::DT_QUINT8:
      *type = itex_xla::U8;
      return Status::OK();
    case itex::DT_UINT16:
    case itex::DT_QUINT16:
      *type = itex_xla::U16;
      return Status::OK();
    case itex::DT_UINT32:
      *type = itex_xla::U32;
      return Status::OK();
    case itex::DT_UINT64:
      *type = itex_xla::U64;
      return Status::OK();
    case itex::DT_BFLOAT16:
      *type = itex_xla::BF16;
      return Status::OK();
    case itex::DT_HALF:
      *type = itex_xla::F16;
      return Status::OK();
    case itex::DT_FLOAT:
      *type = itex_xla::F32;
      return Status::OK();
    case itex::DT_DOUBLE:
      *type = itex_xla::F64;
      return Status::OK();
    case itex::DT_COMPLEX64:
      *type = itex_xla::C64;
      return Status::OK();
    case itex::DT_COMPLEX128:
      *type = itex_xla::C128;
      return Status::OK();
    default:
      return errors::InvalidArgument(
          "Unsupported type in DataTypeToPrimitiveType: '",
          DataTypeString(data_type), "'");
  }
}

StatusOr<DataType> EncodePrimitiveTypeAsDataType(itex_xla::PrimitiveType type) {
  static const absl::flat_hash_map<itex_xla::PrimitiveType, DataType>&
      data_type_map =
          *new absl::flat_hash_map<itex_xla::PrimitiveType, DataType>({
              {itex_xla::PRED, DT_BOOL},
              {itex_xla::BF16, DT_BFLOAT16},
              {itex_xla::F16, DT_HALF},
              {itex_xla::F32, DT_FLOAT},
              {itex_xla::F64, DT_DOUBLE},
              {itex_xla::C64, DT_COMPLEX64},
              {itex_xla::S8, DT_INT8},
              {itex_xla::S16, DT_INT16},
              {itex_xla::S32, DT_INT32},
              {itex_xla::S64, DT_INT64},
              {itex_xla::U8, DT_UINT8},
              {itex_xla::U16, DT_UINT16},
              {itex_xla::U32, DT_UINT32},
              {itex_xla::U64, DT_UINT64},
              {itex_xla::C128, DT_COMPLEX128},
          });

  auto it = data_type_map.find(type);
  if (it == data_type_map.end()) {
    return errors::InvalidArgument(
        "Unsupported type in PrimitiveTypeToDataType ", type);
  }
  return it->second;
}

}  // namespace itex
