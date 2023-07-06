/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/graph/utils/symbolic_shapes.h"

namespace itex {
namespace graph {
namespace {

BCast::Vec ShapeDims(const TensorShapeProto& shape) {
  BCast::Vec dims;
  dims.reserve(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i)
    dims.push_back(shape.dim(i).size());
  return dims;
}

}  // namespace

bool IsUnknown(const TensorShapeProto::Dim& dim) { return dim.size() == -1; }

bool ShapesSymbolicallyEqual(const TensorShapeProto& left,
                             const TensorShapeProto& right) {
  if (left.unknown_rank() || right.unknown_rank() ||
      left.dim_size() != right.dim_size()) {
    return false;
  }
  for (int i = 0; i < left.dim_size(); ++i) {
    const auto& ldim = left.dim(i);
    const auto& rdim = right.dim(i);
    if (IsUnknown(ldim) || IsUnknown(rdim) || ldim.size() != rdim.size()) {
      return false;
    }
  }
  return true;
}

bool ShapesSymbolicallyEqualExceptBatch(const TensorShapeProto& left,
                                        const TensorShapeProto& right) {
  if (left.unknown_rank() || right.unknown_rank() ||
      left.dim_size() != right.dim_size()) {
    return false;
  }
  for (int i = 1; i < left.dim_size(); ++i) {
    const auto& ldim = left.dim(i);
    const auto& rdim = right.dim(i);
    if (IsUnknown(ldim) || IsUnknown(rdim) || ldim.size() != rdim.size()) {
      return false;
    }
  }
  return true;
}

int Rank(const TensorShapeProto& shape) {
  if (shape.unknown_rank()) {
    return -1;
  }
  return shape.dim_size();
}

int64_t NumCoefficients(const TensorShapeProto& shape) {
  if (shape.unknown_rank()) {
    return -1;
  }
  int64_t num_coefficients = 1;
  for (const auto& dim : shape.dim()) {
    if (dim.size() < 0) {
      return -1;
    }
    num_coefficients *= dim.size();
  }
  return num_coefficients;
}

bool ShapeIsSymbolicallyDefined(const TensorShapeProto& shape) {
  return !shape.unknown_rank() &&
         std::all_of(
             shape.dim().begin(), shape.dim().end(),
             [](const TensorShapeProto::Dim& dim) { return !IsUnknown(dim); });
}

bool ShapeIsSymbolicallyDefined(const OpInfo::TensorProperties& properties) {
  return ShapeIsSymbolicallyDefined(properties.shape());
}

bool ShapesBroadcastable(const TensorShapeProto& left,
                         const TensorShapeProto& right) {
  if (!ShapeIsSymbolicallyDefined(left) || !ShapeIsSymbolicallyDefined(right)) {
    return false;
  }
  BCast bcast(ShapeDims(left), ShapeDims(right),
              /*fewer_dims_optimization*/ false);
  return bcast.IsValid();
}

bool ShapesBroadcastable(const OpInfo::TensorProperties& left,
                         const OpInfo::TensorProperties& right) {
  return ShapesBroadcastable(left.shape(), right.shape());
}

}  // end namespace graph
}  // end namespace itex
