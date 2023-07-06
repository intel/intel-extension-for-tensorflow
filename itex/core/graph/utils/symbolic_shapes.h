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

#ifndef ITEX_CORE_GRAPH_UTILS_SYMBOLIC_SHAPES_H_
#define ITEX_CORE_GRAPH_UTILS_SYMBOLIC_SHAPES_H_

#include "itex/core/utils/bcast.h"
#include "protos/op_performance_data.pb.h"
#include "protos/tensor_shape.pb.h"
namespace itex {
namespace graph {
bool IsUnknown(const TensorShapeProto::Dim& dim);

// Shape is symbolically defined, if it has a known rank, and each dimension is
// known (dim_size >= 0), or is a symbolic dimension size (dim_size <= -2).
bool ShapeIsSymbolicallyDefined(const TensorShapeProto& shape);
bool ShapeIsSymbolicallyDefined(const OpInfo::TensorProperties& properties);

// Shapes are symbolically equal, if they have the same rank, they are known or
// symbolically defined, and have matching dimensions.
bool ShapesSymbolicallyEqual(const TensorShapeProto& left,
                             const TensorShapeProto& right);

// Shapes are symbolically equal, if they have the same rank, they are known or
// symbolically defined, and have matching dimensions except batch(first dim).
bool ShapesSymbolicallyEqualExceptBatch(const TensorShapeProto& left,
                                        const TensorShapeProto& right);

// Returns the rank of the shape ir -1 if unknown
int Rank(const TensorShapeProto& shape);

// Returns the number of coefficients in the shape or -1 if unknown.
int64_t NumCoefficients(const TensorShapeProto& shape);

// Check if two shapes can be broadcasted to each other. Both shapes must be at
// least symbolically defined, and the have valid BCast instance.
bool ShapesBroadcastable(const TensorShapeProto& left,
                         const TensorShapeProto& right);
bool ShapesBroadcastable(const OpInfo::TensorProperties& left,
                         const OpInfo::TensorProperties& right);

}  // namespace graph
}  // end namespace itex

#endif  // ITEX_CORE_GRAPH_UTILS_SYMBOLIC_SHAPES_H_
