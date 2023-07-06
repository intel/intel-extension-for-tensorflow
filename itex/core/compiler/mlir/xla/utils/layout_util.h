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

// Utilities for working with XLA layout and shapes.

#ifndef ITEX_CORE_COMPILER_MLIR_XLA_UTILS_LAYOUT_UTIL_H_
#define ITEX_CORE_COMPILER_MLIR_XLA_UTILS_LAYOUT_UTIL_H_

#include <vector>

#include "itex/core/compiler/mlir/xla/utils/xla_argument.h"
#include "itex/core/compiler/mlir/xla/utils/xla_helpers.h"
#include "itex/core/compiler/xla/shape.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/utils/tensor_shape.h"
#include "protos/types.pb.h"
#include "protos/xla_data.pb.h"

namespace itex {

class XlaShapeLayoutHelpers {
 public:
  // The following defines the layout preference of an xla tensor.
  // The return value of LayoutPreferenceFn can be used in
  // XlaHelper::ShapeRepresentationFn.
  typedef std::function<XlaLayoutPreference(const TensorShape&, DataType,
                                            absl::optional<XlaArgument::Kind>)>
      LayoutPreferenceFn;

  // A bundle of LayoutPreferenceFn and ShapeRepresentationFn.
  struct ShapeDeterminationFns {
    // Use no preference function, and identity shape representation function,
    // as default value.
    ShapeDeterminationFns();

    ShapeDeterminationFns(
        LayoutPreferenceFn layout_preference_fn,
        XlaHelpers::ShapeRepresentationFn shape_representation_fn)
        : layout_preference_fn(layout_preference_fn),
          shape_representation_fn(shape_representation_fn) {}

    LayoutPreferenceFn layout_preference_fn;
    XlaHelpers::ShapeRepresentationFn shape_representation_fn;
  };
};

// Return a LayoutPreferenceFn that always uses kNoPreference layout.
XlaShapeLayoutHelpers::LayoutPreferenceFn UseNoPreferenceLayoutFn();

// Rewrites the layout of xla_shape if there is tiled sharding.
Status RewriteLayoutWithShardedShape(
    const absl::optional<itex_xla::HloSharding>& sharding, bool use_fast_memory,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    itex_xla::Shape* xla_shape);

// Adds reshapes to fix the layout of an output, if a shape_representation_fn or
// sharding is present.
StatusOr<itex_xla::XlaOp> ReshapeWithCorrectRepresentationAndSharding(
    itex_xla::XlaBuilder* builder, itex_xla::XlaOp original,
    itex_xla::Shape original_shape,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    absl::optional<itex_xla::OpSharding> sharding, bool fast_mem);

}  // namespace itex

#endif  // ITEX_CORE_COMPILER_MLIR_XLA_UTILS_LAYOUT_UTIL_H_
