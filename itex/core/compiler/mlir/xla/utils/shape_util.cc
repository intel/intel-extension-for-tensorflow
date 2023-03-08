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

#include "itex/core/compiler/mlir/xla/utils/shape_util.h"

#include <numeric>

#include "itex/core/compiler/mlir/xla/utils/type_util.h"
#include "itex/core/compiler/xla/layout_util.h"
#include "itex/core/compiler/xla/shape_util.h"
#include "itex/core/utils/status.h"

namespace itex {
namespace {

Status PopulateInfeedLayoutVector(const itex_xla::Shape& shape,
                                  std::vector<int>* layouts) {
  if (shape.IsTuple()) {
    int64_t tuple_elements = itex_xla::ShapeUtil::TupleElementCount(shape);
    for (int64_t i = 0; i < tuple_elements; ++i) {
      const itex_xla::Shape& subshape =
          itex_xla::ShapeUtil::GetTupleElementShape(shape, i);
      TF_RETURN_IF_ERROR(PopulateInfeedLayoutVector(subshape, layouts));
    }
  } else if (itex_xla::LayoutUtil::HasLayout(shape)) {
    for (auto dim : itex_xla::LayoutUtil::MinorToMajor(shape)) {
      layouts->push_back(dim);
    }
  } else {
    layouts->insert(layouts->end(), shape.rank(), -1);
  }
  return Status::OK();
}

// Populate the output layout unless the minor_to_major array contains all -1
// value, in which case the layout is considered missing and the API returns
// false.
StatusOr<bool> MakeLayout(absl::Span<const int64_t> minor_to_major,
                          itex_xla::Layout* layout) {
  if (std::all_of(minor_to_major.begin(), minor_to_major.end(),
                  [](int64_t dim) { return dim == -1; })) {
    return false;
  }
  std::vector<bool> dim_present(minor_to_major.size(), false);
  for (auto dim : minor_to_major) {
    const int minor_to_major_size = minor_to_major.size();
    if (dim < 0 || dim >= minor_to_major_size) {
      return errors::InvalidArgument("Layout dimension out of range: dim=", dim,
                                     " rank=", minor_to_major.size());
    }
    if (dim_present[dim]) {
      return errors::InvalidArgument("Repeated layout dimension: dim=", dim);
    }
    dim_present[dim] = true;
  }
  *layout = itex_xla::LayoutUtil::MakeLayout(minor_to_major);
  return true;
}

Status AssignLayout(
    absl::Span<const int64_t> minor_to_major,
    const std::function<itex_xla::Layout(const itex_xla::Shape&)>& layout_func,
    itex_xla::Shape* shape) {
  itex_xla::Layout layout;
  TF_ASSIGN_OR_RETURN(bool has_layout, MakeLayout(minor_to_major, &layout));
  if (!has_layout && layout_func) {
    layout = layout_func(*shape);
  }
  *shape->mutable_layout() = layout;
  return Status::OK();
}

}  // namespace

// Convert an XLA Shape into the equivalent TensorFlow shape.
Status XLAShapeToTensorShape(const itex_xla::Shape& shape,
                             TensorShape* tensor_shape) {
  if (shape.IsTuple()) {
    return errors::InvalidArgument("XLA shape ",
                                   itex_xla::ShapeUtil::HumanString(shape),
                                   " cannot be converted to a TensorShape");
  }
  *tensor_shape = TensorShape();
  for (int i = 0; i < shape.rank(); ++i) {
    tensor_shape->AddDim(shape.dimensions(i));
  }
  return Status::OK();
}

// Convert a TensorShape into the equivalent XLA Shape proto.
Status TensorShapeToXLAShape(DataType dtype,
                             const PartialTensorShape& tensor_shape,
                             itex_xla::Shape* shape) {
  itex_xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  *shape = TensorShapeToXLAShape(type, tensor_shape);
  return Status::OK();
}

itex_xla::Shape TensorShapeToXLAShape(itex_xla::PrimitiveType type,
                                      const PartialTensorShape& tensor_shape) {
  if (tensor_shape.unknown_rank()) {
    // For unknown shape, create a rank 1 size 0 tensor.
    return itex_xla::ShapeUtil::MakeShapeWithLayout(type, {0}, {0});
  }
  int rank = tensor_shape.dims();
  std::vector<int64_t> dimensions(rank);
  std::vector<int64_t> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
    if (dimensions[d] < 0) {
      ITEX_LOG(WARNING)
          << "Unable to convert TF shape with dynamic size to XLA "
             "shape; returning unknown sentinel value";
      return itex_xla::ShapeUtil::MakeShapeWithLayout(type, {0}, {0});
    }
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);
  itex_xla::Shape result =
      itex_xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);

  return result;
}

// Convert a TensorShape into the equivalent XLA Shape proto.
Status TensorShapeToXLAShape(DataType dtype, const TensorShape& tensor_shape,
                             itex_xla::Shape* shape) {
  itex_xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  *shape = TensorShapeToXLAShape(type, tensor_shape);
  return Status::OK();
}

StatusOr<itex_xla::Shape> TensorShapeToXLAShape(
    DataType dtype, const TensorShape& tensor_shape) {
  itex_xla::Shape out;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype, tensor_shape, &out));
  return out;
}

itex_xla::Shape TensorShapeToXLAShape(itex_xla::PrimitiveType type,
                                      const TensorShape& tensor_shape) {
  int rank = tensor_shape.dims();
  std::vector<int64_t> dimensions(rank);
  std::vector<int64_t> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);

  return itex_xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
}

StatusOr<std::vector<int>> GetShapeLayoutVector(const itex_xla::Shape& shape) {
  std::vector<int> layouts;
  TF_RETURN_IF_ERROR(PopulateInfeedLayoutVector(shape, &layouts));
  return layouts;
}

Status GetShapeWithLayout(
    const itex_xla::Shape& input_shape,
    absl::Span<const int64_t> minor_to_major,
    const std::function<itex_xla::Layout(const itex_xla::Shape&)>& layout_func,
    itex_xla::Shape* output_shape) {
  if (input_shape.IsTuple()) {
    int64_t tuple_elements =
        itex_xla::ShapeUtil::TupleElementCount(input_shape);
    std::vector<itex_xla::Shape> shapes;
    shapes.reserve(tuple_elements);
    size_t position = 0;
    for (int64_t i = 0; i < tuple_elements; ++i) {
      const itex_xla::Shape& shape =
          itex_xla::ShapeUtil::GetTupleElementShape(input_shape, i);
      if (shape.IsTuple()) {
        return errors::InvalidArgument(
            "Nested tuples not supported: ",
            itex_xla::ShapeUtil::HumanString(input_shape));
      }
      int64_t rank = shape.rank();
      if (position + rank > minor_to_major.size()) {
        return errors::InvalidArgument(
            "Not enough layout attribute elements: position=", position,
            " rank=", rank, " elements=", minor_to_major.size());
      }
      shapes.push_back(shape);
      TF_RETURN_IF_ERROR(AssignLayout(
          absl::Span<const int64_t>(minor_to_major).subspan(position, rank),
          layout_func, &shapes.back()));
      position += rank;

      ITEX_VLOG(4) << "Shape[" << i << "] = "
                   << itex_xla::ShapeUtil::HumanStringWithLayout(shapes.back());
    }
    if (position != minor_to_major.size()) {
      return errors::InvalidArgument(
          "Too many elements passed in the layout attribute: position=",
          position, " size=", minor_to_major.size());
    }
    *output_shape = itex_xla::ShapeUtil::MakeTupleShape(shapes);
  } else {
    int64_t rank = input_shape.rank();
    const int64_t minor_to_major_size = minor_to_major.size();
    if (rank != minor_to_major_size) {
      return errors::InvalidArgument(
          "Wrong number of layout attribute elements: rank=", rank,
          " elements=", minor_to_major.size());
    }
    *output_shape = input_shape;
    TF_RETURN_IF_ERROR(AssignLayout(minor_to_major, layout_func, output_shape));

    ITEX_VLOG(4) << "Shape[] = "
                 << itex_xla::ShapeUtil::HumanStringWithLayout(*output_shape);
  }
  return Status::OK();
}

}  // namespace itex
