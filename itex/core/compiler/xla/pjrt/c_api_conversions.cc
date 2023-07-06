/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/compiler/xla/pjrt/c_api_conversions.h"

#include <algorithm>
namespace ITEXApiConverter {
// Helper functions for copying data to possibly-inlined C arrays.

// 'Src' and 'Dst' are allowed to be different types to make this usable with
// memory-identical types, e.g. int64_t and int64_t. This should not be used
// with types that require a static_cast.
template <typename Src, typename Dst, typename DstList>
static void CreateVectorBase(const absl::Span<Src> src, DstList* dst) {
  dst->size = src.size();
  if (dst->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new Dst[dst->size];
    std::copy(src.begin(), src.end(), dst->heap);
  } else {
    std::copy(src.begin(), src.end(), dst->inlined);
  }
}

void CreateVector(const absl::Span<const int> src, IntList* dst) {
  return CreateVectorBase<const int, int, IntList>(src, dst);
}
void CreateVector(const absl::Span<const int64_t> src, Int64List* dst) {
  return CreateVectorBase<const int64_t, int64_t, Int64List>(src, dst);
}
void CreateVector(const absl::Span<const float> src, FloatList* dst) {
  return CreateVectorBase<const float, float, FloatList>(src, dst);
}
void CreateVector(const absl::Span<const bool> src, BoolList* dst) {
  return CreateVectorBase<const bool, bool, BoolList>(src, dst);
}
static void CreateVector(const absl::Span<const itex_xla::DimLevelType> src,
                         IntList* dst) {
  CreateVectorBase<const itex_xla::DimLevelType, int, IntList>(src, dst);
}
static void CreateVector(const absl::Span<const bool> src, IntList* dst) {
  CreateVectorBase<const bool, int, IntList>(src, dst);
}

static void CreateVector(const absl::Span<const itex_xla::Tile> src,
                         TileList* dst) {
  dst->size = src.size();
  XLA_Tile* c_tiles;
  if (dst->size > 6) {
    dst->heap = new XLA_Tile[dst->size];
    c_tiles = dst->heap;
  } else {
    c_tiles = dst->inlined;
  }
  for (int i = 0; i < dst->size; ++i) {
    ToC(src[i], &c_tiles[i]);
  }
}

void ToC(const itex_xla::Tile& tile, XLA_Tile* c_tile) {
  CreateVector(tile.dimensions(), &c_tile->dimensions);
}

void ToC(const itex_xla::Layout& layout, XLA_Layout* c_layout) {
  CreateVector(layout.minor_to_major(), &c_layout->minor_to_major);
  CreateVector(layout.dim_level_types(), &c_layout->dim_level_types);
  CreateVector(layout.dim_unique(), &c_layout->dim_unique);
  CreateVector(layout.dim_ordered(), &c_layout->dim_ordered);
  c_layout->index_primitive_type = layout.index_primitive_type();
  c_layout->pointer_primitive_type = layout.pointer_primitive_type();
  c_layout->memory_space = layout.memory_space();
  c_layout->dynamic_shape_metadata_prefix_bytes =
      layout.dynamic_shape_metadata_prefix_bytes();
  CreateVector(layout.tiles(), &c_layout->tiles);
}
}  // namespace ITEXApiConverter
