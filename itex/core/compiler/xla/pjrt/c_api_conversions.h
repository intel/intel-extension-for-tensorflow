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
#ifndef ITEX_CORE_COMPILER_XLA_PJRT_C_API_CONVERSIONS_H_
#define ITEX_CORE_COMPILER_XLA_PJRT_C_API_CONVERSIONS_H_
#include "itex/core/compiler/c/pjrt_c_api.h"
#include "itex/core/compiler/xla/shape.h"

// APIs for converting between internal and external versions of
// XLA/StreamExecutor data structures.
namespace ITEXApiConverter {

void CreateVector(const absl::Span<const int> src, FloatList* dst);
void CreateVector(const absl::Span<const int64_t> src, Int64List* dst);
void CreateVector(const absl::Span<const float> src, FloatList* dst);
void CreateVector(const absl::Span<const bool> src, BoolList* dst);

void ToC(const itex_xla::Tile& tile, XLA_Tile* c_tile);
void ToC(const itex_xla::Layout& layout, XLA_Layout* c_layout);

}  // namespace ITEXApiConverter
#endif  // ITEX_CORE_COMPILER_XLA_PJRT_C_API_CONVERSIONS_H_
