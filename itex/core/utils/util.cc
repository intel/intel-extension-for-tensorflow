/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/utils/util.h"

#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/strcat.h"
#include "itex/core/utils/types.h"

namespace itex {

std::string SliceDebugString(const TensorShape& shape, const int64 flat) {
  // Special case rank 0 and 1
  const int dims = shape.dims();
  if (dims == 0) return "";
  if (dims == 1) return strings::StrCat("[", flat, "]");

  // Compute strides
  gtl::InlinedVector<int64, 32> strides(dims);
  strides.back() = 1;
  for (int i = dims - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape.dim_size(i + 1);
  }

  // Unflatten index
  int64 left = flat;
  string result;
  for (int i = 0; i < dims; i++) {
    strings::StrAppend(&result, i ? "," : "[", left / strides[i]);
    left %= strides[i];
  }
  strings::StrAppend(&result, "]");
  return result;
}

}  // namespace itex
