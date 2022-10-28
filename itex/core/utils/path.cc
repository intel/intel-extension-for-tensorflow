/* Copyright (c) 2022 Intel Corporation

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

#include "itex/core/utils/path.h"

namespace itex {
namespace io {
namespace internal {

const char kPathSep[] = "/";

string JoinPathImpl(std::initializer_list<StringPiece> paths) {
  string result;

  for (StringPiece path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = string(path);
      continue;
    }

    if (IsAbsolutePath(path)) path = path.substr(1);

    if (result[result.size() - 1] == kPathSep[0]) {
      strings::StrAppend(&result, path);
    } else {
      strings::StrAppend(&result, kPathSep, path);
    }
  }

  return result;
}

}  // namespace internal

bool IsAbsolutePath(StringPiece path) {
  return !path.empty() && path[0] == '/';
}

}  // namespace io
}  // namespace itex
