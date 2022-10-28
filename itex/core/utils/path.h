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

#ifndef ITEX_CORE_UTILS_PATH_H_
#define ITEX_CORE_UTILS_PATH_H_

#include <string>

#include "absl/strings/string_view.h"
#include "itex/core/utils/strcat.h"
#include "itex/core/utils/stringpiece.h"

namespace itex {
using std::string;

namespace io {
namespace internal {
std::string JoinPathImpl(std::initializer_list<StringPiece> paths);
}  // namespace internal

// Join multiple paths together, without introducing unnecessary path
// separators.
// For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+----------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//
// Usage:
// string path = io::JoinPath("/mydir", filename);
// string path = io::JoinPath(FLAGS_test_srcdir, filename);
// string path = io::JoinPath("/full", "path", "to", "filename");
template <typename... T>
std::string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}

bool IsAbsolutePath(StringPiece path);

}  // namespace io
}  // namespace itex

#endif  // ITEX_CORE_UTILS_PATH_H_
