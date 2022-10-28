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

#ifndef ITEX_CORE_UTILS_ERRORS_H_
#define ITEX_CORE_UTILS_ERRORS_H_

#include <sstream>
#include <string>

#include "absl/strings/str_join.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/str_util.h"
#include "itex/core/utils/strcat.h"

namespace itex {
namespace errors {

namespace internal {

// The DECLARE_ERROR macro below only supports types that can be converted
// into StrCat's AlphaNum. For the other types we rely on a slower path
// through std::stringstream. To add support of a new type, it is enough to
// make sure there is an operator<<() for it:
//
//   std::ostream& operator<<(std::ostream& os, const MyType& foo) {
//     os << foo.ToString();
//     return os;
//   }
// Eventually absl::strings will have native support for this and we will be
// able to completely remove PrepareForStrCat().
template <typename T>
typename std::enable_if<!std::is_constructible<strings::AlphaNum, T>::value,
                        std::string>::type
PrepareForStrCat(const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}
inline const strings::AlphaNum& PrepareForStrCat(const strings::AlphaNum& a) {
  return a;
}

}  // namespace internal

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)                            \
  do {                                                     \
    ::itex::Status _status(__VA_ARGS__);                   \
    if (ITEX_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

// For setting tf_status and propagating errors when calling a function.
#define SET_STATUS_IF_ERROR(tf_status, ...)    \
  do {                                         \
    ::itex::Status _status(__VA_ARGS__);       \
    if (ITEX_PREDICT_FALSE(!_status.ok())) {   \
      TF_StatusFromStatus(_status, tf_status); \
      return;                                  \
    }                                          \
  } while (0)

// Log errors when calling a function.
#define TF_ABORT_IF_ERROR(...)               \
  do {                                       \
    ::itex::Status _status(__VA_ARGS__);     \
    if (ITEX_PREDICT_FALSE(!_status.ok()))   \
      ITEX_LOG(FATAL) << _status.ToString(); \
  } while (0)

// Convenience functions for generating and using error status.
// Example usage:
//   status.Update(errors::InvalidArgument("The ", foo, " isn't right."));
//   if (errors::IsInvalidArgument(status)) { ... }
//   switch (status.code()) { case error::INVALID_ARGUMENT: ... }

#define DECLARE_ERROR(FUNC, CONST)                                             \
  template <typename... Args>                                                  \
  ::itex::Status FUNC(Args... args) {                                          \
    return ::itex::Status(                                                     \
        TF_##CONST, ::itex::strings::StrCat(                                   \
                        ::itex::errors::internal::PrepareForStrCat(args)...)); \
  }                                                                            \
  inline bool Is##FUNC(const ::itex::Status& status) {                         \
    return status.code() == TF_##CONST;                                        \
  }

DECLARE_ERROR(Cancelled, CANCELLED)
DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DECLARE_ERROR(NotFound, NOT_FOUND)
DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DECLARE_ERROR(Unavailable, UNAVAILABLE)
DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
DECLARE_ERROR(Internal, INTERNAL)
DECLARE_ERROR(Aborted, ABORTED)
DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DECLARE_ERROR(DataLoss, DATA_LOSS)
DECLARE_ERROR(Unknown, UNKNOWN)
DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)

#undef DECLARE_ERROR

// Produces a formatted string pattern from the name which can uniquely identify
// this node upstream to produce an informative error message. The pattern
// followed is: {{node <name>}}
// Note: The pattern below determines the regex _NODEDEF_NAME_RE in the file
// tensorflow/python/client/session.py
// LINT.IfChange
inline std::string FormatNodeNameForError(const std::string& name) {
  return strings::StrCat("{{node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/client/session.py)
template <typename T>
std::string FormatNodeNamesForError(const T& names) {
  return absl::StrJoin(
      names, ", ", [](std::string* output, const std::string& s) {
        ::itex::strings::StrAppend(output, FormatNodeNameForError(s));
      });
}
// LINT.IfChange
inline std::string FormatColocationNodeForError(const std::string& name) {
  return strings::StrCat("{{colocation_node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/framework/error_interpolation.py)
template <typename T>
std::string FormatColocationNodeForError(const T& names) {
  return absl::StrJoin(
      names, ", ", [](std::string* output, const std::string& s) {
        ::itex::strings::StrAppend(output, FormatColocationNodeForError(s));
      });
}

inline std::string FormatFunctionForError(const std::string& name) {
  return strings::StrCat("{{function_node ", name, "}}");
}

}  // namespace errors
}  // namespace itex

#endif  // ITEX_CORE_UTILS_ERRORS_H_
