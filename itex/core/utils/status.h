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

#ifndef ITEX_CORE_UTILS_STATUS_H_
#define ITEX_CORE_UTILS_STATUS_H_

#include <memory>
#include <string>
#include <utility>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/stringpiece.h"
#include "itex/core/utils/types.h"
#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_status.h"

namespace itex {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
class TF_MUST_USE_RESULT Status;
#endif

/// @ingroup core
/// Denotes success or failure of a call in Tensorflow.
class Status {
 public:
  /// Create a success status.
  Status() {
    code_ = TF_OK;
    message_ = std::string("");
    message_.reserve(128);
  }

  ~Status() {}

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(TF_Code code, itex::StringPiece msg);

  /// Copy the specified status.
  Status(const Status& s);
  Status& operator=(const Status& s);
#ifndef SWIG
  Status(Status&& s) noexcept;
  Status& operator=(Status&& s) noexcept;
#endif  // SWIG

  /// return a OK status.
  static Status OK() { return Status(); }

  /// Returns true if the status indicates success.
  bool ok() const { return code_ == TF_OK; }

  TF_Code code() const { return code_; }

  const std::string& error_message() const { return message_; }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  std::string ToString() const;

  void IgnoreError();

 private:
  TF_Code code_;
  std::string message_;
};

inline Status::Status(const Status& s)
    : code_(s.code()), message_(s.error_message()) {}

inline Status& Status::operator=(const Status& s) {
  code_ = s.code();
  message_ = s.error_message();
  return *this;
}

#ifndef SWIG
inline Status::Status(Status&& s) noexcept {
  code_ = s.code();
  message_ = std::move(s.error_message());
}

inline Status& Status::operator=(Status&& s) noexcept {
  code_ = s.code_;
  message_ = std::move(s.error_message());
  return *this;
}
#endif  // SWIG

inline bool Status::operator==(const Status& x) const {
  return ToString() == x.ToString();
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

typedef std::function<void(const Status&)> StatusCallback;

extern std::string* TfCheckOpHelperOutOfLine(const ::itex::Status& v,
                                             const char* msg);

inline std::string* TfCheckOpHelper(::itex::Status v, const char* msg) {
  if (v.ok()) return nullptr;
  return TfCheckOpHelperOutOfLine(v, msg);
}

#define ITEX_DO_CHECK_OK(val, level)                        \
  while (auto _result = ::itex::TfCheckOpHelper(val, #val)) \
  ITEX_LOG(level) << *(_result)

#define ITEX_CHECK_OK(val) ITEX_DO_CHECK_OK(val, FATAL)
#define ITEX_QCHECK_OK(val) ITEX_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of ITEX_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define ITEX_DCHECK_OK(val) ITEX_CHECK_OK(val)
#else
#define ITEX_DCHECK_OK(val) \
  while (false && (::itex::Status::OK() == (val))) ITEX_LOG(FATAL)
#endif

// Returns a "status" from "tf_status".
Status StatusFromTF_Status(const TF_Status* tf_status);

/// \brief Copy the status to tf_status. It will return back the status back.
TF_Status* TF_StatusFromStatus(const Status& status, TF_Status* tf_status);

struct StatusDeleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

using StatusUniquePtr = std::unique_ptr<TF_Status, StatusDeleter>;

}  // namespace itex

#endif  // ITEX_CORE_UTILS_STATUS_H_
