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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/types/optional.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/stringpiece.h"

#ifdef ITEX_BUILD_JAX

#include "protos/error_codes.pb.h"
#ifndef TF_Code
#define TF_Code itex::error::Code
#endif
#ifndef TF_OK
#define TF_OK itex::error::OK
#endif
#ifndef TF_CANCELLED
#define TF_CANCELLED itex::error::CANCELLED
#endif
#ifndef TF_UNKNOWN
#define TF_UNKNOWN itex::error::UNKNOWN
#endif
#ifndef TF_INVALID_ARGUMENT
#define TF_INVALID_ARGUMENT itex::error::INVALID_ARGUMENT
#endif
#ifndef TF_DEADLINE_EXCEEDED
#define TF_DEADLINE_EXCEEDED itex::error::DEADLINE_EXCEEDED
#endif
#ifndef TF_NOT_FOUND
#define TF_NOT_FOUND itex::error::NOT_FOUND
#endif
#ifndef TF_ALREADY_EXISTS
#define TF_ALREADY_EXISTS itex::error::ALREADY_EXISTS
#endif
#ifndef TF_PERMISSION_DENIED
#define TF_PERMISSION_DENIED itex::error::PERMISSION_DENIED
#endif
#ifndef TF_UNAUTHENTICATED
#define TF_UNAUTHENTICATED itex::error::UNAUTHENTICATED
#endif
#ifndef TF_RESOURCE_EXHAUSTED
#define TF_RESOURCE_EXHAUSTED itex::error::RESOURCE_EXHAUSTED
#endif
#ifndef TF_FAILED_PRECONDITION
#define TF_FAILED_PRECONDITION itex::error::FAILED_PRECONDITION
#endif
#ifndef TF_ABORTED
#define TF_ABORTED itex::error::ABORTED
#endif
#ifndef TF_OUT_OF_RANGE
#define TF_OUT_OF_RANGE itex::error::OUT_OF_RANGE
#endif
#ifndef TF_UNIMPLEMENTED
#define TF_UNIMPLEMENTED itex::error::UNIMPLEMENTED
#endif
#ifndef TF_INTERNAL
#define TF_INTERNAL itex::error::INTERNAL
#endif
#ifndef TF_UNAVAILABLE
#define TF_UNAVAILABLE itex::error::UNAVAILABLE
#endif
#ifndef TF_DATA_LOSS
#define TF_DATA_LOSS itex::error::DATA_LOSS
#endif

#else
#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_status.h"
#endif

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
  Status() {}

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
  bool ok() const { return (state_ == nullptr); }

  TF_Code code() const { return ok() ? TF_OK : state_->code; }

  const std::string& error_message() const {
    return ok() ? empty_string() : state_->msg;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  ///
  /// By default, it returns combination of the error code name, the message and
  /// any associated payload messages. This string is designed simply to be
  /// human readable and its exact format should not be load bearing. Do not
  /// depend on the exact format of the result of `ToString()` which is subject
  /// to change.
  std::string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  //----------------------------------------------------------------------------
  // Payload Management APIs (Cloned from absl::Status)
  //----------------------------------------------------------------------------
  // A payload may be attached to a status to provide additional context to an
  // error that may not be satisfied by an existing `itex::error::Code`.
  // Typically, this payload serves one of several purposes:
  //
  //   * It may provide more fine-grained semantic information about the error
  //     to facilitate actionable remedies.
  //   * It may provide human-readable contexual information that is more
  //     appropriate to display to an end user.
  //
  // A payload consists of a [key,value] pair, where the key is a string
  // referring to a unique "type URL" and the value is an object of type
  // `absl::Cord` to hold the contextual data.
  //
  // The "type URL" should be unique and follow the format of a URL
  // (https://en.wikipedia.org/wiki/URL) and, ideally, provide some
  // documentation or schema on how to interpret its associated data. For
  // example, the default type URL for a protobuf message type is
  // "type.googleapis.com/packagename.messagename". Other custom wire formats
  // should define the format of type URL in a similar practice so as to
  // minimize the chance of conflict between type URLs.
  // Users should ensure that the type URL can be mapped to a concrete
  // C++ type if they want to deserialize the payload and read it effectively.
  //
  // To attach a payload to a status object, call `Status::SetPayload()`,
  // passing it the type URL and an `absl::Cord` of associated data. Similarly,
  // to extract the payload from a status, call `Status::GetPayload()`. You
  // may attach multiple payloads (with differing type URLs) to any given
  // status object, provided that the status is currently exhibiting an error
  // code (i.e. is not OK).
  // TODO(b/197552541): Use absl::Cord for payload value type.

  // The Payload-related APIs are cloned from absl::Status.
  //
  // Returns the payload of a status given its unique `type_url` key, if
  // present.
  absl::optional<itex::StringPiece> GetPayload(
      itex::StringPiece type_url) const;

  // Sets the payload for a non-ok status using a `type_url` key, overwriting
  // any existing payload for that `type_url`.
  //
  // This function does nothing if the Status is ok.
  void SetPayload(itex::StringPiece type_url, itex::StringPiece payload);

  // Erases the payload corresponding to the `type_url` key.  Returns `true` if
  // the payload was present.
  bool ErasePayload(itex::StringPiece type_url);

  // Iterates over the stored payloads and calls the
  // `visitor(type_key, payload)` callable for each one.
  //
  // The order of calls to `visitor()` is not specified and may change at
  // any time and any mutation on the same Status object during visitation is
  // forbidden and could result in undefined behavior.
  void ForEachPayload(
      const std::function<void(itex::StringPiece, itex::StringPiece)>& visitor)
      const;

 private:
  static const std::string& empty_string();
  struct State {
    TF_Code code;
    std::string msg;
    std::unordered_map<std::string, std::string> payloads;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State* src);
};

// OkStatus()
//
// Returns an OK status, equivalent to a default constructed instance. Prefer
// usage of `OkStatus()` when constructing such an OK status.
Status OkStatus();

inline Status::Status(const Status& s)
    : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {}

inline Status& Status::operator=(const Status& s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
  return *this;
}

#ifndef SWIG
inline Status::Status(Status&& s) noexcept : state_(std::move(s.state_)) {}

inline Status& Status::operator=(Status&& s) noexcept {
  if (state_ != s.state_) {
    state_ = std::move(s.state_);
  }
  return *this;
}
#endif  // SWIG

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
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

#ifndef ITEX_BUILD_JAX
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
#endif
}  // namespace itex

#endif  // ITEX_CORE_UTILS_STATUS_H_
