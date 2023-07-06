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

#include "itex/core/utils/status.h"

#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "itex/core/utils/stacktrace.h"

namespace itex {

Status::Status(TF_Code code, itex::StringPiece msg) {
  assert(code != TF_OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = std::string(msg);
  ITEX_VLOG(5) << "Generated non-OK status: \"" << *this << "\". "
               << CurrentStackTrace();
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const std::string& Status::empty_string() {
  static string* empty = new string;
  return *empty;
}

void Status::IgnoreError() const {
  // do nothing
}

void Status::SetPayload(itex::StringPiece type_url, itex::StringPiece payload) {
  if (ok()) return;
  state_->payloads[std::string(type_url)] = std::string(payload);
}

absl::optional<itex::StringPiece> Status::GetPayload(
    itex::StringPiece type_url) const {
  if (ok()) return absl::nullopt;
  auto payload_iter = state_->payloads.find(std::string(type_url));
  if (payload_iter == state_->payloads.end()) return absl::nullopt;
  return itex::StringPiece(payload_iter->second);
}

bool Status::ErasePayload(itex::StringPiece type_url) {
  if (ok()) return false;
  auto payload_iter = state_->payloads.find(std::string(type_url));
  if (payload_iter == state_->payloads.end()) return false;
  state_->payloads.erase(payload_iter);
  return true;
}

void Status::ForEachPayload(
    const std::function<void(itex::StringPiece, itex::StringPiece)>& visitor)
    const {
  if (ok()) return;
  for (const auto& payload : state_->payloads) {
    visitor(payload.first, payload.second);
  }
}

string error_name(TF_Code code) {
  switch (code) {
    case TF_OK:
      return "OK";
      break;
    case TF_CANCELLED:
      return "Cancelled";
      break;
    case TF_UNKNOWN:
      return "Unknown";
      break;
    case TF_INVALID_ARGUMENT:
      return "Invalid argument";
      break;
    case TF_DEADLINE_EXCEEDED:
      return "Deadline exceeded";
      break;
    case TF_NOT_FOUND:
      return "Not found";
      break;
    case TF_ALREADY_EXISTS:
      return "Already exists";
      break;
    case TF_PERMISSION_DENIED:
      return "Permission denied";
      break;
    case TF_UNAUTHENTICATED:
      return "Unauthenticated";
      break;
    case TF_RESOURCE_EXHAUSTED:
      return "Resource exhausted";
      break;
    case TF_FAILED_PRECONDITION:
      return "Failed precondition";
      break;
    case TF_ABORTED:
      return "Aborted";
      break;
    case TF_OUT_OF_RANGE:
      return "Out of range";
      break;
    case TF_UNIMPLEMENTED:
      return "Unimplemented";
      break;
    case TF_INTERNAL:
      return "Internal";
      break;
    case TF_UNAVAILABLE:
      return "Unavailable";
      break;
    case TF_DATA_LOSS:
      return "Data loss";
      break;
    default:
      char tmp[30];
      ITEX_CHECK(snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                          static_cast<int>(code)) >= 0)
          << "Encoding error occurs";
      return tmp;
      break;
  }
}

string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    string result(error_name(code()));
    result += ": ";
    result += error_message();

    for (const std::pair<const std::string, std::string>& element :
         state_->payloads) {
      absl::StrAppend(&result, " [", element.first, "='",
                      absl::CHexEscape(element.second), "']");
    }
    return result;
  }
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

Status OkStatus() { return Status(); }

std::string* TfCheckOpHelperOutOfLine(const ::itex::Status& v,
                                      const char* msg) {
  string r("Non-OK-status: ");
  r += msg;
  r += " status: ";
  r += v.ToString();
  // Leaks string but this is only to be used in a fatal error message
  return new string(r);
}

#ifndef ITEX_BUILD_JAX
Status StatusFromTF_Status(const TF_Status* tf_status) {
  TF_Code code = TF_GetCode(tf_status);
  if (code == TF_OK) return Status();
  std::string message(TF_Message(tf_status));
  return Status(code, message);
}

TF_Status* TF_StatusFromStatus(const Status& status, TF_Status* tf_status) {
  if (!tf_status) {
    ITEX_LOG(FATAL) << "tf_status should not be nullptr";
  }

  TF_SetStatus(tf_status, status.code(), status.error_message().c_str());

  return tf_status;
}
#endif

}  // namespace itex
