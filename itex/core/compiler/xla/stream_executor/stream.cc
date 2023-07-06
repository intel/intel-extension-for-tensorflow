/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/compiler/xla/stream_executor/stream.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "itex/core/compiler/xla/stream_executor/lib/stacktrace.h"
#include "itex/core/compiler/xla/stream_executor/platform.h"
#include "itex/core/compiler/xla/stream_executor/platform/logging.h"
#include "itex/core/compiler/xla/stream_executor/platform/port.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor_internal.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "third_party/eigen3/Eigen/Core"

namespace stream_executor {

namespace {
std::string ToVlogString(const void* ptr) {
  if (ptr == nullptr) {
    return "null";
  }

  // StrCat does not convert pointers to text.
  std::ostringstream out;
  out << ptr;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::complex<T>& c) {
  // StrCat does not convert std::complex to text.
  std::ostringstream out;
  out << c;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::function<T>& f) {
  return f == nullptr ? "null" : "<non-null function>";
}

std::string ToVlogString(const DeviceMemoryBase& memory) {
  return ToVlogString(memory.opaque());
}

std::string ToVlogString(const DeviceMemoryBase* memory) {
  return memory == nullptr ? "null" : ToVlogString(*memory);
}

std::string ToVlogString(const Eigen::half& h) {
  return absl::StrCat(static_cast<float>(h));
}

std::string ToVlogString(int i) { return absl::StrCat(i); }

std::string ToVlogString(uint32_t i) { return absl::StrCat(i); }

std::string ToVlogString(uint64_t i) { return absl::StrCat(i); }

std::string ToVlogString(int64_t i) { return absl::StrCat(i); }

std::string ToVlogString(float f) { return absl::StrCat(f); }

std::string ToVlogString(double d) { return absl::StrCat(d); }

template <class T>
std::string ToVlogString(port::ArraySlice<T> elements) {  // non-absl ok
  std::string str =
      absl::StrCat(ToVlogString(reinterpret_cast<const void*>(elements.data())),
                   "[", elements.size(), "]{");
  const char* separator = "";
  size_t max_to_show = std::numeric_limits<size_t>::max();
  if (!ITEX_VLOG_IS_ON(2)) {
    max_to_show = 5;
  } else if (!ITEX_VLOG_IS_ON(3)) {
    max_to_show = 20;
  } else if (!ITEX_VLOG_IS_ON(11)) {
    max_to_show = 1000;
  }
  for (size_t i = 0; i < elements.size(); ++i) {
    if (i == max_to_show) {
      str += ", ...";
      break;
    }
    absl::StrAppend(&str, separator, ToVlogString(elements[i]));
    separator = ", ";
  }
  str += "}";
  return str;
}

template <class T>
std::string ToVlogString(port::MutableArraySlice<T> elements) {  // non-absl ok
  return ToVlogString(port::ArraySlice<T>(elements));            // non-absl ok
}

// Used together with PARAM to VLOG calls made to the stream. Intended
// to be used like this:
//
//   VLOG(1) << CallStr("MyFunction", this, {PARAM(a), PARAM(b)});
//
// where a and b are the parameters to MyFunction.
//
// See VLOG_CALL for a short-hand for this. This way of doing it saves
// a tremendous amount of boilerplate code given how many functions
// there are on Stream and how many parameters they each have.
std::string CallStr(const char* function_name, Stream* stream,
                    std::vector<std::pair<const char*, std::string>> params) {
  // Do not call this function unless VLOG is on since just
  // constructing all the strings in params is expensive.
  ITEX_CHECK(ITEX_VLOG_IS_ON(1));

  std::string str = absl::StrCat(stream->DebugStreamPointers(),
                                 " Called Stream::", function_name, "(");
  const char* separator = "";
  for (const auto& param : params) {
    absl::StrAppend(&str, separator, param.first, "=", param.second);
    separator = ", ";
  }
  absl::StrAppend(&str, ")");
  if (ITEX_VLOG_IS_ON(10)) {
    absl::StrAppend(&str, " ", port::CurrentStackTrace(), "\n");
  }
  return str;
}

// Use this macro to avoid having to type every parameter twice to log
// it with VLOG and CallStr.
#define PARAM(parameter) \
  { #parameter, ToVlogString(parameter) }

// Use this macro to avoid having to type out the name of each
// function and to save some boilerplate. Intended to be used like this:
//
//   VLOG_CALL(PARAM(a), PARAM(b))
//
// This saves a tremendous amount of boilerplate compared to the alternative:
//
//   VLOG(1) << "Calling MyFunction(a=" << ToVlogString(a)
//           << ", b=" << ToVlogString(b);
//
// Note here that most of the parameter names are not short and that
// most of the functions take many more than 2 parameters.
// #define VLOG_CALL(...) VLOG(1) << CallStr(__func__, this, {__VA_ARGS__})

}  // namespace

Stream::Stream(StreamExecutor* parent)
    : parent_(parent),
      implementation_(parent->implementation()->GetStreamImplementation()),
      allocated_(false),
      status_(port::InternalError("Uninitialized stream")),
      temporary_memory_manager_(this) {
  // VLOG_CALL(PARAM(parent));
}

Stream::~Stream() {
  // VLOG_CALL();

  // Ensure the stream is completed.
  auto status = BlockHostUntilDone();
  if (!status.ok()) {
    ITEX_LOG(WARNING) << "Error blocking host until done in stream destructor: "
                      << status;
  }
  temporary_memory_manager_.ForceDeallocateAll();
  RunAfterBlockHostUntilDoneCallbacks();

  if (allocated_) {
    parent_->DeallocateStream(this);
  }
}

port::Status Stream::RefreshStatus() {
  port::Status status = parent_->GetStatus(this);
  // We should not put the stream in an error state, just because the GetStatus
  // method is unimplemented.
  if (status != port::Status(itex::error::UNIMPLEMENTED,
                             "GetStatus is not supported on this executor.")) {
    CheckStatus(status);
  }
  return status;
}

Stream& Stream::Init() {
  // VLOG_CALL();

  absl::MutexLock lock(&mu_);
  ITEX_CHECK_EQ(false, allocated_)
      << "stream appears to already have been initialized";
  ITEX_CHECK(!status_.ok())
      << "stream should be in !ok() state pre-initialization";

  if (parent_->AllocateStream(this)) {
    // Successful initialization!
    allocated_ = true;
    status_ = itex::Status::OK();
  } else {
    ITEX_LOG(ERROR) << "failed to allocate stream during initialization";
  }

  return *this;
}

Stream& Stream::InitTimer(Timer* timer) {
  // VLOG_CALL(PARAM(timer));

  CheckError(parent_->AllocateTimer(timer));
  return *this;
}

Stream& Stream::InitWithTimer(Timer* timer) {
  // VLOG_CALL(PARAM(timer));

  return Init().InitTimer(timer);
}

Stream& Stream::ThenRecordEvent(Event* event) {
  // VLOG_CALL(PARAM(event));

  port::Status status = parent_->RecordEvent(this, event);
  if (!status.ok()) {
    ITEX_LOG(ERROR)
        << "Error recording event in stream: " << status.error_message()
        << "; not marking stream as bad, as the Event object may be "
        << "at fault. Monitor for further errors.";
  }

  return *this;
}

Stream* Stream::GetOrCreateSubStream() {
  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::vector<std::unique_ptr<Stream>> bad_streams;

  absl::MutexLock lock(&mu_);

  // Look for the first reusable sub_stream that is ok, dropping !ok sub_streams
  // we encounter along the way.
  for (size_t index = 0; index < sub_streams_.size();) {
    std::pair<std::unique_ptr<Stream>, bool>& pair = sub_streams_[index];
    if (pair.second) {
      // The sub_stream is reusable.
      Stream* sub_stream = pair.first.get();
      if (sub_stream->ok()) {
        ITEX_VLOG(1) << DebugStreamPointers() << " reusing sub_stream "
                     << sub_stream->DebugStreamPointers();
        pair.second = false;
        return sub_stream;
      }

      // The stream is reusable and not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      bad_streams.push_back(std::move(sub_streams_.back().first));
      sub_streams_.pop_back();
      ITEX_VLOG(1) << DebugStreamPointers() << " dropped !ok sub_stream "
                   << sub_stream->DebugStreamPointers();
    } else {
      // The sub_stream is not reusable, move on to the next one.
      ++index;
    }
  }

  // No streams are reusable; create a new stream.
  sub_streams_.emplace_back(std::unique_ptr<Stream>{new Stream{parent_}},
                            false);
  Stream* sub_stream = sub_streams_.back().first.get();
  sub_stream->Init();
  if (!sub_stream->ok()) {
    ITEX_LOG(ERROR) << "sub-stream failed to be initialized";
  }
  ITEX_VLOG(1) << DebugStreamPointers() << " created new sub_stream "
               << sub_stream->DebugStreamPointers();

  return sub_stream;
}

void Stream::ReturnSubStream(Stream* sub_stream) {
  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::unique_ptr<Stream> bad_stream;

  absl::MutexLock lock(&mu_);

  // Look for the sub-stream.
  for (int64_t index = 0, end = sub_streams_.size(); index < end; ++index) {
    std::pair<std::unique_ptr<Stream>, bool>& pair = sub_streams_[index];
    if (pair.first.get() != sub_stream) {
      continue;
    }

    // Found the sub_stream.
    if (sub_stream->ok()) {
      ITEX_VLOG(1) << DebugStreamPointers() << " returned ok sub_stream "
                   << sub_stream->DebugStreamPointers();
      pair.second = true;
    } else {
      // The returned stream is not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      ITEX_VLOG(1) << DebugStreamPointers() << " returned !ok sub_stream "
                   << sub_stream->DebugStreamPointers();
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      std::swap(bad_stream, sub_streams_.back().first);
      sub_streams_.pop_back();
    }
    return;
  }

  ITEX_LOG(FATAL) << DebugStreamPointers()
                  << " did not create the returned sub-stream "
                  << sub_stream->DebugStreamPointers();
}

Stream& Stream::ThenStartTimer(Timer* t) {
  // VLOG_CALL(PARAM(t));

  CheckError(parent_->StartTimer(this, t));
  return *this;
}

Stream& Stream::ThenStopTimer(Timer* t) {
  // VLOG_CALL(PARAM(t));

  CheckError(parent_->StopTimer(this, t));
  return *this;
}

Stream& Stream::ThenWaitFor(Stream* other) {
  // VLOG_CALL(PARAM(other));

  ITEX_CHECK(this != other) << "stream cannot wait for itself";
  if (ok() && other->ok()) {
    CheckError(parent_->CreateStreamDependency(this, other));
  } else {
    SetError();
    ITEX_LOG(INFO) << DebugStreamPointers() << " did not wait for "
                   << other->DebugStreamPointers();
  }
  return *this;
}

Stream& Stream::ThenWaitFor(Event* event) {
  // VLOG_CALL(PARAM(event));

  if (ok()) {
    port::Status status = parent_->WaitForEvent(this, event);
    if (!status.ok()) {
      ITEX_LOG(ERROR)
          << "Error waiting for event in stream: " << status.error_message()
          << "; not marking stream as bad, as the Event object may be "
          << "at fault. Monitor for further errors.";
    }
  } else {
    ITEX_LOG(INFO) << DebugStreamPointers() << " did not wait for an event.";
  }
  return *this;
}

Stream& Stream::ThenMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                           uint64_t size) {
  // VLOG_CALL(PARAM(host_dst), PARAM(gpu_src), PARAM(size));

  CheckError(parent_->Memcpy(this, host_dst, gpu_src, size));
  return *this;
}

Stream& Stream::ThenMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                           uint64_t size) {
  // VLOG_CALL(PARAM(gpu_dst), PARAM(host_src), PARAM(size));

  CheckError(parent_->Memcpy(this, gpu_dst, host_src, size));
  return *this;
}

Stream& Stream::ThenMemcpy(DeviceMemoryBase* gpu_dst,
                           const DeviceMemoryBase& gpu_src, uint64_t size) {
  // VLOG_CALL(PARAM(gpu_dst), PARAM(gpu_src), PARAM(size));

  CheckError(parent_->MemcpyDeviceToDevice(this, gpu_dst, gpu_src, size));
  return *this;
}

Stream& Stream::ThenMemZero(DeviceMemoryBase* location, uint64_t size) {
  // VLOG_CALL(PARAM(location), PARAM(size));

  CheckStatus(parent_->MemZero(this, location, size));
  return *this;
}

Stream& Stream::ThenMemset32(DeviceMemoryBase* location, uint32_t pattern,
                             uint64_t size) {
  // VLOG_CALL(PARAM(location), PARAM(pattern), PARAM(size));

  CheckStatus(parent_->Memset32(this, location, pattern, size));
  return *this;
}

Stream& Stream::ThenDoHostCallback(std::function<void()> callback) {
  // VLOG_CALL(PARAM(callback));

  if (!ok()) {
    ITEX_LOG(INFO) << DebugStreamPointers()
                   << " was in error state before adding host callback";
  }
  CheckError(parent_->HostCallback(this, std::move(callback)));
  return *this;
}

Stream& Stream::ThenDoHostCallbackWithStatus(
    std::function<port::Status()> callback) {
  // VLOG_CALL(PARAM(callback));

  if (!ok()) {
    ITEX_LOG(INFO) << DebugStreamPointers()
                   << " was in error state before adding host callback";
  }
  CheckError(parent_->HostCallback(this, std::move(callback)));
  return *this;
}

Stream& Stream::ThenRunAfterNextBlockHostUntilDone(
    std::function<void()> callback) {
  // VLOG_CALL(PARAM(callback));

  if (!ok()) {
    ITEX_LOG(INFO)
        << DebugStreamPointers()
        << " was in error state before adding callback to be run after "
           "next block-host-until-done.";
  }
  absl::MutexLock lock(&mu_);
  after_block_host_until_done_callbacks_.push_back(std::move(callback));
  return *this;
}

void Stream::CheckError(bool operation_retcode) {
  if (operation_retcode) {
    return;
  }
  absl::MutexLock lock(&mu_);
  status_ = port::InternalError("Unknown error");
}

// It looks confusing, but all this is doing is inserting a callback at the
// present point in the stream to then enqueue a task on the host executor.
Stream& Stream::ThenEnqueueOnBackgroundThread(
    std::function<void(StreamExecutor*)> task) {
  // VLOG_CALL(PARAM(task));

  StreamExecutor* stream_executor = this->parent_;
  std::function<void()> bound_task = std::bind(task, stream_executor);

  return ThenDoHostCallback([stream_executor, bound_task]() {
    stream_executor->EnqueueOnBackgroundThread(bound_task);
  });
}

port::Status Stream::BlockHostUntilDone() {
  // VLOG_CALL();

  if (!ok()) {
    absl::MutexLock lock(&mu_);
    ITEX_LOG(INFO) << status_.ToString();
    port::Status status = port::Status(
        itex::error::INTERNAL,
        "stream did not block host until done; was already in an error state");
    ITEX_LOG(INFO) << DebugStreamPointers() << " " << status;
    return status;
  }

  temporary_memory_manager_.DeallocateFinalizedTemporaries();

  port::Status error = parent_->BlockHostUntilDone(this);
  CheckError(error.ok());

  RunAfterBlockHostUntilDoneCallbacks();
  return error;
}

void Stream::RunAfterBlockHostUntilDoneCallbacks() {
  std::vector<std::function<void()>> callbacks;
  {
    absl::MutexLock lock(&mu_);
    std::swap(callbacks, after_block_host_until_done_callbacks_);
  }
  for (const auto& fn : callbacks) {
    fn();
  }
}

std::string Stream::DebugStreamPointers() const {
  // Relies on the ToVlogString(const void*) overload above.
  return absl::StrCat("[stream=", ToVlogString(this),
                      ",impl=", ToVlogString(implementation_.get()), "]");
}

void Stream::CheckStatus(port::Status status) {
  if (status.ok()) {
    return;
  }
  ITEX_LOG(ERROR) << status;
  absl::MutexLock lock(&mu_);
  status_ = status;
}

}  // namespace stream_executor
