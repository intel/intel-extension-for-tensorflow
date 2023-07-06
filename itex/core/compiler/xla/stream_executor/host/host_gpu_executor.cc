/* Copyright (c) 2023 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Implementation of HostExecutor class [of those methods not defined in the
// class declaration].
#include "itex/core/compiler/xla/stream_executor/host/host_gpu_executor.h"

#include <stdint.h>
#include <string.h>

#include <cstdint>
#include <memory>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "itex/core/compiler/xla/stream_executor/host/host_platform_id.h"
#include "itex/core/compiler/xla/stream_executor/host/host_stream.h"
#include "itex/core/compiler/xla/stream_executor/host/host_timer.h"
#include "itex/core/compiler/xla/stream_executor/lib/statusor.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor_internal.h"
#include "itex/core/utils/mem.h"

namespace stream_executor {
namespace host {

HostStream* AsHostStream(Stream* stream) {
  ITEX_DCHECK(stream != nullptr);
  return dynamic_cast<HostStream*>(stream->implementation());
}

HostExecutor::HostExecutor(const PluginConfig& plugin_config)
    : plugin_config_(plugin_config) {}

HostExecutor::~HostExecutor() {}

port::Status HostExecutor::Init(int device_ordinal,
                                DeviceOptions device_options) {
  auto it =
      device_options.non_portable_tags.find("host_thread_stack_size_in_bytes");
  if (it != device_options.non_portable_tags.end()) {
    if (!absl::SimpleAtoi(it->second, &thread_stack_size_in_bytes_)) {
      return port::InvalidArgumentError(absl::StrCat(
          "Unable to parse host_thread_stack_size_in_bytes as an integer: ",
          it->second));
    }
  }
  return ::itex::OkStatus();
}

bool HostExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  itex::port::MemoryInfo mem_info = tsl::port::GetMemoryInfo();
  *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
  *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
  return true;
}

DeviceMemoryBase HostExecutor::Allocate(uint64_t size, int64_t memory_space) {
  ITEX_CHECK_EQ(memory_space, 0);
  // Use a minimum alignment of 64 bytes to be friendly to AVX512 code.
  // This should probably be kept in sync with
  // itex::Allocator::kAllocatorAlignment.
  return DeviceMemoryBase(
      itex::port::AlignedMalloc(size, /*minimum_alignment=*/64), size);
}

void* HostExecutor::GetSubBuffer(DeviceMemoryBase* parent,
                                 uint64_t offset_bytes, uint64_t size_bytes) {
  return reinterpret_cast<char*>(parent->opaque()) + offset_bytes;
}

void HostExecutor::Deallocate(DeviceMemoryBase* mem) {
  itex::port::AlignedFree(mem->opaque());
}

port::Status HostExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
  memset(location->opaque(), 0, size);
  return ::itex::OkStatus();
}

port::Status HostExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                             int value, uint64_t size) {
  memset(location->opaque(), value, size);
  return ::itex::OkStatus();
}

bool HostExecutor::Memcpy(Stream* stream, void* host_dst,
                          const DeviceMemoryBase& gpu_src, uint64_t size) {
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  AsHostStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool HostExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                          const void* host_src, uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool HostExecutor::MemcpyDeviceToDevice(Stream* stream,
                                        DeviceMemoryBase* gpu_dst,
                                        const DeviceMemoryBase& gpu_src,
                                        uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  // Enqueue this [asynchronous] "device-to-device" (i.e., host-to-host, given
  // the nature of the HostExecutor) memcpy  on the stream (HostStream)
  // associated with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return true;
}

port::Status HostExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                   uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return ::itex::OkStatus();
}

port::Status HostExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                  uint8 pattern, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return ::itex::OkStatus();
}

port::Status HostExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                    uint32_t pattern, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return ::itex::OkStatus();
}

port::Status HostExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  memcpy(gpu_dst->opaque(), host_src, size);
  return ::itex::OkStatus();
}

port::Status HostExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  memcpy(host_dst, gpu_src.opaque(), size);
  return ::itex::OkStatus();
}

port::Status HostExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
  memcpy(gpu_dst->opaque(), gpu_src.opaque(), size);
  return ::itex::OkStatus();
}

bool HostExecutor::HostCallback(Stream* stream,
                                std::function<port::Status()> callback) {
  AsHostStream(stream)->EnqueueTaskWithStatus(callback);
  return true;
}

bool HostExecutor::AllocateStream(Stream* stream) { return true; }

void HostExecutor::DeallocateStream(Stream* stream) {}

bool HostExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  auto event = std::make_shared<absl::Notification>();
  AsHostStream(other)->EnqueueTask([event]() { event->Notify(); });
  AsHostStream(dependent)->EnqueueTask(
      [event]() { event->WaitForNotification(); });
  return true;
}

class HostEvent : public internal::EventInterface {
 public:
  HostEvent() : notification_(std::make_shared<absl::Notification>()) {}

  std::shared_ptr<absl::Notification>& notification() { return notification_; }

 private:
  // We use a std::shared_ptr here because the client may delete the HostEvent
  // object while there are still RecordEvent and WaitForEvent callbacks pending
  // on a stream.
  std::shared_ptr<absl::Notification> notification_;
};

std::unique_ptr<internal::EventInterface>
HostExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new HostEvent());
}

static HostEvent* AsHostEvent(Event* event) {
  ITEX_DCHECK(event != nullptr);
  return static_cast<HostEvent*>(event->implementation());
}

port::Status HostExecutor::AllocateEvent(Event* /*event*/) {
  return ::itex::OkStatus();
}

port::Status HostExecutor::DeallocateEvent(Event* /*event*/) {
  return ::itex::OkStatus();
}

port::Status HostExecutor::RecordEvent(Stream* stream, Event* event) {
  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->notification();
  AsHostStream(stream)->EnqueueTask([notification]() {
    ITEX_CHECK(!notification->HasBeenNotified());
    notification->Notify();
  });
  return ::itex::OkStatus();
}

port::Status HostExecutor::WaitForEvent(Stream* stream, Event* event) {
  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->notification();
  AsHostStream(stream)->EnqueueTask(
      [notification]() { notification->WaitForNotification(); });
  return ::itex::OkStatus();
}

Event::Status HostExecutor::PollForEventStatus(Event* event) {
  absl::Notification& notification = *AsHostEvent(event)->notification();
  return notification.HasBeenNotified() ? Event::Status::kComplete
                                        : Event::Status::kPending;
}

bool HostExecutor::StartTimer(Stream* stream, Timer* timer) {
  dynamic_cast<HostTimer*>(timer->implementation())->Start(stream);
  return true;
}

bool HostExecutor::StopTimer(Stream* stream, Timer* timer) {
  dynamic_cast<HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

port::Status HostExecutor::BlockHostUntilDone(Stream* stream) {
  return AsHostStream(stream)->BlockUntilDone();
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
HostExecutor::CreateDeviceDescription(int device_ordinal) {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  // TODO(rspringer): How to report a value that's based in reality but that
  // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
  builder.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);

  float cycle_counter_frequency = static_cast<float>(
      itex::profile_utils::CpuUtils::GetCycleCounterFrequency());
  builder.set_clock_rate_ghz(cycle_counter_frequency / 1e9);

  builder.set_name("Host");
  builder.set_platform_version("Default Version");

  return builder.Build();
}

std::unique_ptr<internal::StreamInterface>
HostExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(
      new HostStream(thread_stack_size_in_bytes_));
}

}  // namespace host
}  // namespace stream_executor
