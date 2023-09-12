/* Copyright (c) 2021-2022 Intel Corporation

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

//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef ITEX_CORE_PROFILER_UTILS_H_
#define ITEX_CORE_PROFILER_UTILS_H_

#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "itex/core/profiler/pti_assert.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define MAX_STR_SIZE 1024

#define BYTES_IN_MBYTES (1024 * 1024)

#define NSEC_IN_USEC 1000
#define MSEC_IN_SEC 1000
#define NSEC_IN_MSEC 1000000
#define NSEC_IN_SEC 1000000000

namespace utils {

std::atomic<int> g_immediate_command_list_enabled(1);

inline bool IsImmediateCommandListEnabled() {
  return g_immediate_command_list_enabled.load(std::memory_order_acquire);
}

inline void ImmediateCommandListDisabled() {
  g_immediate_command_list_enabled.store(0, std::memory_order_release);
}

struct Comparator {
  template <typename T>
  bool operator()(const T& left, const T& right) const {
    if (left.second != right.second) {
      return left.second > right.second;
    }
    return left.first > right.first;
  }
};

#if defined(__gnu_linux__)
inline uint64_t ConvertClockMonotonicToRaw(uint64_t clock_monotonic) {
  int status = 0;

  timespec monotonic_time;
  status = clock_gettime(CLOCK_MONOTONIC, &monotonic_time);
  PTI_ASSERT(status == 0);

  timespec raw_time;
  status = clock_gettime(CLOCK_MONOTONIC_RAW, &raw_time);
  PTI_ASSERT(status == 0);

  uint64_t raw = raw_time.tv_nsec + NSEC_IN_SEC * raw_time.tv_sec;
  uint64_t monotonic =
      monotonic_time.tv_nsec + NSEC_IN_SEC * monotonic_time.tv_sec;
  if (raw > monotonic) {
    return clock_monotonic + (raw - monotonic);
  } else {
    return clock_monotonic - (monotonic - raw);
  }
}
#endif

inline std::string GetExecutablePath() {
  char buffer[MAX_STR_SIZE] = {0};
#if defined(_WIN32)
  DWORD status = GetModuleFileNameA(nullptr, buffer, MAX_STR_SIZE);
  PTI_ASSERT(status > 0);
#else
  ssize_t status = readlink("/proc/self/exe", buffer, MAX_STR_SIZE);
  PTI_ASSERT(status > 0);
#endif
  std::string path(buffer);
  return path.substr(0, path.find_last_of("/\\") + 1);
}

inline std::string GetExecutableName() {
  char buffer[MAX_STR_SIZE] = {0};
#if defined(_WIN32)
  DWORD status = GetModuleFileNameA(nullptr, buffer, MAX_STR_SIZE);
  PTI_ASSERT(status > 0);
#else
  ssize_t status = readlink("/proc/self/exe", buffer, MAX_STR_SIZE);
  PTI_ASSERT(status > 0);
#endif
  std::string path(buffer);
  return path.substr(path.find_last_of("/\\") + 1);
}

inline std::vector<uint8_t> LoadBinaryFile(const std::string& path) {
  std::vector<uint8_t> binary;
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  if (!stream.good()) {
    return binary;
  }

  stream.seekg(0, std::ifstream::end);
  size_t size = stream.tellg();
  stream.seekg(0, std::ifstream::beg);
  if (size == 0) {
    return binary;
  }

  binary.resize(size);
  stream.read(reinterpret_cast<char*>(binary.data()), size);
  return binary;
}

inline void SetEnv(const char* name, const char* value) {
  PTI_ASSERT(name != nullptr);
  PTI_ASSERT(value != nullptr);

  int status = 0;
#if defined(_WIN32)
  std::string str = std::string(name) + "=" + value;
  status = _putenv(str.c_str());
#else
  status = setenv(name, value, 1);
#endif
  PTI_ASSERT(status == 0);
}

inline std::string GetEnv(const char* name) {
  PTI_ASSERT(name != nullptr);
#if defined(_WIN32)
  char* value = nullptr;
  errno_t status = _dupenv_s(&value, nullptr, name);
  PTI_ASSERT(status == 0);
  if (value == nullptr) {
    return std::string();
  }
  std::string result(value);
  free(value);
  return result;
#else
  const char* value = getenv(name);
  if (value == nullptr) {
    return std::string();
  }
  return std::string(value);
#endif
}

inline uint32_t GetPid() {
#if defined(_WIN32)
  return GetCurrentProcessId();
#else
  return getpid();
#endif
}

inline uint32_t GetTid() {
#if defined(_WIN32)
  return GetCurrentThreadId();
#else
#ifdef SYS_gettid
  return syscall(SYS_gettid);
#else
#error "SYS_gettid is unavailable on this system"
#endif
#endif
}

}  // namespace utils

#endif  // ITEX_CORE_PROFILER_UTILS_H_
