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

#ifndef ITEX_CORE_PROFILER_ZE_API_COLLECTOR_H_
#define ITEX_CORE_PROFILER_ZE_API_COLLECTOR_H_

#include <level_zero/layers/zel_tracing_api.h>

#include <chrono>  // NOLINT(build/c++11)
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>  // NOLINT(build/c++11)
#include <set>
#include <string>

#include "itex/core/profiler/correlator.h"
#include "itex/core/profiler/utils.h"
#include "itex/core/profiler/ze_utils.h"

struct ZeFunction {
  uint64_t total_time;
  uint64_t min_time;
  uint64_t max_time;
  uint64_t call_count;

  bool operator>(const ZeFunction& r) const {
    if (total_time != r.total_time) {
      return total_time > r.total_time;
    }
    return call_count > r.call_count;
  }

  bool operator!=(const ZeFunction& r) const {
    if (total_time == r.total_time) {
      return call_count != r.call_count;
    }
    return true;
  }
};

using ZeFunctionInfoMap = std::map<std::string, ZeFunction>;

typedef void (*OnZeFunctionFinishCallback)(void* data, const std::string& id,
                                           const std::string& name,
                                           uint64_t started, uint64_t ended);

class ZeApiCollector {
 public:  // User Interface
  static ZeApiCollector* Create(Correlator* correlator,
                                ApiCollectorOptions options = {false, false,
                                                               false},
                                OnZeFunctionFinishCallback callback = nullptr,
                                void* callback_data = nullptr) {
    PTI_ASSERT(correlator != nullptr);
    ZeApiCollector* collector =
        new ZeApiCollector(correlator, options, callback, callback_data);
    PTI_ASSERT(collector != nullptr);

    ze_result_t status = ZE_RESULT_SUCCESS;
    zel_tracer_desc_t tracer_desc = {ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC,
                                     nullptr, collector};
    zel_tracer_handle_t tracer = nullptr;

    status = zelTracerCreate(&tracer_desc, &tracer);
    if (status != ZE_RESULT_SUCCESS || tracer == nullptr) {
      std::cerr << "[WARNING] Unable to create L0 tracer" << std::endl;
      delete collector;
      return nullptr;
    }

    collector->tracer_ = tracer;
    SetTracingAPIs(tracer);

    status = zelTracerSetEnabled(tracer, true);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    return collector;
  }

  void DisableTracing() {
    PTI_ASSERT(tracer_ != nullptr);
    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zelTracerSetEnabled(tracer_, false);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  }

  const ZeFunctionInfoMap& GetFunctionInfoMap() const {
    return function_info_map_;
  }

  ~ZeApiCollector() {
    if (tracer_ != nullptr) {
      ze_result_t status = zelTracerDestroy(tracer_);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
  }

 private:  // Tracing Interface
  uint64_t GetTimestamp() const {
    PTI_ASSERT(correlator_ != nullptr);
    return correlator_->GetTimestamp();
  }

  void AddFunctionTime(const std::string& name, uint64_t time) {
    const std::lock_guard<std::mutex> lock(lock_);
    if (function_info_map_.count(name) == 0) {
      function_info_map_[name] = {time, time, time, 1};
    } else {
      ZeFunction& function = function_info_map_[name];
      function.total_time += time;
      if (time < function.min_time) {
        function.min_time = time;
      }
      if (time > function.max_time) {
        function.max_time = time;
      }
      ++function.call_count;
    }
  }

 private:  // Implementation Details
  ZeApiCollector(Correlator* correlator, ApiCollectorOptions options,
                 OnZeFunctionFinishCallback callback, void* callback_data)
      : correlator_(correlator),
        callback_(callback),
        callback_data_(callback_data) {
    PTI_ASSERT(correlator_ != nullptr);
  }

#include "itex/core/profiler/tracing.h"  // Auto-generated callbacks

 private:  // Data
  zel_tracer_handle_t tracer_ = nullptr;

  ZeFunctionInfoMap function_info_map_;
  std::mutex lock_;

  Correlator* correlator_ = nullptr;

  OnZeFunctionFinishCallback callback_ = nullptr;
  void* callback_data_ = nullptr;

  static const uint32_t kFunctionLength = 10;
  static const uint32_t kCallsLength = 12;
  static const uint32_t kTimeLength = 20;
  static const uint32_t kPercentLength = 10;
};

#endif  // ITEX_CORE_PROFILER_ZE_API_COLLECTOR_H_
