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

#ifndef ITEX_CORE_PROFILER_ZE_TRACER_H_
#define ITEX_CORE_PROFILER_ZE_TRACER_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "itex/core/profiler/correlator.h"
#include "itex/core/profiler/trace_options.h"
#include "itex/core/profiler/utils.h"
#include "itex/core/profiler/ze_api_collector.h"
#include "itex/core/profiler/ze_kernel_collector.h"
#include "itex/core/utils/time_utils.h"
#include "protos/xplane.pb.h"

const char* kChromeTraceFileName = "zet_trace";

struct zePluggableTracerHostEvent {
  zePluggableTracerHostEvent(std::string name, uint64_t start, uint64_t end)
      : api_name(name), start_time(start), end_time(end) {}
  std::string api_name;
  uint64_t start_time;
  uint64_t end_time;
};

class ZeTracer {
 public:
  static ZeTracer* Create(const TraceOptions& options) {
    ZeTracer* tracer = new ZeTracer(options);

    ZeKernelCollector* kernel_collector = nullptr;
    if (tracer->CheckOption(TRACE_DEVICE_TIMING) ||
        tracer->CheckOption(TRACE_DEVICE_TIMING_VERBOSE) ||
        tracer->CheckOption(TRACE_DEVICE_TIMELINE) ||
        tracer->CheckOption(TRACE_CHROME_DEVICE_TIMELINE) ||
        tracer->CheckOption(TRACE_CHROME_DEVICE_STAGES)) {
      PTI_ASSERT(!(tracer->CheckOption(TRACE_CHROME_DEVICE_TIMELINE) &&
                   tracer->CheckOption(TRACE_CHROME_DEVICE_STAGES)));

      OnZeKernelFinishCallback callback = nullptr;
      kernel_collector = ZeKernelCollector::Create(
          &(tracer->correlator_),
          tracer->CheckOption(TRACE_DEVICE_TIMING_VERBOSE), callback, tracer);
      if (kernel_collector == nullptr) {
        std::cerr << "[WARNING] Unable to create kernel collector" << std::endl;
        delete tracer;
        return nullptr;
      }
      tracer->kernel_collector_ = kernel_collector;
    }

    ZeApiCollector* api_collector = nullptr;
    if (tracer->CheckOption(TRACE_CALL_LOGGING) ||
        tracer->CheckOption(TRACE_CHROME_CALL_LOGGING) ||
        tracer->CheckOption(TRACE_HOST_TIMING) ||
        tracer->CheckOption(TRACE_HOST_RUNTIME_TIMING)) {
      OnZeFunctionFinishCallback callback = HostRuntimeTracingCallback;
      ApiCollectorOptions options{false, false, false};
      options.call_tracing = tracer->CheckOption(TRACE_CALL_LOGGING);
      options.need_tid = tracer->CheckOption(TRACE_TID);
      options.need_pid = tracer->CheckOption(TRACE_PID);

      api_collector = ZeApiCollector::Create(&(tracer->correlator_), options,
                                             callback, tracer);
      if (api_collector == nullptr) {
        std::cerr << "[WARNING] Unable to create API collector" << std::endl;
        delete tracer;
        return nullptr;
      }
      tracer->api_collector_ = api_collector;
    }

    return tracer;
  }

  static std::vector<zePluggableTracerHostEvent>&
  GetzePluggableTracerHostEventList() {
    static std::vector<zePluggableTracerHostEvent> host_event_list_;
    return host_event_list_;
  }

  void Start() {
    start_walltime_ns_ = itex::profiler::GetCurrentTimeNanos();
    utils::SetEnv("ZE_ENABLE_TRACING_LAYER", "1");
    utils::SetEnv("NEOReadDebugKeys", "1");
    utils::SetEnv("UseCyclesPerSecondTimer", "1");
  }

  void Stop() {
    // Report();
  }

  uint64_t GetStartWallTime() { return start_walltime_ns_; }

  uint64_t GetGPUStartTime() { return correlator_.GetStartPoint(); }

  uint64_t GetTimestamp() { return correlator_.GetTimestamp(); }

  ~ZeTracer() {
    total_execution_time_ = correlator_.GetTimestamp();

    if (api_collector_ != nullptr) {
      api_collector_->DisableTracing();
    }
    if (kernel_collector_ != nullptr) {
      kernel_collector_->DisableTracing();
    }

    if (api_collector_ != nullptr) {
      delete api_collector_;
    }
    if (kernel_collector_ != nullptr) {
      delete kernel_collector_;
    }
  }

  bool CheckOption(unsigned option) { return options_.CheckFlag(option); }

  ZeTracer(const ZeTracer& copy) = delete;
  ZeTracer& operator=(const ZeTracer& copy) = delete;

 private:
  explicit ZeTracer(const TraceOptions& options)
      : options_(options), correlator_() {}

  static void HostRuntimeTracingCallback(void* data, const std::string& id,
                                         const std::string& name,
                                         uint64_t started, uint64_t ended) {
    static std::mutex lock_;
    const std::lock_guard<std::mutex> lock(lock_);
    ZeTracer::GetzePluggableTracerHostEventList().emplace_back(name, started,
                                                               ended);
  }

 private:
  TraceOptions options_;
  uint64_t start_walltime_ns_;
  std::string chrome_trace_file_name_;
  Correlator correlator_;
  uint64_t total_execution_time_ = 0;

  ZeApiCollector* api_collector_ = nullptr;
  ZeKernelCollector* kernel_collector_ = nullptr;
};

#endif  // ITEX_CORE_PROFILER_ZE_TRACER_H_
