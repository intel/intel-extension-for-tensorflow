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

#ifndef ITEX_CORE_PROFILER_GPU_COLLECTOR_H_
#define ITEX_CORE_PROFILER_GPU_COLLECTOR_H_

#include <level_zero/ze_api.h>

#include <string>
#include <utility>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>
#else
#error "Unsupported compiler"
#endif

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "itex/core/profiler/utils/parse_annotation.h"
#include "itex/core/profiler/utils/xplane_builder.h"
#include "itex/core/profiler/utils/xplane_schema.h"
#include "itex/core/profiler/ze_tracer.h"
#include "itex/core/utils/annotation_stack.h"
#include "itex/core/utils/mutex.h"

namespace itex {
namespace profiler {

inline std::string ToXStat(const ZeKernelProps& prop) {
  return strings::StrCat(" SIMD width:", prop.simd_width,
                         " grid:", prop.group_count[0], ",",
                         prop.group_count[1], ",", prop.group_count[2],
                         " block:", prop.group_size[0], ",", prop.group_size[1],
                         ",", prop.group_size[2]);
}

class PerDeviceCollector {
 public:
  PerDeviceCollector(int device_id, uint64_t start_walltime_ns,
                     uint64_t start_gpu_ns)
      : start_walltime_ns_(start_walltime_ns), start_gpu_ns_(start_gpu_ns) {
    ITEX_GPUDevice* device_h;
    ITEX_GPUGetDevice(&device_h, device_id);
    std::vector<ITEX_GPUStream*> stream_pool;
    ITEX_GPUGetStreamPool(device_h, &stream_pool);
    auto l0_native_queue =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*device_h);
    zePluggableTracerQueueList queue_list = ZeKernelCollector::
        GetzePluggableTracerDeviceQueueMap()[l0_native_queue];
    queues_.assign(queue_list.begin(), queue_list.end());
  }

  void CreateXEvent(const zePluggableTracerEventList& event_list,
                    XPlaneBuilder* plane, XLineBuilder* line) {
    for (const zePluggableTracerEvent& event : event_list) {
      std::string kernel_name = event.kernel_name;
      if (event.append_time + start_gpu_ns_ < start_walltime_ns_) {
        ITEX_VLOG(2) << "Skip events have abnormal timestamps:"
                     << event.kernel_name
                     << " start time(ns): " << event.append_time + start_gpu_ns_
                     << " start wall time(ns): " << start_walltime_ns_;
        continue;
      }
      XEventMetadata* event_metadata =
          plane->GetOrCreateEventMetadata(std::move(kernel_name));
      XEventBuilder xevent = line->AddEvent(*event_metadata);
      xevent.SetTimestampNs(event.host_start_time + start_gpu_ns_);
      xevent.SetEndTimestampNs(event.host_end_time + start_gpu_ns_);

      if (event.kernel_props.bytes_transferred > 0) {
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                std::string("Memory transfered bytes")),
                            event.kernel_props.bytes_transferred);
      } else {
        xevent.AddStatValue(
            *plane->GetOrCreateStatMetadata(
                GetStatTypeStr(StatType::kKernelDetails)),
            *plane->GetOrCreateStatMetadata(ToXStat(event.kernel_props)));
      }
      std::vector<Annotation> annotation_stack =
          ParseAnnotationStack(event.annotation);
      if (!annotation_stack.empty()) {
        xevent.AddStatValue(
            *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
            *plane->GetOrCreateStatMetadata(annotation_stack.begin()->name));
      }

      // If multiple metadata have the same key name, show the values from the
      // top of the stack (innermost annotation). Concatenate the values from
      // "hlo_op".
      absl::flat_hash_set<absl::string_view> key_set;

      for (auto annotation = annotation_stack.rbegin();
           annotation != annotation_stack.rend(); ++annotation) {
        for (const Annotation::Metadata& metadata : annotation->metadata) {
          if (key_set.insert(metadata.key).second) {
            xevent.ParseAndAddStatValue(
                *plane->GetOrCreateStatMetadata(metadata.key), metadata.value);
          }
        }
      }
    }
  }

  void CreateHostXEvent(XPlaneBuilder* plane, XLineBuilder* line) const {
    for (zePluggableTracerHostEvent& event :
         ZeTracer::GetzePluggableTracerHostEventList()) {
      if (event.start_time + start_gpu_ns_ < start_walltime_ns_) continue;
      std::string api_name = event.api_name;
      XEventMetadata* event_metadata =
          plane->GetOrCreateEventMetadata(std::move(api_name));
      XEventBuilder xevent = line->AddEvent(*event_metadata);

      xevent.SetTimestampNs(event.start_time + start_gpu_ns_);
      xevent.SetEndTimestampNs(event.end_time + start_gpu_ns_);
    }
  }

  void Flush(XPlaneBuilder* device_plane) {
    mutex_lock lock(&mutex_);
    zePluggableTracerEventMap& event_map =
        ZeKernelCollector::GetzePluggableTracerEventMap();
    for (int i = 0; i < queues_.size(); i++) {
      int64_t line_id = i;
      XLineBuilder line = device_plane->GetOrCreateLine(line_id);
      line.SetTimestampNs(start_walltime_ns_);
      CreateXEvent(event_map[queues_[i]], device_plane, &line);
    }
    {  // Host Runtime API
      int64_t line_id = queues_.size();
      XLineBuilder line = device_plane->GetOrCreateLine(line_id);
      line.SetTimestampNs(start_walltime_ns_);
      CreateHostXEvent(device_plane, &line);
    }

    device_plane->ForEachLine([&](XLineBuilder line) {
      if (line.Id() < queues_.size())
        line.SetName(strings::StrCat("XPU queue/", line.Id()));
      else
        line.SetName("Host Runtime Call");
    });
  }

 private:
  std::vector<ze_command_queue_handle_t> queues_;
  uint64_t start_walltime_ns_;
  uint64_t start_gpu_ns_;
  mutex mutex_;
};

}  // namespace profiler
}  // namespace itex
#endif  // ITEX_CORE_PROFILER_GPU_COLLECTOR_H_
