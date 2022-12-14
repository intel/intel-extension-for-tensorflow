/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_
#define ITEX_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_

#include <stddef.h>

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "itex/core/profiler/utils/timespan.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/protobuf.h"
#include "itex/core/utils/time_utils.h"
#include "itex/core/utils/types.h"
#include "protos/xplane.pb.h"

namespace itex {
namespace profiler {

class XPlaneBuilder;

template <typename T>
class XStatsBuilder {
 public:
  explicit XStatsBuilder(T* stats_owner, XPlaneBuilder* stats_metadata_owner)
      : stats_owner_(stats_owner),
        stats_metadata_owner_(stats_metadata_owner) {}

  void AddStatValue(const XStatMetadata& metadata, uint32 value) {
    AddStat(metadata)->set_uint64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata,
                    unsigned long value) {  // NOLINT
    AddStat(metadata)->set_uint64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata,
                    unsigned long long value) {  // NOLINT
    AddStat(metadata)->set_uint64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, int32 value) {
    AddStat(metadata)->set_int64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, long value) {  // NOLINT
    AddStat(metadata)->set_int64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, long long value) {  // NOLINT
    AddStat(metadata)->set_int64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, double value) {
    AddStat(metadata)->set_double_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, absl::string_view value) {
    AddStat(metadata)->set_str_value(std::string(value));
  }
  void AddStatValue(const XStatMetadata& metadata, std::string&& value) {
    AddStat(metadata)->set_str_value(std::move(value));
  }
  void AddStatValue(const XStatMetadata& metadata, const XStatMetadata& value) {
    AddStat(metadata)->set_ref_value(value.id());
  }
  void AddStatValue(const XStatMetadata& metadata,
                    const protobuf::MessageLite& proto) {
    auto* bytes = AddStat(metadata)->mutable_bytes_value();
    proto.SerializeToString(bytes);
  }

  // Adds a stat by copying a stat from another XPlane. Does not check if a stat
  // with the same metadata already exists in the event. To avoid duplicated
  // stats, use the variant below.
  void AddStat(const XStatMetadata& metadata, const XStat& src_stat,
               const XPlane& src_plane) {
    CopyStatValue(src_stat, src_plane, AddStat(metadata));
  }
  // Same as above but overrides an existing stat with the same metadata.
  void SetOrAddStat(const XStatMetadata& metadata, const XStat& src_stat,
                    const XPlane& src_plane) {
    CopyStatValue(src_stat, src_plane, FindOrAddStat(metadata));
  }

  void ParseAndAddStatValue(const XStatMetadata& metadata,
                            absl::string_view value) {
    int64 int_value;
    uint64 uint_value;
    double double_value;
    if (absl::SimpleAtoi(value, &int_value)) {
      AddStatValue(metadata, int_value);
    } else if (absl::SimpleAtoi(value, &uint_value)) {
      AddStatValue(metadata, uint_value);
    } else if (absl::SimpleAtod(value, &double_value)) {
      AddStatValue(metadata, double_value);
    } else {
      AddStatValue(metadata, GetOrCreateStatMetadata(value));
    }
  }

  void ReserveStats(size_t num_stats) {
    stats_owner_->mutable_stats()->Reserve(num_stats);
  }

 private:
  XStat* AddStat(const XStatMetadata& metadata) {
    XStat* stat = stats_owner_->add_stats();
    stat->set_metadata_id(metadata.id());
    return stat;
  }

  XStat* FindOrAddStat(const XStatMetadata& metadata) {
    for (auto& stat : *stats_owner_->mutable_stats()) {
      if (stat.metadata_id() == metadata.id()) {
        return &stat;
      }
    }
    return AddStat(metadata);
  }

  void CopyStatValue(const XStat& src_stat, const XPlane& src_plane,
                     XStat* dst_stat) {
    switch (src_stat.value_case()) {
      case XStat::VALUE_NOT_SET:
        break;
      case XStat::kInt64Value:
        dst_stat->set_int64_value(src_stat.int64_value());
        break;
      case XStat::kUint64Value:
        dst_stat->set_uint64_value(src_stat.uint64_value());
        break;
      case XStat::kDoubleValue:
        dst_stat->set_double_value(src_stat.double_value());
        break;
      case XStat::kStrValue:
        dst_stat->set_str_value(src_stat.str_value());
        break;
      case XStat::kRefValue: {
        const auto& stat_metadata_by_id = src_plane.stat_metadata();
        const auto it = stat_metadata_by_id.find(src_stat.ref_value());
        if (ITEX_PREDICT_TRUE(it != stat_metadata_by_id.end())) {
          absl::string_view value = it->second.name();
          dst_stat->set_ref_value(GetOrCreateStatMetadata(value).id());
        }
        break;
      }
      case XStat::kBytesValue:
        dst_stat->set_bytes_value(src_stat.bytes_value());
        break;
    }
  }

  const XStatMetadata& GetOrCreateStatMetadata(absl::string_view value);

  T* stats_owner_;
  XPlaneBuilder* stats_metadata_owner_;
};

class XEventBuilder : public XStatsBuilder<XEvent> {
 public:
  XEventBuilder(const XLine* line, XPlaneBuilder* plane, XEvent* event)
      : XStatsBuilder<XEvent>(event, plane), line_(line), event_(event) {}

  int64 OffsetPs() const { return event_->offset_ps(); }
  int64 MetadataId() const { return event_->metadata_id(); }

  void SetOffsetPs(int64 offset_ps) { event_->set_offset_ps(offset_ps); }

  void SetOffsetNs(int64 offset_ns) { SetOffsetPs(NanosToPicos(offset_ns)); }

  void SetTimestampNs(int64 timestamp_ns) {
    SetOffsetPs(NanosToPicos(timestamp_ns - line_->timestamp_ns()));
  }

  void SetNumOccurrences(int64 num_occurrences) {
    event_->set_num_occurrences(num_occurrences);
  }

  void SetDurationPs(int64 duration_ps) {
    event_->set_duration_ps(duration_ps);
  }
  void SetDurationNs(int64 duration_ns) {
    SetDurationPs(NanosToPicos(duration_ns));
  }

  void SetEndTimestampPs(int64 end_timestamp_ps) {
    SetDurationPs(end_timestamp_ps - PicosToNanos(line_->timestamp_ns()) -
                  event_->offset_ps());
  }
  void SetEndTimestampNs(int64 end_timestamp_ns) {
    SetDurationPs(NanosToPicos(end_timestamp_ns - line_->timestamp_ns()) -
                  event_->offset_ps());
  }

  Timespan GetTimespan() const {
    return Timespan(NanosToPicos(line_->timestamp_ns()) + event_->offset_ps(),
                    event_->duration_ps());
  }

 private:
  const XLine* line_;
  XEvent* event_;
};

class XLineBuilder {
 public:
  explicit XLineBuilder(XLine* line, XPlaneBuilder* plane)
      : line_(line), plane_(plane) {}

  // Returns the owner plane.
  XPlaneBuilder* Plane() const { return plane_; }

  int64 Id() const { return line_->id(); }
  void SetId(int64 id) { line_->set_id(id); }

  int64 NumEvents() const { return line_->events_size(); }

  void SetName(absl::string_view name) { line_->set_name(std::string(name)); }

  void SetNameIfEmpty(absl::string_view name) {
    if (line_->name().empty()) SetName(name);
  }

  int64 TimestampNs() const { return line_->timestamp_ns(); }
  // This will set the line start timestamp.
  // WARNING: The offset_ps of existing events will not be altered.
  void SetTimestampNs(int64 timestamp_ns) {
    line_->set_timestamp_ns(timestamp_ns);
  }
  // This will set the line start timestamp to specific time, and adjust
  // the offset_ps of all existing events.
  void SetTimestampNsAndAdjustEventOffsets(int64 timestamp_ns);

  void SetDurationPs(int64 duration_ps) { line_->set_duration_ps(duration_ps); }

  void ReserveEvents(size_t num_events) {
    line_->mutable_events()->Reserve(num_events);
  }

  void SetDisplayNameIfEmpty(absl::string_view display_name) {
    if (line_->display_name().empty()) {
      line_->set_display_name(std::string(display_name));
    }
  }

  XEventBuilder AddEvent(const XEventMetadata& metadata);
  XEventBuilder AddEvent(const XEvent& event);

 private:
  XLine* line_;
  XPlaneBuilder* plane_;
};

// Provides methods to build an XPlane.
// NOTE: avoid to use two builders to wrap the same XPlane.
class XPlaneBuilder : public XStatsBuilder<XPlane> {
 public:
  explicit XPlaneBuilder(XPlane* plane);

  int64 Id() const { return plane_->id(); }
  void SetId(int64 id) { plane_->set_id(id); }

  void SetName(absl::string_view name) { plane_->set_name(std::string(name)); }

  void ReserveLines(size_t num_lines) {
    plane_->mutable_lines()->Reserve(num_lines);
  }

  template <typename ForEachLineFunc>
  void ForEachLine(ForEachLineFunc&& for_each_line) {
    for (XLine& line : *plane_->mutable_lines()) {
      for_each_line(XLineBuilder(&line, this));
    }
  }

  // Returns a builder for the line with the given id. Creates a new line if the
  // id was unused, otherwise the builder will add events to an existing line.
  XLineBuilder GetOrCreateLine(int64 line_id);

  // Returns a new event metadata with an automatically generated metadata_id.
  // WARNING: If calling this function, don't call GetOrCreateEventMetadata.
  XEventMetadata* CreateEventMetadata();

  // Returns event metadata with the given id. Creates a new metadata if the id
  // was unused.
  // WARNING: If calling this function, don't call the string overloads below
  // on the same instance.
  XEventMetadata* GetOrCreateEventMetadata(int64 metadata_id);

  // Returns event metadata with the given name. The id is internally assigned.
  // Creates a new metadata if the name was unused.
  // Using these overloads guarantees names are unique.
  // WARNING: If calling any of these overloads, do not call the integer one
  // above on the same instance.
  XEventMetadata* GetOrCreateEventMetadata(absl::string_view name);
  XEventMetadata* GetOrCreateEventMetadata(std::string&& name);
  XEventMetadata* GetOrCreateEventMetadata(const char* name) {
    return GetOrCreateEventMetadata(absl::string_view(name));
  }

  // Returns a new stat metadata with an automatically generated metadata_id.
  // WARNING: If calling this function, don't call GetOrCreateEventMetadata.
  XStatMetadata* CreateStatMetadata();

  // Returns stat metadata with the given id. Creates a new metadata if the id
  // was unused.
  // WARNING: If calling this function, don't call the string overloads below
  // on the same instance.
  XStatMetadata* GetOrCreateStatMetadata(int64 metadata_id);

  // Returns stat metadata with the given name. The id is internally assigned.
  // Creates a new metadata if the name was unused.
  // Using these overloads guarantees names are unique.
  // WARNING: If calling any of these overloads, do not call the integer one
  // above on the same instance.
  XStatMetadata* GetOrCreateStatMetadata(absl::string_view name);
  XStatMetadata* GetOrCreateStatMetadata(std::string&& name);
  XStatMetadata* GetOrCreateStatMetadata(const char* name) {
    return GetOrCreateStatMetadata(absl::string_view(name));
  }

 private:
  XPlane* plane_;

  // Artifacts to accelerate the builders.
  int64 last_event_metadata_id_ = 0LL;
  int64 last_stat_metadata_id_ = 0LL;
  absl::flat_hash_map<std::string, XEventMetadata*> event_metadata_by_name_;
  absl::flat_hash_map<std::string, XStatMetadata*> stat_metadata_by_name_;
  absl::flat_hash_map<int64, XLine*> lines_by_id_;
};

template <typename T>
const XStatMetadata& XStatsBuilder<T>::GetOrCreateStatMetadata(
    absl::string_view value) {
  return *stats_metadata_owner_->GetOrCreateStatMetadata(value);
}

}  // namespace profiler
}  // namespace itex

#endif  // ITEX_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_
