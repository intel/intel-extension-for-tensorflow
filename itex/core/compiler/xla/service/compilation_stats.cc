/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/compiler/xla/service/compilation_stats.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "itex/core/compiler/xla/types.h"
#include "itex/core/utils/env.h"

namespace itex_xla {

class NoopStats : public CompilationStats {
 public:
  NoopStats() = default;

  void StartPass(absl::string_view pass_name) override {}

  void EndPass(absl::string_view pass_name) override {}

  void CompilationReport() override {}

  int GetPassesSize() override { return 0; }
};

class Stats : public CompilationStats {
 public:
  Stats() = default;

  void StartPass(absl::string_view pass_name) override;

  void EndPass(absl::string_view pass_name) override;

  void CompilationReport() override;

  int GetPassesSize() override;

 private:
  struct PassInfo {
    PassInfo(absl::string_view name, double duration)
        : name(name), duration_ms(duration) {}

    std::string name;
    int num_runs = 1;
    double duration_ms;
  };

  // Info about the passes that have been run so far.
  std::vector<PassInfo> passes_;
  // Used to avoid nested calls to StartPass.
  bool pass_running_ = false;
  std::string current_pass_;
  // The start time of the currently running pass.
  uint64_t start_micros_;
};

/* static */
std::unique_ptr<CompilationStats> CompilationStats::MakeNoopStats() {
  return absl::make_unique<NoopStats>();
}

/* static */
std::unique_ptr<CompilationStats> CompilationStats::MakeStats() {
  return absl::make_unique<Stats>();
}

void Stats::StartPass(absl::string_view pass_name) {
  ITEX_CHECK(!pass_running_)
      << "Can't start " << pass_name << " while running " << current_pass_;
  pass_running_ = true;
  current_pass_ = std::string(pass_name);
  start_micros_ = itex::Env::Default()->NowMicros();
}

void Stats::EndPass(absl::string_view pass_name) {
  ITEX_CHECK(pass_running_);
  ITEX_CHECK_EQ(current_pass_, std::string(pass_name));
  pass_running_ = false;
  uint64_t end_micros = itex::Env::Default()->NowMicros();
  double duration_ms = (end_micros - start_micros_) / 1000.0;
  passes_.push_back(PassInfo(current_pass_, duration_ms));
}

void Stats::CompilationReport() {
  ITEX_CHECK(!pass_running_) << "EndPass never called for " << current_pass_;
  absl::flat_hash_map<std::string, PassInfo> summary;
  double total_duration = 0;

  for (auto& pass_run : passes_) {
    auto pass_name = pass_run.name;
    total_duration += pass_run.duration_ms;
    auto it = summary.find(pass_name);
    if (it == summary.end()) {
      summary.insert(std::make_pair(pass_name, pass_run));
    } else {
      ++summary.at(pass_name).num_runs;
      summary.at(pass_name).duration_ms += pass_run.duration_ms;
    }
  }

  std::vector<PassInfo> sorted_summary;
  sorted_summary.reserve(summary.size());
  for (auto& it : summary) {
    sorted_summary.push_back(it.second);
  }
  absl::c_sort(sorted_summary, [](const PassInfo& a, const PassInfo& b) {
    // Sort passes that take the longest first, break ties using pass names.
    return std::make_pair(b.duration_ms, a.name) <
           std::make_pair(a.duration_ms, b.name);
  });
  ITEX_LOG(INFO) << "Total runtime (ms) of HLO passes: " << total_duration;
  ITEX_LOG(INFO) << "Pass name, num runs, time (ms)";
  for (auto& pass_info : sorted_summary) {
    ITEX_LOG(INFO) << pass_info.name << ", " << pass_info.num_runs << ", "
                   << pass_info.duration_ms;
  }
}

int Stats::GetPassesSize() { return passes_.size(); }

}  // namespace itex_xla
