/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_STREAM_ASSIGNMENT_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_STREAM_ASSIGNMENT_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "itex/core/compiler/xla/service/hlo_instruction.h"
#include "itex/core/compiler/xla/service/hlo_module.h"

namespace itex_xla {
namespace gpu {

// This class encapsulates the assignment of GPU streams to each HloInstruction.
class StreamAssignment {
 public:
  int StreamCount() const { return stream_count_; }
  int StreamNumberForHlo(const HloInstruction& hlo) const;
  bool HasStreamAssigned(const HloInstruction& hlo) const;
  // `hlo` needs to outlive this StreamAssignment object.
  void AssignStreamToHlo(const HloInstruction* hlo, int stream_num);

 private:
  int stream_count_ = 1;  // At least the main stream.
  absl::flat_hash_map<const HloInstruction*, int> hlo_to_stream_number_;
};

// Assigns GPU streams to instructions in `module`.
std::unique_ptr<StreamAssignment> AssignStreams(const HloModule& module);

}  // namespace gpu
}  // namespace itex_xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_STREAM_ASSIGNMENT_H_
