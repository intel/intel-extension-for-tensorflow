/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/service/gpu/runtime_intrinsics.h"

#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "itex/core/compiler/xla/service/custom_call_status.h"
#include "itex/core/compiler/xla/service/custom_call_target_registry.h"
#include "itex/core/compiler/xla/shape.h"
#include "itex/core/compiler/xla/shape_util.h"
#include "itex/core/compiler/xla/status.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/compiler/xla/util.h"

namespace itex_xla {

extern const char* const kXlaGpuAssertCustomCallTag = "__xla_gpu_assert";

static Status AssertOnGpu(void* stream_handle, void* buffer,
                          absl::string_view error_msg) {
  auto stream = reinterpret_cast<ITEX_GPUStream*>(stream_handle);
  int8_t* expected = sycl::malloc_host<int8_t>(1, *stream);
  int64_t byte_size = sizeof(int8_t);
  ITEX_CHECK_EQ(byte_size,
                ShapeUtil::ByteSizeOfPrimitiveType(PrimitiveType::PRED));
  stream->memcpy(expected, buffer, byte_size);
  stream->wait();
  if (!static_cast<bool>(*expected)) {
    return InternalError("%s", error_msg);
  }
  return OkStatus();
}

static void AssertionCustomCall(void* stream_handle, void** buffers,
                                const char* opaque, int opaque_len,
                                ItexXlaCustomCallStatus* status) {
  Status s =
      AssertOnGpu(stream_handle, buffers[0],
                  absl::string_view{opaque, static_cast<uint64_t>(opaque_len)});
  if (!s.ok()) {
    ItexXlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                      s.error_message().size());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(kXlaGpuAssertCustomCallTag,
                                         AssertionCustomCall, "XPU");

}  // namespace itex_xla
