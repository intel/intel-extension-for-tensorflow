/* Copyright (c) 2023 Intel Corporation

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
#ifndef ITEX_CORE_COMPILER_XLA_PJRT_XPU_PJRT_CLIENT_H_
#define ITEX_CORE_COMPILER_XLA_PJRT_XPU_PJRT_CLIENT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "itex/core/compiler/c/pjrt_c_api.h"
#include "itex/core/compiler/xla/pjrt/event_pool.h"
#include "itex/core/compiler/xla/pjrt/pjrt_client.h"
#include "itex/core/compiler/xla/pjrt/pjrt_compiler.h"

struct PJRT_Error {
  itex_xla::Status status;
};

struct PJRT_Client {
  std::unique_ptr<itex_xla::PjRtClient> client;
  std::vector<PJRT_Device> owned_devices;
  // `devices` contains the addresses of the contents of `owned_devices`.
  std::vector<PJRT_Device*> devices;
  // `addressable_devices` contains pointers to the `owned_devices` that the
  // client can issue commands to.
  std::vector<PJRT_Device*> addressable_devices;
  // Map from wrapped C++ devices to C devices. The values are the same as
  // `owned_devices`.
  absl::flat_hash_map<itex_xla::PjRtDevice*, PJRT_Device*>
      c_device_from_cpp_device;
};

// PJRT_Devices are owned by their corresponding PJRT_Client.
struct PJRT_Device {
  // The xla::PjRtDevice* is owned by the corresponding xla::PjRtClient.
  itex_xla::PjRtDevice* device;
  // The device specific attributes which are initialized once per device.
  std::vector<PJRT_NamedValue> attributes;
};

struct PJRT_Executable {
  // Must be shared_ptr so that we can share with PJRT_LoadedExecutable.
  std::shared_ptr<itex_xla::PjRtExecutable> executable;

  explicit PJRT_Executable(
      std::shared_ptr<itex_xla::PjRtExecutable> executable);

  const itex_xla::PjRtExecutable* get() const { return executable.get(); }
  itex_xla::PjRtExecutable* get() { return executable.get(); }
};

struct PJRT_LoadedExecutable {
  // Must be shared_ptr so that we can share with PJRT_Executable.
  std::shared_ptr<itex_xla::PjRtLoadedExecutable> executable;
  PJRT_Client* client;
  // These pointers are a subset of `client`'s `addressable_devices`, i.e. those
  // addressed by the compiled executable program. `client` owns the objects
  // these point to.
  std::vector<PJRT_Device*> addressable_devices;

  mutable absl::Mutex mutex;
  // Cost analysis properties and name strings are populated after cost analysis
  // has been run. These are returned from cost analysis calls, and do not
  // change after the first call.
  bool cost_analysis_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<std::string> cost_analysis_names;
  std::vector<PJRT_NamedValue> cost_analysis_properties;

  PJRT_LoadedExecutable(
      std::shared_ptr<itex_xla::PjRtLoadedExecutable> executable,
      PJRT_Client* client);

  const itex_xla::PjRtLoadedExecutable* get() const { return executable.get(); }
  itex_xla::PjRtLoadedExecutable* get() { return executable.get(); }
};

struct PJRT_Buffer {
  std::unique_ptr<itex_xla::PjRtBuffer> buffer;
  PJRT_Client* client;
};

struct PJRT_Event {
  itex_xla::PjRtFuture<itex_xla::Status> future;
  // Set and stored upon future.Await(), as PjRtFuture only allows its result to
  // be queried through Await() and Await() can only safely be called once. This
  // variable allows C API users to check for error status any time after
  // Await() has been called.
  std::optional<itex_xla::Status> status;
};

struct PJRT_SerializedExecutable {
  std::string serialized;
};

struct PJRT_DeviceTopology {
  std::unique_ptr<itex_xla::PjRtDeviceTopology> topology;
};

namespace itex_xla {
// Helper macros and functions

#define PJRT_RETURN_IF_ERROR(expr)                                \
  do {                                                            \
    itex_xla::Status _status = (expr);                            \
    if (!_status.ok()) {                                          \
      PJRT_Error* _c_status = new PJRT_Error{std::move(_status)}; \
      return _c_status;                                           \
    }                                                             \
  } while (false)

#define PJRT_ASSIGN_OR_RETURN(lhs, rexpr)                                  \
  _PJRT_ASSIGN_OR_RETURN_IMPL(_PJRT_CONCAT(_status_or_value, __COUNTER__), \
                              lhs, rexpr,                                  \
                              _PJRT_CONCAT(_c_status, __COUNTER__));

#define _PJRT_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr, c_status) \
  auto statusor = (rexpr);                                          \
  if (!statusor.ok()) {                                             \
    PJRT_Error* c_status = new PJRT_Error();                        \
    c_status->status = statusor.status();                           \
    return c_status;                                                \
  }                                                                 \
  lhs = std::move(*statusor)

#define _PJRT_CONCAT(x, y) _PJRT_CONCAT_IMPL(x, y)
#define _PJRT_CONCAT_IMPL(x, y) x##y

// Helper function for checking C API argument struct sizes. Returns a non-OK
// status if the expected and actual sizes aren't equal (i.e. no ABI
// compatibility guarantees).
itex_xla::Status CheckMatchingStructSizes(absl::string_view struct_name,
                                          size_t expected_size,
                                          size_t actual_size);

// Helper function
std::string StructSizeErrorMsg(absl::string_view struct_name,
                               size_t expected_size, size_t actual_size);

// Returns a specific error message when the program format is unknown.
// Does not check the program format itself.
std::string ProgramFormatErrorMsg(absl::string_view program_format);
}  // namespace itex_xla
#endif  // ITEX_CORE_COMPILER_XLA_PJRT_XPU_PJRT_CLIENT_H_
