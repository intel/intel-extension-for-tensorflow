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

#include "itex/core/compiler/xla/pjrt/xpu_pjrt_client.h"

#include <iostream>

#include "itex/core/compiler/c/pjrt_c_api.h"
#include "itex/core/compiler/c/pjrt_c_api_tpu.h"
#include "itex/core/compiler/xla/client/xla_computation.h"
#include "itex/core/compiler/xla/literal.h"
#include "itex/core/compiler/xla/pjrt/c_api_conversions.h"
#include "itex/core/compiler/xla/pjrt/mlir_to_hlo.h"
#include "itex/core/compiler/xla/pjrt/pjrt_c_api_helpers.h"
#include "itex/core/compiler/xla/pjrt/se_xpu_pjrt_client.h"
#include "itex/core/compiler/xla/service/compiler.h"
#include "itex/core/compiler/xla/service/gpu/gpu_transfer_manager.h"
#include "itex/core/compiler/xla/service/gpu/spir_compiler.h"
#include "itex/core/compiler/xla/service/gpu/target_constants.h"
#include "itex/core/compiler/xla/shape.h"
#include "itex/core/compiler/xla/shape_util.h"
#include "itex/core/compiler/xla/status.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_platform_id.h"
#include "itex/core/utils/logging.h"
#include "itex/core/version.h"
#include "protos/hlo.pb.h"

namespace itex_xla {

itex_xla::Status CheckMatchingStructSizes(absl::string_view struct_name,
                                          size_t expected_size,
                                          size_t actual_size) {
  if (expected_size != actual_size) {
    return itex::errors::InvalidArgument(
        StructSizeErrorMsg(struct_name, expected_size, actual_size));
  }
  return itex::OkStatus();
}

std::string StructSizeErrorMsg(absl::string_view struct_name,
                               size_t expected_size, size_t actual_size) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ",
                      expected_size, ", got ", actual_size,
                      ". Check installed software versions.");
}

std::string ProgramFormatErrorMsg(absl::string_view program_format) {
  return absl::StrCat("Unknown program format '", program_format, "'.");
}

// Returns C device from wrapped C++ device.
static PJRT_Device* GetCDevice(const PJRT_Client* client,
                               const itex_xla::PjRtDevice* device) {
  auto c_device_map = client->c_device_from_cpp_device;
  auto iter = c_device_map.find(device);
  ITEX_CHECK(iter != c_device_map.end());
  return iter->second;
}

// Performs one-time cost-analysis on an executable if not done already, and
// populates its cost analysis properties. After this returns successfully,
// cost analysis properties of the executable can be accessed without mutex.
static itex_xla::Status PopulateExecutableCostAnalysisIfNeeded(
    PJRT_LoadedExecutable* executable) {
  absl::MutexLock lock(&executable->mutex);
  if (!executable->cost_analysis_ran) {
    // Call GetCostAnalysis in the underlying PjRtExecutable
    using PropertiesMapType =
        absl::flat_hash_map<std::string, itex_xla::PjRtValueType>;
    TF_ASSIGN_OR_RETURN(const PropertiesMapType properties,
                        executable->get()->GetCostAnalysis());
    // If no output, return empty result
    if (properties.empty()) {
      executable->cost_analysis_ran = true;
      return itex_xla::OkStatus();
    }

    // Copy each returned property to cost analysis vectors in PJRT_Executable
    std::vector<PJRT_NamedValue>& cost_analysis_properties =
        executable->cost_analysis_properties;
    cost_analysis_properties.resize((properties.size()));
    std::vector<std::string>& cost_analysis_names =
        executable->cost_analysis_names;
    cost_analysis_names.resize(properties.size());
    size_t i = 0;
    for (const auto& property : properties) {
      PJRT_NamedValue& cost_analysis_property = cost_analysis_properties[i];
      std::string& property_name = cost_analysis_names[i];

      cost_analysis_property.struct_size = PJRT_NamedValue_STRUCT_SIZE;
      cost_analysis_property.priv = nullptr;

      property_name = property.first;
      cost_analysis_property.name = property_name.c_str();
      cost_analysis_property.name_size = property_name.size();

      const itex_xla::PjRtValueType& property_value = property.second;
      ITEX_CHECK(std::holds_alternative<float>(property_value))
          << property_value.index();
      cost_analysis_property.type = PJRT_NamedValue::PJRT_NamedValue_kFloat;
      cost_analysis_property.float_value = std::get<float>(property_value);
      cost_analysis_property.value_size = 1;

      ++i;
    }
    executable->cost_analysis_ran = true;
  }
  return itex_xla::OkStatus();
}

// ---------------------------------- Errors -----------------------------------

void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args) {
  itex_xla::Status struct_size_check = CheckMatchingStructSizes(
      "PJRT_Error_Destroy_Args", PJRT_Error_Destroy_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    ITEX_LOG(ERROR) << struct_size_check.error_message();
  }
  if (args->struct_size >= PJRT_STRUCT_SIZE(PJRT_Error_Destroy_Args, error)) {
    delete args->error;
  }
}

void PJRT_Error_Message(PJRT_Error_Message_Args* args) {
  itex_xla::Status struct_size_check = CheckMatchingStructSizes(
      "PJRT_Error_Message_Args", PJRT_Error_Message_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    ITEX_LOG(ERROR) << struct_size_check.error_message();
  }
  if (args->struct_size >= PJRT_STRUCT_SIZE(PJRT_Error_Destroy_Args, error)) {
    const itex_xla::Status* status = &args->error->status;
    args->message = status->error_message().data();
    args->message_size = status->error_message().size();
  }
}

PJRT_Error* PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Error_GetCode_Args", PJRT_Error_GetCode_Args_STRUCT_SIZE,
      args->struct_size));
  args->code = StatusCodeToPjrtErrorCode(args->error->status.code());
  return nullptr;
}

// ---------------------------------- Client -----------------------------------

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  std::string jax_version = std::string(GetJaxVersion());
  if (jax_version.compare("0.4.4") < 0)
    PJRT_RETURN_IF_ERROR(itex::errors::Internal(
        "The plugin requires jax version at least 0.4.4"));
  static PJRT_Client client;
  args->client = new PJRT_Client();
  args->client->client =
      std::move(GetStreamExecutorXpuClient(false, 0).value());
  for (auto pjrt_device : args->client->client->devices()) {
    PJRT_Device* c_device = new PJRT_Device();
    c_device->device = pjrt_device;
    PJRT_NamedValue attribute;
    attribute.name = "vendor";
    attribute.name_size = sizeof(attribute.name) + 1;
    attribute.type = PJRT_NamedValue::PJRT_NamedValue_kString;
    attribute.string_value = "Intel";
    attribute.value_size = sizeof(attribute.string_value) + 1;
    c_device->attributes.push_back(attribute);
    args->client->owned_devices.push_back(*c_device);
    args->client->devices.push_back(c_device);
    args->client->addressable_devices.push_back(c_device);
    args->client->c_device_from_cpp_device.insert({pjrt_device, c_device});
  }

  return nullptr;
}

PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Destroy_Args", PJRT_Client_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->client;
  return nullptr;
}

PJRT_Error* PJRT_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_CLient_ProcessIndex_Args",
      PJRT_Client_ProcessIndex_Args_STRUCT_SIZE, args->struct_size));
  args->process_index = args->client->client->process_index();
  return nullptr;
}

PJRT_Error* PJRT_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_PlatformName_Args",
      PJRT_Client_PlatformName_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view platform_name = args->client->client->platform_name();
  args->platform_name = platform_name.data();
  args->platform_name_size = platform_name.size();
  return nullptr;
}

PJRT_Error* PJRT_Client_PlatformVersion(
    PJRT_Client_PlatformVersion_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_CLient_PlatformVersion_Args",
      PJRT_Client_PlatformVersion_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view platform_version = args->client->client->platform_version();
  args->platform_version = platform_version.data();
  args->platform_version_size = platform_version.size();
  return nullptr;
}

PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Devices_Args", PJRT_Client_Devices_Args_STRUCT_SIZE,
      args->struct_size));
  args->num_devices = args->client->devices.size();
  args->devices = args->client->devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_AddressableDevices_Args",
      PJRT_Client_AddressableDevices_Args_STRUCT_SIZE, args->struct_size));
  args->num_addressable_devices = args->client->addressable_devices.size();
  args->addressable_devices = args->client->addressable_devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_LookupDevice_Args",
      PJRT_Client_LookupDevice_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(itex_xla::PjRtDevice * device,
                        args->client->client->LookupDevice(args->id));
  args->device = GetCDevice(args->client, device);
  return nullptr;
}

PJRT_Error* PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_LookupAddressableDevice_Args",
      PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      itex_xla::PjRtDevice * addressable_device,
      args->client->client->LookupAddressableDevice(args->local_hardware_id));
  args->addressable_device = GetCDevice(args->client, addressable_device);
  return nullptr;
}

// Searches `device_list` for a PJRT_Device* that wraps a provided
// `itex_xla::PjRtDevice *` (`cpp_device`). If a match is found, that
// PJRT_Device* is returned. Otherwise, returns nullptr.
static PJRT_Device* FindDeviceWrapper(
    itex_xla::PjRtDevice* cpp_device,
    absl::Span<PJRT_Device* const> device_list) {
  for (PJRT_Device* device : device_list) {
    if (device->device == cpp_device) {
      return device;
    }
  }
  return nullptr;
}

static void PopulatePjrtExecutableAddressableDevices(
    PJRT_LoadedExecutable* executable) {
  ITEX_CHECK(executable->client != nullptr) << ": client was null";
  absl::Span<itex_xla::PjRtDevice* const> cpp_devices =
      executable->executable->addressable_devices();
  const size_t num_addressable_devices = cpp_devices.size();
  std::vector<PJRT_Device*>& exec_devices = executable->addressable_devices;
  exec_devices.reserve(num_addressable_devices);

  const std::vector<PJRT_Device*>& client_devices =
      executable->client->addressable_devices;

  ITEX_CHECK(client_devices.size() >= num_addressable_devices)
      << ": client->addressable_devices is not bigger than "
         "executable->addressable_devices()";

  for (int i = 0; i < num_addressable_devices; ++i) {
    itex_xla::PjRtDevice* cpp_device = cpp_devices[i];
    PJRT_Device* device = FindDeviceWrapper(cpp_device, client_devices);
    ITEX_CHECK(device != nullptr)
        << ": No PJRT_Device* found in client->addressable_devices"
        << " that wraps executable->addressable_devices()[" << i << "] ("
        << cpp_devices[i] << ")";
    exec_devices.push_back(device);
  }
}

namespace {

itex_xla::StatusOr<itex_xla::CompileOptions> ParseCompileOptions(
    absl::string_view options_str) {
  itex_xla::CompileOptionsProto options_proto;
  // Open source ParseFromString doesn't support string_view.
  if (!options_proto.ParseFromArray(options_str.data(), options_str.size())) {
    return itex::errors::InvalidArgument(
        "PJRT_Client_Compile: failed to deserialize CompileOptionsProto");
  }
  return itex_xla::CompileOptions::FromProto(options_proto);
}

using ProgramVariant =
    std::variant<mlir::OwningOpRef<mlir::ModuleOp>, itex_xla::XlaComputation>;
itex_xla::StatusOr<
    std::variant<mlir::OwningOpRef<mlir::ModuleOp>, itex_xla::XlaComputation>>
ParsePjrtProgram(std::optional<mlir::MLIRContext>& context,
                 PJRT_Program* program) {
  auto format_str = absl::string_view(program->format, program->format_size);
  auto module_str = absl::string_view(program->code, program->code_size);

  if (format_str == itex_xla::kMlirFormat) {
    if (!context.has_value()) {
      context.emplace();
    }
    TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                        itex_xla::ParseMlirModuleString(module_str, *context));

    return ProgramVariant(std::move(module));
  } else if (format_str == itex_xla::kHloFormat) {
    itex_xla::HloModuleProto module_proto;
    // Open source ParseFromString doesn't support string_view.
    if (!module_proto.ParseFromArray(module_str.data(), module_str.size())) {
      return itex::errors::InvalidArgument(
          "PJRT_Client_Compile: failed to deserialize HloModuleProto");
    }
    return ProgramVariant(itex_xla::XlaComputation(module_proto));
  } else {
    return itex::errors::InvalidArgument(ProgramFormatErrorMsg(format_str));
  }
}

mlir::ModuleOp UnpackPjrtProgram(mlir::OwningOpRef<mlir::ModuleOp>& module) {
  return *module;
}
const itex_xla::XlaComputation& UnpackPjrtProgram(
    const itex_xla::XlaComputation& computation) {
  return computation;
}

}  // namespace

PJRT_Error* PJRT_Client_Compile(PJRT_Client_Compile_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Compile_Args", PJRT_Client_Compile_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Program", PJRT_Program_STRUCT_SIZE, args->program->struct_size));

  PJRT_ASSIGN_OR_RETURN(
      itex_xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  std::optional<mlir::MLIRContext> context;
  PJRT_ASSIGN_OR_RETURN(auto module_or_hlo,
                        ParsePjrtProgram(context, args->program));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<itex_xla::PjRtLoadedExecutable> executable,
      std::visit(
          [args, &options](auto& program) {
            return args->client->client->Compile(UnpackPjrtProgram(program),
                                                 options);
          },
          module_or_hlo));
  args->executable =
      new PJRT_LoadedExecutable(std::move(executable), args->client);
  return nullptr;
}

static void PopulateDeviceAssignment(
    int* const device_assignment_buffer, int num_replicas, int num_partitions,
    itex_xla::DeviceAssignment device_assignment) {
  int* iterator = device_assignment_buffer;
  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int partition = 0; partition < num_partitions; ++partition) {
      *(iterator++) = device_assignment(replica, partition);
    }
  }
}

PJRT_Error* PJRT_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args) {
  const int replicas = args->num_replicas;
  const int partitions = args->num_partitions;
  const size_t buffer_size = args->default_assignment_size;
  if (buffer_size < replicas * partitions) {
    itex_xla::Status status = itex::errors::FailedPrecondition(
        absl::StrCat(__func__, ": `default_assignment_size` ", buffer_size,
                     " < `num_replicas * num_partitions`, ", replicas, " * ",
                     partitions, " = ", replicas * partitions));
    return new PJRT_Error{status};
  }

  PJRT_ASSIGN_OR_RETURN(
      itex_xla::DeviceAssignment device_assignment,
      args->client->client->GetDefaultDeviceAssignment(replicas, partitions));

  PopulateDeviceAssignment(args->default_assignment, replicas, partitions,
                           std::move(device_assignment));
  return nullptr;
}

PJRT_Error* PJRT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_BufferFromHostBuffer_Args",
      PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE, args->struct_size));

  absl::Span<const int64_t> dims =
      absl::Span<const int64_t>(args->dims, args->num_dims);

  std::optional<absl::Span<int64_t const>> byte_strides = std::nullopt;
  if (args->byte_strides != nullptr) {
    byte_strides =
        absl::Span<const int64_t>(args->byte_strides, args->num_byte_strides);
  }

  itex_xla::PjRtFuture<itex_xla::Status>::Promise promise =
      itex_xla::PjRtFuture<itex_xla::Status>::CreatePromise();

  std::function<void()> on_done_with_host_buffer = [promise]() mutable {
    promise.Set(itex_xla::OkStatus());
  };

  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<itex_xla::PjRtBuffer> buffer,
      args->client->client->BufferFromHostBuffer(
          args->data, ::itex_xla::ConvertFromPjRtBufferType(args->type), dims,
          byte_strides,
          ::itex_xla::ConvertFromPjRtHostBufferSemantics(
              args->host_buffer_semantics),
          on_done_with_host_buffer, args->device->device));

  args->buffer = new PJRT_Buffer{std::move(buffer), args->client};
  args->done_with_host_buffer = new PJRT_Event{
      itex_xla::PjRtFuture<itex_xla::Status>(std::move(promise))};

  return nullptr;
}

// --------------------------------- Devices -----------------------------------

PJRT_Error* PJRT_Device_Id(PJRT_Device_Id_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes("PJRT_Device_Id_Args",
                                                PJRT_Device_Id_Args_STRUCT_SIZE,
                                                args->struct_size));
  args->id = args->device->device->id();
  return nullptr;
}

PJRT_Error* PJRT_Device_ProcessIndex(PJRT_Device_ProcessIndex_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_ProcessIndex_Args",
      PJRT_Device_ProcessIndex_Args_STRUCT_SIZE, args->struct_size));
  args->process_index = args->device->device->process_index();
  return nullptr;
}

PJRT_Error* PJRT_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_IsAddressable_Args",
      PJRT_Device_IsAddressable_Args_STRUCT_SIZE, args->struct_size));
  args->is_addressable = args->device->device->IsAddressable();
  return nullptr;
}

PJRT_Error* PJRT_Device_Attributes(PJRT_Device_Attributes_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_Attributes_Args", PJRT_Device_Attributes_Args_STRUCT_SIZE,
      args->struct_size));
  // Returns the attributes that were initialized during PJRT_Device creation.
  args->num_attributes = (args->device->attributes).size();
  args->attributes = args->device->attributes.data();

  return nullptr;
}

PJRT_Error* PJRT_Device_Kind(PJRT_Device_Kind_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_Kind_Args", PJRT_Device_Kind_Args_STRUCT_SIZE,
      args->struct_size));

  args->device_kind = args->device->device->device_kind().data();
  args->device_kind_size = args->device->device->device_kind().size();
  return nullptr;
}

PJRT_Error* PJRT_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_LocalHardwareId_Args",
      PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE, args->struct_size));
  args->local_hardware_id = args->device->device->local_hardware_id();
  return nullptr;
}

PJRT_Error* PJRT_Device_DebugString(PJRT_Device_DebugString_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_DebugString_Args", PJRT_Device_DebugString_Args_STRUCT_SIZE,
      args->struct_size));

  args->debug_string = args->device->device->DebugString().data();
  args->debug_string_size = args->device->device->DebugString().size();
  return nullptr;
}

PJRT_Error* PJRT_Device_ToString(PJRT_Device_ToString_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_ToString_Args", PJRT_Device_ToString_Args_STRUCT_SIZE,
      args->struct_size));
  args->to_string = args->device->device->ToString().data();
  args->to_string_size = args->device->device->ToString().size();
  return nullptr;
}

// ------------------------------- Executables ---------------------------------

PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_Destroy_Args", PJRT_Executable_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->executable;
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_LoadedExecutable_Destroy_Args",
      PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->executable;
  return nullptr;
}

PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_Name_Args", PJRT_Executable_Name_Args_STRUCT_SIZE,
      args->struct_size));
  absl::string_view executable_name = args->executable->executable->name();
  args->executable_name = executable_name.data();
  args->executable_name_size = executable_name.size();
  return nullptr;
}

PJRT_Error* PJRT_Executable_NumReplicas(
    PJRT_Executable_NumReplicas_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_NumReplicas_Args",
      PJRT_Executable_NumReplicas_Args_STRUCT_SIZE, args->struct_size));
  args->num_replicas = args->executable->get()->num_replicas();
  return nullptr;
}

PJRT_Error* PJRT_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_NumPartitions_Args",
      PJRT_Executable_NumPartitions_Args_STRUCT_SIZE, args->struct_size));
  args->num_partitions = args->executable->get()->num_partitions();
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_LoadedExecutable_AddressableDevices_Args",
      PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE,
      args->struct_size));

  args->num_addressable_devices = args->executable->addressable_devices.size();
  args->addressable_devices = args->executable->addressable_devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_NumOutputs_Args",
      PJRT_Executable_NumOutputs_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<itex_xla::HloModule>> hlo_modules,
      args->executable->executable->GetHloModules());
  if (hlo_modules.empty()) {
    return new PJRT_Error{
        itex_xla::InvalidArgument("Can't get number of executable outputs, Hlo "
                                  "modules is empty for executable %s.",
                                  args->executable->executable->name())};
  }
  if (hlo_modules.size() != 1) {
    return new PJRT_Error{itex_xla::Unimplemented(
        "MPMD execution not supported by PJRT C API (in "
        "function PJRT_Executable_NumOutputs).")};
  }
  itex_xla::Shape shape = hlo_modules[0].get()->result_shape();
  if (shape.IsTuple()) {
    args->num_outputs = shape.tuple_shapes_size();
  } else {
    // The output size is 1 is it is not a tuple.
    args->num_outputs = 1;
  }
  return nullptr;
}

PJRT_Error* PJRT_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_SizeOfGeneratedCodeInBytes_Args",
      PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE,
      args->struct_size));

  args->size_in_bytes =
      args->executable->executable->SizeOfGeneratedCodeInBytes();
  return nullptr;
}

static itex_xla::Status VerifyOptimizedProgramArgs(
    PJRT_Executable_OptimizedProgram_Args* args) {
  TF_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_OptimizedProgram_Args",
      PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE, args->struct_size));
  TF_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Program", PJRT_Program_STRUCT_SIZE, args->program->struct_size));
  return itex_xla::OkStatus();
}

static itex_xla::StatusOr<std::shared_ptr<itex_xla::HloModule>>
GetOptimizedProgramModule(const PJRT_Executable_OptimizedProgram_Args* args) {
  TF_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<itex_xla::HloModule>> hlo_modules,
      args->executable->executable->GetHloModules());
  if (hlo_modules.empty()) {
    return itex_xla::InvalidArgument(
        "Can't get the optimized program for executable "
        "`%s`: HLO modules is empty.",
        args->executable->executable->name());
  }
  if (hlo_modules.size() > 1) {
    return itex_xla::Unimplemented(
        "Can't get the optimized program for executable "
        "`%s`: MPMD execution is not supported by PJRT C API",
        args->executable->executable->name());
  }
  return std::move(hlo_modules[0]);
}

PJRT_Error* PJRT_Executable_OptimizedProgram(
    PJRT_Executable_OptimizedProgram_Args* args) {
  PJRT_RETURN_IF_ERROR(VerifyOptimizedProgramArgs(args));
  PJRT_Program* program = args->program;
  program->format = kHloWithConfigFormat.data();
  program->format_size = kHloWithConfigFormat.size();
  PJRT_ASSIGN_OR_RETURN(std::shared_ptr<itex_xla::HloModule> hlo_module,
                        GetOptimizedProgramModule(args));
  PJRT_ASSIGN_OR_RETURN(itex_xla::HloModuleProtoWithConfig proto,
                        hlo_module->ToProtoWithConfig());
  if (program->code == nullptr) {
    program->code_size = proto.ByteSizeLong();
    if (program->code_size >= 2ull * 1024 * 1024 * 1024) {
      return new PJRT_Error{itex_xla::ResourceExhausted(
          "%s: HLO program serialization would require more than the max "
          "supported protobuff size of 2 GiB.",
          __func__)};
    }
    return nullptr;
  } else {
    if (program->code_size < proto.ByteSizeLong()) {
      return new PJRT_Error{
          itex_xla::InvalidArgument(
              "`program->code_size` %d < required bytes %d", program->code_size,
              proto.ByteSizeLong()),
      };
    }
    bool succeeded = proto.SerializeToArray(program->code, program->code_size);
    if (!succeeded) {
      return new PJRT_Error{itex_xla::ResourceExhausted(
          "%s: HLO program serialization exceeds max "
          "supported protobuff size of 2 GiB.",
          __func__)};
    }
    return nullptr;
  }
}

PJRT_Error* PJRT_LoadedExecutable_GetCostAnalysis(
    PJRT_LoadedExecutable_GetCostAnalysis_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_LoadedExecutable_GetCostAnalysis_Args",
      PJRT_LoadedExecutable_GetCostAnalysis_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_RETURN_IF_ERROR(
      PopulateExecutableCostAnalysisIfNeeded(args->executable));

  // Output cost analysis data in PJRT_Executable
  args->num_properties = args->executable->cost_analysis_properties.size();
  if (args->num_properties > 0) {
    args->properties = args->executable->cost_analysis_properties.data();
  } else {
    args->properties = nullptr;
  }
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_LoadedExecutable_Delete_Args",
      PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE, args->struct_size));
  args->executable->get()->Delete();
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_LoadedExecutable_IsDeleted_Args",
      PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE, args->struct_size));
  args->is_deleted = args->executable->get()->IsDeleted();
  return nullptr;
}

static std::vector<std::vector<itex_xla::PjRtBuffer*>>
Convert2DCBuffersToCppBuffers(PJRT_Buffer*** c_lists, size_t outer_size,
                              size_t inner_size) {
  std::vector<std::vector<itex_xla::PjRtBuffer*>> cpp_lists;
  cpp_lists.reserve(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    auto& cpp_list = cpp_lists.emplace_back();
    cpp_list.reserve(inner_size);
    for (int j = 0; j < inner_size; ++j) {
      cpp_list.push_back(c_lists[i][j]->buffer.get());
    }
  }
  return cpp_lists;
}

PJRT_Error* PJRT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_LoadedExecutable_Execute_Args",
      PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE, args->struct_size));
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes("PJRT_ExecuteOptions",
                                                PJRT_ExecuteOptions_STRUCT_SIZE,
                                                args->options->struct_size));
  itex_xla::ExecuteOptions options;
  options.launch_id = args->options->launch_id;
  options.strict_shape_checking = true;
  options.arguments_are_tupled = false;
  options.untuple_result = true;
  options.context = nullptr;
  options.multi_slice_config = nullptr;
  std::vector<std::vector<itex_xla::PjRtBuffer*>> cpp_argument_lists =
      Convert2DCBuffersToCppBuffers(args->argument_lists, args->num_devices,
                                    args->num_args);

  if (args->execute_device == nullptr) {
    std::vector<std::vector<std::unique_ptr<itex_xla::PjRtBuffer>>>
        cpp_buffer_lists;
    if (args->device_complete_events != nullptr) {
      std::optional<std::vector<itex_xla::PjRtFuture<itex_xla::Status>>>
          returned_futures;
      returned_futures.emplace();
      PJRT_ASSIGN_OR_RETURN(cpp_buffer_lists,
                            args->executable->get()->Execute(
                                cpp_argument_lists, options, returned_futures));
      for (int i = 0; i < returned_futures->size(); ++i) {
        args->device_complete_events[i] =
            new PJRT_Event{std::move((*returned_futures)[i])};
      }
    } else {
      PJRT_ASSIGN_OR_RETURN(cpp_buffer_lists, args->executable->get()->Execute(
                                                  cpp_argument_lists, options));
    }
    for (int i = 0; i < cpp_buffer_lists.size(); ++i) {
      for (int j = 0; j < cpp_buffer_lists[i].size(); ++j) {
        args->output_lists[i][j] = new PJRT_Buffer{
            std::move(cpp_buffer_lists[i][j]), args->executable->client};
      }
    }
  } else {
    if (args->num_devices != 1) {
      return new PJRT_Error{itex_xla::InvalidArgument(
          "num_devices and corresponding output list sizes must be 1 when "
          "calling PJRT_LoadedExecutable_Execute with non-null execute_device. "
          "Got "
          "num_devices=%i",
          args->num_devices)};
    }
    std::vector<std::unique_ptr<itex_xla::PjRtBuffer>> cpp_buffer_list;
    std::optional<itex_xla::PjRtFuture<itex_xla::Status>> returned_future;
    bool fill_future = args->device_complete_events != nullptr;
    if (args->executable->get()->num_partitions() == 1 &&
        args->executable->get()->num_replicas() == 1) {
      PJRT_ASSIGN_OR_RETURN(
          cpp_buffer_list,
          args->executable->get()->ExecutePortable(
              cpp_argument_lists[0], args->execute_device->device, options,
              returned_future, fill_future));
    } else {
      PJRT_ASSIGN_OR_RETURN(
          cpp_buffer_list,
          args->executable->get()->ExecuteSharded(
              cpp_argument_lists[0], args->execute_device->device, options,
              returned_future, fill_future));
    }
    for (int i = 0; i < cpp_buffer_list.size(); ++i) {
      args->output_lists[0][i] = new PJRT_Buffer{std::move(cpp_buffer_list[i]),
                                                 args->executable->client};
    }
    if (fill_future) {
      args->device_complete_events[0] =
          new PJRT_Event{std::move((*returned_future))};
    }
  }

  return nullptr;
}

PJRT_Error* PJRT_Executable_Serialize(PJRT_Executable_Serialize_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_Serialize_Args",
      PJRT_Executable_Serialize_Args_STRUCT_SIZE, args->struct_size));
  std::string serialization;
  PJRT_ASSIGN_OR_RETURN(serialization,
                        args->executable->executable->SerializeExecutable());

  PJRT_SerializedExecutable* serialized_exec = new PJRT_SerializedExecutable;
  if (serialized_exec == nullptr) {
    return new PJRT_Error{itex_xla::ResourceExhausted(
        "Out of memory for `PJRT_Executable_Serialize()`")};
  }
  serialized_exec->serialized = std::move(serialization);
  args->serialized_executable = serialized_exec;
  return nullptr;
}

PJRT_Error* PJRT_Executable_DeserializeAndLoad(
    PJRT_Executable_DeserializeAndLoad_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_DeserializeAndLoad_Args",
      PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view serialized(args->serialized_executable,
                               args->serialized_executable_size);

  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<itex_xla::PjRtLoadedExecutable> executable,
      args->client->client->DeserializeExecutable(serialized,
                                                  /*options=*/std::nullopt));

  args->loaded_executable =
      new PJRT_LoadedExecutable(std::move(executable), args->client);
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_LoadedExecutable_GetExecutable_Args",
      PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE, args->struct_size));
  args->executable = new PJRT_Executable{args->loaded_executable->executable};
  return nullptr;
}

// -------------------------- Serialized Executables ---------------------------

PJRT_Error* PJRT_SerializedExecutable_Destroy(
    PJRT_SerializedExecutable_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_SerializedExecutable_Destroy_Args",
      PJRT_SerializedExecutable_Destroy_Args_STRUCT_SIZE, args->struct_size));
  if (args->serialized_executable != nullptr) {
    delete args->serialized_executable;
  }
  return nullptr;
}

PJRT_Error* PJRT_SerializedExecutable_Data(
    PJRT_SerializedExecutable_Data_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_SerializedExecutable_Data_Args",
      PJRT_SerializedExecutable_Data_Args_STRUCT_SIZE, args->struct_size));
  args->data = args->serialized_executable->serialized.c_str();
  args->data_size = args->serialized_executable->serialized.size();
  return nullptr;
}

// ---------------------------------- Buffers ----------------------------------
// TODO(b/238999986): Replace this with decomposed shape methods.
PJRT_Error* PJRT_Buffer_OnDeviceTrimmedShape(
    PJRT_Buffer_OnDeviceTrimmedShape_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_OnDeviceTrimmedShape_Args",
      PJRT_Buffer_OnDeviceTrimmedShape_Args_STRUCT_SIZE, args->struct_size));

  const itex_xla::Shape& shape = args->buffer->buffer->on_device_shape();
  args->element_type = shape.element_type();
  ITEXApiConverter::CreateVector(shape.dimensions(), &args->dimensions);
  ITEXApiConverter::CreateVector(shape.dynamic_dimensions(),
                                 &args->dynamic_dimensions);

  if (shape.has_layout()) {
    args->has_layout = true;
    ITEXApiConverter::ToC(shape.layout(), &args->layout);
  } else {
    args->has_layout = false;
  }

  return nullptr;
}

PJRT_Error* PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_Destroy_Args", PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->buffer;
  return nullptr;
}

PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_OnDeviceSizeInBytes_Args",
      PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(args->on_device_size_in_bytes,
                        args->buffer->buffer->GetOnDeviceSizeInBytes());
  return nullptr;
}

PJRT_Error* PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_Device_Args", PJRT_Buffer_Device_Args_STRUCT_SIZE,
      args->struct_size));
  args->device = FindDeviceWrapper(args->buffer->buffer->device(),
                                   args->buffer->client->addressable_devices);
  ITEX_CHECK(args->device != nullptr)
      << "No PJRT_Device* found in the client's `addressable_devices` that "
         "wraps this "
      << args->buffer->buffer->device()->DebugString();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_Delete_Args", PJRT_Buffer_Delete_Args_STRUCT_SIZE,
      args->struct_size));
  args->buffer->buffer->Delete();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_IsDeleted_Args", PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE,
      args->struct_size));
  args->is_deleted = args->buffer->buffer->IsDeleted();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_CopyToDevice_Args",
      PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<itex_xla::PjRtBuffer> dst_buffer,
      args->buffer->buffer->CopyToDevice(args->dst_device->device));
  args->dst_buffer =
      new PJRT_Buffer{std::move(dst_buffer), args->buffer->client};
  return nullptr;
}

PJRT_Error* PJRT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_ToHostBuffer_Args",
      PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE, args->struct_size));

  const itex_xla::Shape& host_shape =
      itex_xla::ShapeUtil::DeviceShapeToHostShape(
          args->src->buffer->on_device_shape());

  size_t host_buffer_size = itex_xla::ShapeUtil::ByteSizeOfElements(host_shape);

  if (args->dst == nullptr) {
    args->dst_size = host_buffer_size;
    return nullptr;
  }

  if (args->dst_size < host_buffer_size) {
    return new PJRT_Error{
        itex_xla::InvalidArgument("`dst_size` must be >= %zu, got %zu.",
                                  host_buffer_size, args->dst_size)};
  }

  auto literal = std::make_unique<itex_xla::MutableBorrowingLiteral>(
      static_cast<char*>(args->dst), host_shape);
  itex_xla::PjRtFuture<itex_xla::Status> future =
      args->src->buffer->ToLiteral(literal.get());

  args->event = new PJRT_Event{std::move(future)};

  args->event->future.OnReady(
      [literal{std::move(literal)}](itex_xla::Status status) {
        /* To keep literal alive */
      });
  return nullptr;
}

PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_IsOnCpu_Args", PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE,
      args->struct_size));
  args->is_on_cpu = args->buffer->buffer->IsOnCpu();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_ReadyEvent_Args", PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE,
      args->struct_size));
  itex_xla::PjRtFuture<itex_xla::Status> wrapped_promise =
      args->buffer->buffer->GetReadyFuture();
  args->event = new PJRT_Event{std::move(wrapped_promise)};
  return nullptr;
}

PJRT_Error* PJRT_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_UnsafePointer_Args",
      PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE, args->struct_size));

  PJRT_ASSIGN_OR_RETURN(args->buffer_pointer,
                        args->buffer->client->client->UnsafeBufferPointer(
                            args->buffer->buffer.get()));
  return nullptr;
}

// -------------------------------- Events -------------------------------------

PJRT_Error* PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_Destroy", PJRT_Event_Destroy_Args_STRUCT_SIZE,
      args->struct_size));

  delete args->event;
  return nullptr;
}

PJRT_Error* PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_IsReady", PJRT_Event_IsReady_Args_STRUCT_SIZE,
      args->struct_size));

  args->is_ready = args->event->future.IsReady();
  return nullptr;
}

PJRT_Error* PJRT_Event_Await(PJRT_Event_Await_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_Await", PJRT_Event_Await_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event* event = args->event;
  event->status.emplace(event->future.Await());
  PJRT_RETURN_IF_ERROR(event->status.value());
  return nullptr;
}

PJRT_Error* PJRT_Event_Error(PJRT_Event_Error_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_Error", PJRT_Event_Error_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event* event = args->event;
  ITEX_CHECK(event->future.IsReady());
  if (!event->status.has_value()) {
    PJRT_Event_Await_Args await_args;
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.priv = nullptr;
    await_args.event = event;
    return PJRT_Event_Await(&await_args);
  }
  PJRT_RETURN_IF_ERROR(event->status.value());
  return nullptr;
}

PJRT_Error* PJRT_Event_OnReady(PJRT_Event_OnReady_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_OnReady", PJRT_Event_OnReady_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event_OnReadyCallback callback = args->callback;
  void* user_arg = args->user_arg;
  auto impl_callback = [callback, user_arg](itex_xla::Status status) -> void {
    PJRT_Error* error = nullptr;
    if (!status.ok()) {
      error = new PJRT_Error{status};
    }
    callback(error, user_arg);
  };
  // impl_callback(itex::Status::OK());
  args->event->future.OnReady(impl_callback);
  return nullptr;
}

// ------------------------------ Device Topology ------------------------------
PJRT_Error* PJRT_DeviceTopology_Create(PJRT_DeviceTopology_Create_Args* args) {
  return new PJRT_Error{itex::errors::Unimplemented(
      "Topology not supported for XPU compilation.")};
}

PJRT_Error* PJRT_DeviceTopology_Destroy(
    PJRT_DeviceTopology_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_DeviceTopology_Destroy_Args",
      PJRT_DeviceTopology_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->topology;
  return nullptr;
}

PJRT_Error* PJRT_DeviceTopology_PlatformName(
    PJRT_DeviceTopology_PlatformName_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_DeviceTopology_PlatformName_Args",
      PJRT_DeviceTopology_PlatformName_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view platform_name = args->topology->topology->platform_name();
  args->platform_name = platform_name.data();
  args->platform_name_size = platform_name.size();
  return nullptr;
}

PJRT_Error* PJRT_DeviceTopology_PlatformVersion(
    PJRT_DeviceTopology_PlatformVersion_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_DeviceTopology_PlatformVersion_Args",
      PJRT_DeviceTopology_PlatformVersion_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view platform_version =
      args->topology->topology->platform_version();
  args->platform_version = platform_version.data();
  args->platform_version_size = platform_version.size();
  return nullptr;
}

PJRT_Error* PJRT_Compile(PJRT_Compile_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Compile_Args", PJRT_Compile_Args_STRUCT_SIZE, args->struct_size));
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Program", PJRT_Program_STRUCT_SIZE, args->program->struct_size));

  itex_xla::PjRtClient* client = nullptr;
  if (args->client != nullptr) {
    client = args->client->client.get();
  }
  PJRT_ASSIGN_OR_RETURN(
      itex_xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  std::optional<mlir::MLIRContext> context;
  PJRT_ASSIGN_OR_RETURN(auto module_or_hlo,
                        ParsePjrtProgram(context, args->program));
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<itex_xla::PjRtExecutable> executable,
                        std::visit(
                            [&](auto& program) {
                              return PjRtCompile(
                                  options, UnpackPjrtProgram(program),
                                  *args->topology->topology, client);
                            },
                            module_or_hlo));
  args->executable = new PJRT_Executable(std::move(executable));
  return nullptr;
}

}  // namespace itex_xla

static bool InitCompilerModule() {
  itex_xla::Compiler::RegisterCompilerFactory(
      stream_executor::gpu::kSyclPlatformId,
      []() { return std::make_unique<itex_xla::gpu::SPIRCompiler>(); });
  return true;
}
static bool compiler_module_initialized = InitCompilerModule();

static std::unique_ptr<itex_xla::TransferManager> CreateDPCPPTransferManager() {
  return std::make_unique<itex_xla::gpu::GpuTransferManager>(
      /*id=*/stream_executor::gpu::kSyclPlatformId,
      /*pointer_size=*/llvm::DataLayout(itex_xla::gpu::spir::DataLayout())
          .getPointerSize(0 /* default address space */));
}

static bool InitTransferManagerModule() {
  itex_xla::TransferManager::RegisterTransferManager(
      stream_executor::gpu::kSyclPlatformId, &CreateDPCPPTransferManager);
  return true;
}

static bool transfer_manager__module_initialized = InitTransferManagerModule();

const PJRT_Api* GetPjrtApi() {
  static PJRT_Api c_api;
  c_api.struct_size = PJRT_Api_STRUCT_SIZE;
  c_api.priv = nullptr;

  c_api.PJRT_Error_Destroy = itex_xla::PJRT_Error_Destroy;
  c_api.PJRT_Error_Message = itex_xla::PJRT_Error_Message;
  c_api.PJRT_Error_GetCode = itex_xla::PJRT_Error_GetCode;

  c_api.PJRT_Event_Destroy = itex_xla::PJRT_Event_Destroy;
  c_api.PJRT_Event_IsReady = itex_xla::PJRT_Event_IsReady;
  c_api.PJRT_Event_Error = itex_xla::PJRT_Event_Error;
  c_api.PJRT_Event_Await = itex_xla::PJRT_Event_Await;
  c_api.PJRT_Event_OnReady = itex_xla::PJRT_Event_OnReady;

  c_api.PJRT_Client_Create = itex_xla::PJRT_Client_Create;
  c_api.PJRT_Client_Destroy = itex_xla::PJRT_Client_Destroy;
  c_api.PJRT_Client_PlatformName = itex_xla::PJRT_Client_PlatformName;
  c_api.PJRT_Client_ProcessIndex = itex_xla::PJRT_Client_ProcessIndex;
  c_api.PJRT_Client_PlatformVersion = itex_xla::PJRT_Client_PlatformVersion;
  c_api.PJRT_Client_Devices = itex_xla::PJRT_Client_Devices;
  c_api.PJRT_Client_AddressableDevices =
      itex_xla::PJRT_Client_AddressableDevices;
  c_api.PJRT_Client_LookupAddressableDevice =
      itex_xla::PJRT_Client_LookupAddressableDevice;
  c_api.PJRT_Client_LookupDevice = itex_xla::PJRT_Client_LookupDevice;
  c_api.PJRT_Client_Compile = itex_xla::PJRT_Client_Compile;
  c_api.PJRT_Client_DefaultDeviceAssignment =
      itex_xla::PJRT_Client_DefaultDeviceAssignment;
  c_api.PJRT_Client_BufferFromHostBuffer =
      itex_xla::PJRT_Client_BufferFromHostBuffer;

  c_api.PJRT_Device_Id = itex_xla::PJRT_Device_Id;
  c_api.PJRT_Device_ProcessIndex = itex_xla::PJRT_Device_ProcessIndex;
  c_api.PJRT_Device_IsAddressable = itex_xla::PJRT_Device_IsAddressable;
  c_api.PJRT_Device_Attributes = itex_xla::PJRT_Device_Attributes;
  c_api.PJRT_Device_Kind = itex_xla::PJRT_Device_Kind;
  c_api.PJRT_Device_LocalHardwareId = itex_xla::PJRT_Device_LocalHardwareId;
  c_api.PJRT_Device_DebugString = itex_xla::PJRT_Device_DebugString;
  c_api.PJRT_Device_ToString = itex_xla::PJRT_Device_ToString;

  c_api.PJRT_Executable_Destroy = itex_xla::PJRT_Executable_Destroy;
  c_api.PJRT_Executable_Name = itex_xla::PJRT_Executable_Name;
  c_api.PJRT_Executable_NumReplicas = itex_xla::PJRT_Executable_NumReplicas;
  c_api.PJRT_Executable_NumPartitions = itex_xla::PJRT_Executable_NumPartitions;
  c_api.PJRT_Executable_NumOutputs = itex_xla::PJRT_Executable_NumOutputs;
  c_api.PJRT_Executable_SizeOfGeneratedCodeInBytes =
      itex_xla::PJRT_Executable_SizeOfGeneratedCodeInBytes;
  c_api.PJRT_Executable_OptimizedProgram =
      itex_xla::PJRT_Executable_OptimizedProgram;
  c_api.PJRT_Executable_Serialize = itex_xla::PJRT_Executable_Serialize;

  c_api.PJRT_LoadedExecutable_Destroy = itex_xla::PJRT_LoadedExecutable_Destroy;
  c_api.PJRT_LoadedExecutable_GetExecutable =
      itex_xla::PJRT_LoadedExecutable_GetExecutable;
  c_api.PJRT_LoadedExecutable_AddressableDevices =
      itex_xla::PJRT_LoadedExecutable_AddressableDevices;
  c_api.PJRT_LoadedExecutable_GetCostAnalysis =
      itex_xla::PJRT_LoadedExecutable_GetCostAnalysis;
  c_api.PJRT_LoadedExecutable_Delete = itex_xla::PJRT_LoadedExecutable_Delete;
  c_api.PJRT_LoadedExecutable_IsDeleted =
      itex_xla::PJRT_LoadedExecutable_IsDeleted;
  c_api.PJRT_LoadedExecutable_Execute = itex_xla::PJRT_LoadedExecutable_Execute;
  c_api.PJRT_Executable_DeserializeAndLoad =
      itex_xla::PJRT_Executable_DeserializeAndLoad;

  c_api.PJRT_SerializedExecutable_Destroy =
      itex_xla::PJRT_SerializedExecutable_Destroy;
  c_api.PJRT_SerializedExecutable_Data =
      itex_xla::PJRT_SerializedExecutable_Data;

  c_api.PJRT_Buffer_Destroy = itex_xla::PJRT_Buffer_Destroy;
  c_api.PJRT_Buffer_OnDeviceTrimmedShape =
      itex_xla::PJRT_Buffer_OnDeviceTrimmedShape;
  c_api.PJRT_Buffer_OnDeviceSizeInBytes =
      itex_xla::PJRT_Buffer_OnDeviceSizeInBytes;
  c_api.PJRT_Buffer_Device = itex_xla::PJRT_Buffer_Device;
  c_api.PJRT_Buffer_Delete = itex_xla::PJRT_Buffer_Delete;
  c_api.PJRT_Buffer_IsDeleted = itex_xla::PJRT_Buffer_IsDeleted;
  c_api.PJRT_Buffer_CopyToDevice = itex_xla::PJRT_Buffer_CopyToDevice;
  c_api.PJRT_Buffer_ToHostBuffer = itex_xla::PJRT_Buffer_ToHostBuffer;
  c_api.PJRT_Buffer_IsOnCpu = itex_xla::PJRT_Buffer_IsOnCpu;
  c_api.PJRT_Buffer_ReadyEvent = itex_xla::PJRT_Buffer_ReadyEvent;
  c_api.PJRT_Buffer_UnsafePointer = itex_xla::PJRT_Buffer_UnsafePointer;

  c_api.PJRT_DeviceTopology_Create = itex_xla::PJRT_DeviceTopology_Create;
  c_api.PJRT_DeviceTopology_Destroy = itex_xla::PJRT_DeviceTopology_Destroy;
  c_api.PJRT_DeviceTopology_PlatformName =
      itex_xla::PJRT_DeviceTopology_PlatformName;
  c_api.PJRT_DeviceTopology_PlatformVersion =
      itex_xla::PJRT_DeviceTopology_PlatformVersion;

  c_api.PJRT_Compile = itex_xla::PJRT_Compile;

  return &c_api;
}

PJRT_Executable::PJRT_Executable(
    std::shared_ptr<itex_xla::PjRtExecutable> executable)
    : executable(std::move(executable)) {}

PJRT_LoadedExecutable::PJRT_LoadedExecutable(
    std::shared_ptr<itex_xla::PjRtLoadedExecutable> executable,
    PJRT_Client* client)
    : executable(std::move(executable)), client(client) {
  itex_xla::PopulatePjrtExecutableAddressableDevices(this);
}
