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

#include "itex/core/compiler/xla/pjrt/pjrt_c_api_helpers.h"

#include <memory>

#include "itex/core/compiler/c/pjrt_c_api.h"
#include "itex/core/compiler/xla/pjrt/pjrt_client.h"
#include "itex/core/compiler/xla/primitive_util.h"
#include "protos/xla_data.pb.h"

namespace itex_xla {

const absl::string_view kHloFormat = "hlo";
const absl::string_view kMlirFormat = "mlir";
const absl::string_view kHloWithConfigFormat = "hlo_with_config";

PJRT_ClientDeleter MakeClientDeleter(const PJRT_Api* api) {
  return [api](PJRT_Client* client) -> void {
    PJRT_Client_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.client = client;

    PJRT_Error* error = api->PJRT_Client_Destroy(&destroy_args);
    // TODO(b/236710439): handle the error and remove this ITEX_CHECK() call
    ITEX_CHECK(error == nullptr);
  };
}

PJRT_ErrorDeleter MakeErrorDeleter(const PJRT_Api* api) {
  return [api](PJRT_Error* error) -> void {
    PJRT_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.error = error;

    api->PJRT_Error_Destroy(&destroy_args);
  };
}

PJRT_BufferDeleter MakeBufferDeleter(const PJRT_Api* api) {
  return [api](PJRT_Buffer* buffer) -> void {
    PJRT_Buffer_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.buffer = buffer;

    itex_xla::LogFatalIfPjrtError(api->PJRT_Buffer_Destroy(&destroy_args), api);
  };
}

PJRT_ExecutableDeleter MakeExecutableDeleter(const PJRT_Api* api) {
  return [api](PJRT_Executable* executable) -> void {
    PJRT_Executable_Destroy_Args args;
    args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.executable = executable;
    itex_xla::LogFatalIfPjrtError(api->PJRT_Executable_Destroy(&args), api);
  };
}

itex_xla::Status PjrtErrorToStatus(const PJRT_Error* error,
                                   const PJRT_Api* api) {
  itex_xla::Status status;
  if (error != nullptr) {
    status = itex_xla::Status(PjrtErrorToStatusCode(error, api),
                              GetPjrtErrorMessage(error, api));
  }
  return status;
}

itex::error::Code PjrtErrorToStatusCode(const PJRT_Error* error,
                                        const PJRT_Api* api) {
  PJRT_Error_GetCode_Args args;
  args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.error = error;
  api->PJRT_Error_GetCode(&args);
  PJRT_Error_Code code = args.code;
  switch (code) {
    case PJRT_Error_Code_CANCELLED:
      return itex::error::CANCELLED;
    case PJRT_Error_Code_UNKNOWN:
      return itex::error::UNKNOWN;
    case PJRT_Error_Code_INVALID_ARGUMENT:
      return itex::error::INVALID_ARGUMENT;
    case PJRT_Error_Code_DEADLINE_EXCEEDED:
      return itex::error::DEADLINE_EXCEEDED;
    case PJRT_Error_Code_NOT_FOUND:
      return itex::error::NOT_FOUND;
    case PJRT_Error_Code_ALREADY_EXISTS:
      return itex::error::ALREADY_EXISTS;
    case PJRT_Error_Code_PERMISSION_DENIED:
      return itex::error::PERMISSION_DENIED;
    case PJRT_Error_Code_RESOURCE_EXHAUSTED:
      return itex::error::RESOURCE_EXHAUSTED;
    case PJRT_Error_Code_FAILED_PRECONDITION:
      return itex::error::FAILED_PRECONDITION;
    case PJRT_Error_Code_ABORTED:
      return itex::error::ABORTED;
    case PJRT_Error_Code_OUT_OF_RANGE:
      return itex::error::OUT_OF_RANGE;
    case PJRT_Error_Code_UNIMPLEMENTED:
      return itex::error::UNIMPLEMENTED;
    case PJRT_Error_Code_INTERNAL:
      return itex::error::INTERNAL;
    case PJRT_Error_Code_UNAVAILABLE:
      return itex::error::UNAVAILABLE;
    case PJRT_Error_Code_DATA_LOSS:
      return itex::error::DATA_LOSS;
    case PJRT_Error_Code_UNAUTHENTICATED:
      return itex::error::UNAUTHENTICATED;
  }
}

PJRT_Error_Code StatusCodeToPjrtErrorCode(itex::error::Code code) {
  switch (code) {
    case itex::error::CANCELLED:
      return PJRT_Error_Code::PJRT_Error_Code_CANCELLED;
    case itex::error::UNKNOWN:
      return PJRT_Error_Code::PJRT_Error_Code_UNKNOWN;
    case itex::error::INVALID_ARGUMENT:
      return PJRT_Error_Code::PJRT_Error_Code_INVALID_ARGUMENT;
    case itex::error::DEADLINE_EXCEEDED:
      return PJRT_Error_Code::PJRT_Error_Code_DEADLINE_EXCEEDED;
    case itex::error::NOT_FOUND:
      return PJRT_Error_Code::PJRT_Error_Code_NOT_FOUND;
    case itex::error::ALREADY_EXISTS:
      return PJRT_Error_Code::PJRT_Error_Code_ALREADY_EXISTS;
    case itex::error::PERMISSION_DENIED:
      return PJRT_Error_Code::PJRT_Error_Code_PERMISSION_DENIED;
    case itex::error::UNAUTHENTICATED:
      return PJRT_Error_Code::PJRT_Error_Code_UNAUTHENTICATED;
    case itex::error::RESOURCE_EXHAUSTED:
      return PJRT_Error_Code::PJRT_Error_Code_RESOURCE_EXHAUSTED;
    case itex::error::FAILED_PRECONDITION:
      return PJRT_Error_Code::PJRT_Error_Code_FAILED_PRECONDITION;
    case itex::error::ABORTED:
      return PJRT_Error_Code::PJRT_Error_Code_ABORTED;
    case itex::error::OUT_OF_RANGE:
      return PJRT_Error_Code::PJRT_Error_Code_OUT_OF_RANGE;
    case itex::error::UNIMPLEMENTED:
      return PJRT_Error_Code::PJRT_Error_Code_UNIMPLEMENTED;
    case itex::error::INTERNAL:
      return PJRT_Error_Code::PJRT_Error_Code_INTERNAL;
    case itex::error::UNAVAILABLE:
      return PJRT_Error_Code::PJRT_Error_Code_UNAVAILABLE;
    case itex::error::DATA_LOSS:
      return PJRT_Error_Code::PJRT_Error_Code_DATA_LOSS;
    case itex::error::OK:
      ITEX_CHECK(false)
          << "Status::OK() cannot be converted to PJRT_Error code, "
             "use nullptr instead";
  }
}

absl::string_view GetPjrtErrorMessage(const PJRT_Error* error,
                                      const PJRT_Api* api) {
  PJRT_Error_Message_Args message_args;
  message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message_args.priv = nullptr;
  message_args.error = error;
  api->PJRT_Error_Message(&message_args);
  return absl::string_view(message_args.message, message_args.message_size);
}

void LogFatalIfPjrtError(PJRT_Error* error, const PJRT_Api* api) {
  std::unique_ptr<PJRT_Error, itex_xla::PJRT_ErrorDeleter> _error(
      error, MakeErrorDeleter(api));
  itex_xla::Status _status = PjrtErrorToStatus(_error.get(), api);
  if (!_status.ok()) {
    ITEX_LOG(FATAL) << "Unexpected error status " << _status.error_message();
  }
}

PJRT_EventDeleter MakeEventDeleter(const PJRT_Api* api) {
  ITEX_CHECK(api != nullptr);
  return [api](PJRT_Event* managed) {
    PJRT_Event_Destroy_Args args;
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.event = managed;

    LogFatalIfPjrtError(api->PJRT_Event_Destroy(&args), api);
  };
}

PJRT_Buffer_Type ConvertToPjRtBufferType(itex_xla::PrimitiveType type) {
  switch (type) {
    case itex_xla::PrimitiveType::PRIMITIVE_TYPE_INVALID:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID;
    case itex_xla::PrimitiveType::PRED:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_PRED;
    case itex_xla::PrimitiveType::S8:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S8;
    case itex_xla::PrimitiveType::S16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S16;
    case itex_xla::PrimitiveType::S32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S32;
    case itex_xla::PrimitiveType::S64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S64;
    case itex_xla::PrimitiveType::U8:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U8;
    case itex_xla::PrimitiveType::U16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U16;
    case itex_xla::PrimitiveType::U32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U32;
    case itex_xla::PrimitiveType::U64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U64;
    case itex_xla::PrimitiveType::F16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F16;
    case itex_xla::PrimitiveType::F32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F32;
    case itex_xla::PrimitiveType::BF16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_BF16;
    case itex_xla::PrimitiveType::F64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F64;
    case itex_xla::PrimitiveType::C64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_C64;
    case itex_xla::PrimitiveType::C128:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_C128;
    default:
      ITEX_CHECK(false)
          << "Element type of the shape is not supported in C API layer: "
          << itex_xla::primitive_util::LowercasePrimitiveTypeName(type);
  }
}

itex_xla::PrimitiveType ConvertFromPjRtBufferType(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type::PJRT_Buffer_Type_PRED:
      return itex_xla::PrimitiveType::PRED;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S8:
      return itex_xla::PrimitiveType::S8;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S16:
      return itex_xla::PrimitiveType::S16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S32:
      return itex_xla::PrimitiveType::S32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S64:
      return itex_xla::PrimitiveType::S64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U8:
      return itex_xla::PrimitiveType::U8;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U16:
      return itex_xla::PrimitiveType::U16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U32:
      return itex_xla::PrimitiveType::U32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U64:
      return itex_xla::PrimitiveType::U64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F16:
      return itex_xla::PrimitiveType::F16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F32:
      return itex_xla::PrimitiveType::F32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_BF16:
      return itex_xla::PrimitiveType::BF16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F64:
      return itex_xla::PrimitiveType::F64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_C64:
      return itex_xla::PrimitiveType::C64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_C128:
      return itex_xla::PrimitiveType::C128;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID:
      ITEX_CHECK(false) << "Buffer type is not supported in C API layer.";
  }
}

const char* HostBufferSemanticsToString(
    itex_xla::PjRtClient::HostBufferSemantics h) {
  switch (h) {
    case itex_xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall:
      return "itex_xla::PjRtClient::HostBufferSemantics::"
             "kImmutableOnlyDuringCall";
    case itex_xla::PjRtClient::HostBufferSemantics::kZeroCopy:
      return "itex_xla::PjRtClient::HostBufferSemantics::kZeroCopy";
    case itex_xla::PjRtClient::HostBufferSemantics::
        kImmutableUntilTransferCompletes:
      return "itex_xla::PjRtClient::HostBufferSemantics::"
             "kImmutableUntilTransferCompletes";
  }
}

PJRT_HostBufferSemantics ConvertToPjRtHostBufferSemantics(
    itex_xla::PjRtClient::HostBufferSemantics buffer_semantics) {
  switch (buffer_semantics) {
    case itex_xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    case itex_xla::PjRtClient::HostBufferSemantics::kZeroCopy:
      return PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kZeroCopy;
    default:
      ITEX_CHECK(false)
          << "Input host buffer semantics is not supported in C API layer: "
          << HostBufferSemanticsToString(buffer_semantics);
  }
}

itex_xla::PjRtClient::HostBufferSemantics ConvertFromPjRtHostBufferSemantics(
    PJRT_HostBufferSemantics buffer_semantics) {
  switch (buffer_semantics) {
    case PJRT_HostBufferSemantics::
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall:
      return itex_xla::PjRtClient::HostBufferSemantics::
          kImmutableOnlyDuringCall;
    case PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kZeroCopy:
      return itex_xla::PjRtClient::HostBufferSemantics::kZeroCopy;
  }
}

}  // namespace itex_xla
